import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt
import time

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


CLASSES = ['Pedestrian', 'Car', 'Cyclist']

class_num = len(CLASSES)
input_h = 384
input_w = 1280

object_thresh = 0.5

output_h = 96
output_w = 320

downsample_ratio = 4


class ScoreXY:
    def __init__(self, score, c, h, w):
        self.score = score
        self.c = c
        self.h = h
        self.w = w


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]



def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image /= 255.0
    image -= mean
    image /= std

    image = image.transpose((2, 0, 1))
    image = np.ascontiguousarray(image)
    return image


def nms(heatmap, heatmapmax):
    keep_heatmap = []
    for b in range(1):
        for c in range(class_num):
            for h in range(output_h):
                for w in range(output_w):
                    if heatmapmax[c * output_h * output_w + h * output_w + w] == heatmap[c * output_h * output_w + h * output_w + w] and heatmap[c * output_h * output_w + h * output_w + w] > object_thresh:
                        temp = ScoreXY(heatmap[c * output_h * output_w + h * output_w + w], c, h, w)
                        keep_heatmap.append(temp)
    return keep_heatmap


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def postprocess(outputs): 
    offset_2d = outputs[0]
    size_2d = outputs[1]
    heatmap = outputs[2]
    heatmapmax = outputs[3]

    keep_heatmap = nms(heatmap, heatmapmax)
    top_heatmap = sorted(keep_heatmap, key=lambda t: t.score, reverse=True)

    boxes2d = []
    for i in range(len(top_heatmap)):
        if i > 50:
            break

        classId = top_heatmap[i].c
        score = top_heatmap[i].score
        w = top_heatmap[i].w
        h = top_heatmap[i].h

        bx = (w + offset_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        by = (h + offset_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bw = (size_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bh = (size_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio

        xmin = (bx - bw / 2) / input_w
        ymin = (by - bh / 2) / input_h
        xmax = (bx + bw / 2) / input_w
        ymax = (by + bh / 2) / input_h

        keep_flag = 0
        for j in range(len(boxes2d)):
            xmin1 = boxes2d[j].xmin
            ymin1 = boxes2d[j].ymin
            xmax1 = boxes2d[j].xmax
            ymax1 = boxes2d[j].ymax
            if IOU(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.45:
                keep_flag += 1
                break
        if keep_flag == 0:
            bbox = DetectBox(classId, score, xmin, ymin, xmax, ymax)
            boxes2d.append(bbox)
    return boxes2d



def main():
    engine_file_path = './centernet.trt'
    image_path = './test.png'

    origin_image = cv2.imread(image_path)
    image_h, image_w = origin_image.shape[0:2]
    resize_image = cv2.resize(origin_image, (input_w, input_h))
    image = precess_image(resize_image, input_w, input_h)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        inputs[0].host = image
        t1  = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        t2 = time.time()
        print('run tiems time:', (t2 - t1))

        outputs = []
        for i in range(len(trt_outputs)):
            print(len(trt_outputs[i]))
            outputs.append(trt_outputs[i].reshape(-1))
    
        result = postprocess(outputs)
    
        print('detect num is:', len(result))
    
        for i in range(len(result)):
            classid = result[i].classId
            score = result[i].score
            xmin = int(result[i].xmin * image_w + 0.5)
            ymin = int(result[i].ymin * image_h + 0.5)
            xmax = int(result[i].xmax * image_w + 0.5)
            ymax = int(result[i].ymax * image_h + 0.5)
    
            cv2.rectangle(origin_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            ptext = (xmin, ymin)
            title = '%s:%.2f' % (CLASSES[classid], score)
            cv2.putText(origin_image, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
        cv2.imwrite('./test_onnx_result.jpg', origin_image)


if __name__ == '__main__':
    print('This is main ...')
    main()
