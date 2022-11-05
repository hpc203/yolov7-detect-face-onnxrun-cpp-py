import cv2
import numpy as np
import onnxruntime
import argparse


class YOLOv7_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        # Initialize model
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        # self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.session = onnxruntime.InferenceSession(path, sess_options=session_option)
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        self.scale = np.array(
            [self.img_width / self.input_width, self.img_height / self.input_height, self.img_width / self.input_width,
             self.img_height / self.input_height], dtype=np.float32)
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        # input_img, newh, neww, top, left = self.resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) ###也可以使用保持高宽比resize的pad填充

        # Scale input pixel values to 0 to 1
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]
        return input_tensor

    def detect(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.session.run(self.output_names, {input_name: input_tensor for input_name in self.input_names})
        boxes, scores, kpts = self.process_output(outputs)
        return boxes, scores, kpts

    def process_output(self, output):
        predictions = np.squeeze(output[0]).reshape((-1, 21))

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5] *= obj_conf

        # Get the scores
        scores = predictions[:, 5]

        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get bounding boxes for each object
        boxes, kpts = self.extract_boxes(predictions)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold,
                                   self.iou_threshold)
        return boxes[indices], scores[indices], kpts[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4] * self.scale
        kpts = predictions[:, 6:]  ###x1,y1,score1, ...., x5,y5,score5
        kpts *= np.tile(np.array([self.scale[0], self.scale[1], 1], dtype=np.float32), (1, 5))

        # Convert boxes to xywh format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        return boxes_, kpts

    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            label = self.class_names[0]
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 1, (0, 255, 0), thickness=-1)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='selfie.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='onnx_havepost_models/yolov7-lite-e.onnx',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv7_face object detector
    YOLOv7_face_detector = YOLOv7_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, scores, kpts = YOLOv7_face_detector.detect(srcimg)

    # Draw detections
    dstimg = YOLOv7_face_detector.draw_detections(srcimg, boxes, scores, kpts)
    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
