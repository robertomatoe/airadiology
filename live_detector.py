#!/usr/bin/env python3
import sys

sys.path.append("..")

import cv2
import argparse
import numpy as np
import hashlib
import time

from lib.robot import Robot
from lib.camera_v2 import Camera
from lib.ros_environment import ROSEnvironment

# SET YOLO CONFIG
cfg_path = "./custom_config/custom.cfg"
weight_path = "./custom_config/custom.weights"
class_name_path = "./custom_config/classes.txt"

# SET  VARIABLE
center_buffer_ratio = 0.2
move_step = 0.06
move_time_gap = 5

classes = None
COLORS = None


def loadClasses(filename):
    """Load classes into 'classes' list and assign a random but stable color to each class."""
    global classes, COLORS

    try:
        with open(filename, "r") as file:
            classes = [line.strip() for line in file.readlines()]
    except EnvironmentError:
        print("Error: cannot load classes from '{}'.".format(filename))
        quit()

    COLORS = []
    for idx, c in zip(range(0, len(classes)), classes):
        cstr = hashlib.md5(c.encode()).hexdigest()[0:6]
        c = tuple(int(cstr[i : i + 2], 16) for i in (0, 2, 4))
        COLORS.append(c)

    try:
        COLORS[0] = (241, 95, 75)
    except:
        pass


def drawAnchorbox(frame, class_id, confidence, box):
    """Draw an anchorbox identified by `box' onto frame and label it with the class name and confidence."""
    global classes, COLORS

    conf_str = "{:.2f}".format(confidence).lstrip("0")
    label = "{:s} ({:s})".format(classes[class_id], conf_str)
    color = COLORS[class_id]

    lx = max(box[0] + 5, 0)
    ly = max(box[1] + 15, 0)

    cv2.rectangle(frame, box, color, 2)


def main():
    global cfg_path, weight_path, class_name_path, classes, COLORS

    scale = 1.0 / 255  # scale factor: normalize pixel value to 0...1
    meansub = (0, 0, 0)  # we do not use mean subtraction
    outsize = (416, 416)  # output size (=expected input size for YOLOv3)
    conf_threshold = 0.50  # confidence threshold
    nms_threshold = 0.4  # threshold for non-maxima suppression (NMS)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to input image")
    ap.add_argument(
        "-c", "--confidence", required=False, help="confidence threshold", type=float
    )
    ap.add_argument("-m", "--nms", required=False, help="NMS threshold", type=float)

    args = ap.parse_args()

    if args.confidence is not None:
        conf_threshold = args.confidence
    if args.nms is not None:
        nms_threshold = args.nms

    print(
        "Configuration:\n"
        "  Network:\n"
        "    config:      {}\n"
        "    weights:     {}\n"
        "    classes:     {}\n"
        "  Preprocessing:\n"
        "    scale        {:.3f}\n"
        "    mean subtr.  {}\n"
        "    output size  {}\n"
        "  Detection:\n"
        "    conf. thld   {:.3f}\n"
        "    nms. thld    {:.3f}"
        "\n".format(
            cfg_path,
            weight_path,
            class_name_path,
            scale,
            meansub,
            outsize,
            conf_threshold,
            nms_threshold,
        )
    )

    net = cv2.dnn.readNet(weight_path, cfg_path)
    loadClasses(class_name_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    print(
        "Network information:\n"
        "  layer names:\n    {}"
        "  output layers:\n    {}"
        "\n".format(layer_names, output_layers)
    )

    cv2.namedWindow("ObjectDetection")

    ROSEnvironment()

    camera = Camera()
    camera.start()

    robot = Robot()
    robot.start()
    robot_x_rad = 0
    robot_y_rad = 0
    robot.move(robot_x_rad, robot_y_rad)

    while True:
        # Step 1: Focusing target
        while True:
            input_image = camera.getImage()
            if input_image is None:
                print("Error: cannot load image '{}'.".format(args.image))
                quit()

            width = input_image.shape[1]
            height = input_image.shape[0]

            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            input_image_tmp = 255 - input_image
            bgrLower = np.array([180, 180, 180])
            bgrUpper = np.array([255, 255, 255])

            img_mask = cv2.inRange(input_image_tmp, bgrLower, bgrUpper)
            bgrResult = cv2.bitwise_and(input_image_tmp, input_image_tmp, mask=img_mask)

            gray = cv2.cvtColor(bgrResult, cv2.COLOR_BGR2GRAY)

            invGamma = 1.0 / 0.3
            table = np.array(
                [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")

            gray = cv2.LUT(gray, table)

            ret, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(
                thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours is None or len(contours) == 0:
                continue
            
            def biggestRectangle(contours):
                biggest = None
                max_area = 0
                indexReturn = -1
                for index in range(len(contours)):
                    i = contours[index]
                    area = cv2.contourArea(i)
                    if area > 100:
                        peri = cv2.arcLength(i, True)
                        approx = cv2.approxPolyDP(i, 0.1 * peri, True)
                        if area > max_area:
                            biggest = approx
                            max_area = area
                            indexReturn = index
                return indexReturn

            indexReturn = biggestRectangle(contours)
            hull = cv2.convexHull(contours[indexReturn])

            x_list = []
            for i in range(len(hull)):  # x
                x_list.append(hull[i][0][0])

            y_list = []
            for i in range(len(hull)):  # y
                y_list.append(hull[i][0][1])

            obj_center_x = int((max(x_list) + min(x_list)) / 2)
            obj_center_y = int((max(y_list) + min(y_list)) / 2)

            obj_center_pos = (obj_center_x, obj_center_y)

            center_x = int(width / 2)
            center_y = int(height / 2)
            img_center_pos = (center_x, center_y)

            x_buffer = width * center_buffer_ratio
            y_buffer = height * center_buffer_ratio

            img_center_x_min = center_x - int(x_buffer / 2)
            img_center_x_max = center_x + int(x_buffer / 2)
            img_center_y_min = center_y - int(y_buffer / 2)
            img_center_y_max = center_y + int(y_buffer / 2)

            is_lr_move = True
            is_ud_move = True

            # LEFT / RIGHT
            if img_center_x_max < obj_center_x:
                # Should move LEFT <--
                robot_x_rad -= move_step
            elif obj_center_x < img_center_x_min:
                # Should move RIGHT -->
                robot_x_rad += move_step
            else:
                is_lr_move = False

            # UP / DOWN
            if img_center_y_max < obj_center_y:
                # Should move DOWN
                robot_y_rad -= move_step
            elif obj_center_y < img_center_y_min:
                # Should move UP
                robot_y_rad += move_step
            else:
                is_ud_move = False

            if is_lr_move or is_ud_move:
                robot.move(robot_x_rad, robot_y_rad)
            else:
                break

            input_image = cv2.circle(input_image, img_center_pos, 10, (0, 0, 255), 3)
            input_image = cv2.circle(input_image, obj_center_pos, 10, (255, 0, 0), 3)
            input_image = cv2.rectangle(
                input_image,
                (img_center_x_min, img_center_y_min),
                (img_center_x_max, img_center_y_max),
                (255, 0, 0),
                3,
            )
            input_image = cv2.drawContours(input_image, [hull], 0, (0, 255, 0), 3)
            cv2.imshow("ObjectDetection", input_image[..., ::-1])
            key = cv2.waitKey(move_time_gap)
            if key > 0:
                break

        input_image = cv2.circle(input_image, img_center_pos, 10, (0, 0, 255), 3)
        input_image = cv2.circle(input_image, obj_center_pos, 10, (255, 0, 0), 3)
        input_image = cv2.rectangle(
            input_image,
            (img_center_x_min, img_center_y_min),
            (img_center_x_max, img_center_y_max),
            (255, 0, 0),
            3,
        )
        input_image = cv2.drawContours(input_image, [hull], 0, (0, 255, 0), 3)
        cv2.imshow("ObjectDetection", input_image[..., ::-1])
        key = cv2.waitKey(move_time_gap)
        if key > 0:
            break

        print(f"center_x:{center_x}, center_y:{center_y}")
        print(f"obj_center_x:{obj_center_x}, obj_center_y:{obj_center_y}")
        print(f"img_center_x_max:{img_center_x_max}, img_center_y_max:{img_center_y_max}")
        print(f"img_center_x_min:{img_center_x_min}, img_center_y_min:{img_center_y_min}")
        print("[FIT~~~~!!!]")

        # Step 2: Detect Tumor
        input_image = camera.getImage()
        if input_image is None:
            print("Error: cannot load image '{}'.".format(args.image))
            quit()

        width = input_image.shape[1]
        height = input_image.shape[0]

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(
            input_image, scale, outsize, meansub, swapRB=False, crop=False
        )

        net.setInput(blob)
        preds = net.forward(output_layers)

        class_ids = []
        confidence_values = []
        bounding_boxes = []

        for pred in preds:
            for anchorbox in pred:
                class_id = np.argmax(anchorbox[5:])
                confidence = anchorbox[4] * anchorbox[class_id + 5]

                if confidence >= conf_threshold or confidence >= 0:
                    cx = int(anchorbox[0] * width)
                    cy = int(anchorbox[1] * height)
                    dx = int(anchorbox[2] * width)
                    dy = int(anchorbox[3] * height)

                    x = int(cx - dx / 2)
                    y = int(cy - dy / 2)

                    print(
                        "[ ({:3d}/{:3d}) - ({:3d}/{:3d}): Pobj: {:.3f}, Pclass: {:.3f} -> Ptotal: {:.3f}, class: {:s} ]".format(
                            x,
                            y,
                            x + dx,
                            y + dy,
                            anchorbox[4],
                            anchorbox[class_id + 5],
                            confidence,
                            classes[class_id],
                        )
                    )

                    if confidence >= conf_threshold:
                        class_ids.append(class_id)
                        confidence_values.append(float(confidence))
                        bounding_boxes.append([x, y, dx, dy])

        if nms_threshold >= 0:
            indices = cv2.dnn.NMSBoxes(
                bounding_boxes, confidence_values, conf_threshold, nms_threshold
            )
        else:
            indices = range(0, len(bounding_boxes))

        indices = np.reshape(indices, -1)

        if len(indices) == 0 or indices is None:
            continue

        for idx in indices:
            drawAnchorbox(
                input_image, class_ids[idx], confidence_values[idx], bounding_boxes[idx]
            )

        cv2.imshow("ObjectDetection", input_image[..., ::-1])

        key = cv2.waitKey(5000)
        if key > 0:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
