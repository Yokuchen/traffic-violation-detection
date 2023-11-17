import numpy as np
import glob
import os
import time
import torch
import cv2
import time
import sys
import math
import utils
from CV2_RGB_signal import *
from deep_sort_realtime.deepsort_tracker import DeepSort

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
model_path = "dnn_model/v5m/yolov5m.onnx"
classes_path = "dnn_model/v5m/classes.txt"
video_path = "dataset/sfm_3.mp4"


def build_model(cuda, model_p):
    model_dir = model_p
    net = cv2.dnn.readNet(model_dir)
    if cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def load_capture(video_p):
    cap = cv2.VideoCapture(video_p)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_size = (frame_width, frame_height)
    return cap, input_size, input_fps


def load_classes():
    class_list = []
    with open(classes_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


class_list = load_classes()


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > .25:
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[
                    3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(format_frame):
    row, col, _ = format_frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = format_frame
    return result


def plot_boxes(boxes_o, ids, pb_img, height, width, confidence=0.3):
    labels, cord = ids, boxes_o
    detects = []

    n = len(labels)
    x_shape, y_shape = width, height

    for i in range(n):
        row = cord[i]

        if row[4] >= confidence:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(
                row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)

            if class_list[i]:
                # == 'car'
                x_center = x1 + (x2 - x1)
                y_center = y1 + ((y2 - y1) / 2)

                tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)],
                                  dtype=np.float32)
                confidence = float(row[4].item())
                feature = class_list[i]

                detects.append(([x1, y1, int(x2 - x1), int(y2 - y1)],
                                row[4].item(), class_list[i]))

    return pb_img, detects


point_matrix = []
click_counter = 0


def click_event(event, x_ce, y_ce, flags, params):
    # checking for left mouse clicks
    # global click_counter, point_matrix
    global click_counter, point_matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x_ce, ' ', y_ce)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x_ce) + ',' +
                    str(y_ce), (x_ce, y_ce), font,
                    1, (255, 0, 0), 2)
        cv2.circle(img, (x_ce, y_ce), 1, (0, 255, 255),
                   2)
        cv2.imshow('selection', img)
        point_matrix.append((x_ce, y_ce))
        click_counter = click_counter + 1

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x_ce, ' ', y_ce)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y_ce, x_ce, 0]
        g = img[y_ce, x_ce, 1]
        r = img[y_ce, x_ce, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x_ce, y_ce), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('selection', img)
        point_matrix.append((x_ce, y_ce))
        click_counter = click_counter + 1
    return [x_ce, y_ce]


colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
# print(torch.cuda.is_available())
# print(torch.version.cuda)
is_cuda = torch.cuda.is_available()

net = build_model(is_cuda, model_path)
capture, original_size, original_fps = load_capture(video_path)
vidout = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'XVID'),
                         original_fps, original_size)

# globals
start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1

center_points_cur_frame = []
center_points_prev_frame = []
tracking_objects = {}
track_id = 0
cur_tuple = []
tracks = []
signal = 0
signal_str = ['NONE', 'RED', 'GREEN', 'YELLOW']
redlight_track = {}
redlight_run = []
allowed_turn = 45
Hough = False
travel = {}
presented = []
travel_maintain = []
check_pos = False
clear_cpc = False
check_pos_count = 0
frame_rate = capture.get(cv2.CAP_PROP_FPS)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# globals end

object_tracker = DeepSort(max_age=5,
                          n_init=2,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

# select areas
_, img = capture.read()
cv2.imshow('selection', img)
print("select 4 points (clockwise), press any key to exit")
cv2.setMouseCallback('selection', click_event)
# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()

print(point_matrix)
wait_zone = [point_matrix[0], point_matrix[1], point_matrix[2], point_matrix[3]]

# detect on frame
while True:

    _, frame = capture.read()

    if frame is None:
        print("End of stream")
        break

    check_pos_count += 1
    """
    # skip 1 frame out of 3
    if count % 3 != 0:
        continue
    """
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    (class_ids, scores, boxes) = wrap_detection(inputImage, outs[0])

    frame_count += 1
    total_frames += 1

    # Detect objects on frame

    # Point current frame
    center_points_cur_frame.clear()
    # (class_ids, scores, boxes) = od.detect(frame)

    # bbs expected to be a list of detections, each in tuples of
    # ( [left,top,w,h], confidence, detection_class )
    # print(boxes[1])
    # cur_tuple = (boxes, scores, class_ids)
    cur_tuple.clear()
    for i in range(len(boxes)):
        # vehicles
        if class_ids[i] == 0:
            cur_tuple.append((boxes[i], scores[i], class_ids[i]))

        # traffic lights
        if class_ids[i] == 3:
            # print((int(boxes[i][1]), int(boxes[i][1] + boxes[i][3])))
            t_signal = frame[
                       int(boxes[i][1]):int(boxes[i][1] + boxes[i][3]),
                       int(boxes[i][0]):int(boxes[i][0] + boxes[i][2])]
            # tl br xy
            low_leftx = int(boxes[i][0])
            low_lefty = int(boxes[i][1])
            # hough method
            if Hough:
                signal, circle_cords, circles = detect_signal(t_signal)
                if signal != 0:
                    circle = (circle_cords[0] + low_leftx,
                              circle_cords[1] + low_lefty)
                    cv2.circle(frame, circle, circles[1], circles[2],
                               circles[3])

                cv2.rectangle(frame,
                              (int(boxes[i][0]), int(boxes[i][1])),
                              (int(boxes[i][0] + boxes[i][2]),
                               int(boxes[i][1] + boxes[i][3])),
                              circles[2], 2)
                cv2.putText(frame, "traffic light: " + str(signal_str[signal]),
                            (int(boxes[i][0]), int(boxes[i][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, circles[2], 2)
            # musk method
            else:
                # ['NONE', 'RED', 'GREEN', 'YELLOW']
                signal, color, pixels = detect_mask(t_signal)
                # print(pixels)
                cv2.rectangle(frame,
                              (int(boxes[i][0]), int(boxes[i][1])),
                              (int(boxes[i][0] + boxes[i][2]),
                               int(boxes[i][1] + boxes[i][3])),
                              color, 2)
                cv2.putText(frame, "traffic light: " + str(signal_str[signal]),
                            (int(boxes[i][0]), int(boxes[i][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)

    # print(cur_tuple)
    if len(cur_tuple) != 0:
        tracks = object_tracker.update_tracks(cur_tuple,
                                              frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        # print(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # xyl xyr
        x_track = int((int(bbox[0]) + int(bbox[2])) / 2)
        y_track = int((int(bbox[1]) + int(bbox[3])) / 2)
        cv2.circle(frame, (x_track, y_track), radius=0, color=(255, 0, 0),
                   thickness=7)
        # speed_detection()
        if int(track_id) not in presented:
            travel[int(track_id)] = [track_id,
                                     x_track, y_track, x_track, y_track, -1, 0]
            presented.append(int(track_id))

        if utils.point_in_polygon(travel[int(track_id)], wait_zone):
            # print(travel[int(track_id)])
            # print("in zone")
            cv2.circle(frame, (x_track, y_track), radius=0, color=(0, 0, 255),
                       thickness=7)

        if check_pos_count > (frame_rate / 10):
            if int(track_id) in presented:
                travel_tmp = travel.get(int(track_id))
                # print(travel_tmp)
                if travel_tmp[5] != -1:
                    # move direction
                    # (id, curr[12], prev[34], dir, lifetime)
                    """
                    travel_dir = utils.angle_of_vectors(x_track - travel_tmp[3],
                                                        y_track - travel_tmp[4],
                                                        0, 1)
                    """
                    # degree in top left +y -x
                    travel_dir = utils.vector_to_deg([x_track - travel_tmp[3],
                                                      y_track - travel_tmp[4]])
                    travel[int(track_id)][1] = x_track
                    travel[int(track_id)][2] = y_track
                    travel[int(track_id)][3] = travel_tmp[1]
                    travel[int(track_id)][4] = travel_tmp[2]
                    travel[int(track_id)][5] = travel_dir
                    travel[int(track_id)][6] = 0
                    '''
                    travel[int(track_id)] = [int(track_id),
                                             x_track, y_track,
                                             travel_tmp[3], travel_tmp[4],
                                             travel_dir,
                                             0]
                    '''
                    # travel[] updated
                clear_cpc = True
        else:
            if int(track_id) in presented:
                travel_tmp = travel.get(int(track_id))
                if travel_tmp[5] != -1:
                    travel[int(track_id)][1] = x_track
                    travel[int(track_id)][2] = y_track

        if travel[int(track_id)][5] == -1:
            travel[int(track_id)][5] = 1
        # print(travel[int(track_id)])

        # run through
        # ['NONE', 'RED', 'GREEN', 'YELLOW']

        if signal == 1:

            if utils.point_in_polygon(travel[int(track_id)], wait_zone) and\
                    travel[int(track_id)][5] > 180:
                # print(track_id)

                if int(track_id) not in redlight_track:
                    redlight_track[int(track_id)] = [int(track_id),
                                                     travel[int(track_id)][5],
                                                     0]
            if travel[int(track_id)][5] > 180 and\
                    int(track_id) in redlight_track:
                redlight_track[int(track_id)][2] += 1

            if int(track_id) in redlight_track:
                # print(redlight_track)

                rt_tmp = redlight_track.get(int(track_id))
                if rt_tmp[2] > frame_rate:
                    if abs(rt_tmp[1]-travel[int(track_id)][5]) < allowed_turn\
                            and int(track_id) not in redlight_run:
                        redlight_run.append(int(track_id))
                        # print(redlight_run)
                    elif abs(rt_tmp[1]-travel[int(track_id)][5]) > allowed_turn\
                            and int(track_id) in redlight_run:
                        redlight_run.remove(int(track_id))
                        redlight_track.pop(int(track_id))

                if rt_tmp[2] > frame_rate * 2:
                    if int(track_id) in redlight_run:
                        redlight_track.pop(int(track_id))

        else:
            redlight_track.clear()

        if int(track_id) in redlight_run:
            vehicle_color = (0, 0, 255)
        else:
            vehicle_color = (255, 255, 255)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), vehicle_color, 2)
        cv2.putText(frame, "ID:" + str(track_id) + " |dir:" +
                    str(int(travel[int(track_id)][5])),
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        if travel[int(track_id)][5] != 1:
            arr_start_point = (int((x_track + travel[int(track_id)][3]) / 2),
                               int((y_track + travel[int(track_id)][4]) / 2))
            direction_vector = utils.deg_to_unive(travel[int(track_id)][5])
            arr_end_point = [int(x_track - direction_vector[0] * 30),
                             int(y_track + direction_vector[1] * 30)]

            cv2.arrowedLine(frame, (x_track, y_track), arr_end_point,
                            (0, 255, 0), 2, tipLength=0.3)

    # destroy on lifetime
    for x in list(travel.keys()):
        travel_tmp = travel.get(x)
        # print(travel_tmp)
        travel_tmp[6] += 1
        travel[x][6] += 1
        if travel[x][6] > 3 * frame_rate:
            travel.pop(x)
            # print(x, presented)
            presented.remove(x)

    # print(check_pos_count)
    if clear_cpc:
        check_pos_count = 0
        clear_cpc = False

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        # print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=-1)

    cv2.putText(frame, f'Frame: {int(frame_count)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(wait_zone, np.int32)], True,
                  (15, 220, 10), 3)
    cv2.imshow('img', frame)
    vidout.write(frame)
    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
vidout.release()
cv2.destroyAllWindows()

print("Total frames: " + str(total_frames))
