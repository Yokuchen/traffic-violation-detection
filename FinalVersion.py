import numpy as np
import os
import time
import torch
import cv2
import time
import utils
from CV2_RGB_signal import *
from deep_sort_realtime.deepsort_tracker import DeepSort
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
import datetime

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

model_path = "dnn_model/v5m/yolov5m.onnx"
classes_path = "dnn_model/v5m/classes.txt"
# video_path = "videos/red_light_5.mp4"
# video_path = "videos/speeding.mp4"

crosswalkAuto = False
totalViolations = 0
speedingOccurs = False

save_dir = "speeding_test"

formatted_time = ""

def videoName(video_path):
    name = video_path.split("/")[-1]
    return name.split(".")[0]

def extractFrame(video_path, count):
    cap = cv2.VideoCapture(video_path)
    cap.set(1, count)
    ret, frame = cap.read()  
    cv2.imwrite("images/frame%d.jpg" % count, frame) 

def build_model(cuda, model_p):
    model_dir = model_p
    net = cv2.dnn.readNet(model_dir)
    if cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
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
    return cap, input_size, input_fps, frame_height,frame_width

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


#######################################
def sa_v3(distance, box, w, h, id, speed_limit,im0,save_dir, vid_cap, frame_index): # for each frame
    # distance: the actutal estimated distance in meter
    # box: the coordinates of the bounding box as an array
    # w, h : width and height of video (frame)
    # id: car id as an integer
    # speed limit: user defined speed threshold
    # im0: the current frame image
    # save_dir: the directory to be saved to
    # vid_cap: cv2.VideoCapture(path)
    # frame_index: the current frame index (current frame)
    global id_tracker, pass_time, time_tracker,time_tracker2, av_speed, end_time, av_time_1, av_time_2, id_tracker2, id_tracker3, start_time, av_session_1, av_session_2,id_flag, violation_type
    global totalViolations
    global speedingOccurs

    vehicle_vertial_pos = int((int(box[1]) + int(box[3])) / 2)
    vehicle_horizontal_pos = int((int(box[0]) + int(box[2])) / 2)

    if (vehicle_vertial_pos > (h-875)) and (vehicle_horizontal_pos < w) and id not in id_tracker2:
        if (id not in id_tracker) and vehicle_vertial_pos< (h-850):
            id_tracker.append(id)
            f = vid_cap.get(cv2.CAP_PROP_FPS)
            time_in_sec = round(frame_index/f, 5)
            time_tracker[id]=time_in_sec
            start_time[id]= time_tracker[id]
            #print(id_tracker)
        elif id in id_tracker and vehicle_vertial_pos >= (h-850):
            f = vid_cap.get(cv2.CAP_PROP_FPS)
            time_in_sec = round(frame_index/f, 5)
            av_time_1[id] = time_in_sec
            av_session_1[id] = (av_time_1[id] + time_tracker[id])/2
            id_tracker.remove(id)
            #print("pass:"+ f'{id_tracker}')
        elif id not in id_tracker and vehicle_vertial_pos >=(h-300):
            f = vid_cap.get(cv2.CAP_PROP_FPS)
            time_in_sec = round(frame_index/f, 5)
            time_tracker2[id]=time_in_sec # session 2 start time
            id_tracker2.append(id)
            id_flag.append(id)
            #print("pass2:"+ f'{id_tracker2}')

    elif id in id_tracker2 and (vehicle_horizontal_pos < w): # it passed the session 2 start line
        if vehicle_vertial_pos >= (h-150) and id in id_flag: # just need to record once so added the flag
            f = vid_cap.get(cv2.CAP_PROP_FPS)
            time_in_sec = round(frame_index/f, 5)
            av_time_2[id]= time_in_sec
            end_time[id] = av_time_2[id]
            av_session_2[id]= (av_time_2[id]+ time_tracker2[id])/2 
            pass_time[id] = round(av_session_2[id] - av_session_1[id],3)
            av_speed[id] = int((distance/pass_time[id])*3.6)
            #id_tracker2.remove(id)
            id_flag.remove(id)
            if av_speed[id] > speed_limit and id not in id_tracker3:
                speedingOccurs = True
                id_tracker3.append(id)
                x,y,w,h=int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])                   
                img_ = im0.astype(np.uint8)
                crop_img=img_[y:y+ h, x:x + w]                          
                #!!rescale image !!!
                # Get the fps (frame per second)
                f = vid_cap.get(cv2.CAP_PROP_FPS)
                # Calculate the time the violation occurs
                time_in_sec = round(frame_index/f, 2)
                # Save the violated car as a pic and name it
                filename= f'{id}'+ '_'+ f'{violation_type[0]}'+ '_'+f'{time_in_sec}'+'.jpg' # Have to add the time information as well  
                # Save it to a user defined path
                filepath=os.path.join(save_dir, filename)
                completeName = os.path.join(save_dir, videoName(video_path) + '-output-' + formatted_time + '.txt').replace("\\", "/") 
                # os.getcwd().replace("\\", "/")
                #print(filepath)
                # Get the cropped pic and save to path
                file1 = open(completeName, "a")
                toFile = '[Vehicle ID]: '+ f'{id}  '+ '[Violation Type]: '+ f'{violation_type[0]}  '+ '[Time Occured]: '+ f'{time_in_sec}'
                file1.write(toFile+'\n')
                file1.close()
                cv2.imwrite(filepath, crop_img)
        else:
            pass

def draw_reference_line(im0,h):
    global count
    color=(0,255,0) #green

    #The coordinate system starts from the top left, y increases as going down

    upper_section = [{"start_point":(750, h-875), "end_point":(1250, h-875)}, {"start_point":(700, h-850), "end_point":(1300, h-850)}]
    lower_section = [{"start_point":(310, h-550), "end_point":(1700, h-550)}, {"start_point":(200, h-500), "end_point":(1800, h-500)}]
    # start_point = (0, h-350)
    # end_point = (w, h-350)
    # Define the line location
    # cv2.line(im0, upper_line['start_point'], upper_line['end_point'], color, thickness=2)
    # cv2.line(im0, lower_line["start_point"], lower_line["end_point"], color, thickness=2)

    cv2.line(im0, upper_section[0]['start_point'], upper_section[0]['end_point'], (255,0,0), thickness=2)
    cv2.line(im0, upper_section[1]['start_point'], upper_section[1]['end_point'], color, thickness=2)

    cv2.line(im0, lower_section[0]['start_point'], lower_section[0]['end_point'], (255,0,0), thickness=2)
    cv2.line(im0, lower_section[1]['start_point'], lower_section[1]['end_point'], color, thickness=2)
    
    org = (150, 150)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
    thickness=3
#######################################

class Window(Frame):
    def __init__(self, master=None, currentFrame=None, videoPath=None, outputTxt=None, speedingDetect=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.title("GUI")
        self.config(bg="white")
        self.pack(fill=BOTH, expand=1)
        self.video_path = videoPath
        self.currentFrame = currentFrame
        self.image = None
        self.filename = ""
        self.outputTxt = outputTxt
        self.counter = 0
        self.crosswalkAuto = False
        self.speedingDetect = speedingDetect

        # First menu
        file_menu = Menu(master, font=("Times", 20), tearoff=False, bg="white", fg="black", activebackground="#ADD8E6", activeforeground="white")
        file_menu.add_command(label="Open File", command=self.open_file)
        file_menu.add_command(label="Exit", command=self.client_exit)

        # Second menu
        detection_menu = Menu(master, font=("Times", 20), tearoff=False, bg="white", fg="black", activebackground="#ADD8E6", activeforeground="white")
        detection_menu.add_command(label="Auto Detect", command=self.auto_detect)
        detection_menu.add_command(label="Manual", command=self.manual_detect)

        main_menu = Menu(master, font=("Times", 20))
        main_menu.add_cascade(label="File", menu=file_menu)
        main_menu.add_cascade(label="Crosswalk Detection", menu=detection_menu)

        master.config(menu=main_menu)

        img = Image.open("dataset/trafficFlow.jpg")
        img = img.resize((1000, 550), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(img)

        self.text_label = Label(self, text="Traffic Violation Detection", font=("Times", 30), fg="#63c9eb", bg="white")
        self.text_label.config(width=30)
        self.text_label.pack(pady=20)

        self.label = Label(self, image=self.photo)
        self.label.pack()

        self.button = Button(self, text="Start", command=self.button_click, width=10, height=1, font=("Times", 20), bg="#ADD8E6", fg="white", relief="groove", bd=5, activebackground="#c8b0f7", activeforeground="white")
        self.button.pack(side=LEFT, padx=5, pady=5)

        self.button2 = Button(self, text="Result", command=self.button_click2, width=10, height=1, font=("Times", 20), bg="#ADD8E6", fg="white", relief="groove", bd=5, activebackground="#c8b0f7", activeforeground="white")
        self.button2.pack(side=RIGHT, padx=5, pady=5)

        self.button.pack_configure(anchor=CENTER)
        self.button2.pack_configure(anchor=CENTER)

    def auto_detect(self):
        self.crosswalkAuto = True

    def manual_detect(self):
        self.crosswalkAuto = False

    def open_file(self):
        self.filename = filedialog.askopenfilename()
        self.video_path = self.filename

    def button_click(self):
        if(self.filename == ""):
            messagebox.showinfo("Try Again", "No file selected")
        else:
            root.destroy()

    def open_file_2(self):
        text_box = Text(self)
        text_box.pack()
        if self.outputTxt:
            with open(self.outputTxt, "r") as file:
                text_box.delete("1.0", "end")
                text_box.insert("1.0", file.read())

    def open_file_3(self):
        with open(self.outputTxt, 'r') as f:
            file_contents = f.read()
        new_window = Toplevel()
        file_label = Label(new_window, text=file_contents)
        file_label.pack()

        window_width = 1000
        window_height = 500
        screen_width = new_window.winfo_screenwidth()
        screen_height = new_window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2) - 50
        new_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def button_click2(self):
        if self.currentFrame == None or self.video_path == None:
            messagebox.showinfo("Try Again", "No results collected")
        elif not self.speedingDetect:
            if self.counter == 0:
                count = 2
                for frame in self.currentFrame:
                    extractFrame(self.video_path, frame)
                    image_path = r'images/frame%d.jpg' % frame
                    self.update_image(image_path)
                    count += 1
                if self.outputTxt != None:
                    self.button3 = Button(self, text="Output File", command=self.open_file_2, width=10, height=1, font=("Times", 20), bg="#ADD8E6", fg="white", relief="groove", bd=5, activebackground="#c8b0f7", activeforeground="white")
                    self.button3.place(relx=0.5, rely=0.90, anchor="center")
                self.counter += 1
            else:
                messagebox.showinfo("Error", "Results already shown")
        else:
            if self.outputTxt != None:
                self.button3 = Button(self, text="Output File", command=self.open_file_3, width=10, height=1, font=("Times", 20), bg="#ADD8E6", fg="white", relief="groove", bd=5, activebackground="#c8b0f7", activeforeground="white")
                self.button3.place(relx=0.5, rely=0.93, anchor="center")

    def update_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((1000, 500), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

    def client_exit(self):
        exit()

root = Tk()
app = Window(root)

window_width = 1000
window_height = 750

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2) - 50

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.title("Traffic Violation Detection")
root.mainloop()
video_path = app.video_path
crosswalkAuto = app.crosswalkAuto


colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
is_cuda = torch.cuda.is_available()

net = build_model(is_cuda, model_path)
capture, original_size, original_fps, fh, fw = load_capture(video_path)
# if video_path is not None:
#     vidout = cv2.VideoWriter('output/' + videoName(video_path) + '-output.mp4', cv2.VideoWriter_fourcc(*'XVID'),
#                             original_fps, original_size)
vidout = cv2.VideoWriter('output/' + videoName(video_path) + '-output.mp4', cv2.VideoWriter_fourcc(*'XVID'), original_fps, original_size)


current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%m-%d-%H_%M_%S")


##########################################
#Speeding global
count = 0
data = []
id_tracker=[]
id_tracker2=[]
id_tracker3=[]
id_flag=[]
time_tracker={}
time_tracker2={}
av_speed={}
end_time={}
start_time={}
pass_time={}
av_time_1={}
av_time_2={}
av_session_1={}
av_session_2={}
violation_type=['Speeding', 'Red Light']
speeding_run = []
###########################################




# globals
start = time.time_ns()
current_frame = []
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

outputStr = []
trackIdList = []

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
######################
# select areas
if video_path == os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
    wait_zone = [0,0,0,0]
#######################
else:
    if not crosswalkAuto:
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
    else:
        wait_zone = []

######################################
if video_path != os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
    OUTPUT_FILE = videoName(video_path) + '-output-' + formatted_time + '.txt' 
    with open(os.getcwd() +'//output//'+ OUTPUT_FILE,'w+') as output:
        output.writelines("Traffic Violation Detection Report\n\n")

else:
    completeName = os.path.join(save_dir, videoName(video_path) + '-output-' + formatted_time + '.txt').replace("\\", "/") 
    file1 = open(completeName, "a")
    file1.write("Traffic Violation Detection Report\n\n")
    file1.close()
######################################

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

    center_points_cur_frame.clear()
    cur_tuple.clear()
    for i in range(len(boxes)):
        # vehicles
        if class_ids[i] == 0:
            cur_tuple.append((boxes[i], scores[i], class_ids[i]))

        if video_path != os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
            if crosswalkAuto:
                # crosswalks
                if class_ids[i] == 1:
                    wait_zone.append([[int(boxes[i][0]), int(boxes[i][1]) + int(boxes[i][3])], 
                                [int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])], 
                                [int(boxes[i][0] + boxes[i][2]), int(boxes[i][1])], 
                                [int(boxes[i][0]), int(boxes[i][1])]])

        # traffic lights
        if class_ids[i] == 3:
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
                cv2.putText(frame, "Traffic Light: " + str(signal_str[signal]),
                            (int(boxes[i][0]), int(boxes[i][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, circles[2], 2)
            # musk method
            else:
                # ['NONE', 'RED', 'GREEN', 'YELLOW']
                signal, color, pixels = detect_mask(t_signal)
                cv2.rectangle(frame,
                              (int(boxes[i][0]), int(boxes[i][1])),
                              (int(boxes[i][0] + boxes[i][2]),
                               int(boxes[i][1] + boxes[i][3])),
                              color, 2)
                cv2.putText(frame, "Traffic Light: " + str(signal_str[signal]),
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

######################################################
        if video_path == os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
            sa_v3(30,bbox,fw,fh,int(track_id),90,frame,save_dir,capture,frame_count)
            if speedingOccurs:
                totalViolations += 1
                speeding_run.append(int(track_id))
                speedingOccurs = False
       
        if video_path == os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
            pass
##########################################################        
        else:
            if not crosswalkAuto:
                if utils.point_in_polygon(travel[int(track_id)], wait_zone):
                    print(travel[int(track_id)])
                    print("in zone")
                    cv2.circle(frame, (x_track, y_track), radius=0, color=(0, 0, 255), thickness=7)

            else:
                if utils.point_in_polygon(travel[int(track_id)], wait_zone[0]):
                    print(travel[int(track_id)])
                    print("in zone")
                    cv2.circle(frame, (x_track, y_track), radius=0, color=(0, 0, 255), thickness=7)

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
            if not crosswalkAuto:
                if utils.point_in_polygon(travel[int(track_id)], wait_zone) and\
                        travel[int(track_id)][5] > 180:
                    # print(track_id)

                    if int(track_id) not in redlight_track:
                        redlight_track[int(track_id)] = [int(track_id),
                                                        travel[int(track_id)][5],
                                                        0]
            else:
                if utils.point_in_polygon(travel[int(track_id)], wait_zone[0]) and\
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
                rt_tmp = redlight_track.get(int(track_id))
                if rt_tmp[2] > frame_rate:
                    if abs(rt_tmp[1]-travel[int(track_id)][5]) < allowed_turn\
                            and int(track_id) not in redlight_run:
                        current_frame.append(frame_count)
                        redlight_run.append(int(track_id))
                    elif abs(rt_tmp[1]-travel[int(track_id)][5]) > allowed_turn\
                            and int(track_id) in redlight_run:
                        redlight_run.remove(int(track_id))
                        redlight_track.pop(int(track_id))

                if rt_tmp[2] > frame_rate * 2:
                    if int(track_id) in redlight_run:
                        redlight_track.pop(int(track_id))

        else:
            redlight_track.clear()

        count = 0
        if int(track_id) in redlight_run:
            vehicle_color = (0, 0, 255)
            tempStr = "[Car ID]:  " + str(int(track_id)) + "  [Frame Number]:  " + str(current_frame[count]) + "  [Violation Type]:  Running Red Light\n"
            if tempStr not in outputStr:
                    outputStr.append(tempStr)
            count += 1
        
        elif int(track_id) in speeding_run:
            vehicle_color = (0, 0, 255)

        else:
            vehicle_color = (255, 255, 255)

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), vehicle_color, 2)
        
        cv2.putText(frame, "ID:" + str(track_id) + " | Dir:" +
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

###############################################
    if video_path == os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
        draw_reference_line(frame,fh)
###############################################

    temp = redlight_run.copy()
    for id in temp:
        if int(id) not in trackIdList:
            trackIdList.append(int(id))
        else:
            temp.remove(int(id))
    totalViolations += len(temp)

    cv2.putText(frame, f'Frame: {int(frame_count)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Violation Num: {totalViolations}', (350, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 2)
    
    if video_path != os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
        if not crosswalkAuto:
            cv2.polylines(frame, [np.array(wait_zone, np.int32)], True,
                        (15, 220, 10), 3)
        else:
            cv2.polylines(frame, [np.array(wait_zone[0], np.int32)], True,
                        (15, 220, 10), 3)
            
    new = cv2.resize(frame, (1200, 720))
    cv2.imshow('img', new)
    cv2.moveWindow('img', 150, 40)
    vidout.write(frame)
    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
vidout.release()
cv2.destroyAllWindows()

########################################################
print("Total frames: " + str(total_frames))
print("Speeding vehicle id: "+ f'{id_tracker3}')
print("Start_time: "+ f'{start_time}')
print("End_time: "+f'{end_time}')
print("Passed time: "+f'{pass_time}')
print("Average speed: "+ f'{av_speed}')
print("Results saved to:" + f'{save_dir}')
#########################################################
root = Tk()

if video_path != os.getcwd().replace("\\", "/") + "/videos/speeding.mp4":
    with open(os.getcwd() +'//output//'+ OUTPUT_FILE,'a') as output:
        for ele in outputStr:
            output.write(ele)
    outputFile = open(os.getcwd() +'//output//'+ OUTPUT_FILE, 'r')
    outputTxt = os.getcwd() +'\output\\'+ OUTPUT_FILE
    app = Window(root, currentFrame = current_frame, videoPath=video_path, outputTxt = outputTxt, speedingDetect = False)

else:
    outputTxt = os.getcwd() + '\\' + save_dir + '\\' + videoName(video_path) + '-output-' + formatted_time + '.txt'
    print(outputTxt + "HHHHHHHHHHHHHHHHHHHHHH")
    app = Window(root, currentFrame = current_frame, videoPath=video_path, outputTxt = outputTxt, speedingDetect = True)


window_width = 1000
window_height = 750
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2) - 50
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.title("Traffic Violation Detection")
root.mainloop()