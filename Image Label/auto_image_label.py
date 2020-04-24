# import all the necessary modules

from tkinter import *    #for GUI
from PIL import ImageTk, Image    # for small scale image processing
from tkinter import filedialog    # for uploading folder containing images
from os import listdir      # for appending the path of folder containing images
import tensorflow as tf     # for using pre trained models last stored checkpoint
import cv2   # for small scale image processing ex(image resize)
import os   # for appending path
import numpy as np  # needed for the pre trained model to perform
from pascal_voc_writer import Writer  # for writing image bounding box into xml file in Pascal_voc format

''' 
declare an array that will contain 3 values either 0 or 1
    1 if FRCNN is selected else 0
    1 if SSD is selected else 0
    1 if MobileNet is selected else 0
'''
select_model_arr=[] 
def select_model():
    select_model_arr.clear()
    if radio.get()==1:
        select_model_arr.append(1)
        select_model_arr.append(0)
        select_model_arr.append(0)
    elif radio.get()==2:
        select_model_arr.append(0)
        select_model_arr.append(1)
        select_model_arr.append(0)
    else:
        select_model_arr.append(0)
        select_model_arr.append(0)
        select_model_arr.append(1)

'''
this array will contain 5 values in total:
1) person
'''
select_label_arr=[]
def select_label():
    select_label_arr.clear()
    select_label_arr.append(var1.get())
    select_label_arr.append(var2.get())
    select_label_arr.append(var3.get())
    select_label_arr.append(var4.get())
    select_label_arr.append(var5.get())

d={ 0:"/home/rohan/Downloads/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
    1:"/home/rohan/Downloads/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
    2:"/home/rohan/Downloads/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
    }

pb_fname=[]
def finalize():
    pb_fname.clear()
    pb_fname.append(d[select_model_arr.index(1)])

def print_det():
    img_path=folder_arr[0]+"/"+img_arr[0][curr[0]]
    run_model(pb_fname[0], img_path)

global img
def show_image(path):
    img=Image.open(str(path))
    img=img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor="nw", image=img)
    canvas.image=img
    
folder_arr=[]
img_arr=[]
    
def select_folder():
    #window.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    window.filename=filedialog.askdirectory()
    folder_arr.append(window.filename)
    img_arr.append(listdir(folder_arr[0]))
    show_image(folder_arr[0]+"/"+img_arr[0][0])

index=[0]
curr=[]
def nxt():
    t=index[0]
    if t<4:
        t+=1
        index.clear()
        index.append(t)
        show_image(folder_arr[0]+"/"+img_arr[0][index[0]])
        curr.clear()
        curr.append(index[0])
def prev():
    t=index[0]
    if t>-1:
        t-=1
        index.clear()
        index.append(t)
        show_image(folder_arr[0]+"/"+img_arr[0][index[0]])
        curr.clear()
        curr.append(index[0])

save_xml_arr=[]
def run_model(graph, img):
    def get_frozen_graph(graph_file):
        """Read Frozen Graph file from disk."""
        with tf.gfile.FastGFile(graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

# The TensorRT inference graph file downloaded from Colab or your local machine.
    pb_fname = graph
    trt_graph = get_frozen_graph(pb_fname)

    input_names = ['image_tensor']

# Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')  
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


    IMAGE_PATH = img
    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (300, 300))

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
    tf_input: image[None, ...]
    })
    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    from IPython.display import Image as DisplayImage

# Boxes unit in pixels (image coordinates).
    boxes_pixels = []
    for i in range(num_detections):
        # scale box to image coordinates
        box = boxes[i] * np.array([image.shape[0],
                               image.shape[1], image.shape[0], image.shape[1]])
        box = np.round(box).astype(int)
        boxes_pixels.append(box)
    boxes_pixels = np.array(boxes_pixels)

    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

    class_id=[1,17,18,44,62]
    for i in range(num_detections):
        if (scores[i] > dthresh ):
            if int(classes[i])==1 and select_label_arr[0]==1:
                label='Person'
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                # Draw bounding box.
                image = cv2.rectangle(
                    image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                draw_label(image, (box[1], box[0]), label)
                select_label_arr[0]=0

            elif int(classes[i])==17 and select_label_arr[1]==1:
                label='Cat'
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                # Draw bounding box.
                image = cv2.rectangle(
                    image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                draw_label(image, (box[1], box[0]), label)
                select_label_arr[1]=0

            elif int(classes[i])==18 and select_label_arr[2]==1:
                label='Dog'
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                # Draw bounding box.
                image = cv2.rectangle(
                    image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                draw_label(image, (box[1], box[0]), label)
                select_label_arr[2]=0

            elif int(classes[i])==44 and select_label_arr[3]==1:
                label='Bottle'
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                # Draw bounding box.
                image = cv2.rectangle(
                    image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                draw_label(image, (box[1], box[0]), label)
                select_label_arr[3]=0

            elif int(classes[i])==62 and select_label_arr[4]==1:
                label='Chair'
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                # Draw bounding box.
                image = cv2.rectangle(
                    image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                draw_label(image, (box[1], box[0]), label)
                select_label_arr[4]=0
            
# Save and display the labeled image.
    cv2.imwrite("/home/rohan/out.jpg", image)
    show_image("/home/rohan/out.jpg")
    save_xml_arr.clear()
    img=cv2.imread("/home/rohan/out.jpg")
    height=np.size(img, 1)
    width=np.size(img, 0)
    save_xml_arr.append(width)
    save_xml_arr.append(height)
    save_xml_arr.append(label)
    save_xml_arr.append(box[1])
    save_xml_arr.append(box[0])
    save_xml_arr.append(box[3])
    save_xml_arr.append(box[2])

def save_xml():
    writer=Writer('/home/rohan/out.jpg', save_xml_arr[0], save_xml_arr[1])
    writer.addObject(save_xml_arr[2], save_xml_arr[3], save_xml_arr[4], save_xml_arr[5], save_xml_arr[6])
    writer.save('/home/rohan/out.xml')

    
window = Tk()
window.title("Auto Image Labelling Tool")
window.geometry('560x410')


canvas = Canvas(window, width = 300, height = 300)
canvas.grid(column=1, row=0, columnspan=3, rowspan=15)


select = Button(window, text="Select Folder", command=select_folder)
select.grid(column=0, row=0)
prev = Button(window, text="Prev Image", command=prev)
prev.grid(column=0, row=4)
nxt = Button(window, text="Next Image", command=nxt)
nxt.grid(column=0, row=8)
save = Button(window, text="Save", command=save_xml)
save.grid(column=0, row=12)


lbl=Label(window, text="Select Model").grid(column=4, row=0)
radio = IntVar()  
R1 = Radiobutton(window, text="FRCNN", variable=radio, value=1)  
R1.grid(column=4, row=1)
R2 = Radiobutton(window, text="SSD", variable=radio, value=2)  
R2.grid(column=4, row=2)  
R3 = Radiobutton(window, text="Mobilenet", variable=radio, value=3)  
R3.grid(column=4, row=3) 
model = Button(window, text="Select Model", command=select_model)
model.grid(column=4, row=4)
s='Detection'+'\n'+'Threshold'
lbl=Label(window, text=s).grid(column=4, row=5)
e1=Entry(window, width=3).grid(column=5, row=5)
dthresh=0.05
#thresh = Button(window, text="Set", command=set_threshold)
#thresh.grid(column=6, row=5)

lbl=Label(window, text="Label Filter").grid(column=4, row=6)
var1 = IntVar()
Checkbutton(window, text="Person", variable=var1).grid(column=4, row=7)
var2 = IntVar()
Checkbutton(window, text="Cat", variable=var2).grid(column=4, row=8)
var3 = IntVar()
Checkbutton(window, text="Dog", variable=var3).grid(column=4, row=9)
var4 = IntVar()
Checkbutton(window, text="Bottle", variable=var4).grid(column=4, row=10)
var5 = IntVar()
Checkbutton(window, text="Chair", variable=var5).grid(column=4, row=11)
model = Button(window, text="Label Filter", command=select_label)
model.grid(column=4, row=12)


dt = Button(window, text="Finalize", command=finalize)
dt.grid(column=2, row=15)

pr = Button(window, text="Detect", command=print_det)
pr.grid(column=2, row=16)

window.mainloop()
