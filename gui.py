import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from tkinter import *


from PIL import Image

import numpy as np 
import pandas as pd 
from upload_image import *
import json
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

import tensorflow as tf
from keras.models import load_model
from keras.applications import xception

classes = ['Aloe vera', 'Angel’s_Trumpet', 'Apple_tree', 'Atiso', 'Banana', 'Banyan', 'Cactus', 'Coconut', 'Golden_shower', 'Longan', 'Mango', 'Mangrove', 'Peach_blossom', 'Pine_tree', 'Potato', 'Red_pepper', 'Rhododendron', 'Royal_Poinciana', 'Tomato', 'Triticum aestivum'] 




# Load model
model2 = load_model('res_net.h5')
model2.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

net = load_model('model.h5')
net.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model3 = load_model('effnet.h5')
model3.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# thread1 = threading.Thread(target=net)
# thread2 = threading.Thread(target=model2)

# thread1.start()
# thread2.start()


# thread1.join()
# thread2.join()

win = tk.Tk()
win.geometry("1400x800+100+50")
win['background'] = '#E0E0E0'#58F
win.title('Plant Identification System')
import tkinter as tk
from tkinter import ttk

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# # create logo
# logo = Image.open('./icon.png')
# logo_image = logo.resize((3, 3))
# logo_image = ImageTk.PhotoImage(logo)
# logo_label = Label(win, image= logo_image)
# logo_label.pack()
# logo_label.place(x=0, y=0)


font = ('georgia', 22, 'bold')
title = Label(win, text='Plant Identification System                                                                                                               ')
title.config(bg='medium spring green', fg='dark goldenrod')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 12, 'bold')
text = Text(win, height=20, width=80)

# myscrollbar=Scrollbar(text)
# text.configure(yscrollcommand=myscrollbar.set)
# text.place(x=50, y=120)
# text.config(font=font1)

font = ('black', 10, 'bold')
Imagetext = Label(win, text='IMAGE DISPLAY')
Imagetext.config(bg='#40A944', fg='black')
Imagetext.config(font=font)
Imagetext.config(height=3, width=20)
Imagetext.place(x=660, y=120)

font = ('black', 10, 'bold')
Imagetext2 = Label(win, text='**** PREDICTION ****')
Imagetext2.config(bg='#40A944', fg='black')
Imagetext2.config(font=font)
Imagetext2.config(height=3, width=20)
Imagetext2.place(x=1020, y=120)

font1 = ('times',15, 'bold')
predict = Text(win, height=3, width=20, bg='white')
predict.place(x=1020, y=200)
predict.config(font=font1)


frame = Frame(win, width=500, height=400, bg="white", colormap="new")
frame.pack()
frame.place(x=500, y=200)

from PIL import Image

from keras.preprocessing import image


def get_probabilities(img):
    global predicted_class 
    img1 = img
    img1 = img1.resize((299, 299))
    img1 = np.array(img1)
    # img = img/255
    img1 = np.expand_dims(img1, axis=0)
    img1 = img1.reshape(1,299,299,3)
    img1 = img1.astype('float32')


    img2 = image.load_img(filename, target_size=(224, 224))
    img2 = image.img_to_array(img2)
    img2 = np.expand_dims(img2, axis=0)
    img2 = img2.reshape(1,224,224,3)
    img2 = img2.astype('float32')


    img3 = image.load_img(filename, target_size=(299, 299))
    img3 = image.img_to_array(img3)
    img3 = np.expand_dims(img3, axis=0)
    img3 = img3.reshape(1,299,299,3)
    img3 = img3.astype('float32')

    probabilities1 = net.predict(img1)

    # predicted_class1 = np.argmax(probabilities1)
    # Load data from validation.csv
    validation_data = pd.read_csv('./validation.csv')
    # #Convert predicted label to tree name in validation.csv by using predicted label as index and get actual label
    # predicted_class1 = validation_data['tree_name'][int(predicted_class1)]

    probabilities2 = model2.predict(img2)
    # probabilities = probabilities[0]
    # predicted_class2 = np.argmax(probabilities2)
    # #Convert predicted label to tree name in validation.csv by using predicted label as index and get actual label
    # predicted_class2 = validation_data['tree_name'][int(predicted_class2)]

    probabilities3 = model3.predict(img3)
    
    # probabilities = probabilities[0]
    # predicted_class3 = np.argmax(probabilities3)
    # #Convert predicted label to tree name in validation.csv by using predicted label as index and get actual label
    # predicted_class3 = validation_data['tree_name'][int(predicted_class3)]

    # soft voting for all predicted output
    probabilities = probabilities1 + probabilities2 + probabilities3
    print(probabilities)
    predicted_class = np.argmax(probabilities)
    predicted_class = validation_data['tree_name'][int(predicted_class)]
    print(predicted_class)
    predict.delete("1.0", "end")
    predict.insert(tk.END, predicted_class)
    

def load_file():
    global img
    for widget in frame.winfo_children():
        widget.destroy()
    # f_types = [('Png Files', '*.png')]
    global filename
    filename = filedialog.askopenfilename()
    #try to open the file if it's an image file
    try:
        img = Image.open(filename)
    except:
        print("Not an image file")
        #show error message 
        predict.delete("1.0", "end")
        predict.insert(tk.END, "Not an image file")
        return
    print(filename)
    get_probabilities(img)
    img_resized=img.resize((500,400)) # new width & height
    img=ImageTk.PhotoImage(img_resized)
    label = Label(frame, image = img)
    label.pack()

def upload(predicted_class, file_path, is_private_file=True):
    folder = "/Image/" + predicted_class
    folder = str(folder)
    file_name = predicted_class+".png"
    #read credential from file
    with open('credential.json', 'r') as f:
        credential = json.load(f)
    
    print(folder)
    print(file_name)

    # Initialize ImageKit with your ImageKit Private Key and ImageKit Public Key
    imagekit = ImageKit(
        private_key= credential["private_key"],
        public_key= credential["public_key"],
        url_endpoint= credential["url_endpoint"]
    )


    #Upload image to ImageKit
    options = UploadFileRequestOptions(
        folder=folder,
        is_private_file=is_private_file
    )
    result = imagekit.upload(
        file=open(file_path, 'rb'),
        file_name=file_name,
        options=options
    )

    # Final Result
    print(result)




def close():
   win.destroy()

b1 = tk.Button(win,text='Select',command = lambda:load_file(), bg='#40A944', fg='black')
b1.config(font=('times', 12, 'bold'))
b1.place(x=620, y=620)
tooltip1 = Tooltip(b1, "Select an image to predict")

# font = ('black', 10, 'bold')
# probabilities = Label(win, text='Probabilities of each class')
# probabilities.config(bg='white', fg='dark goldenrod')
# probabilities.config(font=('times', 12, 'bold'))
# probabilities.config(height=3, width=20)
# probabilities.place(x=300, y=600)

exitButton = Button(win, text="Exit", command=close, bg='#40A944', fg='black')
exitButton.place(x=820, y=620)
exitButton.config(font=('times', 12, 'bold'))
#add tooltip
# exitButton = Button(win, text="Exit", command=close, bg='#40A944', fg='black')
# exitButton.place(x=820, y=620)
# exitButton.config(font=('times', 12, 'bold'))
tooltip2= Tooltip(exitButton, "Exit the program")


SaveButton = Button(win, text="Upload", bg='#40A944', fg='black', command=lambda:upload(predicted_class, filename, True))
SaveButton.place(x=720, y=620)
SaveButton.config(font=('times', 12, 'bold'))
tooltip3 =Tooltip(SaveButton, "The upload image is only used for experimental purpose.")

def hover(event):
    event.widget.config(bg="blue")

def hover_leave(event):
    event.widget.config(bg='#40A944')

b1.bind("<Enter>", hover)
b1.bind("<Leave>", hover_leave)
SaveButton.bind("<Enter>", hover)
SaveButton.bind("<Leave>", hover_leave)
    
def exithover(event):
    event.widget.config(bg="red")

def exithover_leave(event):
    event.widget.config(bg='#40A944')

exitButton.bind("<Enter>", exithover)
exitButton.bind("<Leave>", exithover_leave)


win.mainloop()  # Keep the window open

