# from tkinter import *
# import time
# import re
# #Import scikit-learn metrics module for accuracy calculation
# import pickle
# import numpy as np
# from PIL import Image, ImageTk  
# from tensorflow.keras.models import load_model
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import pickle
# attack_list=['Normal','DOS','Probe','R2L','U2R']
# loaded_model1=load_model('alexmodel.model')
# print("Loaded Alexnet model from disk")

# json_file = open('model_lstm1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model_lstm = model_from_json(loaded_model_json)
# # load weights into new model
# model_lstm.load_weights("lstm_weight1.h5")
# print("Loaded LSTM model from disk")

# minivgg_model=load_model('model.model')
# print("Loaded MiniVGG model from disk")

# def pp(a):
#     global mylist
#     mylist.insert(END, a)
#     import tkinter as tk
# from tkinter import messagebox

# def check_credentials():
#     username = username_entry.get()
#     password = password_entry.get()
    
#     # Check if the username and password are valid
#     if username == "admin" and password == "password":
#         messagebox.showinfo("Login Successful", "Welcome, admin!")
        
#         # Add code here to continue with the rest of your program
#     else:
#         messagebox.showerror("Login Failed", "Invalid username or password")

# # Create the main window
# import tkinter as tk
# window = tk.Tk()
# window.title("Login Page")

# # Create and pack the username label and entry
# username_label = tk.Label(window, text="Username:")
# username_label.pack()
# username_entry = tk.Entry(window)
# username_entry.pack()

# # Create and pack the password label and entry
# password_label = tk.Label(window, text="Password:")
# password_label.pack()
# password_entry = tk.Entry(window, show="*")  # Show * instead of the actual password
# password_entry.pack()

# # Create and pack the login button
# login_button = tk.Button(window, text="Login", command=check_credentials)
# login_button.pack()

# # Start the Tkinter event loop
# window.mainloop()
# import tkinter as tk
# from tkinter import messagebox

# # Function to handle login button click event
# def login():
#     username = entry_username.get()
#     password = entry_password.get()

#     # Perform authentication/validation checks
#     if username == "admin" and password == "password":
#         # If authentication is successful, navigate to the index page
#         root.destroy()  # Close the login page window
#         open_index_page()  # Open the index page window
#     else:
#         messagebox.showerror("Login Failed", "Invalid username or password")

# # Function to open the index page
# def open_index_page():
#     index_root = tk.Tk()
#     index_root.title("Index Page")

#     # Create index page contents
#     label = tk.Label( userHome(), text="Welcome to the Index Page!")
#     label.pack()

#     index_root.mainloop()

# # Create the login page
# root = tk.Tk()
# root.title("Login Page")

# # Username label and entry field
# label_username = tk.Label(root, text="Username:")
# label_username.pack()
# entry_username = tk.Entry(root)
# entry_username.pack()

# # Password label and entry field
# label_password = tk.Label(root, text="Password:")
# label_password.pack()
# entry_password = tk.Entry(root, show="*")
# entry_password.pack()

# # Login button
# button_login = tk.Button(root, text="Login", command=login)
# button_login.pack()

# root.mainloop()


# def predict(val):
#     print(val)
#     relist=[]
#     list1=val.split(",")
#     floatlist=[float(x)for x in list1]
#     print(list1)
#     print(floatlist)
#     text=[]
#     text.append(floatlist)
#     featalex=np.array(text)
#     alex_scale=pickle.load( open( "norm.pkl", "rb" ) )
#     featalex=alex_scale.transform(featalex)
#     featalex=np.reshape(featalex,(1,20,2,1))
#     preds = loaded_model1.predict(featalex)[0]
#     alex_result=np.argmax(preds)
#     print("alexnet result==>",alex_result)
#     relist.append(alex_result)
        
#     lstm_trans=pickle.load( open( "minmaxlstm.pkl", "rb" ) )
#     X_test=lstm_trans.transform(text)
#     print(X_test)
#     feat=np.array(X_test)
#     print(feat.shape)
#     feat=np.reshape(feat,(1,40,1))
#     y=model_lstm.predict(feat)
#     print(y)
#     lstm_result=round(y[0][0])
#     print("LSTM result==>",lstm_result)
#     relist.append(lstm_result)
    
#     featvgg=np.array(text)
#     vgg_scale=pickle.load( open( "norm.pkl", "rb" ) )
#     featvgg=vgg_scale.transform(featvgg)
#     featvgg=np.reshape(featvgg,(1,20,2,1))
#     preds = minivgg_model.predict(featvgg)[0]
#     result_vgg=np.argmax(preds)
#     print("Mini VGG result==>",result_vgg)

#     relist.append(result_vgg)

#     print(relist)
#     finalindex=max(relist, key = relist.count)
#     print(finalindex)

#     print("Intrusion type==>",attack_list[finalindex])
    
#     root.after(500, lambda : pp("Input data received"))
#     root.after(1700, lambda : pp("Preprocessing started"))
#     root.after(2000, lambda : pp("Feature scaling"))
#     root.after(2300, lambda : pp("Loaded LSTM,AlexNet and Mini VGGNet models "))
#     root.after(2500, lambda : pp("Prediction using Loaded model"))
#     root.after(2800, lambda : pp("Attack type: "+attack_list[finalindex]))
#     root.after(3000, lambda : pp("============================"))
#     root.after(3100, lambda :shrslt.config(text=attack_list[finalindex],fg="red"))


        
    
    
    
# def userHome():
    # global root, mylist,shrslt
    # root = Tk()
    # root.geometry("1200x700+0+0")
    # root.title("Home Page")

    # image = Image.open("nbg.png")
    # image = image.resize((1200, 700), Image.ANTIALIAS) 
    # pic = ImageTk.PhotoImage(image)
    # lbl_reg=Label(root,image= pic,anchor=CENTER)
    # lbl_reg.place(x=0,y=0)
   
# #-----------------INFO TOP------------
# root = Tk()
# lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="NETWORK INTRUSION DETECTION",fg="white",bg="black",bd=10,anchor='w')
# lblinfo.place(x=350,y=50)
    
# lblinfo3 = Label(root, font=( 'aria' ,20 ),text="Enter Network features",fg="#000955",anchor='w')
# lblinfo3.place(x=780,y=310)
# E1 = Entry(root,width=30,font="veranda 20")
# E1.place(x=650,y=360)

# lblinfo4 = Label(root, font=( 'aria' ,17 ),text="Process",fg="white",anchor='w',bg="black")
# lblinfo4.place(x=180,y=250)
# mylist = Listbox(root,width=50, height=20,bg="white")

# mylist.place( x = 80, y = 300 )
# btntrn=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Detect", bg="green",command=lambda:predict(E1.get()))
# btntrn.place(x=800, y=420)
# # btnhlp=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Help?", bg="blue",command=lambda:predict(E1.get()))
# # btnhlp.place(x=50, y=450)
# rslt = Label(root, font=( 'aria' ,20, ),text="Attack type :",fg="black",bg="white",anchor=W)
# rslt.place(x=630,y=580)
# shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
# shrslt.place(x=790,y=580)

# def qexit():
#     root.destroy()
     

#     root.mainloop()


# userHome()

# from tkinter import *
# import tkinter.messagebox as messagebox
# from PIL import Image, ImageTk
# import pickle
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np

# attack_list = ['Normal', 'DOS', 'Probe', 'R2L', 'U2R']
# loaded_model1 = load_model('alexmodel.model')
# print("Loaded Alexnet model from disk")

# json_file = open('model_lstm1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model_lstm = tf.keras.models.model_from_json(loaded_model_json)
# model_lstm.load_weights("lstm_weight1.h5")
# print("Loaded LSTM model from disk")

# minivgg_model = load_model('model.model')
# print("Loaded MiniVGG model from disk")

# def pp(a):
#     mylist.insert(END, a)

# def predict(val):
#     print(val)
#     relist = []
#     list1 = val.split(",")
#     floatlist = [float(x) for x in list1]
#     print(list1)
#     print(floatlist)
#     text = []
#     text.append(floatlist)
#     featalex = np.array(text)
#     alex_scale = pickle.load(open("norm.pkl", "rb"))
#     featalex = alex_scale.transform(featalex)
#     featalex = np.reshape(featalex, (1, 20, 2, 1))
#     preds = loaded_model1.predict(featalex)[0]
#     alex_result = np.argmax(preds)
#     print("alexnet result==>", alex_result)
#     relist.append(alex_result)

#     lstm_trans = pickle.load(open("minmaxlstm.pkl", "rb"))
#     X_test = lstm_trans.transform(text)
#     print(X_test)
#     feat = np.array(X_test)
#     print(feat.shape)
#     feat = np.reshape(feat, (1, 40, 1))
#     y = model_lstm.predict(feat)
#     print(y)
#     lstm_result = round(y[0][0])
#     print("LSTM result==>", lstm_result)
#     relist.append(lstm_result)

#     featvgg = np.array(text)
#     vgg_scale = pickle.load(open("norm.pkl", "rb"))
#     featvgg = vgg_scale.transform(featvgg)
#     featvgg = np.reshape(featvgg, (1, 20, 2, 1))
#     preds = minivgg_model.predict(featvgg)[0]
#     result_vgg = np.argmax(preds)
#     print("Mini VGG result==>", result_vgg)

#     relist.append(result_vgg)

#     print(relist)
#     finalindex = max(relist, key=relist.count)
#     print(finalindex)

#     print("Intrusion type==>", attack_list[finalindex])

#     pp("Input data received")
#     pp("Preprocessing started")
#     pp("Feature scaling")
#     pp("Loaded LSTM, AlexNet, and Mini VGGNet models")
#     pp("Prediction using Loaded model")
#     pp("Attack type: " + attack_list[finalindex])
#     pp("============================")
#     shrslt.config(text=attack_list[finalindex], fg="red")

# def check_credentials():
#     username = username_entry.get()
#     password = password_entry.get()

#     # Check if the username and password are valid
#     if username == "admin" and password == "password":
#         messagebox.showinfo("Login Successful", "Welcome, admin!")
#         userHome()
#     else:
#         messagebox.showerror("Login Failed", "Invalid username or password")

# def userHome():
#     global root, mylist, shrslt
#     root.destroy()  # Close the login page window

#     root = Tk()
#     root.geometry("1200x700+0+0")
#     root.title("Home Page")

#     image = Image.open("nbg.png")
#     image = image.resize((1200, 700), Image.ANTIALIAS)
#     pic = ImageTk.PhotoImage(image)
#     lbl_reg = Label(root, image=pic, anchor=CENTER)
#     lbl_reg.place(x=0, y=0)

      

#     # -----------------INFO TOP------------
#     lblinfo = Label(root, font=('aria', 20, 'bold'), text="NETWORK INTRUSION DETECTION", fg="white", bg="black",
#                     bd=10, anchor='w')
#     lblinfo.place(x=350, y=50)

#     lblinfo3 = Label(root, font=('aria', 20), text="Enter Network features", fg="#000955", anchor='w')
#     lblinfo3.place(x=780, y=310)
#     E1 = Entry(root, width=30, font="veranda 20")
#     E1.place(x=650, y=360)

#     lblinfo4 = Label(root, font=('aria', 17), text="Process", fg="white", anchor='w', bg="black")
#     lblinfo4.place(x=180, y=250)
#     mylist = Listbox(root, width=50, height=20, bg="white")
#     mylist.place(x=80, y=300)
#     btntrn = Button(root, padx=16, pady=8, bd=6, fg="white", font=('ariel', 16, 'bold'), width=10, text="Detect",
#                     bg="green", command=lambda: predict(E1.get()))
#     btntrn.place(x=800, y=420)

#     rslt = Label(root, font=('aria', 20), text="Attack type:", fg="black", bg="white", anchor=W)
#     rslt.place(x=630, y=580)
#     shrslt = Label(root, font=('aria', 20), text="", fg="blue", bg="white", anchor=W)
#     shrslt.place(x=790, y=580)

# root = Tk()
# root.title("Login Page")

# # Username label and entry field
# username_label = Label(root, text="Username:")
# username_label.pack()
# username_entry = Entry(root)
# username_entry.pack()

# # Password label and entry field
# password_label = Label(root, text="Password:")
# password_label.pack()
# password_entry = Entry(root, show="*")  # Show * instead of the actual password
# password_entry.pack()

# # Login button
# login_button = Button(root, text="Login", command=check_credentials)
# login_button.pack()

# root.mainloop()
from tkinter import *
from PIL import Image, ImageTk
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# ======= Models & labels =======
attack_list = ['Normal', 'DOS', 'Probe', 'R2L', 'U2R']

# Load models (ensure these files exist in the same folder)
loaded_model1 = tf.keras.models.load_model('alexmodel.h5')   # AlexNet-like model
print("Loaded AlexNet model")

with open('model_lstm1.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model_lstm = tf.keras.models.model_from_json(loaded_model_json)
model_lstm.load_weights("lstm_weight1.h5")
print("Loaded LSTM model")

minivgg_model = load_model('model.model')                    # MiniVGG-like model
print("Loaded MiniVGG model")

# ======= Globals assigned in userHome() =======
root = None
mylist = None
shrslt = None

def pp(msg: str):
    """Append a line in the Process listbox."""
    if mylist is not None:
        mylist.insert(END, msg)

def predict(val: str):
    """Run prediction pipeline on comma-separated numeric features."""
    try:
        # Parse and shape input
        list_str = [s.strip() for s in val.split(",") if s.strip() != ""]
        floatlist = [float(x) for x in list_str]
        text = [floatlist]

        # --- AlexNet path ---
        featalex = np.array(text)
        alex_scale = pickle.load(open("norm.pkl", "rb"))
        featalex = alex_scale.transform(featalex)
        featalex = np.reshape(featalex, (1, 20, 2, 1))
        preds_alex = loaded_model1.predict(featalex)[0]
        alex_result = int(np.argmax(preds_alex))

        # --- LSTM path ---
        lstm_trans = pickle.load(open("minmaxlstm.pkl", "rb"))
        X_test = lstm_trans.transform(text)
        feat = np.array(X_test).reshape(1, 40, 1)
        y = model_lstm.predict(feat)
        lstm_result = int(round(float(y[0][0])))

        # --- MiniVGG path ---
        featvgg = np.array(text)
        vgg_scale = pickle.load(open("norm.pkl", "rb"))
        featvgg = vgg_scale.transform(featvgg)
        featvgg = np.reshape(featvgg, (1, 20, 2, 1))
        preds_vgg = minivgg_model.predict(featvgg)[0]
        result_vgg = int(np.argmax(preds_vgg))

        # Majority vote
        votes = [alex_result, lstm_result, result_vgg]
        final_index = max(votes, key=votes.count)
        final_label = attack_list[final_index]

        # UI updates
        pp("Input data received")
        pp("Preprocessing started")
        pp("Feature scaling")
        pp("Loaded LSTM, AlexNet, and Mini VGGNet models")
        pp("Prediction using loaded models")
        pp(f"Attack type: {final_label}")
        pp("============================")
        if shrslt is not None:
            shrslt.config(text=final_label, fg="red")

    except Exception as e:
        pp(f"Error: {e}")

def userHome():
    """Build and run the main GUI (no login)."""
    global root, mylist, shrslt
    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("CyberEyes – Network Intrusion Detection")

    # Background image (optional)
    try:
        image = Image.open("nbg.png").resize((1200, 700), Image.LANCZOS)
        bg_image = ImageTk.PhotoImage(image)
        bg_label = Label(root, image=bg_image)
        bg_label.image = bg_image  # prevent GC
        bg_label.place(x=0, y=0)
    except Exception:
        # If the background image is missing, continue without it
        pass

    # Header
    lblinfo = Label(root, font=('Arial', 20, 'bold'),
                    text="CYBEREYES", fg="white", bg="black", bd=10, anchor='w')
    lblinfo.place(x=350, y=50)

    # Input
    Label(root, font=('Arial', 20),
          text="Enter Network features", fg="#000955", anchor='w').place(x=780, y=310)
    E1 = Entry(root, width=30, font=("Verdana", 20))
    E1.place(x=650, y=360)

    # Process list
    Label(root, font=('Arial', 17),
          text="Process", fg="white", bg="black", anchor='w').place(x=180, y=250)
    mylist = Listbox(root, width=50, height=20, bg="white")
    mylist.place(x=80, y=300)

    # Detect button
    Button(root, padx=16, pady=8, bd=6, fg="white",
           font=('Arial', 16, 'bold'), width=10, text="Detect", bg="green",
           command=lambda: predict(E1.get())).place(x=800, y=420)

    # Result
    Label(root, font=('Arial', 20),
          text="Attack type :", fg="black", bg="white", anchor=W).place(x=630, y=580)
    shrslt = Label(root, font=('Arial', 20),
                   text="", fg="blue", bg="white", anchor=W)
    shrslt.place(x=790, y=580)

    root.mainloop()

if __name__ == "__main__":
    userHome()
