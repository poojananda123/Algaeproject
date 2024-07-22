from django.shortcuts import render
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
#import cv2
#import os
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64

# Create your views here.
def loginpage(request):
    return render(request,'core/loginpage.html')


def predictionscreen(request):
    return render(request,'core/predictionscreen.html')      

        
def prediction(request):
    predictionscreen=''

    if request.method=='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if(username=="mlac" and password=="12345"):
            predictionscreen='yes'

    return render(request,returnPath(predictionscreen))  
            

def returnPath(a):

    if a=='yes':
        return 'core/predictionscreen.html'
    else:
        return 'core/loginpage.html'
    



def predictimage(request):
    # E:\algae12
    train=ImageDataGenerator(rescale=1/255)
    validation=ImageDataGenerator(rescale=1/255)
    train_dataset = train.flow_from_directory('E:/algae12/training/predict',
                                         target_size= (200, 200),
                                         batch_size = 3,
                                         class_mode = 'binary')
    validation_dataset = validation.flow_from_directory('E:/algae12/validation/predict',
                                                   target_size= (200, 200),
                                                   batch_size = 3,
                                                   class_mode = 'binary')
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]) 

    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])


    history=model.fit(train_dataset, 
          epochs=5,
          validation_data=validation_dataset)

    accuracy=history.history['accuracy']
    plt.plot(accuracy)
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.show()
    fig=plt.gcf()
    buf=io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string=base64.b64encode(buf.read())
    uri=urllib.parse.quote(string)

    print(validation_dataset.class_indices)
    print(request.FILES["image"])
    # upload = request.FILES['image']
    path = 'C:/Users/nandakumar/Desktop/algae_classification_cnn_project/algae_prediction/prediction/static/images/'+str(request.FILES["image"])

    img=image.load_img(path,target_size=(200,200))
    # plt.imshow(img)
    # plt.show()
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=model.predict(images)
    
    # # if val==0:
    # #   print("blue-green")
    # # else:
    # #   print("red")
    print(val)
    # print(uri)
    params={
        'class':val,
        'imgpath':'images/'+str(request.FILES["image"]),
        'data':uri
    }
    return render(request,'core/predictimage.html',params) 

    






   


    

