import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

if __name__ == "__main__":
        input_dir = sys.argv[1]
        test_dir=sys.argv[2]
        features=[]
        test_features=[]
        model = VGG16(weights='imagenet',include_top=False)
        for sample in sorted(os.listdir(input_dir)):
                sample_path=os.path.join(input_dir,sample)
             
                frame_features=[]
                
                for frame in sorted(os.listdir(sample_path)):
                        frame_path=os.path.join(sample_path,frame)
                        image = load_img(frame_path, target_size=(224, 224))
### convert the image pixels to a numpy array
                        image = img_to_array(image)
### reshape data for the model
                        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
### prepare the image for the VGG model
                        image = preprocess_input(image)
### predict the probability across all output classes
                        yhat = np.asarray(model.predict(image))
                        ## yhat is of the shape (7*7*512) which is reshaped to a row vector
                        reshaped_features=yhat.reshape(1,-1)

                        frame_features.append(reshaped_features) 
                        
                current_video_features=np.vstack(frame_features)

                reduced_feature=np.mean(current_video_features,axis=0) ## Computes the mean of all frame feature vectors to get a single feature for each video
                features.append(reduced_feature)

##                if not features.size:
##                        features=reduced_feature
##                else:
##                        ##for each frame processed, stack it's feature vector with the features of other frames.
##                        features=np.vstack((features,reduced_feature))
        final_features=np.vstack(features)
        print(final_features.shape)
        dbsc=DBSCAN(eps=.5,min_samples=2)
        dbsc.fit(final_features)
     


        for test_sample in sorted(os.listdir(test_dir)):
                test_sample_path=os.path.join(test_dir,test_sample)
             
                test_frame_features=[]
                
                for test_frame in sorted(os.listdir(test_sample_path)):
                        test_frame_path=os.path.join(test_sample_path,test_frame)
                        test_image = load_img(test_frame_path, target_size=(224, 224))
### convert the image pixels to a numpy array
                        test_image = img_to_array(test_image)
### reshape data for the model
                        test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
### prepare the image for the VGG model
                        test_image = preprocess_input(test_image)
### predict the probability across all output classes
                        test_yhat = np.asarray(model.predict(test_image))
                        ## yhat is of the shape (7*7*512) which is reshaped to a row vector
                        test_reshaped_features=test_yhat.reshape(1,-1)

                        test_frame_features.append(test_reshaped_features) 
                        
                test_current_video_features=np.vstack(test_frame_features)

                test_reduced_feature=np.mean(test_current_video_features,axis=0) ## Computes the mean of all frame feature vectors to get a single feature for each video
                test_features.append(test_reduced_feature)


        test_final_features=np.vstack(test_features)
        labels=dbsc.fit_predict(test_final_features)
        print(labels)
       


        



