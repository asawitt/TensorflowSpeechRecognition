from keras.models import load_model
import os
import numpy as np
import PIL

IMG_WIDTH = 60
IMG_HEIGHT = 65

model_filename = '60x65_all_labels.h5'
test_set_directory = '../Datasets/Test/Processed/audio/'
categories = [
    'bed', 'dog', 'five', 'happy', 'marvin', 
    'off', 'right', 'six', 'tree', 'wow', 'bird', 
    'down', 'four', 'house', 'nine', 'on', 
    'seven', 'stop', 'two', 'yes', 'cat', 'eight', 
    'go', 'left', 'no', 'one', 'sheila', 'three', 
    'up', 'zero', 'silence'
]
output_categories = set('yes no up down left right on off stop go silence'.split())
num_categories = len(categories)

cat_to_index = {cat:i for i,cat in enumerate(categories)}
one_hot = [[0]*num_categories for i in range(num_categories)] 
for i in range(num_categories):
     one_hot[i][i] = 1

prediction_filename = model_filename[:-3] + "_predictions.txt"

def main():
    print_prediction_header(prediction_filename)
    model = load_model(model_filename)

    test_data = []
    filenames = []
    index = 0
    for test_img_filename in os.listdir(test_set_directory):
        test_img = load_img(os.path.join(test_set_directory,test_img_filename))
        test_data.append(test_img)
        filenames.append(test_img_filename)
        index += 1
        if not index%100:
            print(index/1580,"% done")
        
    test_data = np.array(test_data).reshape(len(test_data),IMG_WIDTH,IMG_HEIGHT,1)

    predictions = model.predict(test_data,batch_size=100)

    predictions = zip(filenames,predictions)
    print_predictions(predictions,prediction_filename)

def load_img(test_img_filename):
    return normalize(np.asarray(PIL.Image.open(test_img_filename),dtype=np.float32))

def print_prediction_header(filename):
    print('yep')
    with open(filename,'w') as file:
        file.write('fname,label\n')


def print_predictions(predictions,filename):
    with open(filename,'a') as file:
        for prediction in predictions:
            category = categories[np.argmax(prediction[1])]
            if category not in output_categories:
                category = 'unknown'
            file.write(prediction[0][:-3] + "wav," + category + "\n")




def normalize(img):
    img.setflags(write=1)
    for r,row in enumerate(img):
        for c,item in enumerate(row):
            img[r][c] = item/255
    return img

    


if __name__ == '__main__':
    main()