---
layout: post
comments: true
mathjax: true
priority: -1000
title: "Write Code Professionally"
excerpt: “A simple tutorial in understanding Matrix capsules with EM Routing in Capsule Networks”
date: 2017-11-14 11:00:00
---

As a student in CS, you may often ask yourself what is good code and what is bad code. In this post, I will take as an example a code snippet written by a Phd student to briefly discuss the issue.  

Functionality of snippet is pretty simple, that is resizing images or image segmentation in a specified directory and output the results into another specified directory.
中文

```python
"""






"""

import os
import glob
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt


"""
os.chdir ('/home/zhewei/Zhewei/CamVid_MultiScale/test/')
for files in glob.glob('*.png'):
    print (files)
    img = cv2.imread(files)
    print (img.shape)
    norm_img = copy.deepcopy(img)

    blur = cv2.GaussianBlur(norm_img, (5,5), 0)
    Half_size = cv2.resize(blur, (0,0), fx=0.5, fy=0.5)
    blur2 = cv2.GaussianBlur(Half_size, (5,5), 0)
    Quarter_size = cv2.resize(blur2, (0,0), fx=0.5, fy=0.5)

    norm_img = cv2.normalize(Quarter_size, dst=norm_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img = np.uint8(norm_img*255)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    cl1 = clahe.apply(norm_img[:,:,0])
    cl2 = clahe.apply(norm_img[:,:,1])
    cl3 = clahe.apply(norm_img[:,:,2])
    cl = cv2.merge((cl1, cl2, cl3))
    #cv2.imshow('result', cl)
    #cv2.imshow('origin', img)
    #cv2.waitKey(0)
    #print (a)
    cv2.imwrite('/home/zhewei/Zhewei/CamVid_MultiScale/test_small/'+files, cl)
"""


os.chdir ('/home/zhewei/Zhewei/CamVid_MultiScale/trainannot/')
for files in glob.glob('*.png'):
    print (files)
    img = cv2.imread(files,0)
    #tmp = copy.deepcopy(img)
    print (img.shape)
    all_pixel = list()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            all_pixel.append(img[i,j])
            #tmp[i,j] = img[i,j]*10

    class_list = list(set(all_pixel))
    for classNO, classPixel in enumerate(class_list):
        tmp = copy.deepcopy(img)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                if img[i,j] != classPixel:
                    tmp[i,j] = 0
                else:
                    tmp[i,j] = 255

        blur = cv2.GaussianBlur(tmp, (5,5), 0)
        Half_size = cv2.resize(blur, (0,0), fx=0.5, fy=0.5)
        blur2 = cv2.GaussianBlur(Half_size, (5,5), 0)
        Quarter_size = cv2.resize(blur2, (0,0), fx=0.5, fy=0.5)

        norm_img = cv2.normalize(Quarter_size, dst=tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = np.uint8(norm_img*255)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cl1 = clahe.apply(norm_img)
        cl1 = cl1.reshape((1, cl1.shape[0], cl1.shape[1]))
        if classNO == 0:
            stack_img = cl1
        else:
            stack_img = np.concatenate((stack_img, cl1), axis=0)


    # go back to grey image
    final_img = np.zeros((stack_img.shape[1], stack_img.shape[2]),dtype=np.int)
    for i_s in range(stack_img.shape[1]):
        for j_s in range(stack_img.shape[2]):
            tmp_array = stack_img[:,i_s,j_s]
            max_index = np.argmax(tmp_array)
            final_img[i_s, j_s] = class_list[max_index]
            #print tmp_array
    #cv2.imshow('result', final_img)
    #cv2.waitKey(0)
    #cv2.imwrite('final.png', final_img)
    #print (a)
    cv2.imwrite('/home/zhewei/Zhewei/CamVid_MultiScale/trainannot_small/'+files, final_img)
```

I will refactor the code step by step, talking about the following three aspects of a good code respectively.

* easy to use
* easy to read
* easy to modify/update

##### Easy to Use

Although this snippet is supposed to use internally, usability is pretty important because easy-to-use code could enchance productivity substentially and avoid unnecessary. For a programmer, good usability always means well-designed interface, whatever it is Graphical User Interface (GUI) or Command-line Interface (CLI). We decide to apply CLI in this case.

Design comes from requirements. According to functionally of the code, the interface at least should let a user to specify 3 things, that is the operation (resize original images or image segmentations), the folder the input data resides and the folder where the results ouput. Concretely, I will do the following 2 refactorings: 

1. Use the native command parsing tool of Python to parse a command.
2. In the highest level, add two methods to handle image resizing and segmentation resizing respectively.

The new code is as following.

```python

import os
import glob
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt


def resize_images(input_dir, output_dir):
    os.chdir (input_dir)
    for files in glob.glob('*.png'):
        print (files)
        img = cv2.imread(files)
        print (img.shape)
        norm_img = copy.deepcopy(img)

        blur = cv2.GaussianBlur(norm_img, (5,5), 0)
        Half_size = cv2.resize(blur, (0,0), fx=0.5, fy=0.5)
        blur2 = cv2.GaussianBlur(Half_size, (5,5), 0)
        Quarter_size = cv2.resize(blur2, (0,0), fx=0.5, fy=0.5)

        norm_img = cv2.normalize(Quarter_size, dst=norm_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = np.uint8(norm_img*255)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cl1 = clahe.apply(norm_img[:,:,0])
        cl2 = clahe.apply(norm_img[:,:,1])
        cl3 = clahe.apply(norm_img[:,:,2])
        cl = cv2.merge((cl1, cl2, cl3))
        #cv2.imshow('result', cl)
        #cv2.imshow('origin', img)
        #cv2.waitKey(0)
        #print (a)
        cv2.imwrite(output_dir+files, cl)



def resize_labels(input_dir, output_dir):
    os.chdir (input_dir)
    for files in glob.glob('*.png'):
        print (files)
        img = cv2.imread(files,0)
        #tmp = copy.deepcopy(img)
        print (img.shape)
        all_pixel = list()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                all_pixel.append(img[i,j])
                #tmp[i,j] = img[i,j]*10

        class_list = list(set(all_pixel))
        for classNO, classPixel in enumerate(class_list):
            tmp = copy.deepcopy(img)
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    if img[i,j] != classPixel:
                        tmp[i,j] = 0
                    else:
                        tmp[i,j] = 255

            blur = cv2.GaussianBlur(tmp, (5,5), 0)
            Half_size = cv2.resize(blur, (0,0), fx=0.5, fy=0.5)
            blur2 = cv2.GaussianBlur(Half_size, (5,5), 0)
            Quarter_size = cv2.resize(blur2, (0,0), fx=0.5, fy=0.5)

            norm_img = cv2.normalize(Quarter_size, dst=tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_img = np.uint8(norm_img*255)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            cl1 = clahe.apply(norm_img)
            cl1 = cl1.reshape((1, cl1.shape[0], cl1.shape[1]))
            if classNO == 0:
                stack_img = cl1
            else:
                stack_img = np.concatenate((stack_img, cl1), axis=0)


        # go back to grey image
        final_img = np.zeros((stack_img.shape[1], stack_img.shape[2]),dtype=np.int)
        for i_s in range(stack_img.shape[1]):
            for j_s in range(stack_img.shape[2]):
                tmp_array = stack_img[:,i_s,j_s]
                max_index = np.argmax(tmp_array)
                final_img[i_s, j_s] = class_list[max_index]
                #print tmp_array
        #cv2.imshow('result', final_img)
        #cv2.waitKey(0)
        #cv2.imwrite('final.png', final_img)
        #print (a)
        cv2.imwrite(output_dir+files, final_img)



FLAGS = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        default='images',
        help="Types: 'images' or 'labels'."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the data.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/data',
        help='Output dir.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    input_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    
    if FLAGS.type == 'images':
        resize_images(input_dir, output_dir)
    else:
        resize_labels(input_dir, output_dir)

```

Looks a little bit better, right? Let's move on!

#### Easy to read

If you want to be a professional programmer, you must bear in your mind that your code is to read by other people. And this other people might be yourself. The following refactors could be made to current code to improve its readability:

1. Naming. Use names with clear meaning and avoid vague names such as 'blur', 'norm_img'.
2. Consistent code style, including capitalization, whitespaces, indentation.

### Easy to modify/update

As a programmer, you will face legacy code oneday. To make coding based on legacy code easily, you must enhance the modifiability of your code. Remember the bad smell in code (credit belonging to Martin Fowler and Robert Martin):

* Duplicate code (most notorious one, add a code for it instantly)
* Long methods (what are you talking about with so long a sentence)
* Needless complexity

After the two improvements, the final code snippet is rolled out.

```python
import os
import argparse
import glob
import cv2
import numpy as np


def resize_img(img, scale):
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    resized_img = cv2.resize(blurred_img, (0, 0), fx=0.5, fy=0.5)
        
    if scale == 0.25:
        blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
        resized_img = cv2.resize(blurred_img, (0, 0), fx=0.5, fy=0.5)

    normalized_img = cv2.normalize(resized_img, dst=resized_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result_img = np.uint8(normalized_img * 255)
    
    return result_img


def resize_images(input_dir, output_dir, scale=0.5, apply_clahe=False):
    os.chdir(input_dir)

    for img_file in glob.glob('*.png'):
        print(img_file)
        img = cv2.imread(img_file)
        #print (img.shape)
        
        result_img = resize_img(img, scale)

        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            result_img = clahe.apply(result_img)

        cv2.imwrite(output_dir + '/' + img_file, result_img)


def resize_labels(input_dir, output_dir, class_count, scale=0.5, apply_clahe=False):
    os.chdir(input_dir)    

    for label_file in glob.glob('*.png'):
        print(label_file)
        label_img = cv2.imread(label_file, 0)
        label_img_height = label_img.shape[0]
        label_img_width = label_img.shape[1]
        label_pixel_list = np.reshape(label_img, (label_img_height * label_img_width))

        resized_img_height = int(round(label_img_height * scale))
        resized_img_width = int(round(label_img_width * scale))
        stacked_class_img = np.array([]).reshape((0, resized_img_height, resized_img_width))

        for class_value in range(class_count):
            print('class value:' + str(class_value))
            class_label_list = [(0.0 if label_pixel_list[i] != class_value else 255.0) for i, label in enumerate(label_pixel_list)]
            class_label_img = np.reshape(class_label_list, (label_img.shape[0], label_img.shape[1]))
            #print(class_label_img)

            resized_class_img = resize_img(class_label_img, scale)

            if apply_clahe:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                resized_class_img = clahe.apply(resized_class_img)

            resized_class_img = np.reshape(resized_class_img, (1, resized_class_img.shape[0], resized_class_img.shape[1]))
            stacked_class_img = np.vstack((stacked_class_img, resized_class_img))

        result_img = np.argmax(stacked_class_img, axis=0)
        cv2.imwrite(output_dir + '/' + label_file, result_img)


FLAGS = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        default='images',
        help="Types: 'images' or 'labels'."
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale of the resizing.'
    )
    parser.add_argument(
        '--class_count',
        type=int,
        default=32,
        help='Total class count.'
    )
    parser.add_argument(
        '--improve_contrast',
        type=bool,
        default=False,
        help='Whether to improve contrast.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the data.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/data',
        help='Output dir.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    input_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    scale = FLAGS.scale
    improve_contrast = FLAGS.improve_contrast

    if FLAGS.type == 'images':
        print("Starting resizing images...")
        resize_images(input_dir, output_dir, scale, improve_contrast)
    else:
        resize_labels(input_dir, output_dir, FLAGS.class_count, scale, improve_contrast)

```





    
