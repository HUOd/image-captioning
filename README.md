# image-captioning
1. You can run the program on cmd by enter : python Imagecaptioning.py
2. The program will automatically test the first picture(index start from 0) of Flicker8k test data set.
   If you want to check other pictures, add '-i index': python Imagecaptioning.py -i index
3. If you want to test your personal pictures, add the picture to the floder: "Flicker8k_Dataset/PersonaltestImages/" , 
   make sure every picture has it's own name.
   Then enter: python Imagecaptioning.py -p -i index

The program is build under anaconda python 3.6, and we use the library keras 2.0. 
There is a big difference between keras 1.0 and keras 2.0. So the program might not work out under keras 1.0.
Here are some other libraries I use: 
   Keras 
   Tensorflow
   tqdm
   numpy
   pandas
   pickle
   glob
    
