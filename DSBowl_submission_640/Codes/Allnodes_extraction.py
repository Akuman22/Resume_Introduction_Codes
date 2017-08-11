import os
import dicom
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import collections
import scipy.misc
from sklearn.neural_network import MLPClassifier
import pandas
import matplotlib.image as mpimg
import skimage.feature
nnin = []*500
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

datadir = 'D:\Artificial Intelligence\sample_images'
datadir2 = 'D:\Artificial Intelligence\\finalNodes'
posnames = ['X','Y']
data = pandas.read_csv("stage1_sample_submission.csv",names = posnames)
dataname = list(data['X'])
dataclass = list(data['Y'])
patients = os.listdir(datadir)
setall = collections.defaultdict(list)
for pat in range(len(dataname)):
    if str(dataname[pat]) not in os.listdir(datadir2):
            if(str([pat]) in dataname):
            os.mkdir(datadir2+'/'+str(patients[pat]))
            print(patients[pat])
            p1 = [dicom.read_file(datadir+'/'+patients[pat]+'/'+s ) for s in os.listdir(datadir+'/'+patients[pat])]
            p1.sort(key = lambda x : int(x.ImagePositionPatient[2]))
            try:
                slice_thickness = np.abs(p1[0].ImagePositionPatient[2] - p1[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(p1[0].SliceLocation - p1[1].SliceLocation)
            for s in p1:
                s.SliceThickness = slice_thickness
            p2 = get_pixels_hu(p1)
            p2, spacing = resample(p2, p1, [1,1,1])
            p2+=1024
            i = 0;
            xs = 0;
            for line in range(10,int(len(p2)/5)-1):
                print line*5
                num = 0
                for lin in range(50,len(p2[line*5])-50):
                    xs = 0;
                    for li in range(50,len(p2[line*5][lin])-50):
                        xy = 0
                        tc = []
                        if(1000> p2[line*5][lin][li] > 600):
                            xs+=1
                            li+=1
                            i = 0
                            xytmp = 0
                            while(1000> p2[line*5][lin+i][li] > 600):
                                if(lin+i >= (len(p2[line*5])-50 )):
                                    break
                                xytmp+=1
                                i+=1
                            if(xytmp>xy):
                                xy = xytmp
                        if(xs>6) or (xy>6):
                            xs = 0
                            xy = 0
                        nps = 0
                        if(xs>2) & (xy>2):
                            for kk in range(lin-8,lin+7):
                                for ll in range (li-int(xs/2)-8,li-int(xs/2)+7):
                                    if(p2[line*5][kk][ll]>900):
                                        nps+=1
                        if(xs<6) & (xs>2) and (xy<6) and (xy >2) and (nps<30):
                            
                            plt.imshow(p2[line*5][lin-15:lin+14,li-int(xs/2)-15:li-int(xs/2)+14])
                            plt.show()
                            mpimg.imsave(datadir2+'/'+str(patients[pat])+'/'+str(line*5)+str(num)+'.png' , p2[line*5][lin-15:lin+14,li-int(xs/2)-15:li-int(xs/2)+14])
        #                    plt.imshow(p2[line*5][lin-8:lin+7,li-int(xs/2)-8:li-int(xs/2)+7])
        #                    plt.show()
                            num+=1
                            xs = 0
                      
             
