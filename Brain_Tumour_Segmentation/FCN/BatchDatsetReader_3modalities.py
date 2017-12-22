"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import SimpleITK as sitk

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        
        self.files = records_list
        self.image_options = image_options
        self._read_images()
    
    def _read_images(self):        

        ### Slice into 2D images and Resize it 224 by 224
        imageList = []
        annotList = []
        for filename in self.files:
            imageList.append(self._transform_and_stack(filename['T1c'], filename['T2'], filename['FLAIR']))
            annotList.append(self._transform_and_stack_annotation(filename['annotation']))        
        self.images = np.concatenate(imageList, axis=0)
        self.annotations = np.concatenate(annotList, axis=0)        
        
        print (self.images.shape)
        print (self.annotations.shape)  
    
    
    def _transform_and_stack(self, T1c, T2, Flair):        
        T1c_image = sitk.GetArrayFromImage(sitk.ReadImage(T1c))
        T2_image = sitk.GetArrayFromImage(sitk.ReadImage(T2))
        Flair_image = sitk.GetArrayFromImage(sitk.ReadImage(Flair))
        
        z = T1c_image.shape[0]
        slicedList = []
        stackList = []        
        for index in range(z):
            #T1_slice = T1_image[index, :, :]
            T1c_slice = T1c_image[index, :, :]
            T2_slice = T2_image[index, :, :]
            Flair_slice = Flair_image[index, :, :]
            
            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                #T1_resize = misc.imresize(T1_slice, [resize_size, resize_size], interp='nearest')
                T1c_resize = misc.imresize(T1c_slice, [resize_size, resize_size], interp='nearest')
                T2_resize = misc.imresize(T2_slice, [resize_size, resize_size], interp='nearest')
                Flair_resize = misc.imresize(Flair_slice, [resize_size, resize_size], interp='nearest')
                
            else:
                #T1_resize = T1_slice
                T1c_resize = T1c_slice
                T2_resize = T2_slice
                Flair_resize = Flair_slice
                
            del T1c_slice, T2_slice, Flair_slice
            stacks = np.array([T1c_resize, T2_resize, Flair_resize])            
            del T1c_resize, T2_resize, Flair_resize
            stackList.append(stacks.T)            
            del stacks           

        return stackList
    
    def _transform_and_stack_annotation(self, GT):        
        GT_image = sitk.GetArrayFromImage(sitk.ReadImage(GT))
        z = GT_image.shape[0]        
        GT_List = []
        for index in range(z):            
            GT_slice = GT_image[index, :, :]
            
            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])                
                GT_resize = misc.imresize(GT_slice, [resize_size, resize_size], interp='nearest')
            else:                
                GT_resize = GT_slice
            del GT_slice            
            GT_stacks = np.array([GT_resize, GT_resize, GT_resize])
            del GT_resize            
            GT_List.append(GT_stacks.T)
            del GT_stacks  

        return GT_List
    
    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    
class TestDatset(BatchDatset):   
    
    
    def _read_images(self):        

        ### Slice into 2D images and Resize it 224 by 224
        imageList = []        
        for filename in self.files:
            imageList.append(self._transform_and_stack(filename['T1c'], filename['T2'], filename['FLAIR']))                   
        self.images = np.concatenate(imageList, axis=0)            
        
        print (self.images.shape)
        

    