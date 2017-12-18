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
        sliced_images = []
        sliced_annotations = []

        ### Slice into 2D images and Resize it 224 by 224
        for filename in self.files:
            new_images = self._transform(filename['image'])
            sliced_images.append(new_images)
            del new_images
            new_annot = self._transform(filename['annotation'])
            sliced_annotations.append(new_annot)
            del new_annot

        self.images = np.array(sliced_images)
        del sliced_images
        self.annotations = np.array(sliced_annotations)
        del sliced_annotations
        print (self.images.shape)
        print (self.annotations.shape)        

    def _transform(self, filename):
        _3D_image = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        z = _3D_image.shape[0]
        slicedList = []
        for index in range(z):
            _2D_slice = _3D_image[index, :, :]
            #_2D_slice = np.array([_2D_slice for i in range(3)])

            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                resize_image = misc.imresize(_2D_slice, [resize_size, resize_size], interp='nearest')
            else:
                resize_image = _2D_slice
            slicedList.append(resize_image)

        return slicedList

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

    
class TestDatset:
    files = []
    images = []    
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
        print("Initializing Test Dataset Reader...")
        
        self.files = records_list
        self.image_options = image_options
        self._read_images()


    def _read_images(self):
        sliced_images = []

        ### Slice into 2D images and Resize it 224 by 224
        for filename in self.files:
            new_images = self._transform(filename['image'])
            sliced_images.append(new_images)
            del new_images

        self.images = np.array(sliced_images)
        del sliced_images

        print(self.images.shape)


    def _transform(self, filename):
        _3D_image = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        z = _3D_image.shape[0]
        slicedList = []
        for index in range(z):
            _2D_slice = _3D_image[index, :, :]
            # _2D_slice = np.array([_2D_slice for i in range(3)])

            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                resize_image = misc.imresize(_2D_slice, [resize_size, resize_size], interp='nearest')
            else:
                resize_image = _2D_slice
            slicedList.append(resize_image)

        return slicedList

    def get_records(self):
        return self.images

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
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes]