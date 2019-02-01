from functools import *
import cv2
import numpy as np
from matplotlib import pyplot as plt

class cvBase():
    """
    Base Class for OpenCV Servings
    """

    def __init__(self):
        # processing
        self.avg_kernel = np.ones((63, 63), dtype='float32') / 3969
        self.facelocs, self.eyeloc_ = np.zeros(shape=(2, 1, 4), dtype='uint8')
        self.imgloc = np.unicode_(None)
        # cascades
        self.HAAR_frontalface = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
        self.HAAR_mcs_nose = cv2.CascadeClassifier('detector_architectures/haarcascade_mcs_nose.xml')
        self.HAAR_smile = cv2.CascadeClassifier('detector_architectures/haarcascade_smile.xml')
        self.HAAR_eye = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')
        self.getfaces = partial(self.HAAR_frontalface.detectMultiScale,
                                scaleFactor=1.25,
                                minNeighbors=6,
                                minSize=(16, 16))
        self.geteyes = partial(self.HAAR_eye.detectMultiScale,
                               scaleFactor=1.05,
                               minNeighbors=3,
                               minSize=(8, 8))
        self.cvfilter = partial(cv2.filter2D,
                              ddepth=-1,
                              kernel=self.avg_kernel,
                              anchor=(-1, -1),
                              borderType=cv2.BORDER_REFLECT)
        self.cvblur = partial(cv2.blur,ksize=127,
                              anchor=(-1,-1),
                              borderType=cv2.BORDER_REFLECT)


        # partials
        self.imread = partial(cv2.imread, )



class cvImage(cvBase):
    """
    Images Generator Class for OpenCV Servings
    """

    def __init__(self):
        super(cvImage, self).__init__()


    def getimage(self, imageloc: np.unicode_, transform: bool = True, grey: bool = False) -> np.ndarray:
        """
        get the image, and do elementary transforms if needed.
        :param imageloc: str, path for image file.
        :param transform: bool, flag for calling color_transform on output
        :param grey: bool, for greyscale output from color_transform call
        :return: numpy.ndarray, with dtype=uint8, transformed image array(RGB/GREY) or Raw(BGR)
        """
        # load
        self.imgloc = imageloc
        image = self.imread(imageloc).astype('uint8')
        if transform:
            return self.color_transform(image, isBGR=True, togrey=grey)
        else:
            # return raw
            return image

    def showimage(image: np.ndarray, text: np.unicode_ = '') -> None:
        """
        plots image inline, for % matplotlib inline display
        :param image: numpy.ndarray, opencv cv2.imread() image array
        :return: None
        """
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Image: ' + text)
        ax1.imshow(image, cmap='gray')

    def plotimage(self, imageloc: np.unicode_) -> None:
        """
        helper function to plot an image from path.
        calls self.showimage for display.
        :param imageloc: numpy.unicode_, path to image
        :return: None
        """
        image = self.getimage(imageloc, transform=True)
        self.showimage(image)

    @staticmethod
    def color_transform(image: np.ndarray, togrey: bool = False, isBGR:bool=False) -> np.ndarray:
        """
        function to perform elementary transforms on opencv image.
        :param image: numpy.ndarray, opencv cv2.imread() image array (BGR)
        :param togrey: bool, for greyscale output
        :param isBGR: bool, for BGR images
        :return: numpy.ndarray, with dtype=uint8, transformed image array (RGB/GREY)
        """
        if togrey and isBGR:
            _img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #self.showimage(_img, 'BGR=>GREY')
            return _img.astype('uint8')
        elif togrey and not(isBGR):
            _img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #self.showimage(_img, 'RGB=>GREY')
            return _img.astype('uint8')
        elif not(togrey) and isBGR:
            _img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #self.showimage(_img, 'BGR=>RGB')
            return _img.astype('uint8')
        elif not(togrey) and not(isBGR):
            return image.astype('uint8')

    @staticmethod
    def draw_img_bbox(image: np.ndarray,
                      boxlocs: np.ndarray,
                      color: (int, int, int) = (255, 0, 0),
                      thickness: int = 3) -> np.ndarray:
        """
        helper function to draw bounding boxes on images
        :param image: numpy.ndarray, opencv image array
        :param boxlocs: numpy.ndarray of type [[int,int,int,int],..], bounding box (x,y,w,h), from cascade detections
        :param color:  tuple, of type (int,int,int), color tuple for drawing boxes
        :param thickness: int, box frame thickness
        :return: numpy.ndarray,with dtype=uint8, image array with bounding boxes filled in.
        """
        for ix_ in range(len(boxlocs)):
            x, y, w, h = boxlocs[ix_]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        return image

    def detect_faces(self, image: np.ndarray,
                     show: bool = False,
                     useParams: bool = False,
                     fparams: (float, int, (int, int)) = (4, 6, (31, 31)),
                     sens_scale: np.float32 = 1,
                     isBGR: bool = False) -> np.ndarray:
        """
        HAAR cascade face detector
        :param img_: numpy.ndarray, image array [BGR] with bounding boxes marked in.
        :param show: bool, to show image as inline.
        :param useParams: bool, to use non default detector parameters.
        :param fparams: tuple, of type (float,int,(int,int)), parameter tuples for passing detector params.
        :param sens_scale: float, detector sensitivity scaling factor
        :param isBGR: bool, to work with non BGR image arrays
        :return: numpy.ndarray, with dtype=uint8, image as numpy array
        """
        # get detected img_ with faces
        scale, min_neighbors, min_size = fparams[0] * sens_scale, int(fparams[1] * sens_scale), fparams[-1]
        if useParams:
            self.facelocs = self.getfaces(self.color_transform(image, togrey=True, isBGR=isBGR),
                                       scaleFactor=scale,
                                       minNeighbors=min_neighbors,
                                       minSize=min_size)
        else:
            # use opencv defaults
            self.facelocs = self.getfaces(self.color_transform(image, togrey=True, isBGR=isBGR))

        # Draw bounding box for each detected face
        img_ = self.draw_img_bbox(image, self.facelocs, color=((0, 0, 255) if isBGR else (255, 0, 0)))

        if show:  # Display the img_ with the detections.
            message = 'number of faces detected: ' + str(len(self.faces))
            self.showimage(img_, message)
        return img_.astype('uint8')

    def detect_eyesonfaces(self,
                           image: np.ndarray,
                           show:bool=False,
                           useParams:bool=False,
                           eyeparams:(float, int,(int,int))=(1.05, 6, (16, 16)),
                           sens_scale:np.float32 = 1,
                           isBGR:bool = False) -> np.ndarray:
        """
        HAAR cascade eye detector
        :param image: numpy.ndarray, image array with bounding boxes marked in.
        :param show: bool, to show image as inline.
        :param useParams: bool, to use non default detector parameters.
        :param eyeparams:  tuple, of type (float,int,(int,int)), parameter tuples for detector.
        :param sens_scale: float, detector sensitivity scaling factor
        :param isBGR: bool, to work with non BGR image arrays
        :return: numpy.ndarray, with dtype=uint8, image as numpy array
        """
        scale, min_neighbors, min_size = eyeparams
        #grey_img = self.color_transform(image, togrey=True, isBGR=isBGR)
        self.detect_faces(image, useParams=True, sens_scale=sens_scale)
        col_img = self.draw_img_bbox(image, self.faces, color=((0, 0, 255) if isBGR else (255, 0, 0)))
        self.showimage(col_img, str(len(self.faces)))
        partialimg_ = lambda x,y,w,h: self.color_transform(col_img[y:y + h, x:x + w], isBGR=isBGR)
        if useParams:
            for (x,y,w,h) in self.faces:
                self.eyeloc_ = self.geteyes(partialimg_(x, y, w, h),
                                            scaleFactor=scale,
                                            minNeighbors=min_neighbors,
                                            minSize=min_size)
                # Draw bounding box for each eye detected in faces sub array
                col_img[y:y + h, x:x + w] = self.draw_img_bbox(col_img[y:y + h, x:x + w],
                                                               self.eyeloc_, color=(0, 255, 0),
                                                               thickness=2)
        else:
            # use opencv defaults
            for (x,y,w,h) in self.faces:
                self.eyeloc_ = self.geteyes(partialimg_(x, y, w, h))
                # Draw bounding box for each eye detected in faces sub array
                col_img[y:y + h, x:x + w] = self.draw_img_bbox(col_img[y:y + h, x:x + w],
                                                               self.eyeloc_, color=(0, 255, 0),
                                                               thickness=2)
        if show:  # Display the image with the detections.
            message = 'number of faces detected: ' + str(len(self.faces))
            self.showimage(image, message)
        return col_img.astype('uint8')

    def do_blur(self, image_arr: np.ndarray) -> np.ndarray:
        """
        blurs image using an averaging kernel.
        :param image_arr: numpy.ndarray, cv.imread() image input
        :return: numpy.ndarray, Image (BGR) with Blurred Patch on Detected Area.
        """
        applyfilterbychannel = lambda x: cv2.merge([self.cvfilter(x) for x in cv2.split(x)])
        # blur_start = time.time()
        applyfilterbychannel(image_arr)
        # print(time.time()-blur_start)
        return image_arr


    def blur_overlay(self, image: np.ndarray, N: int = 1) -> np.ndarray:
        """
        creates blur overlay on face detections
        :param image: numpy.ndarray, cv.imread() image input
        :param N: Number of Blurs in Sequence
        :return: numpy.ndarray, Image (RGB) with Blurred Patch on Detected Areas.
        """
        faces = self.detect_faces(self.color_transform(image))
        for _ in range(N):
            for _ix in range(len(faces)):
                x, y, w, h = faces[_ix]
                segmented = image[y:y + h, x:x + w]
                segmented = self.do_blur(segmented)
                image[y:y + h, x:x + w] = segmented

        return self.color_transform(image)


class cvVideo(cvBase):
    """
    Videos Generator Class for OpenCV Servings
    """

    def __init__(self):
        super(cvVideo, self).__init__()

    @staticmethod
    def color_transform(image: np.ndarray, togrey: bool = False, isBGR: bool = False) -> np.ndarray:
        """
        function to perform elementary transforms on opencv image.
        :param image: numpy.ndarray, opencv cv2.imread() image array (BGR)
        :param togrey: bool, for greyscale output
        :param isBGR: bool, for BGR images
        :return: numpy.ndarray, with dtype=uint8, transformed image array (RGB/GREY)
        """
        if togrey and isBGR:
            _img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # self.showimage(_img, 'BGR=>GREY')
            return _img.astype('uint8')
        elif togrey and not (isBGR):
            _img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # self.showimage(_img, 'RGB=>GREY')
            return _img.astype('uint8')
        elif not (togrey) and isBGR:
            _img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # self.showimage(_img, 'BGR=>RGB')
            return _img.astype('uint8')
        elif not (togrey) and not (isBGR):
            return image.astype('uint8')

    @staticmethod
    def draw_img_bbox(image: np.ndarray,
                      boxlocs: np.ndarray,
                      color: (int, int, int) = (255, 0, 0),
                      thickness: int = 3) -> np.ndarray:
        """
        helper function to draw bounding boxes on images
        :param image: numpy.ndarray, opencv image array
        :param boxlocs: numpy.ndarray of type [[int,int,int,int],..], bounding box (x,y,w,h), from cascade detections
        :param color:  tuple, of type (int,int,int), color tuple for drawing boxes
        :param thickness: int, box frame thickness
        :return: numpy.ndarray,with dtype=uint8, image array with bounding boxes filled in.
        """
        for ix_ in range(len(boxlocs)):
            x, y, w, h = boxlocs[ix_]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        return image

    def detect_faces(self, image: np.ndarray,
                     show: bool = False,
                     useParams: bool = False,
                     fparams: (float, int, (int, int)) = (4, 6, (31, 31)),
                     sens_scale: np.float32 = 1,
                     isBGR: bool = False) -> np.ndarray:
        """
        HAAR cascade face detector
        :param img_: numpy.ndarray, image array [BGR] with bounding boxes marked in.
        :param show: bool, to show image as inline.
        :param useParams: bool, to use non default detector parameters.
        :param fparams: tuple, of type (float,int,(int,int)), parameter tuples for passing detector params.
        :param sens_scale: float, detector sensitivity scaling factor
        :param isBGR: bool, to work with non BGR image arrays
        :return: numpy.ndarray, with dtype=uint8, image as numpy array
        """
        # get detected img_ with faces
        scale, min_neighbors, min_size = fparams[0] * sens_scale, int(fparams[1] * sens_scale), fparams[-1]
        if useParams:
            self.faces = self.getfaces(self.color_transform(image, togrey=True, isBGR=isBGR),
                                       scaleFactor=scale,
                                       minNeighbors=min_neighbors,
                                       minSize=min_size)
        else:
            # use opencv defaults
            self.faces = self.getfaces(self.color_transform(image, togrey=True, isBGR=isBGR))

        # Draw bounding box for each detected face
        img_ = self.draw_img_bbox(image, self.faces, color=((0, 0, 255) if isBGR else (255, 0, 0)))

        if show:  # Display the img_ with the detections.
            message = 'number of faces detected: ' + str(len(self.faces))
            self.showimage(img_, message)
        return img_.astype('uint8')

    def detect_eyesonfaces(self,
                           image: np.ndarray,
                           show: bool = False,
                           useParams: bool = False,
                           eyeparams: (float, int, (int, int)) = (1.05, 6, (16, 16)),
                           sens_scale: np.float32 = 1,
                           isBGR: bool = False) -> np.ndarray:
        """
        HAAR cascade eye detector
        :param image: numpy.ndarray, image array with bounding boxes marked in.
        :param show: bool, to show image as inline.
        :param useParams: bool, to use non default detector parameters.
        :param eyeparams:  tuple, of type (float,int,(int,int)), parameter tuples for detector.
        :param sens_scale: float, detector sensitivity scaling factor
        :param isBGR: bool, to work with non BGR image arrays
        :return: numpy.ndarray, with dtype=uint8, image as numpy array
        """
        scale, min_neighbors, min_size = eyeparams
        # grey_img = self.color_transform(image, togrey=True, isBGR=isBGR)
        self.detect_faces(image, useParams=True, sens_scale=sens_scale)
        col_img = self.draw_img_bbox(image, self.faces, color=((0, 0, 255) if isBGR else (255, 0, 0)))
        self.showimage(col_img, str(len(self.faces)))
        partialimg_ = lambda x, y, w, h: self.color_transform(col_img[y:y + h, x:x + w], isBGR=isBGR)
        if useParams:
            for (x, y, w, h) in self.faces:
                self.eyeloc_ = self.geteyes(partialimg_(x, y, w, h),
                                            scaleFactor=scale,
                                            minNeighbors=min_neighbors,
                                            minSize=min_size)
                # Draw bounding box for each eye detected in faces sub array
                col_img[y:y + h, x:x + w] = self.draw_img_bbox(col_img[y:y + h, x:x + w],
                                                               self.eyeloc_, color=(0, 255, 0),
                                                               thickness=2)
        else:
            # use opencv defaults
            for (x, y, w, h) in self.faces:
                self.eyeloc_ = self.geteyes(partialimg_(x, y, w, h))
                # Draw bounding box for each eye detected in faces sub array
                col_img[y:y + h, x:x + w] = self.draw_img_bbox(col_img[y:y + h, x:x + w],
                                                               self.eyeloc_, color=(0, 255, 0),
                                                               thickness=2)
        if show:  # Display the image with the detections.
            message = 'number of faces detected: ' + str(len(self.faces))
            self.showimage(image, message)
        return col_img.astype('uint8')

    def do_blur(self, image_arr: np.ndarray) -> np.ndarray:
        """
        blurs image using an averaging kernel.
        :param image_arr: numpy.ndarray, cv.imread() image input
        :return: numpy.ndarray, Image (BGR) with Blurred Patch on Detected Area.
        """
        applyfilterbychannel = lambda x: cv2.merge([self.cvfilter(x) for x in cv2.split(x)])
        # blur_start = time.time()
        applyfilterbychannel(image_arr)
        # print(time.time()-blur_start)
        return image_arr

    def blur_overlay(self, image: np.ndarray, N: int = 1) -> np.ndarray:
        """
        creates blur overlay on face detections
        :param image: numpy.ndarray, cv.imread() image input
        :param N: Number of Blurs in Sequence
        :return: numpy.ndarray, Image (RGB) with Blurred Patch on Detected Areas.
        """
        faces = self.detect_faces(self.color_transform(image))
        for _ in range(N):
            for _ix in range(len(faces)):
                x, y, w, h = faces[_ix]
                segmented = image[y:y + h, x:x + w]
                segmented = self.do_blur(segmented)
                image[y:y + h, x:x + w] = segmented

        return self.color_transform(image)
