import cv2
import dlib
import numpy as np
cimport numpy as np

dirpath = 'F:\\git\\aind\\deeplearn\\cv_keypoints_capstone\\'

def getcolorimage(imagepath: str):
    return cv2.imread(imagepath, 1).astype(np.int)


def getgrayimage(imagepath: str):
    return cv2.imread(imagepath, 0).astype(np.int)


cpdef np.ndarray[np.int, ndim=2] equalizeGRAY(np.ndarray[np.int, ndim=2] imagearray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(imagearray)


cpdef np.ndarray[np.int, ndim=3] equalizeCOLOR(np.ndarray[np.int, ndim=3] imagearray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.merge([clahe.apply(channels) for channels in cv2.split(imagearray)])


cdef fcascade = cv2.CascadeClassifier(dirpath+'/detector_architectures/haarcascade_frontalface_default.xml')


cpdef np.ndarray[np.int, ndim=3] markfacesCV(str imagepath):
    cdef np.ndarray[np.int, ndim=2] fdetects
    cdef np.ndarray[np.int, ndim=2] imagegray
    cdef np.ndarray[np.int, ndim=3] imagecol

    imagegray, imagecol = equalizeGRAY(getgrayimage(imagepath)), getcolorimage(imagepath)
    fdetects = fcascade.detectMultiScale(imagegray, 3, 6)
    [cv2.rectangle(imagecol, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3) for (x, y, w, h) in fdetects]
    return imagecol[:, :, ::-1]


cdef dlibdetect = dlib.get_frontal_face_detector()


cpdef np.ndarray markfacesdlib(imagepath: str):
    cdef np.ndarray[np.int, ndim=2] fdetects
    cdef np.ndarray[np.int, ndim=2] imagegray
    cdef np.ndarray[np.int, ndim=3] imagecol

    imagegray = equalizeGRAY(cv2.resize(getgrayimage(imagepath), fx=0.5, fy=0.5))
    imagecol = getcolorimage(imagepath)
    fdetect = dlibdetect(imagegray, 1)
    for face in fdetect:
        (x, y, w, h) = np.array([face.left(), face.top(), face.right() - face.left(),
                                 face.bottom() - face.top()]) * 2
        cv2.rectangle(imagecol, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
    return imagecol
