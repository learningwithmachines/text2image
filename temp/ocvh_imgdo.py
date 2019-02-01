import time
from functools import *
import cv2
import dlib as dlb
import numpy as np

from core.memoizers import ocvh_libmemo, ocvh_memoclear

dotpath = 'F:/git/aind/deeplearn/cv_keypoints_capstone/detector_architectures'
testruns_NUM = 10

# methods #fullface
dlibfacedet = dlb.get_frontal_face_detector()
# cv2 face
cv2faceC = cv2.CascadeClassifier(dotpath+'/haarcascade_frontalface_default.xml')
cv2detface = cv2faceC.detectMultiScale
# cv2 eyes
cv2eyesC = cv2.CascadeClassifier(dotpath+'/haarcascade_eye.xml')
cv2deteyes = cv2eyesC.detectMultiScale
# cv2 nose
cv2noseC = cv2.CascadeClassifier(dotpath+'/haarcascade_frontalface_default.xml')
cv2detnose = cv2noseC.detectMultiScale
# cv2 smile
cv2smileC = cv2.CascadeClassifier(dotpath+'/haarcascade_frontalface_default.xml')
cv2detsmile = cv2smileC.detectMultiScale


@ocvh_libmemo.cache
def readfile(loctxt: np.unicode_) -> (np.ndarray, np.ndarray):
    img_bgr = cv2.imread(loctxt, 1)
    img_gray = np.average(img_bgr, axis=-1)
    return img_bgr.astype(np.uint8), img_gray


@ocvh_libmemo.cache
def mem_rescale(input_npmat: np.ndarray, dsize: (int, int)) -> np.ndarray:
    result = cv2.resize(src=input_npmat, dsize=dsize, interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)
    return result


@ocvh_libmemo.cache
def eq_hist(input_npmat: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(input_npmat)
    return result


@ocvh_libmemo.cache
def dlibface(imgarray) -> np.ndarray:
    faces = dlibfacedet(imgarray, 1)
    return faces


@ocvh_libmemo.cache
def cv2face(imgarray) -> np.ndarray:
    return cv2detface(imgarray, scaleFactor=1.5, minNeighbors=6, flags=0, minSize=(31, 31))


@ocvh_libmemo.cache
def cvrectify(imgarray: np.ndarray, bboxes, color: (int, int, int), upscaleby: int = 2):
    for face in bboxes:
        x, y, w, h = face * upscaleby
        cv2.rectangle(imgarray, (x, y), (x + w, y + h), color=color, thickness=2)
    return imgarray


@ocvh_libmemo.cache
def dlibrectify(imgarray: np.ndarray, bboxes, color: (int, int, int), upscaleby: int = 2):
    for face in bboxes:
        (x, y, w, h) = np.array([face.left(), face.top(), face.right() - face.left(),
                                 face.bottom() - face.top()]) * upscaleby
        cv2.rectangle(imgarray, (x, y), (x + w, y + h), color=color, thickness=2)
    return imgarray


class BaseImg:
    def __init__(self):
        pass

    @staticmethod
    def cvimgwindow(imgarray: np.ndarray) -> None:
        winname: str = 'CV2 Image: Shape: {} '.format(list(imgarray.shape))
        cv2.startWindowThread()
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 960, 540)
        cv2.imshow(winname, imgarray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None


class OCVH(BaseImg):
    def __init__(self, imageloctxt: np.unicode_, dlibfacesens: float = 0.5):
        # init
        super(OCVH, self).__init__()
        # vars
        self.dlibsens = dlibfacesens
        self.faces = None
        self.path = np.unicode_(imageloctxt)
        self.imagebgr, self.imagegray = self.filereadcv(self.path)
        self.bgrshape, self.grayshape = self.imagegray.shape, self.imagegray.shape
        self.rescaledshape = tuple(int(x * self.dlibsens) for x in self.grayshape[::-1])
        self.rescaled = self.rescalecpu(self.imagegray, dsize=self.rescaledshape)
        self.faces = self.facedlib()
        self.facesbgr = self.getboxes()

    def filereadcv(self, imageloctxt: np.unicode_) -> (np.ndarray, np.ndarray):
        return readfile(imageloctxt)

    def rescalecpu(self, imgarray: np.ndarray, dsize: (int, int) = (0, 0)) -> np.ndarray:
        return cv2.merge([eq_hist(mem_rescale(x, dsize)) for x in cv2.split(imgarray)])

    def facedlib(self):
        return cv2face(self.rescaled)

    def getboxes(self):
        f_ = int(1 / self.dlibsens)
        color = (0, 0, 255)
        return cvrectify(np.copy(self.imagebgr), self.faces, color, f_)

    def __del__(self, for_each_device=False):
        # self.file2img_cache.clear()
        ocvh_memoclear()


##tests
def _test(runsnum=testruns_NUM):
    import time
    pathx = 'F:/git/aind/deeplearn/cv_keypoints_capstone/images/test_image_1.jpg'
    printmask = "time:{} , loop:{}, bgrshape:{}, faces:{}"
    finalprintmask = "\nX--X \ntotal time:{} , per image:{} \nX--X"
    testocvh = OCVH
    timedg = time.time()

    def getimageinloop(pathx):
        run: int = 0
        for _ in range(runsnum):
            run += 1
            timed = time.time()
            tested = testocvh(pathx)
            faces = tested.facedlib()
            print(printmask.format((time.time() - timed) * 1e3, run, tested.imagebgr.shape, len(faces)))
        return run

    run = getimageinloop(pathx)
    run = run if run > 0 else 1
    print(finalprintmask.format((time.time() - timedg) * 1e3,
                                ((time.time() - timedg) * 1e3 / (run))))
    # optional = testohcv.cvimgwindow(testohcv.rescalefrompath(pathx,dsize=(192,108)))
    testocvh.cvimgwindow(testocvh(pathx).facesbgr)


# .. main
_test()
