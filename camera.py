import cv2
import numpy as np

# mindvision camera sdk
import mvsdk


class WebCam:
    def __init__(self, cameraNumber = 0):
        self.capture = None 
        self.cameraNumber = cameraNumber
      
    def __enter__(self):
        self.capture = cv2.VideoCapture(self.cameraNumber)
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        self.capture.release()
    
    def processCapture(self):
        ret, frame = self.capture.read()
        if not ret:
            raise ValueError("camera.py : not ret #TODO understand") #TODO understand

        # Flip image so it matches the training input
        frame = cv2.flip(frame, 1)
        return frame

class MVCam:
    def __init__(self, cameraNumber = 0):
        self.cameraNumber = cameraNumber
        self.hCamera = 0

    def __enter__(self):
        # Enumerate the camera
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            raise ValueError("camera.py : No camera was found!")

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
            #TODO make these print available in debug mode only.
        DevInfo = DevList[self.cameraNumber] 
        print(DevInfo)

        # Turn on the camera
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("camera.py : CameraInit Failed({}): {}".format(e.error_code, e.message) )
            return e

        # Get a description of camera characteristics
        cap = mvsdk.CameraGetCapability(self.hCamera)

        # Determine whether it is a black and white camera or a color camera
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # The black and white camera allows the ISP to directly output MONO data instead of expanding to a 24-bit grayscale of R=G=B
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Switch the camera mode to continuous acquisition
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # Manual exposure, exposure time 30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, 30 * 1000)

        # Let the SDK internal drawing thread start working
        mvsdk.CameraPlay(self.hCamera)

        #Calculate the size required for the RGB buffer, here it is directly allocated according to the maximum resolution of the camera
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # Allocate RGB buffer to store images output by ISP
        # Remarks: RAW data is transmitted from the camera to the PC side, and it is converted to RGB data through the software ISP on the PC side (if it is a black and white camera, you don't need to convert the format, but the ISP has other processing, so you also need to allocate this buffer)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        # Turn off the camera
        mvsdk.CameraUnInit(self.hCamera) #self
        # Release the frame cache
        mvsdk.CameraAlignFree(self.pFrameBuffer) #self
    
    def processCapture(self):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            
            # At this time, the picture is already stored in pFrameBuffer. For color cameras, pFrameBuffer =RGB data, and black and white cameras, pFrameBuffer=8-bit grayscale data.
            # Convert pFrameBuffer into opencv image format for subsequent algorithm processing
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            return frame
            
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("camera.py : CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

