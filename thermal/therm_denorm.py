import cv2, pytesseract, re, joblib
from typing import Literal
import blosc
import numpy as np
import matplotlib.pyplot as plt

#helper function for calling ocr detection and compiling
def ocr_number(img):
    cfg = r'--psm 7 -c tessedit_char_whitelist=0123456789.-'
    text = pytesseract.image_to_string(img, config=cfg)
    m = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(m.group()) if m else None

#helper function for preparing the windows for ocr interpretation
def prep(roi):
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (3,3), 0)
            g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            return g


def denorm_temps_bw_thermal(
    video_path  :   str =   "../../.local_data/midbelt_BW.mp4",
    output_name :   str =   "thermal_denorm",
    compression :   Literal['none','light','extreme'] = 'extreme'
):
    '''
    INFO
    ----
    This function takes in a given MP4 path to a black/white thermal video and denormalizes all temperatue values.<br><br>
    This allows a user to get the original temperature data in the form of a 3d numpy ndarray.

    SAVES
    -----
    A 2 item list, first item is array below, second item is describing string for compression method for loading.<br>
    All approximate raw temperature values of the frame in a 3d matrix in form of (Frame number, m (y-axis), n (x-axis))
    '''

    cap = cv2.VideoCapture(video_path)
    
    ok, frame0 = cap.read()

    #these values should be unchanged, as location of heatbar should be identical in all videos
    x1 = 190
    y1 = 57
    y2 = 240

    w1 = 60
    h1 = 18

    #plt.imshow(frame0[:, :, 0]>100, cmap='gray')
    #plt.scatter(w1+x1, y1+h1, s=50, c='blue')
    #plt.scatter(x1, y1, s=50, c='blue')
    #plt.scatter(w1+x1, y2+h1, s=50, c='blue')
    #plt.scatter(x1, y2, s=50, c='blue')
    #plt.title("tesseract ranges")
    #plt.show()


    # Define ROIs in (x,y,w,h)\
    ROI_HIGH = (x1, y1, w1, h1)
    ROI_LOW  = (x1, y2, w1, h1)

    #variable initiation, frame_idx is frame number, results is tuple stacker for each frame
    results = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("Scraping heatbar temps with tesseract. This will take a while..\n Expect this step to take >1x video length time.")

    #frame loop
    while True:

        #sanity check
        ok, frame = cap.read()
        if not ok: break
        
        #get seconds value instance
        t = frame_idx / fps
        
		#only run every four frames
        if(frame_idx%4>0):
              #iterate and append last results
              frame_idx+=1
              results.append((frame_idx, t, hi_val, lo_val))
              continue
        
        #declare windows for ocr
        x,y,w,h = ROI_HIGH
        hi_roi = frame[y:y+h, x:x+w]
        x,y,w,h = ROI_LOW
        lo_roi = frame[y:y+h, x:x+w]

        #attempt detection of numbers in these windows
        hi_val = ocr_number(prep(hi_roi))
        lo_val = ocr_number(prep(lo_roi))

        #append results 
        results.append((frame_idx, t, hi_val, lo_val))
        
        #printout every 100 frames to ensure running in time
        if(frame_idx%2000==0):
            print(f"frame idx: {frame_idx}")

        frame_idx += 1

    cap.release()
    del cap
    del frame


    temp_rngs = np.asarray(results)

    del results, frame0

    #spit out all 100th frame temps for quick validation by eye if need to see what is going wrong
    for i in range(temp_rngs.shape[0]):
        if i%2000==0:
            print(temp_rngs[i])

    #plt.plot(temp_rngs[:, 3])
    #plt.plot(temp_rngs[:, 2])
    #plt.show()

    #go through all frames of temp data
    for m in range(1, temp_rngs.shape[0]):
        #go through all values saved in this frame
        for n in range(1,temp_rngs.shape[1]):
            #check to see if there are any missing values
            if((temp_rngs[m,n]) is None):
                #if so, replace it with last samples value
                temp_rngs[m,n]=temp_rngs[m-1,n]

    #plt.plot(np.clip(temp_rngs[:, 3], 0, 100))
    #plt.plot(temp_rngs[:, 2])
    #plt.show()

    #logical bounds according to this video
    #need to remap off values of highs and lows to match actual values
    #lets do this iteratively

    #found these ranges to adequately describe what is being observed, likely generally.
    #these constants can be changed if need be
    #iterating through to make sure partial numbers werent missed or added
    #forcing high and low values to be confined to REAL and logical windows.
    for m in range(1, temp_rngs.shape[0]):
        if(temp_rngs[m, 2] < 55 or temp_rngs[m, 2] > 62):
            temp_rngs[m, 2] = temp_rngs[m-1, 2]
        if(temp_rngs[m, 3] < 14 or temp_rngs[m, 3] > 17):
            temp_rngs[m, 3] = temp_rngs[m-1, 3]
    temp_rngs[:8, 2:] = temp_rngs[8, 2:]

    #plt.plot(temp_rngs[:, 3])
    #plt.plot(temp_rngs[:, 2])
    #plt.show()


    #reopening video for denormalization of mp4 intensity values
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()

    #first frame validation printout
    m = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
    #MUST BE 0, 255
    print(np.min(m), np.max(m))
    m = m.astype(np.float32) / 255
    #MUST BE 0, 1.0
    print(np.min(m), np.max(m))

    #plt.imshow(m,cmap='gray')
    #plt.show()

    del frame0

    #same initialization as above
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    #two values to hold frame local high and low temps
    lo_t = 0
    hi_t = 0

    #variable for denormalized temp matrix stacking
    therm_denorm = []

    #frame loop
    while True:

        #sanity
        ok, frame = cap.read()
        if not ok: break

        #reducing redundant 3 channels to 1
        norm_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #scale val range [0, 255] to [0.0, 1.0]
        norm_frame = norm_frame.astype(np.float32) / 255

        #get the low and high values grabbed earlier in the function
        lo_t = temp_rngs[frame_idx, 3]
        hi_t = temp_rngs[frame_idx, 2]
        #get the value range for proper scaling
        df_t = hi_t-lo_t

        #sanity check, should only get caught if there was errors in earlier parts of the function
        if(df_t<0):
            raise ValueError(f"negative hi lo temp difference, impossible. check logic. GOT (hi, lo) as ({hi_t}, {lo_t})")

        #scale the matrix by the real temp range
        norm_frame *= df_t
        #displace the values of the scaled matrix to sit in the real temperature values 
        norm_frame += lo_t

        #validation (should be same as original scraped range above)
        #print(np.min(norm_frame), np.max(norm_frame))

        #stack the matrix
        therm_denorm.append(norm_frame)
        
        frame_idx += 1

    #turn it into a 3d matrix with shape (frame#, m, n)
    therm_denorm = np.stack(therm_denorm, axis=0)

    cap.release()
    del cap

    print(f"Compressing...")

    #made a few different methods of compressing.
    #saving and loading NOTE should be all autonomous so there is NOTE no need in knowning what is going on here
    #for anyone reading this unless an error ise thrown during loading. NOTE Ask Logan if needed
    match(compression):

        #if no compression is selected (NOTE NOT RECOMMENDED NOTE est. 5Min > 3GB, accuracy ~0.0001, excessive)
        case 'none':
            joblib.dump([
                    therm_denorm, 
                    {"method":compression, "shape":therm_denorm.shape}
                ], 
                output_name+".data"
            )
        
        #if light compression is selected (NOTE est. 5Min > 1.25GB, accuracy ~0.25, good enough)
        case 'light':
            therm_denorm *= 4
            therm_denorm = therm_denorm.astype(np.uint8)
            joblib.dump([
                    therm_denorm, 
                    {"method":compression, "shape":therm_denorm.shape}
                ], 
                output_name+".data"
            )
        
        #if extreme compression is selected (NOTE RECOMMENDED NOTE est. 5Min > 50MB, accuracy ~0.25, good enough)
        case 'extreme':
            #make quantile, this is only possible when temps dont exceed 63.9 deg C, which should never happen.
            therm_denorm *= 4
            therm_denorm = np.round(therm_denorm).astype(np.uint8)

            #use blosc with zstd for second level compression using this as quantile
            c = blosc.compress(therm_denorm, typesize=1, cname="zstd", clevel=9, shuffle=blosc.SHUFFLE)
            joblib.dump([
                    c, 
                    {"method":compression, "shape":therm_denorm.shape}
                ], 
                output_name+".data"
            )



def load_temps(
    file_name   :   str =   ''
)   ->  np.ndarray:
     
    data, meta = joblib.load(file_name)

    match(meta["method"]):
        case 'none':
            return data
        
        case 'light':
            return data.astype(np.float16)*4
        
        case 'extreme':
            quant = np.frombuffer(blosc.decompress(data), dtype=np.uint8).reshape(meta["shape"])
            # NOTE NOTE NOTE 0.25 IS COMPLETELY SUBJECTIVE NOTE NOTE NOTE
            # THIS VALUE ONLY WORKS FOR HOW THESE FUNCTIONS ARE WRITTEN AS OF COMPLETION 9/20/2025
            # NOTE by modifying compression methods for "extreme", this data loading method is subject to corrupting upon loading.
            # NOTE IF compression method is adjusted please add "step" instance to meta that depicts value accuracy.
            # NOTE right now the step is 0.25 because we are scaling by 4 to make use of uint8 range. (0.25==1/4, adjust if needed)
            # I am leaving as is because I do not forsee changing this functionality at least for a long time.
            return quant.astype(np.float320) * 0.25

        case _:
            raise NotImplementedError(f"Loaded file '{file_name}' with unrecognized compression method. GOT ({meta}\nCONTAINING\n{meta["method"]}). Ask Logan K about this.")