### Collection of Boston Scientific RT related utility functions 
import numpy as np
import cv2
from tqdm import tqdm 
import os

def get_ilab_rt (rt_file, frame_index = None):
    ""
    # Reads binary RT file into Nx256x256 numpy array 
    ""
    
    st = os.stat(rt_file)
    #print (st.st_size)
    nr, nc = 256, 256

    with open(rt_file, 'rb') as f:
        rtSize = np.fromfile(f,dtype = np.dtype('uint32'), count = 1)
        rtSize = rtSize[0]
        #print(rtSize)
        bkOffset, vOffset = 0, 0
        if rtSize == 128: #iLab3.0
            bkOffset = 96 # frame header size
            vOffset = 16  # vector header size
        elif rtSize == 106: #iLab
            f.seek(102)  # seek(offset, from_what), from_what: 0 (default)- begin, 1 - current, 2 - end
            bkOffset = np.fromfile(f,dtype = np.dtype('int32'), count = 1)
        # print (rtSize, bkOffset, vOffset)
        frame_size = bkOffset + (nr+vOffset)*nc
        nframe = st.st_size//frame_size
        #print (nframe)
        # nframe = 2
        if rtSize == 128:
            rtSize = 128+2656+frame_size*15
            nframe = nframe - 15  # 7 zero-frames

        #print(rtSize, bkOffset, frame_size)
        f.seek(rtSize) # + bkOffset)
        rt_frames = np.fromfile(f,dtype = np.dtype('uint8'), count = nframe * frame_size)

    #print (rt_frames.shape[0])
    rt_frames  = rt_frames.reshape(nframe, frame_size)
    rt_frames = rt_frames[:, bkOffset:]
    rt_frames = rt_frames.reshape(nframe, nc, nr + vOffset)
    #print (rt_frames.shape)
    rt_frames = rt_frames[:,:,vOffset:]
    #print (rt_frames.shape)
    #plt.imshow(rt_frames[0,:,:], cmap = 'gray')
    if frame_index is not None:
        return rt_frames[frame_index,:,:]
    return rt_frames


def polar2cart(im_polar, zeropad=32, filled_mask = False):
    """
    Converts single frame polar array (256,256) to cartesian (512,512)
    """
    
    #add cath mask
    if filled_mask:
        cath_mask = np.ones((256,zeropad))
    else:
        cath_mask = np.zeros((256, zeropad))
    processed_polar = cv2.hconcat([cath_mask.astype(np.float64), im_polar.astype(np.float64)])
    
    #wrap to cartesian coordinates 
    dst = np.zeros((512, 512))
    flags = cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS
    im_cart = cv2.warpPolar(processed_polar, (512, 512), (256, 256), 256, flags)

    return im_cart

def cart2polar(im_cart):
    '''
    Convert Cartesian (H,H) to Polar (H,H)
    The Zeropad (Catheter) is not removed
    '''
    flags = cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS
    H,W = im_cart.shape
    assert H%2 == 0
    assert H==W
    im_polar = cv2.warpPolar(im_cart, (H,H), (H//2, H//2),H//2, flags)
    return im_polar

    

def border_to_mask(border_csv, cath_mask_radius=32):
    """
    Converts ALA csv format lumen/media border to cartesian segmentation mask (Nx512x512) 
    inputs: lumen/media border as numpy array (Nx256) , cath_mask_radius as integer 
    output: lumen/media border as segmentation mask
    """
    
    border_1d_frames = np.loadtxt(border_csv).astype(int)
    outbound_mask = np.zeros((len(border_1d_frames),512,512))
    
    for idx in tqdm(range(len(border_1d_frames))):
        border = border_1d_frames[idx,:]
        mask = np.zeros((256,256))

        #populate the mask 
        for i in range(256):
            for j in range(border[i]+1):
                mask[i,j]=1

        #add to array 
        outbound_mask[idx,:,:] = polar2cart(mask, zeropad = cath_mask_radius, filled_mask=True)
        
    return outbound_mask