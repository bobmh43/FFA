import numpy as np
import cv2
from skimage import filters, segmentation, morphology
# Current assumptions: blood vessel image only of the first image in the sequence. All other operations are for all images in the sequence.
# Take note: opencv's grayscale morphology ops are limited to flat structuring elements only! (Fine for our case.)
# Note: use cv2.add() instead of np.add() as the former does saturated arithmetic and the latter does modular arithmetic
#
#
# PROBLEM FIXES:
# - improved the removal of small vessels by first applying a median filter of size 19 (after comparing the results of various smoothing filters)
# - fixed the issue with make_large_vessels_mask: unsigned integer subtraction. I needed cv2.absdiff.
# - optimized large_vessel_size to exclude as much small vessel as possible
# make sure the original image is not overwritten


# ================================substeps================================
def create_line_element(length, angle):
    '''
    Create a line structuring element of `length` and at `angle`.\n
    Parameter `angle` must be in degrees (counterclockwise).
    '''
    # horizontal line
    structuring_element = np.zeros((length, length), dtype=np.uint8)
    structuring_element[length // 2] = 1

    # rotated to the angle.
    rot_mat = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1.0)
    structuring_element = cv2.warpAffine(structuring_element, rot_mat, structuring_element.shape)

    return structuring_element
    

def log_gabor_filter(shape, center_freq, sigma_on_f, theta, sigma_theta):
    """
    Create a 2D log-Gabor filter in the FFT ordering (matching np.fft.fft2 output).
    
    Args:
        shape: (rows, cols) output filter shape
        center_freq: center frequency in cycles/pixel (0 < center_freq <= 0.5)
        sigma_on_f: bandwidth parameter (ratio). Typical 0.4-0.9. Smaller -> narrower band.
        theta: orientation (radians). 0 -> horizontal cycles in image space.
        sigma_theta: angular bandwidth (radians).
    Returns:
        2D numpy array (float) of shape `shape` representing the filter (real, >=0).
    """
    rows, cols = shape
    # frequency grids in cycles/pixel (range [-0.5, 0.5) along each axis)
    fy = np.fft.fftfreq(rows)[:, None]   # shape (rows, 1)
    fx = np.fft.fftfreq(cols)[None, :]   # shape (1, cols)

    # radial frequency (always non-negative)
    radius = np.sqrt(fx**2 + fy**2)

    # avoid log(0) and define DC = 0 explicitly
    tiny = 1e-16
    radius_nozero = radius.copy()
    radius_nozero[radius_nozero == 0] = tiny

    # radial (log-Gaussian) component
    # log-Gabor: Gaussian on log-frequency axis
    # denominator uses log(sigma_on_f) following standard formulations
    log_rad = np.log(radius_nozero / float(center_freq))
    # if sigma_on_f <= 0 or ==1, log(sigma_on_f) becomes problematic; ensure reasonable value
    if sigma_on_f <= 0:
        raise ValueError("sigma_on_f must be > 0 (typical values: 0.4 - 0.9).")
    radial = np.exp(-(log_rad**2) / (2.0 * (np.log(sigma_on_f) ** 2)))

    # angular (orientation) component: Gaussian on angle difference, wrapped
    angle = np.arctan2(fy, fx)  # -pi..pi
    # minimal angular difference, handles wrap-around
    ang_diff = np.arctan2(np.sin(angle - theta), np.cos(angle - theta))
    angular = np.exp(-(ang_diff**2) / (2.0 * (sigma_theta ** 2)))

    # combine; explicitly set DC (radius==0) to 0 (log-Gabor has no DC)
    log_gabor = radial * angular
    log_gabor[radius == 0] = 0.0

    return log_gabor

def apply_log_gabor(img):
    return img

    # setup parameters
    N = max(img.shape)
    num_scales = int(np.log2(N)) - 1
    center_freqs = [1 / 2 ** s for s in range(1, num_scales + 1)]

    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    
    B = 2.0                      # bandwidth in octaves
    sigma_ln = (B * np.sqrt(np.log(2))) / 2.0
    sigma_on_f = np.exp(-sigma_ln)   # ~0.435 for B=2

    n_orient = 8
    sigma_theta = (np.pi / n_orient / 2.0) / np.sqrt(np.log(2))  # ~0.236 rad

    # creating and applying the filter bank and combining the results, per the paper.
    qn_list = []
    for center_freq in center_freqs:
        resps_of_one_scale = []
        for theta in thetas:
            img_f = np.fft.fft2(img)
            lg = log_gabor_filter(img.shape, center_freq, sigma_on_f, theta, sigma_theta)
            filtered_f = img_f * lg
            resp = np.fft.ifft(filtered_f)
            resps_of_one_scale.append(resp)
        q_n = np.sum(resps_of_one_scale, axis=0)
        qn_list.append(q_n)

    qn_arr = np.array(qn_list)
    P_numerator = np.sum(qn_arr * np.abs(qn_arr) ** 3, axis=0)
    P_denominator = np.sum(np.abs(qn_arr) ** 3, axis=0)
    P = P_numerator / P_denominator

    sigma = 3
    P_reg = P * np.abs(P) / (np.abs(P) ** 2 + sigma ** 2)
    res = np.real(P_reg)
    return res

# ================================steps================================
def remove_small_vessels(img, median_blur_radius, small_vessel_size):
    # NEW: median blur helps denoise the image for better results.
    img = cv2.medianBlur(img, median_blur_radius)

    # morphological openings using line elements
    thetas = (180 / 8 * i for i in range(8))
    elements = (create_line_element(small_vessel_size, angle) for angle in thetas)
    res_imgs = [cv2.morphologyEx(img, cv2.MORPH_OPEN, se) for se in elements]

    # applying these openings in all directions
    res_img = np.minimum.reduce(res_imgs)
    return res_img

def make_large_vessels_mask(img, large_vessel_size):
    # 2a. Difference between grayscale openings (line elements and their perpendiculars)
    thetas = (180 / 8 * i for i in range(8))
    thetas_perp = (180 * (1 / 2 +  i / 8) for i in range(8))

    elements = (create_line_element(large_vessel_size, angle) for angle in thetas)
    elements_perp = (create_line_element(large_vessel_size, angle) for angle in thetas_perp)

    res_imgs = [cv2.morphologyEx(img, cv2.MORPH_OPEN, se) for se in elements]
    res_imgs_perp = [cv2.morphologyEx(img, cv2.MORPH_OPEN, se_perp) for se_perp in elements_perp]

    delta_imgs = [cv2.absdiff(im, im_perp) for im, im_perp in zip(res_imgs, res_imgs_perp)]
    large_vessels_img = np.maximum.reduce(delta_imgs) # leakages suppressed
    after_diff = large_vessels_img.copy()

    # 2b. log Gabor quadrature filter
    large_vessels_img = apply_log_gabor(large_vessels_img)
    after_log_gabor = large_vessels_img.copy()

    # 2c. hystersis thresholding
    high = filters.threshold_otsu(large_vessels_img)
    low = high * 0.7
    large_vessels_img = filters.apply_hysteresis_threshold(large_vessels_img, high, low)

    return after_diff, after_log_gabor, large_vessels_img


def make_leakage_image(img_large_leak, large_vessels_mask):
    tmp = img_large_leak.copy()
    tmp[large_vessels_mask] = 0
    bg_intensity = round(np.mean(tmp[tmp > 0])) # approximate background intensity
    tmp[large_vessels_mask] = bg_intensity

    return tmp


def segment_blood_vessels(img):
    # 3a. Apply Log Gabor quadrature filters
    res_img = apply_log_gabor(img)

    #3b. Perform Chan-Vese graph cut segmentation
    res_img = segmentation.chan_vese(res_img)
    return res_img

# TODO
def get_reference_point(large_vessels_mask):
    # 1. skeletonization by thinning
    skeleton_mask = morphology.skeletonize(large_vessels_mask)

    # 2. removing short branch of the skeleton 
    # simple version: pruning
    # 3. building a graph and repeatedly removing short branches

    # 3. straight line adjustment (linear regression) and accumulation

    pass

def register_images(img_list, reference_point_list):
    # translate the images to the reference point of the first image
    center = reference_point_list[0]
    translation_mats = [np.array([[1, 0, center[0] - refpt[0]], [0, 1, center[1] - refpt[1]]]) for refpt in reference_point_list[1:]]
    translated_imgs = [cv2.warpAffine(img, mat, img.shape) for img, mat in zip(img_list[1:],translation_mats)]
    translated_imgs = [img_list[0]] + translated_imgs

    # rotate optimally
    first_img = img_list[0]

    def rotate_optimally(img):
        rot_mats = (cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), deg, 1.0) for deg in range(360))
        rot_imgs = [cv2.warpAffine(img, rot_mat, img.shape) for rot_mat in rot_mats]
        opt_deg = np.argmax([np.sum(rot_img * first_img) for rot_img in rot_imgs])
        return rot_imgs[opt_deg]
    
    registered_imgs = [first_img] + [rotate_optimally(img) for img in img_list[1:]]

    return registered_imgs


# ================================MAIN ROUTINES================================
def main_task_old():
    # TODO: load images: shell script, sys.argv, be able to handle glob.
    # TODO: get values for MEDIAN_BLUR_RADIUS, SMALL_VESSEL_SIZE, LARGE_VESSEL_SIZE
    img_list = None

    # Leakage and Blood Vessel Detection
    # 1. remove small vessels
    img_large_leak_list = [remove_small_vessels(img) for img in img_list]

    # 2. large vessels mask (boolean array)
    large_vessels_mask_list = [make_large_vessels_mask(img) for img in img_list]

    # leakage image (leakage only)
    leakage_image_list = [make_leakage_image(img_large_leak, large_vessels_mask) for img_large_leak, large_vessels_mask in zip(img_large_leak_list, large_vessels_mask_list)]

    # 3. blood vessels segmented
    blood_vessel_img = segment_blood_vessels(img_list[0])

    # Image registration
    # 1. find reference points
    ref_pt_list = [get_reference_point(large_vessels_mask) for large_vessels_mask in large_vessels_mask_list]

    # 2. image registration
    registered_image_list = register_images(img_list, ref_pt_list)

    # TODO: save these images or do more analysis

# an abridged version of the task for testing (using only one image).
def main_task():
    # default parameters
    LARGE_VESSEL_SIZE = 49
    SMALL_VESSEL_SIZE = 15
    MEDIAN_BLUR_RADIUS = 19

    # script arguments
    import argparse

    parser = argparse.ArgumentParser(description="A script to perform analysis on a Fundus Fluorescein Angiography (FFA) image. Possible output images include the leakage regions and the blood vessels. See the options below.")
    parser.add_argument("input_file", help="Image file to be processed")
    parser.add_argument("-l", "--large-vessel-size", type=int, default=LARGE_VESSEL_SIZE
    , help=f"Upper bound on the diameter of the large blood vessels, in pixels. The default is {LARGE_VESSEL_SIZE}.")
    parser.add_argument("-s", "--small-vessel-size", type=int, default=SMALL_VESSEL_SIZE, help=f"Upper bound on the diameter of the small blood vessels, in pixels. The default is {SMALL_VESSEL_SIZE}.")
    parser.add_argument("-m", "--median-blur-radius", type=int, default=MEDIAN_BLUR_RADIUS, help=f"Size of the median blur filter, in pixels. The default is {MEDIAN_BLUR_RADIUS}.")
    parser.add_argument("-i", "--images-to-output", choices=["leakage", "vessels", "both"], default="both", help="Whether the script should output just the leakage regions, just the blood vessels, or both (default).")
    parser.add_argument("-o", "--output-dir", help="Output directory. The outputs would be <output-dir>/<input_file>_leakage and <output-dir>/<input_file>_vessels. If not provided, will create the output files in the same directory as the input image file.")
    parser.add_argument("-d", "--display", action="store_true", help="Whether to display the input and output images.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Controls script verbosity.")
    
    args = parser.parse_args()

    # verbose
    if args.verbose:
        print(f"Processing {args.input_file}, with large vessel size {args.large_vessel_size}, small vessel size {args.small_vessel_size} and median blur radius {args.median_blur_radius}.")

    # load image
    img = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)

    # Leakage and Blood Vessel Detection
    # 1. remove small vessels
    img_large_leak = remove_small_vessels(img, args.median_blur_radius, args.small_vessel_size)

    # 2. large vessels mask
    after_diff, _, large_vessels_mask = make_large_vessels_mask(img, args.large_vessel_size)

    # leakage image (leakage only)
    leakage_img = make_leakage_image(img_large_leak, large_vessels_mask)

    # 3. blood vessels segmented
    if args.images_to_output != "leakage":
        blood_vessel_img = segment_blood_vessels(after_diff)

    # save images
    import os
    dtry, fle = os.path.split(args.input_file)
    name_no_ext, file_ext = os.path.splitext(fle)

    from pathlib import Path 
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dtry = args.output_dir
    
    if args.images_to_output != "vessels": 
        leak_file = name_no_ext + "_leakage" + file_ext      
        cv2.imwrite(os.path.join(dtry, leak_file), leakage_img)
        if args.verbose:
            print(f"Saved leakage image to {leak_file}.")
    if args.images_to_output != "leakage":   
        vessels_file = name_no_ext + "_vessels" + file_ext
        cv2.imwrite(os.path.join(dtry, vessels_file), blood_vessel_img.astype(np.uint8) * 255)  #bool to grayscale
        if args.verbose:
            print(f"Saved vessels image to {vessels_file}.")

    # display results
    if not args.display:
        return 
    
    import matplotlib.pyplot as plt
    title_font = {
        'size': 16,
        'weight': 'bold'
    }

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(131)
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Original", title_font)

    if args.images_to_output != "vessels":
        ax = fig.add_subplot(132)
        ax.imshow(leakage_img, cmap="gray")
        ax.set_axis_off()
        ax.set_title("Leakage Image", title_font)

    if args.images_to_output != "leakage":
        ax = fig.add_subplot(133)
        ax.imshow(blood_vessel_img, cmap="gray")
        ax.set_axis_off()
        ax.set_title("Segmented Blood Vessels", title_font)
        
    plt.show()



# an abridged version of the task for testing (using only one image).
def my_test():
    # setup
    FILENAME = "resources\\testing\\test_img.jpg"
    SMALL_VESSEL_SIZE = 15      # upper bound
    LARGE_VESSEL_SIZE= 49       # upper bound
    MEDIAN_BLUR_RADIUS = 19

    # load image
    img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)

    # Leakage and Blood Vessel Detection
    # 1. remove small vessels
    img_large_leak = remove_small_vessels(img, MEDIAN_BLUR_RADIUS, SMALL_VESSEL_SIZE)

    # 2. large vessels mask
    after_diff, _, large_vessels_mask = make_large_vessels_mask(img, LARGE_VESSEL_SIZE)

    # leakage image (leakage only)
    leakage_img = make_leakage_image(img_large_leak, large_vessels_mask)

    # 3. blood vessels segmented
    blood_vessel_img = segment_blood_vessels(after_diff)

    # save images
    import os
    name_no_ext, file_ext = os.path.splitext(FILENAME)
    cv2.imwrite(name_no_ext + "_no_small" + file_ext, img_large_leak)
    cv2.imwrite(name_no_ext + "_large_vessels_mask" + file_ext, large_vessels_mask.astype(np.uint8) * 255)  #bool to grayscale
    cv2.imwrite(name_no_ext + "_leakage" + file_ext, leakage_img)
    cv2.imwrite(name_no_ext + "_vessels" + file_ext, blood_vessel_img.astype(np.uint8) * 255)  #bool to grayscale

    # display results
    import matplotlib.pyplot as plt
    title_font = {
        'size': 16,
        'weight': 'bold'
    }

    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(231)
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Original", title_font)

    ax = fig.add_subplot(232)
    ax.imshow(img_large_leak, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Without Small Vessels", title_font)

    ax = fig.add_subplot(233)
    ax.imshow(large_vessels_mask, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Large Vessels Mask", title_font)

    ax = fig.add_subplot(234)
    ax.imshow(leakage_img, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Leakage Image", title_font)

    ax = fig.add_subplot(235)
    ax.imshow(blood_vessel_img, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Segmented Blood Vessels", title_font)

    plt.show()





if __name__ == "__main__":
    main_task()