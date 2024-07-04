# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    px_array_r = []
    px_array_g = []
    px_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        px_array_r.append(pixel_row_r)
        px_array_g.append(pixel_row_g)
        px_array_b.append(pixel_row_b)

    return (image_width, image_height, px_array_r, px_array_g, px_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_px_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_px_array.append(new_row)

    return new_px_array


###########################################
### You can add your own functions here ###
###########################################
def computeHistogram(px_array, image_width, image_height): ###extension
    bins = {}
    for row in range(image_height):
        for col in range(image_width):
            if px_array[row][col] in bins:
                        bins[px_array[row][col]] +=1
            else:
                        bins.update({px_array[row][col]:1})
    return bins

def computeCumulativeHistogram(px_array, image_width, image_height):
    bins = []
    count = image_height * image_width
    for row in range(image_height):
        for col in range(image_width):
            if px_array[row][col] not in bins:
                bins.append(px_array[row][col])
    bins.sort()

    result = len(bins) * [0]
    cumulative = len(bins) * [0]
    for row in range(image_height):
        for col in range(image_width):
            result[bins.index(px_array[row][col])] = result[bins.index(px_array[row][col])] + 1
            
    for i in range(len(bins)):
        for n in range(len(bins)):
            if n >= i:
                cumulative[n] = cumulative[n]+ result[i]

    return (cumulative,count,bins)



def contrastStretch(px_array, image_width, image_height):
    (cumulative,count,bins) = computeCumulativeHistogram(px_array, image_width, image_height)
    alpha = 0.05 * count
    beta = 0.95 * count
    alpha_index = -1
    beta_index = -1
    i=0
    while alpha_index == -1:
            if cumulative[i] > alpha:
                alpha_index = i
            i+=1

    j = len(bins)-1
    while beta_index == -1:
            if cumulative[j] < beta:
                beta_index = j
            j-=1

    qalpha = bins[alpha_index]
    qbeta = bins[beta_index]

    for row in range(image_height):
        for col in range(image_width):
            if ((255/(qbeta-qalpha))*(px_array[row][col]-qalpha)) < 0:
                px_array[row][col] = 0
            elif ((255/(qbeta-qalpha))*(px_array[row][col]-qalpha)) > 255:
                px_array[row][col] = 255
            else:
                px_array[row][col] = ((255/(qbeta-qalpha))*(px_array[row][col]-qalpha))

    return px_array

def computeRGBToGreyscaleandContrastStretch(px_array_r, px_array_g, px_array_b, image_width, image_height):
    
    greyscale_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for row in range(image_height):
        for col in range(image_width):
            greyscale_px_array[row][col] = round(0.3 * px_array_r[row][col]+0.6*px_array_g[row][col]+0.1*px_array_b[row][col])
    px_array = contrastStretch(greyscale_px_array, image_width, image_height)

    return px_array
def horizontalScharrFilter(px_array, image_width, image_height):
    result = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(0.0)
        result.append(new_row)


    for row in range(1,image_height-1):
        for col in range(1,image_width-1):

            topleft = px_array[row-1][col-1] * 3.0
            topmid = px_array[row-1][col] * 0.0
            topright = px_array[row-1][col+1] *-3.0
                
            midleft = px_array[row][col-1] *10.0
            centre = px_array[row][col] * 0.0
            midright = px_array[row][col+1] *-10.0
            
            botleft = px_array[row+1][col-1] * 3.0
            botmid = px_array[row+1][col] * 0.0
            botright = px_array[row+1][col+1] *-3.0

            result[row][col] = (topleft+topmid+topright+midleft+centre+midright+botleft+botmid+botright)/32
    return result

def verticalScharrFilter(px_array, image_width, image_height):
    result = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(0.0)
        result.append(new_row)

    for row in range(1,image_height-1):
        for col in range(1,image_width-1):
            vtopleft = px_array[row-1][col-1] * 3.0
            vtopmid = px_array[row-1][col] * 10.0
            vtopright = px_array[row-1][col+1] *3.0
                
            vmidleft = px_array[row][col-1] *0.0
            vcentre = px_array[row][col] * 0.0
            vmidright = px_array[row][col+1] *0.0
            
            vbotleft = px_array[row+1][col-1] * -3.0
            vbotmid = px_array[row+1][col] * -10.0
            vbotright = px_array[row+1][col+1] *-3.0
            result[row][col]=(vtopleft+vtopmid+vtopright+vmidleft+vcentre+vmidright+vbotleft+vbotmid+vbotright)/32
    return result

def scharrFilter(px_array, image_width, image_height):
    result = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(0.0)
        result.append(new_row)
    horizontal = horizontalScharrFilter(px_array, image_width, image_height)
    vertical = verticalScharrFilter(px_array, image_width, image_height)

    for row in range(1,image_height-1):
        for col in range(1,image_width-1):
            result[row][col] = abs(horizontal[row][col])+abs(vertical[row][col])

    return result

def meanFilter(px_array, image_width, image_height):
    result = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(0.0)
        result.append(new_row)


    for row in range(2,image_height-2):
        for col in range(2,image_width-2):
            topleftmost = px_array[row-2][col-2] * 1.0
            topleft = px_array[row-2][col-1] * 1.0
            topmid = px_array[row-2][col] * 1.0
            topright = px_array[row-2][col+1] * 1.0
            toprightmost = px_array[row-2][col+2] * 1.0

            upperleftmost = px_array[row-1][col-2] * 1.0
            upperleft = px_array[row-1][col-1] * 1.0
            uppermid = px_array[row-1][col] * 1.0
            upperright = px_array[row-1][col+1] * 1.0
            upperrightmost = px_array[row-1][col+2] * 1.0
                
            midleftmost = px_array[row][col-2] * 1.0
            midleft = px_array[row][col-1] * 1.0
            centre = px_array[row][col] * 1.0
            midright = px_array[row][col+1] * 1.0
            midrightmost = px_array[row][col+2] * 1.0
            
            underleftmost = px_array[row+1][col-2] * 1.0
            underleft = px_array[row+1][col-1] * 1.0
            undermid = px_array[row+1][col] * 1.0
            underright = px_array[row+1][col+1] * 1.0
            underrightmost = px_array[row+1][col+2] * 1.0

            botleftmost = px_array[row+2][col-2] * 1.0
            botleft = px_array[row+2][col-1] * 1.0
            botmid = px_array[row+2][col] * 1.0
            botright = px_array[row+2][col+1] * 1.0
            botrightmost = px_array[row+2][col+2] * 1.0
            
            result[row][col] = abs(topleftmost+topleft+topmid+topright+toprightmost+upperleftmost+upperleft+uppermid+upperright+upperrightmost+midleftmost+midleft+centre+midright+midrightmost+underleftmost+underleft+undermid+underright+underrightmost+botleftmost+botleft+botmid+botright+botrightmost)/25

    return result

def threshold(px_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(image_height):
        for col in range(image_width):
            if px_array[row][col] < 22:
                result[row][col] = 0
            else:
                result[row][col] = 255

    return result

def adaptiveThreshold(px_array, image_width, image_height): ###extension
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    histogram = computeHistogram(px_array,image_width, image_height)
    intensity_hist = computeHistogram(px_array,image_width, image_height)
    intensity_total = 0

    for key in histogram:
        intensity_total+= key*histogram[key]
        intensity_hist.update({key:key*histogram[key]})

    prev_threshold = 0
    threshold = round(intensity_total/(image_height*image_width))

    while prev_threshold != threshold:
        prev_threshold = threshold
        background = 0
        background_intensity = 0
        object = 0
        object_intensity = 0
        for key in histogram:
            if key < prev_threshold:
                background += histogram[key]
                background_intensity += intensity_hist[key]
            else:
                object += histogram[key]
                object_intensity += intensity_hist[key]

        threshold = round((background_intensity/background + object_intensity/object)/2)
    
    for row in range(image_height):
        for col in range(image_width):
            if px_array[row][col] < threshold:
                result[row][col] = 0
            else:
                result[row][col] = 255
    return result

def dilation(px_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    px_array = [[0] * image_width] + px_array
    px_array = [[0] * image_width] + px_array
    px_array.append([0]*image_width)
    px_array.append([0]*image_width)

    for row in range(image_height+4):
        px_array[row] = [0]+[0]+px_array[row]
        px_array[row].append(0)
        px_array[row].append(0)
    

    kernel = [[0,0,1,0,0],
              [0,1,1,1,0], 
              [1,1,1,1,1], 
              [0,1,1,1,0], 
              [0,0,1,0,0]]
    
    for row in range(2,image_height+2):
        for col in range(2,image_width+2):
            intersect = 0
            x = 0
            for check_row in range(row-2, row+3):
                y = 0
                for check_col in range(col-2, col+3):
                    if (px_array[check_row][check_col]*kernel[x][y]) > 0:
                        intersect += 1
                    y += 1
                x+=1
            if intersect > 0:
                result[row-2][col-2] = 255
            else:
                result[row-2][col-2] = 0

        
    return result

def erosion(px_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    px_array = [[0] * image_width] + px_array
    px_array = [[0] * image_width] + px_array
    px_array.append([0]*image_width)
    px_array.append([0]*image_width)

    for row in range(image_height+4):
        px_array[row] = [0]+[0]+px_array[row]
        px_array[row].append(0)
        px_array[row].append(0)
    
    

    kernel = [[0,0,1,0,0],
              [0,1,1,1,0], 
              [1,1,1,1,1], 
              [0,1,1,1,0], 
              [0,0,1,0,0]]
    
    for row in range(2,image_height+2):
        for col in range(2,image_width+2):
            intersect = 0
            x = 0
            for check_row in range(row-2, row+3):
                y = 0
                for check_col in range(col-2, col+3):
                    if (px_array[check_row][check_col]*kernel[x][y]) > 0:
                        intersect += 1
                    y += 1
                x+=1
            if intersect == 13:
                result[row-2][col-2] = 255
            else:
                result[row-2][col-2] = 0

        
    return result

def connectedComponents(px_array, image_width, image_height):
    label = 1
    result_dict = {}
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(image_height):
        for col in range(image_width):
            if px_array[row][col] > 0 and result[row][col] == 0:
                queue = []
                queue.append((row, col))
                result[row][col] = label

                while queue:
                    check = queue.pop(0)
                    check_row = check[0]
                    check_col = check[1]
                    pixel_direction = [(0, -1), (0, 1), (-1, 0), (1, 0)]

                    for i in pixel_direction:
                        d_row = check_row + i[0]
                        d_col = check_col + i[1]
                        if 0 <= d_row < image_height and 0 <= d_col < image_width and px_array[d_row][d_col] > 0 and result[d_row][d_col] == 0:
                            result[d_row][d_col] = label
                            queue.append((d_row, d_col))

                    if label in result_dict:
                        result_dict[label] +=1
                    else:
                        result_dict.update({label:1})
                
                label += 1
            
    for key in list(result_dict.keys()): ###extension
        if result_dict[key] < 10000 or result_dict[key] > 100000:
            del result_dict[key]

    return result, result_dict

                    

def boundingBox(px_array, connected_dict, image_width, image_height):
    bounding_dict = {}
    for key in connected_dict:
        bounding_dict[key] = [image_width,image_height,0,0] 

    for row in range(image_height):
        for col in range(image_width):
            if px_array[row][col] in bounding_dict:
                if col < bounding_dict[px_array[row][col]][0]:
                    bounding_dict[px_array[row][col]][0] = col
                if row < bounding_dict[px_array[row][col]][1]:
                    bounding_dict[px_array[row][col]][1] = row
                if col > bounding_dict[px_array[row][col]][2]:
                    bounding_dict[px_array[row][col]][2] = col
                if row > bounding_dict[px_array[row][col]][3]:
                    bounding_dict[px_array[row][col]][3] = row
    
    return list(bounding_dict.values())


def medianFilter(px_array, image_width, image_height): ###extension
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(2,image_height-2):
        for col in range(2,image_width-2):
            topleftmost = px_array[row-2][col-2] * 1.0
            topleft = px_array[row-2][col-1] * 1.0
            topmid = px_array[row-2][col] * 1.0
            topright = px_array[row-2][col+1] * 1.0
            toprightmost = px_array[row-2][col+2] * 1.0

            upperleftmost = px_array[row-1][col-2] * 1.0
            upperleft = px_array[row-1][col-1] * 1.0
            uppermid = px_array[row-1][col] * 1.0
            upperright = px_array[row-1][col+1] * 1.0
            upperrightmost = px_array[row-1][col+2] * 1.0
                
            midleftmost = px_array[row][col-2] * 1.0
            midleft = px_array[row][col-1] * 1.0
            centre = px_array[row][col] * 1.0
            midright = px_array[row][col+1] * 1.0
            midrightmost = px_array[row][col+2] * 1.0
            
            underleftmost = px_array[row+1][col-2] * 1.0
            underleft = px_array[row+1][col-1] * 1.0
            undermid = px_array[row+1][col] * 1.0
            underright = px_array[row+1][col+1] * 1.0
            underrightmost = px_array[row+1][col+2] * 1.0

            botleftmost = px_array[row+2][col-2] * 1.0
            botleft = px_array[row+2][col-1] * 1.0
            botmid = px_array[row+2][col] * 1.0
            botright = px_array[row+2][col+1] * 1.0
            botrightmost = px_array[row+2][col+2] * 1.0
            
            sorted_list = [topleftmost,topleft,topmid,topright,toprightmost,upperleftmost,upperleft,uppermid,upperright,upperrightmost,midleftmost,midleft,centre,midright,midrightmost,underleftmost,underleft,undermid,underright,underrightmost,botleftmost,botleft,botmid,botright,botrightmost]
            sorted_list.sort()
            result[row][col] = sorted_list[12]


    return result
            
def gaussianFilter(px_array, image_width, image_height): ###extension
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(2,image_height-2):
        for col in range(2,image_width-2):
            topleftmost = px_array[row-2][col-2] * 1.0
            topleft = px_array[row-2][col-1] * 2.0
            topmid = px_array[row-2][col] * 4.0
            topright = px_array[row-2][col+1] * 2.0
            toprightmost = px_array[row-2][col+2] * 1.0

            upperleftmost = px_array[row-1][col-2] * 2.0
            upperleft = px_array[row-1][col-1] * 4.0
            uppermid = px_array[row-1][col] * 8.0
            upperright = px_array[row-1][col+1] * 4.0
            upperrightmost = px_array[row-1][col+2] * 2.0
                
            midleftmost = px_array[row][col-2] * 4.0
            midleft = px_array[row][col-1] * 8.0
            centre = px_array[row][col] * 16.0
            midright = px_array[row][col+1] * 8.0
            midrightmost = px_array[row][col+2] * 4.0
            
            underleftmost = px_array[row+1][col-2] * 2.0
            underleft = px_array[row+1][col-1] * 4.0
            undermid = px_array[row+1][col] * 8.0
            underright = px_array[row+1][col+1] * 4.0
            underrightmost = px_array[row+1][col+2] * 12.0

            botleftmost = px_array[row+2][col-2] * 1.0
            botleft = px_array[row+2][col-1] * 2.0
            botmid = px_array[row+2][col] * 4.0
            botright = px_array[row+2][col+1] * 2.0
            botrightmost = px_array[row+2][col+2] * 1.0

            result[row][col] = abs(topleftmost+topleft+topmid+topright+toprightmost+upperleftmost+upperleft+uppermid+upperright+upperrightmost+midleftmost+midleft+centre+midright+midrightmost+underleftmost+underleft+undermid+underright+underrightmost+botleftmost+botleft+botmid+botright+botrightmost)/100

    return result

def laplacianFilter(px_array, image_width, image_height): ###extension
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(1,image_height-1):
        for col in range(1,image_width-1):
            upperleft = px_array[row-1][col-1] * 1.0
            uppermid = px_array[row-1][col] * 1.0
            upperright = px_array[row-1][col+1] * 1.0
           
            midleft = px_array[row][col-1] * 1.0
            centre = px_array[row][col] * -8.0
            midright = px_array[row][col+1] * 1.0

            underleft = px_array[row+1][col-1] * 1.0
            undermid = px_array[row+1][col] * 1.0
            underright = px_array[row+1][col+1] * 1.0

            result[row][col] = abs(upperleft+uppermid+upperright+midleft+centre+midright+underleft+undermid+underright)
    return result

# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'hard_case_3'
    input_filename = f'./Images/hard/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################
    
    greyscale_px_array = computeRGBToGreyscaleandContrastStretch(px_array_r, px_array_g, px_array_b, image_width, image_height)

    laplacian_array = laplacianFilter(greyscale_px_array, image_width, image_height)
    scharr_array = scharrFilter(greyscale_px_array, image_width, image_height)

    ###mean_array = meanFilter(scharr_array, image_width, image_height)
    ###second_mean_array = meanFilter(mean_array,image_width, image_height)
    ###third_mean_array = meanFilter(second_mean_array,image_width, image_height)

    median_array = medianFilter(laplacian_array, image_width, image_height)
    second_median_array = medianFilter(median_array, image_width, image_height)
    third_median_array = medianFilter(second_median_array, image_width, image_height)

    ###gaussian_array = gaussianFilter(scharr_array, image_width, image_height)
    ###second_gaussian_array = gaussianFilter(gaussian_array, image_width, image_height)
    ###third_gaussian_array = gaussianFilter(second_gaussian_array, image_width, image_height)

    adaptive_threshold_array = adaptiveThreshold(third_median_array, image_width, image_height)
    
    dilation_array = dilation(adaptive_threshold_array, image_width, image_height)
    second_dilation_array = dilation(dilation_array, image_width, image_height)
    third_dilation_array = dilation(second_dilation_array, image_width, image_height)
    fourth_dilation_array = dilation(third_dilation_array, image_width, image_height)


    connected_result, connected_dict = connectedComponents(fourth_dilation_array, image_width, image_height)

    bounding_box_list = boundingBox(connected_result, connected_dict, image_width, image_height)

    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
    
    ###bounding_box_list = [[150, 140, 200, 190]]  # This is a dummy bounding box list, please comment it out when testing your own code.
    px_array = pyplot.imread(input_filename)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)

        type = "" ###extension
        if (bbox_width * bbox_height) > 75000:
            type = "$2 Coin"
        elif 60000 < (bbox_width * bbox_height) < 75000:
            type = "50c Coin"
        elif 50000 < (bbox_width * bbox_height) < 60000:
            type = "$1 Coin"
        elif  (bbox_width * bbox_height) < 50000:
            type = "10c Coin"
        axs.text(bbox_min_x, bbox_max_y, type)

    total = "Number of Coins: " + str(len(bounding_box_list)) ###extension
    axs.text(0, 0, total)   

    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox_and_extension.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    