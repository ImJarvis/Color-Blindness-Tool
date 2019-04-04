import cv2
import numpy as np
import os
import concurrent.futures
from PIL import Image
from moviepy.editor import VideoFileClip
from skimage.measure import compare_ssim
import argparse
import logging


normalColorList = [
[255,255,255],
[255,255,204],
[255,255,153],
[255,255,102],
[255,255,51],
[255,255,00],
[255,204,255],
[255,204,204],
[255,204,153],
[255,204,102],
[255,204,51],
[255,204,00],
[255,153,255],
[255,153,204],
[255,153,153],
[255,153,102],
[255,153,51],
[255,153,00],
[255,102,255],
[255,102,204],
[255,102,153],
[255,102,102],
[255,102,51],
[255,102,00],
[255,51,255],
[255,51,204],
[255,51,153],
[255,51,102],
[255,51,51],
[255,51,00],
[255,00,255],
[255,00,204],
[255,00,153],
[255,00,102],
[255,00,51],
[255,00,00],
[204,255,255],
[204,255,204],
[204,255,153],
[204,255,102],
[204,255,51],
[204,255,00],
[204,204,255],
[204,204,204],
[204,204,153],
[204,204,102],
[204,204,51],
[204,204,00],
[204,153,255],
[204,153,204],
[204,153,153],
[204,153,102],
[204,153,51],
[204,153,00],
[204,102,255],
[204,102,204],
[204,102,153],
[204,102,102],
[204,102,51],
[204,102,00],
[204,51,255],
[204,51,204],
[204,51,153],
[204,51,102],
[204,51,51],
[204,51,00],
[204,00,255],
[204,00,204],
[204,00,153],
[204,00,102],
[204,00,51],
[204,00,00],
[153,255,255],
[153,255,204],
[153,255,153],
[153,255,102],
[153,255,51],
[153,255,00],
[153,204,255],
[153,204,204],
[153,204,153],
[153,204,102],
[153,204,51],
[153,204,00],
[153,153,255],
[153,153,204],
[153,153,153],
[153,153,102],
[153,153,51],
[153,153,00],
[153,102,255],
[153,102,204],
[153,102,153],
[153,102,102],
[153,102,51],
[153,102,00],
[153,51,255],
[153,51,204],
[153,51,153],
[153,51,102],
[153,51,51],
[153,51,00],
[153,00,255],
[153,00,204],
[153,00,153],
[153,00,102],
[153,00,51],
[153,00,00],
[102,255,255],
[102,255,204],
[102,255,153],
[102,255,102],
[102,255,51],
[102,255,00],
[102,204,255],
[102,204,204],
[102,204,153],
[102,204,102],
[102,204,51],
[102,204,00],
[102,153,255],
[102,153,204],
[102,153,153],
[102,153,102],
[102,153,51],
[102,153,00],
[102,102,255],
[102,102,204],
[102,102,153],
[102,102,102],
[102,102,51],
[102,102,00],
[102,51,255],
[102,51,204],
[102,51,153],
[102,51,102],
[102,51,51],
[102,51,00],
[102,00,255],
[102,00,204],
[102,00,153],
[102,00,102],
[102,00,51],
[102,00,00],
[51,255,255],
[51,255,204],
[51,255,153],
[51,255,102],
[51,255,51],
[51,255,00],
[51,204,255],
[51,204,204],
[51,204,153],
[51,204,102],
[51,204,51],
[51,204,00],
[51,153,255],
[51,153,204],
[51,153,153],
[51,153,102],
[51,153,51],
[51,153,00],
[51,102,255],
[51,102,204],
[51,102,153],
[51,102,102],
[51,102,51],
[51,102,00],
[51,51,255],
[51,51,204],
[51,51,153],
[51,51,102],
[51,51,51],
[51,51,00],
[51,00,255],
[51,00,204],
[51,00,153],
[51,00,102],
[51,00,51],
[51,00,00],
[000,255,255],
[000,255,204],
[000,255,153],
[000,255,102],
[000,255,51],
[000,255,00],
[000,204,255],
[000,204,204],
[000,204,153],
[000,204,102],
[000,204,51],
[000,204,00],
[000,153,255],
[000,153,204],
[000,153,153],
[000,153,102],
[000,153,51],
[000,153,00],
[000,102,255],
[000,102,204],
[000,102,153],
[000,102,102],
[000,102,51],
[000,102,00],
[000,51,255],
[000,51,204],
[000,51,153],
[000,51,102],
[000,51,51],
[000,51,00],
[000,00,255],
[000,00,204],
[000,00,153],
[000,00,102],
[000,00,51]
]

protanopiaColorList = [
[255,250,250],
[255,242,200],
[255,237,162],
[255,234,134],
[255,233,117],
[255,232,113],
[207,215,255],
[222,216,210],
[229,214,157],
[233,212,105],
[236,212, 53],
[236,211, 15],
[170,189,255],
[177,184,224],
[189,182,168],
[196,180,112],
[199,180, 58],
[200,179, 23],
[152,178,255],
[130,160,246],
[153,157,185],
[165,155,124],
[170,154, 66],
[172,154, 30],
[150,177,255],
[119,157,255],
[122,142,206],
[143,140,139],
[152,139, 74],
[154,139, 35],
[150,177,255],
[123,160,255],
[110,137,215],
[136,136,146],
[147,135, 78],
[150,135, 38],
[248,244,248],
[255,241,197],
[255,235,151],
[255,232,115],
[255,231, 92],
[255,230, 85],
[196,206,255],
[207,203,203],
[215,201,151],
[219,199,100],
[221,198, 49],
[221,198, 000],
[152,177,255],
[158,168,215],
[170,165,160],
[177,164,106],
[181,163, 54],
[181,162, 14],
[135,167,255],
[100,140,235],
[127,136,175],
[140,134,117],
[146,133, 60],
[148,132, 21],
[138,170,255],
[67,135,255],
[82,117,200],
[112,115,135],
[123,113, 70],
[126,113, 27],
[140,171,255],
[93,145,255],
[53,111,213],
[100,109,144],
[116,108, 76],
[120,108, 30],
[239,236,244],
[247,233,193],
[251,231,144],
[254,230, 94],
[255,229, 50],
[255,228, 28],
[185,197,250],
[196,193,198],
[203,191,147],
[207,189, 97],
[209,188, 47],
[210,188, 000],
[131,164,255],
[143,156,206],
[155,152,153],
[162,150,101],
[165,149, 50],
[166,149, 000],
[112,155,255],
[73,123,223],
[106,119,165],
[120,116,109],
[126,114, 55],
[127,114, 13],
[125,162,255],
[000,122,246],
[000, 94,191],
[80, 91,128],
[95, 89, 65],
[99, 89, 19],
[131,166,255],
[39,130,255],
[000,103,208],
[57, 83,143],
[85, 81, 74],
[90, 81, 23],
[234,231,240],
[241,228,191],
[246,226,141],
[248,225, 93],
[250,224, 42],
[250,224, 000],
[178,190,245],
[189,187,193],
[196,184,143],
[200,182, 94],
[202,181, 45],
[202,181, 000],
[110,152,254],
[133,147,199],
[145,143,147],
[152,140, 97],
[155,139, 47],
[155,139, 000],
[81,141,255],
[50,110,213],
[89,105,156],
[104,102,102],
[109,100, 50],
[111, 99, 000],
[110,154,255],
[000,116,234],
[000, 89,180],
[50, 70,118],
[70, 67, 58],
[74, 66, 11],
[122,161,255],
[000,124,248],
[000,101,202],
[000, 71,142],
[50, 55, 72],
[60, 54, 15],
[231,229,239],
[238,226,189],
[243,223,140],
[245,222, 92],
[247,221, 41],
[247,221, 000],
[174,187,242],
[185,183,191],
[192,180,142],
[196,178, 93],
[198,178, 44],
[198,177, 000],
[105,147,250],
[127,142,195],
[140,138,144],
[146,135, 94],
[149,134, 46],
[149,134, 000],
[37,129,255],
[32,103,205],
[80, 97,149],
[94, 93, 97],
[100, 91, 47],
[101, 91, 000],
[96,148,255],
[000,112,225],
[000, 84,170],
[25, 55,106],
[52, 51, 51],
[55, 50, 000],
[115,157,255],
[000,121,242],
[000, 98,196],
[000, 70,139],
[000, 36, 72],
[30, 27, 8],
[230,228,238],
[237,225,189],
[242,223,140],
[245,221, 92],
[246,221, 41],
[246,220, 000],
[173,186,242],
[184,182,191],
[191,179,141],
[195,178, 93],
[197,177, 43],
[197,176, 000],
[103,146,248],
[126,141,194],
[138,137,143],
[144,134, 94],
[147,133, 45],
[148,133, 000],
[000,126,254],
[26,102,204],
[78, 95,147],
[92, 91, 95],
[97, 89, 46],
[99, 88, 000],
[91,145,255],
[000,111,222],
[000, 83,166],
[13, 51,102],
[46, 46, 48],
[49, 44, 000],
[113,156,255],
[000,120,240],
[000, 96,193],
[000, 68,135],
[000, 35, 70],
]

deuteranopiaColorList = [

[255,232,239],
[255,223,200],
[255,218,173],
[255,215,157],
[255,213,148],
[255,213,146],
[225,216,253],
[241,210,203],
[252,205,153],
[255,202,111],
[255,200,87],
[255,199,80],
[176,188,249],
[197,181,199],
[210,176,149],
[218,172,98],
[222,171,42],
[223,170,00],
[133,167,245],
[160,159,195],
[177,153,146],
[187,149,94],
[191,147,34],
[192,147,00],
[103,155,242],
[138,146,193],
[157,139,143],
[167,135,92],
[173,132,28],
[174,134,00],
[94,152,241],
[131,143,192],
[151,136,142],
[162,131,91],
[168,128,26],
[169,130,00],
[255,234,253],
[255,222,204],
[255,216,171],
[255,212,151],
[255,211,140],
[255,211,137],
[203,204,255],
[222,198,205],
[234,193,155],
[241,190,106],
[245,188,59],
[246,198,00],
[143,174,251],
[170,167,201],
[186,161,152],
[196,157,101],
[201,155,50],
[202,154,00],
[76,151,246],
[123,141,197],
[145,134,148],
[158,129,97],
[164,126,43],
[166,127,00],
[000,141,239],
[87,126,194],
[118,117,145],
[133,111,95],
[140,108,39],
[143,109,00],
[000,140,236],
[75,122,192],
[110,113,144],
[126,106,94],
[133,103,38],
[136,105,00],
[244,228,255],
[255,221,208],
[255,214,169],
[255,210,145],
[255,209,132],
[255,208,128],
[185,196,255],
[206,189,207],
[219,184,157],
[228,181,108],
[232,179,63],
[233,178,42],
[111,164,253],
[146,155,204],
[166,148,154],
[178,144,104],
[183,142,55],
[166,149,00],
[000,144,243],
[80,127,199],
[115,118,150],
[132,112,100],
[140,109,49],
[142,108,00],
[000,141,232],
[000,115,194],
[73,97,146],
[98,89,97],
[108,85,45],
[112,86,00],
[000,140,229],
[000,115,192],
[56,91,144],
[87,83,95],
[99,78,44],
[103,79,00],
[235,223,255],
[255,221,211],
[255,213,167],
[255,208,139],
[255,207,124],
[255,206,121],
[173,190,255],
[195,183,209],
[210,178,159],
[218,174,110],
[223,172,66],
[224,172,46],
[81,156,254],
[128,147,205],
[151,140,156],
[164,135,106],
[170,133,59],
[172,132,33],
[000,144,238],
[000,118,201],
[88,106,152],
[111,99,103],
[121,95,53],
[123,94,17],
[000,140,228],
[000,118,194],
[000,89,149],
[61,71,99],
[79,64,49],
[83,64,00],
[000,139,225],
[000,118,191],
[000,90,148],
[37,61,96],
[63,53,47],
[69,53,00],
[231,221,255],
[252,219,212],
[255,212,166],
[255,207,136],
[255,205,120],
[255,205,116],
[167,187,255],
[189,180,209],
[204,174,160],
[214,171,111],
[218,169,67],
[219,168,48],
[57,153,255],
[117,143,206],
[143,135,157],
[157,130,107],
[164,128,60],
[165,134,00],
[000,144,236],
[000,120,201],
[67,99,154],
[97,91,104],
[109,87,55],
[112,86,23],
[000,140,226],
[000,120,194],
[000,93,153],
[000,59,101],
[55,49,51],
[61,47,9],
[000,139,223],
[000,118,191],
[000,93,150],
[000,63,103],
[19,30,48],
[34,26,00],
[230,220,255],
[251,218,212],
[255,211,166],
[255,207,135],
[255,205,119],
[255,205,114],
[165,187,255],
[187,179,209],
[203,174,160],
[212,170,111],
[217,168,68],
[218,168,49],
[51,152,255],
[114,142,207],
[141,134,157],
[155,129,108],
[162,126,61],
[163,126,37],
[000,144,236],
[000,120,201],
[61,98,154],
[94,90,105],
[106,85,56],
[109,84,24],
[000,140,226],
[000,120,194],
[000,94,154],
[000,62,104],
[47,45,52],
[54,42,12],
[000,139,223],
[000,118,190],
[000,94,151],
[000,65,104],
[000,33,53]
]

tritanopiaColorList = [
[244,240,255],
[253,239,255],
[255,234,249],
[255,230,245],
[255,229,243],
[255,228,242],
[251,209,225],
[255,202,216],
[255,196,209],
[255,192,205],
[255,191,202],
[255,190,202],
[246,169,181],
[252,159,170],
[255,153,162],
[255,149,158],
[255,148,157],
[255,148,156],
[242,135,145],
[248,121,129],
[253,110,116],
[255,102,108],
[255,101,106],
[255,101,105],
[240,113,120],
[247,94,99],
[251,76,79],
[254,61,62],
[255,51,50],
[255,51,49],
[240,106,113],
[246,85,90],
[250,64,66],
[253,43,40],
[254,26,000],
[254,28,000],
[203,239,255],
[211,239,255],
[217,238,255],
[221,238,255],
[224,238,255],
[224,238,255],
[198,209,225],
[206,202,217],
[211,196,211],
[215,193,207],
[216,191,205],
[217,190,204],
[191,169,182],
[199,159,171],
[205,152,162],
[208,146,157],
[210,144,154],
[211,143,153],
[187,135,145],
[195,121,130],
[201,110,117],
[204,101,108],
[206,96,103],
[207,95,101],
[184,113,120],
[193,94,100],
[198,76,80],
[202,61,63],
[204,51,52],
[204,48,48],
[183,106,113],
[192,85,90],
[198,64,67],
[202,43,43],
[203,23,11],
[204,22,00],
[166,239,255],
[172,239,255],
[176,238,255],
[180,238,255],
[181,237,255],
[182,237,255],
[145,209,225],
[156,202,217],
[163,196,211],
[168,193,207],
[170,191,205],
[171,190,205],
[134,169,182],
[146,159,171],
[154,151,163],
[159,146,157],
[162,144,154],
[162,143,153],
[126,135,145],
[140,121,130],
[148,109,117],
[154,101,108],
[156,96,103],
[157,95,102],
[122,112,120],
[136,93,100],
[145,76,81],
[150,61,64],
[153,51,53],
[154,48,50],
[121,106,113],
[135,85,91],
[144,64,68],
[150,43,44],
[152,22,18],
[153,17,000],
[136,239,255],
[139,239,255],
[142,238,255],
[144,238,255],
[146,237,255],
[146,237,255],
[87,209,225],
[107,202,217],
[119,196,211],
[126,193,207],
[129,191,205],
[130,190,205],
[62,169,182],
[90,159,171],
[105,151,163],
[113,146,157],
[116,144,154],
[117,143,154],
[34,135,145],
[77,121,130],
[94,109,117],
[103,101,108],
[107,96,103],
[108,95,102],
[000,113,121],
[68,93,100],
[88,76,81],
[98,61,65],
[102,51,54],
[103,48,51],
[000,107,115],
[66,84,91],
[86,63,68],
[96,43,45],
[101,22,21],
[102,11,000],
[116,239,255],
[118,239,255],
[120,238,255],
[121,238,255],
[122,237,255],
[122,237,255],
[000,209,224],
[62,202,217],
[84,196,211],
[94,193,207],
[99,191,205],
[100,191,215],
[000,171,183],
[000,160,172],
[57,151,163],
[73,146,157],
[79,144,154],
[81,143,154],
[000,144,153],
[000,127,135],
[20,109,118],
[53,101,109],
[63,96,104],
[65,95,102],
[000,129,135],
[000,106,112],
[000,80,86],
[38,60,65],
[51,50,54],
[54,48,51],
[000,124,130],
[000,100,105],
[000,71,75],
[33,42,45],
[48,21,23],
[51,6,000],
[110,239,255],
[112,239,255],
[114,238,255],
[115,238,255],
[115,237,255],
[115,237,255],
[000,208,223],
[41,202,217],
[71,196,211],
[84,193,207],
[89,191,205],
[90,191,205],
[000,172,183],
[000,161,172],
[31,151,163],
[57,146,157],
[66,144,154],
[68,143,154],
[000,146,154],
[000,130,138],
[000,114,122],
[21,101,109],
[42,96,104],
[45,95,102],
[000,131,137],
[000,111,116],
[000,89,94],
[000,67,71],
[10,50,54],
[23,48,51],
[000,127,133],
[000,106,110],
[000,81,85],
[000,55,57],
[000,28,29]
]

# Getting our data strutcure for fast lookup and applying correct filters
normalFilter = set(tuple(i) for i in normalColorList)
protanopiaFilter = set(tuple(i) for i in protanopiaColorList)
deuteranopiaFilter = set(tuple(i) for i in deuteranopiaColorList)
tritanopiaFilter = set(tuple(i) for i in tritanopiaColorList)
""" 
frameName = "./original/"+str(videoName)+"_Frame_"+str(frameCounter)+".jpg"
pixMarkedName = "./output/"+str(videoName)+"_Frame_"+str(frameCounter)+".jpg"
maskedName = "./difference/"+str(videoName)+"_Frame_"+str(frameCounter)+".jpg" """
#default filter


def chooseFilterList(inputFilterName):
    if inputFilterName == "Proto":
        filterResult = (normalFilter.union(deuteranopiaFilter)).union(tritanopiaFilter)
        logging.debug("Filter selected is Prota." + "Size of filter list = " + str(len(filterResult)))
    elif inputFilterName == 'Deuta':
        filterResult = (normalFilter.union(protanopiaFilter)).union(tritanopiaFilter)
        logging.debug("Filter selected is Deuta"+ "Size of filter list = " + str(len(filterResult)))
    elif inputFilterName == 'Trita':
        filterResult = (normalFilter.union(deuteranopiaFilter)).union(protanopiaFilter)
        logging.debug("Filter selected is Trita"+ "Size of filter list = " + str(len(filterResult)))
    
    return filterResult


def takeLogs():
    #printing all the logs
    try:
        log_file = open("ColorBlindness_Logger.log","w")
    except OSError: 
        print ('Error: Creating log File')
    
    logging.basicConfig(filename='./ColorBlindness_Logger.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
    return

def toskiporNot(imageA,imageB,ssimIndex):
    
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    #print("inside ","getAlltheFrames")
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB,full=True)#,use_sample_covariance=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))
    if score < ssimIndex:
        logging.debug("SSIM Score = "+ str(score))
        return True
    else:
        return False


def countPixel(picture,frameName,maskedName,filterToCheck):
    data = np.asarray(picture)
    numberofMatchedPixel =0
    flag = False
    logging.debug("inside CountPixel")
    imgRGB=cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    inputFrame = Image.fromarray(imgRGB)
    pix_in = inputFrame.load()
    #outFrame = Image.new(inputFrame.mode,inputFrame.size)
    maskedFrame = Image.new(inputFrame.mode,inputFrame.size)
    #pix_out = outFrame.load()
    masked_out = maskedFrame.load()
    i=0
    timestampOld = cv2.getTickCount()
    for elem in data:
        j=0
        for xy in elem:
            #str= np.array2string(xy)
            #print(str)
            if tuple(xy) in normalFilter:
                flag = True
                #logging.debug("Matched Pixel index = "+str(j)+","+str(i) +" and value ="+str(tuple(xy)))
                numberofMatchedPixel +=1
                masked_out[j,i] = (255,255,255,255)
                #pix_out[j,i] = (0,0,0,255)
            else:
                masked_out[j,i] = (0,0,0,255)#pix_in[j,i]
                #pix_out[j,i] = pix_in[j,i]
                #print("Not Matched Pixel index = ",j,i ," and value =",tuple(xy))
            
            j=j+1
                #break
        i=i+1
    timestampNew = cv2.getTickCount()
    totalTime = (timestampNew-timestampOld)/cv2.getTickFrequency()
    logging.debug("Total Time Taken for Looping over all pixels = "+ str( totalTime ))
    
    if(flag == True):
        maskedFrame.save(maskedName)
        #outFrame.save(pixMarkedName)
        cv2.addWeighted(np.asarray(maskedFrame), 0.8, picture, 0.2,0, picture)
        cv2.imwrite(frameName,picture)
    else:  
        inputFrame.close()
        #outFrame.close()
        maskedFrame.close()
    
    logging.debug("Total number of Matched Pixels = "+ str(numberofMatchedPixel))
    timestampNew = cv2.getTickCount()
    totalTime = (timestampNew-timestampOld)/cv2.getTickFrequency()
    logging.debug("Total Time Taken for extracting Frames = "+ str( totalTime ))
    return numberofMatchedPixel
   


def getAlltheFrames(videoSource,videoName,filterToCheck,ssimIndex):
    #print("inside ","getAlltheFrames")
    frameCounter = 0
    success ,firstframe = videoSource.read()
    fps = videoSource.get(cv2.CAP_PROP_FPS)
    resizedframe = cv2.resize(firstframe,(640,360))
    executor = concurrent.futures.ThreadPoolExecutor(2000)
    totalFrame = videoSource.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while(success):
        
        success ,frame = videoSource.read()
        frameName = "./output/"+str(videoName)+"_Frame_"+str(frameCounter)+".jpg"
        #pixMarkedName = "./output/"+str(videoName)+"_Frame_"+str(frameCounter)+".jpg"
        maskedName = "./difference/"+str(videoName)+"_Frame_"+str(frameCounter)+".jpg"
        if success:
            if toskiporNot(firstframe,frame,ssimIndex) or frameCounter == 0:
            
                print("Working on Frame number #",frameCounter+1,"Out of ", int(totalFrame))
                futures = executor.submit(countPixel,resizedframe,frameName,maskedName,filterToCheck)
                logging.debug("Thread No # "+ str(executor._counter()))
                logging.debug("Processed frame No #" + str(frameCounter+1) +" and Bad Pixels No # "+str(futures.result()))
                firstframe = frame
            frameCounter = frameCounter +1
            resizedframe = cv2.resize(frame,(640,360))
        else:
            break
        
    return frameCounter

def checkFilter(inputFilter):
    filterOptions = set({"Proto","Deuta","Trita"})
    if inputFilter in filterOptions:
        return True
    else:
        return False

def main():
    
    #Input and Logging work
    #videoPath = 'test.mp4'
    
############################################################
# Command line parsing.
############################################################
    parser = argparse.ArgumentParser(description='This tool will try to find if the color blindness filter is successfully applied to the input video file.\nIt will '
    +'report those frame where there is still a problem. You can check for the buggy frames in "output" folder.'
    +'\n\nTool author - Siddharth Srivastava' ,
    formatter_class=argparse.RawDescriptionHelpFormatter)
    # Add required, positional, arguments.
    parser.add_argument('inputVideo', type = str,help = 'input Video file full path')
    parser.add_argument('filterName', type =str, help ="input the color blindness filter name which need to be tested. "
    +" Valid options are ---> Proto | Deuta | Trita")
    
    parser.add_argument('--ssim', type = float,default = 0.4, help= 'Only give a value between 0 and 1')
    #+" This is a threshold value for frames comparision. More towards 1 means, no of output frames will increase , More towards 0 means no of output frames will decrease")

    args = parser.parse_args()
    
    # Basic error checking of command line options.
    assert(os.path.isfile(args.inputVideo)),\
         'Input Video (' + args.inputVideo + ') not found.'
    
    videoPath = args.inputVideo
    filterName = args.filterName
    ssimIndex = args.ssim
    
    assert(checkFilter(filterName)),\
        args.filterName+' is Wrong Filter option. Select either --> Proto or Deuta or Trita"'
    
    assert((0< ssimIndex < 1)),\
        args.ssimIndex + " is not a valid value. Please give a value between 0 and 1"

    videoSource = cv2.VideoCapture(videoPath)
    clip = VideoFileClip(videoPath)
    videoDuration =  clip.duration 
    clip.close()
    fileName = os.path.basename(videoPath)
    takeLogs()
    try: 
        if not (os.path.exists('output')):
            os.makedirs('output')
        
        if not (os.path.exists('difference')):
            os.makedirs('difference')
    
    # if not created then raise error 
    except OSError: 
        logging.debug ('Error: Creating directory of data') 
    


############################################################
# Command line parsing ends here.
############################################################
    

    #Core Business Logic Goes Here
    filterToCheck = chooseFilterList(filterName)
    logging.debug("$####################### Input Values #####################################")
    logging.debug("File Path ="+ str(fileName) + "  Filter Options selected = " +str(filterName))
    logging.debug("Filter list elements size =" + str(len(filterToCheck)))
    logging.debug("SSIM index input value = " +str(ssimIndex))
    logging.debug("#######################  Input Ends  ##################################$")

    frameCount = videoSource.get(cv2.CAP_PROP_FRAME_COUNT)
    executor1 = concurrent.futures.ThreadPoolExecutor(5)
    print(executor1._counter())
    timestampOld = cv2.getTickCount()
    futures1 = executor1.submit(getAlltheFrames,videoSource,fileName,filterToCheck,ssimIndex)
    logging.info("Threads for frame result of  is "+str(futures1.result()))

    timestampNew = cv2.getTickCount()
    totalTime = (timestampNew-timestampOld)/cv2.getTickFrequency()
    print()
    print("++++++++++++++++ RESULT SUMMARY ++++++++++++++++")
    print()
    print("Total Time Taken by Tool= ",format(totalTime/60,'.2f')," minutes" , "\n\rFor Video Source        = ", videoPath , "\n\rWhose total length      = ",format(videoDuration/60,'.2f')," minutes " ,
         "\n\rand Video fps           = ",round(videoSource.get(cv2.CAP_PROP_FPS)))
    print()
    print("++++++++++++++++ END ++++++++++++++++")

    #Clean-Up
    videoSource.release()
    cv2.destroyAllWindows()
    logging.debug("Total Time take by tool = "+str(format(totalTime/60,'.2f')+" minutes"))
    return

    

if __name__ == "__main__":
    main()
