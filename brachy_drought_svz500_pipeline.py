
# coding: utf-8

# In[1]:


#!/usr/bin/python
import os
import sys, traceback
import cv2
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv

class options:
    def __init__(self):
        self.debug = "plot"
        self.writeimg= False
        self.result = "./Brachy_results"
        self.coresult = "./Brachy_coresult"
        self.outdir = "."
        
args = options()

pcv.params.debug= 'plot'


# In[90]:


img, path, filename = pcv.readimage(filename='/shares/mgehan_share/mgehan/brachypodium-2020-ella/brachy-drought/B002dr_022315/snapshot276519/VIS_SV_0_z500_h1_g0_e82_380053_0.png')


# In[91]:


img_wb = pcv.white_balance(img, mode='max', roi=(600, 250, 80, 80))


# In[92]:


s = pcv.rgb2gray_hsv(rgb_img=img_wb, channel='s')


# In[121]:


s_thresh = pcv.threshold.binary(gray_img=s, threshold=30, max_value=255, object_type='light')


# In[94]:


b = pcv.rgb2gray_lab(rgb_img=img_wb, channel='b')


# In[95]:


b_thresh = pcv.threshold.binary(gray_img=b, threshold=133, max_value=255, object_type='light')


# In[96]:


bs = pcv.logical_and(bin_img1=s_thresh, bin_img2=b_thresh)


# In[97]:


fill = pcv.fill(bs, 50)


# In[98]:


masked = pcv.apply_mask(img=img_wb, mask=fill, mask_color='white')


# In[99]:


id_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=fill)


# In[100]:


roi_contour, roi_hierarchy= pcv.roi.rectangle(img=img_wb, x=850, y=1100, h=620, w=700)


# In[101]:


roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img_wb, roi_contour, roi_hierarchy, 
                                                              id_objects, obj_hierarchy, 'partial')


# In[102]:


pcv.params.line_thickness = 5


# In[103]:


obj, mask = pcv.object_composition(img=img_wb, contours=roi_objects, hierarchy=hierarchy)


# In[104]:


shape_img = pcv.analyze_object(img=img_wb, obj=obj, mask=mask)


# In[105]:


color_histogram = pcv.analyze_color(img_wb, mask, 'hsv')


# In[106]:


height_img = pcv.analyze_bound_horizontal(img=img_wb, obj=obj, mask=mask, 
                                                     line_position=1755)


# In[107]:


pseudocolored_img = pcv.visualize.pseudocolor(gray_img=s, mask=kept_mask, cmap='jet')


# In[108]:


a = pcv.rgb2gray_hsv(img_wb,"v")
inv_mask = pcv.invert(mask)
pmasked = cv2.applyColorMap(a,cv2.COLORMAP_JET)
masked = pcv.apply_mask(mask=mask,img=pmasked,mask_color="black")
masked1 = pcv.apply_mask(mask=inv_mask,img=img_wb,mask_color="black")
sum_img = pcv.image_add(masked, masked1)


# In[109]:


iy,ix,iz = np.shape(img_wb)
blank = (np.ones((iy,500,3),dtype=np.uint8))*255
vis1 = np.concatenate((img_wb, blank), axis=1)
vis2 = np.concatenate((vis1, sum_img), axis=1)
vis3 = cv2.resize(vis2, (720, 251))
outname = args.outdir+"/"+str(filename[:-4])+"_pseudocolor.png"
pcv.print_image(vis3,outname)


# In[110]:


pcv.print_results(filename=args.result)


# In[111]:


if args.coresult is not None:
        nirpath = pcv.get_nir(path,filename)
        nir, path1, filename1 = pcv.readimage(nirpath)
        nir2 = cv2.imread(nirpath,0)


# In[112]:


nmask = pcv.resize(img=mask, resize_x=0.13, resize_y=0.125)


# In[113]:


newmask = pcv.crop_position_mask(img=nir, mask=nmask, x=20, y=3, 
                                 v_pos="top", h_pos="left")


# In[114]:


nir_objects, nir_hierarchy = pcv.find_objects(img=nir, mask=newmask)


# In[115]:


pcv.params.line_thickness = 1


# In[116]:


nir_combined, nir_combinedmask = pcv.object_composition(img=nir, contours=nir_objects, 
                                                        hierarchy=nir_hierarchy)


# In[117]:


nir_hist = pcv.analyze_nir_intensity(gray_img=nir2, mask=nir_combinedmask, 
                                     bins=256, histplot=True)


# In[118]:


nir_shape_image = pcv.analyze_object(img=nir2, obj=nir_combined, mask=nir_combinedmask)


# In[119]:


pcv.print_image(img=nir_hist, filename=os.path.join(pcv.params.debug_outdir, 'nirhist.png'))
pcv.print_image(img=nir_shape_image, filename=os.path.join(pcv.params.debug_outdir, 'nirshape.png'))


# In[120]:


pcv.print_result(filename=args.coresult)

if __name__ == '__main__':
main()

