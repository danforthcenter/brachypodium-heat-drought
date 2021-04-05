
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


# In[2]:


img, path, filename = pcv.readimage(filename='/shares/mgehan_share/mgehan/brachypodium-2020-ella/brachy-drought/B002dr_022315/snapshot199157/VIS_SV_90_z2500_h2_g0_e82_269326_0.png')


# In[3]:


img_wb = pcv.white_balance(img, mode='max', roi=(600, 250, 80, 80))


# In[4]:


s = pcv.rgb2gray_hsv(rgb_img=img_wb, channel='s')


# In[15]:


s_thresh = pcv.threshold.binary(gray_img=s, threshold=30, max_value=255, object_type='light')


# In[6]:


b = pcv.rgb2gray_lab(rgb_img=img_wb, channel='b')


# In[11]:


b_thresh = pcv.threshold.binary(gray_img=b, threshold=130, max_value=255, object_type='light')


# In[16]:


bs = pcv.logical_and(bin_img1=s_thresh, bin_img2=b_thresh)


# In[17]:


fill = pcv.fill(bs, 50)


# In[18]:


masked = pcv.apply_mask(img=img_wb, mask=fill, mask_color='white')


# In[19]:


id_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=fill)


# In[20]:


roi_contour, roi_hierarchy= pcv.roi.rectangle(img=img_wb, x=700, y=500, h=670, w=920)


# In[21]:


roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img_wb, roi_contour, roi_hierarchy, 
                                                              id_objects, obj_hierarchy, 'partial')


# In[22]:


pcv.params.line_thickness = 5


# In[23]:


obj, mask = pcv.object_composition(img=img_wb, contours=roi_objects, hierarchy=hierarchy)


# In[24]:


shape_img = pcv.analyze_object(img=img_wb, obj=obj, mask=mask)


# In[25]:


color_histogram = pcv.analyze_color(img_wb, mask, 'hsv')


# In[26]:


height_img = pcv.analyze_bound_horizontal(img=img_wb, obj=obj, mask=mask, 
                                                     line_position=1170)


# In[27]:


pseudocolored_img = pcv.visualize.pseudocolor(gray_img=s, mask=kept_mask, cmap='jet')


# In[28]:


a = pcv.rgb2gray_hsv(img_wb,"v")
inv_mask = pcv.invert(mask)
pmasked = cv2.applyColorMap(a,cv2.COLORMAP_JET)
masked = pcv.apply_mask(mask=mask,img=pmasked,mask_color="black")
masked1 = pcv.apply_mask(mask=inv_mask,img=img_wb,mask_color="black")
sum_img = pcv.image_add(masked, masked1)


# In[29]:


iy,ix,iz = np.shape(img_wb)
blank = (np.ones((iy,500,3),dtype=np.uint8))*255
vis1 = np.concatenate((img_wb, blank), axis=1)
vis2 = np.concatenate((vis1, sum_img), axis=1)
vis3 = cv2.resize(vis2, (720, 251))
outname = args.outdir+"/"+str(filename[:-4])+"_pseudocolor.png"
pcv.print_image(vis3,outname)


# In[30]:


pcv.print_results(filename=args.result)


# In[31]:


if args.coresult is not None:
        nirpath = pcv.get_nir(path,filename)
        nir, path1, filename1 = pcv.readimage(nirpath)
        nir2 = cv2.imread(nirpath,0)


# In[32]:


nmask = pcv.resize(img=mask, resize_x=0.13, resize_y=0.125)


# In[35]:


newmask = pcv.crop_position_mask(img=nir, mask=nmask, x=48, y=3, 
                                 v_pos="top", h_pos="left")


# In[36]:


nir_objects, nir_hierarchy = pcv.find_objects(img=nir, mask=newmask)


# In[37]:


pcv.params.line_thickness = 1


# In[38]:


nir_combined, nir_combinedmask = pcv.object_composition(img=nir, contours=nir_objects, 
                                                        hierarchy=nir_hierarchy)


# In[39]:


nir_hist = pcv.analyze_nir_intensity(gray_img=nir2, mask=nir_combinedmask, 
                                     bins=256, histplot=True)


# In[40]:


nir_shape_image = pcv.analyze_object(img=nir2, obj=nir_combined, mask=nir_combinedmask)


# In[41]:


pcv.print_image(img=nir_hist, filename=os.path.join(pcv.params.debug_outdir, 'nirhist.png'))
pcv.print_image(img=nir_shape_image, filename=os.path.join(pcv.params.debug_outdir, 'nirshape.png'))


# In[47]:


pcv.print_result(filename=args.coresult)

if __name__ == '__main__':
main()

