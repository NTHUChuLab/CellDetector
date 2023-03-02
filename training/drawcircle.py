
def drawcircle(file,savepath,radius):
	x = []
	y = []
	frame = []
	from PIL import Image
	import numpy as np
	import csv,cv2
	with open(file) as csvfile:
	     data = csv.DictReader(csvfile)
	     for index in data:
	         frame.append(index['frame'])
	         x.append(round(float(index['x [nm]'])/1800))
	         y.append(round(float(index['y [nm]'])/1800))
	frame.append(0)
	number = 0
	mask = np.zeros((1600, 2000)).astype(np.uint16)
	while number<(len(frame)-1):
	      mask = cv2.circle(mask, (x[number],y[number]), radius, (255,255,255), -1)
	      if frame[number] != frame[number+1]:
	         slice = frame[number]
	         maskimage = Image.fromarray(mask)
	         maskimage.save(savepath+'\\'+str(slice)+'_mask'+'.tif')
	         mask = np.zeros((1600, 2000)).astype(np.uint16)
	      number += 1

drawcircle(r'D:\JH\validation\1.csv', r'D:\JH\validation', 5)