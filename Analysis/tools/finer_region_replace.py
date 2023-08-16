import tifffile

imagepath = '/VMH.tiff'
savepath = './'
annotation = tifffile.imread('/annotation.tiff')

image = tifffile.imread(imagepath)
kuoVMH = 1
kuoVMHDM = image[0,50,369]
kuoVMHC = image[0,67,369]
kuoVMHVL = image[0,74,370]

# set the gray value in annotation file
VMH = 693
VMHDM = 1200
VMHC = 1300
VMHVL = 1400
print(image.shape)
image[image==kuoVMH] = VMH
image[image==kuoVMHDM] = VMHDM
image[image==kuoVMHC] = VMHC
image[image==kuoVMHVL] = VMHVL
image[0:527,0:100,0:455] = 0
tifffile.imwrite(savepath+'/finerVMH.tiff',image)

# replace finer region to annotation file
for x in range(100,370):
    for y in range(150,320):
        for z in range(200,350):
            if VMH[z,y,x]==0:
                continue
            else:
                annotation[z,y,x] = image[z,y,x]
            
tifffile.imwrite(savepath+'/goodannotation.tiff',annotation)