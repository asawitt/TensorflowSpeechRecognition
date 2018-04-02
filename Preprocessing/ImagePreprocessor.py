import FileUtil 
from PIL import Image




#Grayscale
for filename in FileUtil.get_all_processed_filenames():
	print(filename)
	Image.open(filename).convert('L').save(filename)









