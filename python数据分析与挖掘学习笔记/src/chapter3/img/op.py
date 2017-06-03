from PIL import Image

images = [Image.open('temp_1.png'), Image.open('temp_2.png')]

size_0 = images[0].size
size_1 = images[1].size

width = size_0[0] if size_0[0] > size_1[0] else size_1[0]
heigth = size_0[1] + size_1[1]

target = Image.new('RGB', (width, heigth), 255)

y = 0
for image in images:
  target.paste(image, (0, y))
  y += image.size[1]
target.save('spearman.png')
