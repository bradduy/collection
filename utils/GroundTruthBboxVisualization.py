import cv2

# Ground truth path 
filename = 'home/user/'
# Image folder path
image_folder = 'home/user/images'
visualized_image_folder = 'home/user/bbox_images'

with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content]
array = []
listImage = []

# Make list paths of images
for element in content:

    check = element[-2:]

    if str(check) == ' 0':
        array.append(element)

    if str(element[-4:]) == '.png':
        listImage.append(element)

# Based on list paths of images, drawing bounding box on them
for i in range(len(listImage)):

    img = cv2.imread(image_folder + listImage[i][-16:])
    bbox = array[i].split(' ')

    bbox[0] = round(float(bbox[0]))
    bbox[1] = round(float(bbox[1]))
    bbox[2] = round(float(bbox[2]))
    bbox[3] = round(float(bbox[3]))

    y1 = int(bbox[0]) + int(bbox[2])
    y2 = int(bbox[1]) + int(bbox[3])

    image = cv2.rectangle(img, (bbox[0], bbox[1]), (y1, y2), (0, 0, 255), thickness=5)

    cv2.imwrite(visualized_image_folder + listImage[i][-16:], image)
    i += 1
    cv2.waitKey(0)

