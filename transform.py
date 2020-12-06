from lib import *

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	print(np.argmin(diff))
	print(np.argmax(diff))
	# return the ordered coordinates
	return rect
  


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def draw_box(img_OG, mask1):
    #plt.rcParams['figure.figsize'] = [16, 5]
    # img = cv2.imread(mask_path)
    mask = mask1
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_10 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    thresh_di = cv2.dilate(thresh, kernel_10, iterations=5)
    thresh_di_er = cv2.erode(thresh_di, kernel_5, iterations=3)
    # thresh_di_er = cv.morphologyEx(thresh_di_er, cv.MORPH_CLOSE, kernel_5)
    mask = thresh_di_er - thresh_di
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_OG = cv2.imread(img_path)
    rotrect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    img_draw = cv2.drawContours(img_OG, [box], 0, (255,0,255), 5)
    print(box.shape)

    font = cv2.FONT_HERSHEY_COMPLEX 
    # img_OG = cv2.imread(filename)
    rect = []
    for cnt in contours : 

        rotrect = cv2.minAreaRect(contours[0])
        approx = cv2.boxPoints(rotrect)
        approx = np.int0(approx)
        img_draw = cv2.drawContours(img_OG, [approx], 0, (255,0,255), 5)


    # approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
  
    # # draws boundary of contours. 
    # cv2.drawContours(img_OG, [approx], 0, (255, 0, 255), 5)  
  
    # Used to flatted the array containing 
    # the co-ordinates of the vertices. 
        n = approx.ravel()  
        i = 0
        count = 1
        for j in n : 
            if(i % 2 == 0): 
                x = n[i] 
                y = n[i + 1] 
  
            # String containing the co-ordinates. 
                string = str(x) + " " + str(y)  
  
                if(i == 0): 
                # text on topmost co-ordinate. 
                    print('0: '+string)
                    temp = (x,y)
                # rect.append((x,y))
                    cv2.putText(img_OG, "Arrow tip", (x, y), 
                                    font, 0.5, (255, 0, 0))  
                else: 
                # text on remaining co-ordinates. 
                    print('{}: '.format(count)+string)
                    count +=1
                    rect.append((x,y))
                    cv2.putText(img_OG, string, (x, y),  
                              font, 0.5, (0, 255, 0))  
            i = i + 1
    rect.append(temp)

    return img_OG, rect

def perspective_transform(img_OG, mask):
    mask = add_bg(mask)
    img_OG = add_bg(img_OG)
    image, rect = draw_box(img_OG, mask)

    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    pts = np.array(rect)
    warped = four_point_transform(img_OG, pts)

    if warped.shape[0] > warped.shape[1]:
        result = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE) 
    else:
        result = warped
    cv2.imwrite('result.png',result)
    
    result = true_rotate(result)
    

    return result

def true_rotate(result):

    class_names = ['down', 'up']
    num_classes = 2
    img_height = 482
    img_width = 337
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    model.load_weights('checkpoints/classification_3.h5')
    passport_path = "result.png"
    # img = keras.preprocessing.image.load_img(
    #     img, target_size=(img_height, img_width)
    # )
    img = tf.image.resize(result, [img_height,img_width], method='lanczos5')
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    result = cv2.imread(passport_path)
    if class_names[np.argmax(score)] == 'down' and np.max(score)*100 > 70:
        result = cv2.rotate(result, cv2.cv2.ROTATE_180)
    cv2.imwrite('images/result.png',result)
    
    return result

def add_bg(img):
  # convert to a PIL image
  img = Image.fromarray(img)
  w1, h1 = img.size
  img_bg = Image.new('RGB', (w1+50, h1+50), (255, 255, 255))
  w2, h2 = img_bg.size
  img_bg.paste(img, (int(w2/2-w1/2), int(h2/2-h1/2)))
  img_bg.convert('RGB')
  # use numpy to convert the pil_image into a numpy array
  numpy_image=np.array(img_bg)
  # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
  # the color is converted from RGB to BGR format
  img_bg=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
  return img_bg

if __name__ == "__main__":
    img_OG = cv2.imread('ho-chieu-passport-tre-em.jpg')
    mask = cv2.imread('images/mask.png')
    
    # image, rect = draw_box(img_OG, mask)

    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    # pts = np.array(rect)
    # warped = four_point_transform(img_OG, pts)
    
    # warped = true_rotate(warped)
    # show the original and warped images
    #img_OG = add_bg(img_OG)
    #mask = add_bg(mask)
    warped = perspective_transform(img_OG, mask)
    plt.figure(figsize=(16,10))
    plt.imshow(warped[:,:,::-1])
    plt.show()
    