from lib import *
from transform import *


# def load(image):
# #    filename = os.path.join(filename, name) 
# #    np_image = Image.open(filename)
# #    np_image = cv2.imread(filename)
#    np_image = np.array(image).astype('float32')/255
#    np_image = transform.resize(np_image, (352, 480, 3))
# #    np_image = cv2.resize(np_image, (480, 352))
#    np_image = np.expand_dims(np_image, axis=0)
#    return np_image

def segment(np_image):
    size = np_image.shape
    print(size)
    file_size = (size[1],size[0])

    
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (352, 480, 3))
#    np_image = cv2.resize(np_image, (480, 352))
    image = np.expand_dims(np_image, axis=0)

    # image = load(filename)
    print(type(image))
    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 16
    CLASSES = ['passport']
    LR = 0.0001
    EPOCHS = 10

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    model.load_weights('./checkpoints/segmentation.h5')



    predict = model.predict(image)


    pred = predict[0,:,:,0]
    pred = np.dstack([pred, pred, pred])
    pred = (pred * 255).astype(np.uint8)
    img = Image.fromarray(pred, 'RGB')
    img = img.resize(file_size)
    # PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
    # mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    img.save('./images/mask.png')
    img = np.array(img)
    return img


if __name__ == "__main__":

    filename = 'ho-chieu-passport-tre-em.jpg'
    image = cv2.imread(filename)
    mask = segment(image)
    # img = perspective_transform(image, mask)
    
    plt.imshow(mask)
    plt.show()