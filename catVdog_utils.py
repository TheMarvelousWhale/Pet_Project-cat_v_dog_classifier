def load_catvdog_h5_data(h5filepath):
    h5file = h5py.File(h5filepath, "r")
    try:
        class_fieldname,x_fieldname,y_fieldname, = h5file.keys()   #this order is Gfriend files
        print("The keys are: ", h5file.keys())
        set_x_orig = np.array(h5file[x_fieldname][:]) # your test set features
        set_y_orig = np.array(h5file[y_fieldname][:]) # your test set labels
        set_class = list(h5file[class_fieldname][:])
        print("The shape of x_field",set_x_orig.shape)
        print("The shape of y_field",set_y_orig.shape)
        print("Len of class list: ",len(set_class))
        #show_first_20(set_x_orig,set_y_orig[0],offset)
    finally:
        h5file.close()
    return set_x_orig,set_y_orig,set_class

def show_20_images(img_array,label_array,class_array,offset=0):
    
    assert(len(img_array)==len(label_array))
    assert(len(img_array)==len(class_array))
    
    datalen = len(img_array)
    numOfIter = min(datalen,20)
    
    if numOfIter < 20:
        offset = 0  #if numOfIter < 20, it means it equals datalen, which is less than 20 fed in. So no need offset
        
    numOfRow = numOfIter//4 + (numOfIter%4 != 0)   #a ceiling function
    
    plt.rcParams['figure.figsize'] = (60.0, 100.0) 
    print(img_array.shape)
    num_px = img_array.shape[1] #shape is (m, num_px,num_px,3)
    for i in range(numOfIter):
        plt.subplot(5,4,i+1)  #plot 5 by 4 grid
        #plt.imshow(img_array[i+offset])
        plt.imshow(img_array[i+offset].reshape(num_px,num_px,3), interpolation='nearest')  #take only the num_px
        plt.axis('off')
        #print(f'IMG {i+1} is labelled {label_array[i+offset]}')
        _class = class_array[i+offset].decode("utf-8")
        plt.title("Class: " + _class,fontsize = 50)
        
def review_h5_data(h5filepath,offset=0):
    """
    Plot the first 20 images (or entire set if the files have less than 20 images)
    offset - starting index of the 20 photos(no boundary checking condition is implemented.
    """
    
    h5file = h5py.File(h5filepath, "r")
    try:
        class_fieldname,x_fieldname,y_fieldname = h5file.keys()
        print("The keys are: ", h5file.keys())
        set_x_orig = np.array(h5file[x_fieldname][:]) # your test set features
        set_y_orig = np.array(h5file[y_fieldname][:]) # your test set labels
        set_class = list(h5file[class_fieldname][:])
        #print(set_class)
        print("The shape of x_field",set_x_orig.shape)
        print("The shape of y_field",set_y_orig.shape)
        show_20_images(set_x_orig,set_y_orig[0],set_class,offset)
    finally:
        h5file.close()


    """
    Sample usage:
    train_x_orig,train_y_orig,train_classes = load_catvdog_h5_data('./train_catVdog.h5')
    test_x_orig,test_y_orig,test_classes = load_catvdog_h5_data('./test_catVdog.h5')
    review_h5_data('./train_catVdog.h5',offset = 5000) #for dogs
    review_h5_data('./train_catVdog.h5',offset = 0)  #for cats
    """