from sklearn.cluster import KMeans
from collections import Counter
import urllib
import cv2

def jarak(r1,g1,b1, r2,g2,b2):
    a= r1 - r2
    b= g1 - g2
    c= b2 - b2
    d = (a**2)+(b**2)+(c**2)
    return d
# function to replace char in a string
def clean_sw(string, x,y):
    try:
        x = str(string).replace(x,y)
    except:
        x = ""
    return x
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # return the image
    return image,gray
# METHOD #2: get RGB
def get_dominant_color(image, k=5, l=3, image_processing_size = None):
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = []
    for i in range(l):
        x = clt.cluster_centers_[label_counts.most_common(4)[i][0]]
        dominant_color.append(x)
    return dominant_color

# METHOD #2: get RGB
def get_dominant_black(image, k=5, image_processing_size = None):
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_black = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return dominant_black