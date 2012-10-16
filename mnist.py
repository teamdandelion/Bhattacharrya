import cPickle, gzip, numpy

# Load the dataset
def load_mnist():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return (train_set, valid_set, test_set)


def vectorize_images(imgs):
    (n, p) = imgs.shape
    assert p == 784
    vectorized_images = []
    for i in xrange(n):
        img = []
        for j in xrange(784):
            if imgs[i,j] > 0:
                img.append([j%28, 28-j//28, imgs[i,j]])
        vectorized_images.append(numpy.matrix(img))
    return vectorized_images

    

def save_vectorize():
    vim = vectorize_images(train_set[0][0:20,:])
    with open('vectorized_mnist.pkl', 'w') as f:
        cPickle.dump(vim, f)



def main():
    mnist_imgs = load_mnist()
    vim = vectorize_images(mnist_imgs[0][0][0:20,:])
    return (mnist_imgs, vim)