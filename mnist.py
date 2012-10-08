import cPickle, gzip, numpy

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)


def vectorize_images(imgs):
    (n, p) = imgs.shape
    assert p == 784
    vectorized_images = []
    for i in xrange(n):
        img = []
        for j in xrange(784):
            if imgs[i,j] > 0:
                img.append([j//28, j%28, imgs[i,j]])
        vectorized_images.append(numpy.matrix(img))
    return vectorized_images

with open('vectorized_mnst.pkl', 'w') as f:
    vim = vectorize_images(train_set[0][0:20,:])
    cPickle.dump(vim, f)





