"""Python script for Miniproject 1 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
    plb.close('all')
    
    dim = 28*28
    data_range = 255.0
    
    # load in data and labels    
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    # select 4 digits    
    name = 'Manu' # REPLACE BY YOUR OWN NAME
    targetdigits = name2digits(name) # assign the four digits that should be used
    print(targetdigits) # output the digits that were selected
    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    labels = np.array(labels[np.logical_or.reduce([labels==x for x in targetdigits])])
    
    dy, dx = data.shape
    
    #set the size of the Kohonen map. In this case it will be 6 X 6
    size_k = 6
    
    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    sigma = 3.0
    
    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    #set the learning rate
    eta = 0.0001 # HERE YOU HAVE TO SET YOUR OWN LEARNING RATE
    
    num_epochs = 100
    #set the maximal iteration count
    
    weight_changes = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        #set the random order in which the datapoints should be presented
        i_random = np.arange(np.shape(data)[0])
        np.random.shuffle(i_random)
        mean_change = 0;
        for t, i in enumerate(i_random):
            mean_change += som_step(centers, data[i,:],neighbor,eta,sigma)
        
        mean_change /= np.shape(data)[0]*eta
        print(mean_change)
        weight_changes[epoch] = mean_change

    # for visualization, you can use this:
    for i in range(1, size_k**2+1):
        ax = plb.subplot(size_k,size_k,i)
        plb.imshow(np.reshape(centers[i-1,:], [28, 28]),interpolation='bilinear')
        ax.set_title('label ' + str(assign_label(centers[i-1,:],data,labels)))
        plb.axis('off')
    
    plb.subplots_adjust(hspace=1.0)
    plb.figure()
    plb.plot(range(1,num_epochs+1),weight_changes)
    plb.xlabel('Number of epochs')
    plb.ylabel('Mean change in centers')
    # leave the window open at the end of the loop
    plb.show()
    plb.draw()
   

def assign_label(center, data, labels):
    unique_labels = np.unique(labels)
    dist_from_centroid = np.zeros_like(unique_labels)
    for i,label in enumerate(unique_labels):
        indices = np.where(labels==label)[0]
        centroid = np.mean(data[indices,:],0)
        dist_from_centroid[i] = np.linalg.norm(center-centroid)
    
    return int(unique_labels[np.argmin(dist_from_centroid)])

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """
    
    size_k = int(np.sqrt(len(centers)))
    
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
    delta_centers = np.zeros(size_k**2)
        
    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights        
        delta_centers[j] = np.linalg.norm(disc * eta * (data - centers[j,:]))
        centers[j,:] += disc * eta * (data - centers[j,:])
    
    return np.mean(delta_centers)

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.int(np.mod(s,x.shape[0]))

    return np.sort(x[t,:])


if __name__ == "__main__":
    kohonen()

