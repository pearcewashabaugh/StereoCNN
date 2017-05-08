import train

def plotter(layer, plotlev='basic'):
    # This just helps make sure the user doesn't accidentally print far too many images.
    # This bit can clearly be improved upon
    if plotlev == 'none':
        return
    if plotlev == 'basic':
        plt.figure()
        plt.title('Displacement Distance')
        plt.imshow(layer.offset_map[:, :], cmap='hot', interpolation='nearest')

        plt.show()

    if plotlev == 'full':
        if layer.subim1_number > 50:
            print('WARNING: You are about to print %d images to your screen.' % (3 * layer.subim1_number))
            proceed = input('Proceed? (y/n)\n')
            if (proceed == 'n') | (proceed == 'no'):
                print('To print fewer to no images call main with arg plotter = \'basic\' or \'none\' ')
                return

        subpltdim = int(np.ceil(np.sqrt(layer.subim1_number)))
        # plot the subim1 bank
        fig = plt.figure(1)
        plt.title('subim1')
        plt.axis('off')
        for i in range(layer.subim1_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer.subim1_bank[i, :, :, :])

        # plot the subim2 bank
        fig = plt.figure(2)
        plt.title('subim2')
        plt.axis('off')
        for i in range(layer.subim1_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer.subim2_bank[i, :, :, :])
            plt.axhline(y=layer.local_disparity[i][0])
            plt.axvline(x=layer.local_disparity[i][1])

        plt.show()
