from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


def plot_N(images, texts):
    fig = plt.figure(figsize=(32, 10))
    gs = GridSpec(nrows=1,ncols=len(images))
    gs.update( hspace = 0.5, wspace = 0.)

    for i in range(len(images)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(images[i], vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(128, -5,  texts[i], size=20, ha="center", color='Blue')
        
    plt.show()  

