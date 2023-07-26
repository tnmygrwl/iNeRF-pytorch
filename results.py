import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imageio

datadir = './inerf_logs'
folder = './inerf_logs/fern_test/random/10/'
savedir = './imgs/fern/'

def superimpose():
    query_path = f'{folder}query.png'
    query = imageio.imread(query_path)
    images = []
    for i in range(0,300,10):
        path = folder + str(i) + '.png'
        curr = imageio.imread(path)
        rgb = curr*0.5 + query*0.5
        images.append(rgb.astype(np.uint8))
        imageio.imwrite(savedir + str(i) + '.png', rgb.astype(np.uint8))
    imageio.mimsave(f'{savedir}fern.gif', images, duration=0.1)

def plot():
    for exp_path in glob.glob(f'{datadir}/*/'):
        exp_name = exp_path.split('/')[-2]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
        # fig.suptitle(exp_name.split('_')[0])
        ax1.set_xlabel('Number of Steps', fontsize=14)
        ax1.set_ylabel('Avg. Translation Error', fontsize=14)
        ax2.set_xlabel('Number of Steps', fontsize=14)
        ax2.set_ylabel('Avg. of Rotation Error', fontsize=14)

        for sampling_path in glob.glob(f'{exp_path}/*/'):
            sampling_type = sampling_path.split('/')[-2]
            rot_error_matrix = []
            tran_error_matrix = []
            for txt_file in glob.glob(f'{sampling_path}/*/*.txt'):
                tran_error, rot_error = np.loadtxt(txt_file, delimiter=', ', skiprows=1, usecols=(2, 3), unpack=True)
                if len(tran_error) == 300:
                    tran_error_matrix.append(tran_error)
                    rot_error_matrix.append(rot_error)
                else:
                    print(txt_file)

            tran_error_matrix = np.stack(tran_error_matrix)
            rot_error_matrix = np.stack(rot_error_matrix)
            print(tran_error_matrix.shape, rot_error_matrix.shape)
            # tran_error_matrix = tran_error_matrix < 0.05
            # rot_error_matrix = rot_error_matrix < 5
            frac_tran_error = np.sum(tran_error_matrix, axis=0) / len(tran_error_matrix)
            frac_rot_error = np.sum(rot_error_matrix, axis=0) / len(rot_error_matrix)

            steps = np.arange(0, tran_error_matrix.shape[1])
            ax1.plot(steps, frac_tran_error, label=sampling_type)
            ax1.legend(fontsize=14)
            ax2.plot(steps, frac_rot_error, label=sampling_type)
            ax2.legend(fontsize=14)

        fig.savefig(f'{exp_path}plot.png')

plot()
# superimpose()