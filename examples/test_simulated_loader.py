import torch
import numpy as np
import SignalTransformData as st
import matplotlib.pyplot as plt

def debug_loader(plot=True, samples=1000, sig_length=1000):

    data_module = st.data_modules.simulated.SinusoidalDataModule(classes=2, 
                                                                 samples=samples,
                                                                 channels=2,
                                                                 sig_length=sig_length,
                                                                 batch_size=128)
    print(data_module)

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'\n{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            print(f'Batch {batch.keys()}')
            print(f"Feature shape {batch['feature'].shape}, label shape {batch['label'].shape}")
            print(f"label counts {torch.unique(batch['label'], return_counts=True)}")
            print(np.unique(batch['feature'], axis=0).shape)

            if plot:

                fig, (ax1, ax2) = plt.subplots(2, 1)
            
                for l in np.unique(batch['label']):
                    color = 'red' if l == 0 else 'green'
                    print(f"class {l}, color {color}")

                    signals_l = batch['feature'][batch['label'] == l, :]
                    t = np.tile(np.arange(samples), (signals_l.shape[0], 1))
                    
                    ax1.scatter(t, signals_l[:, 0, :], c=color, s=0.5, alpha=0.1)
                    ax2.scatter(t, signals_l[:, 1, :], c=color, s=0.5, alpha=0.1)

                plt.show()
            
            #    for l in [0, 1]:
            #        samples = batch['feature'][batch['label'] == l, :][:, strand, channel, :]
            #        for s in range(samples.shape[0]):
            #            plt.scatter(np.linspace(1, length, length), samples[s, :] + 0.01*np.random.randn(samples[s,:].shape[0]))
            #        plt.title(f'Label {l}')
            #    plt.show()


if __name__ == '__main__':

    debug_loader()

