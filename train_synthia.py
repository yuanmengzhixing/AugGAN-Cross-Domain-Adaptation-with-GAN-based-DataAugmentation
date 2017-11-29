import time
from net_synthiaGAN import *
from data.dataset_synthia import ImageDataset
import click

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--name', default='cyclegan-SoftSharing_NoSim')
@click.option('--root_dir', default='../gan-for-synthia/data')
@click.option('--load_size', default=(286,286), type=tuple)
@click.option('--fine_size', default=256, type=int)
@click.option('--epochs', default=200, type=int)
@click.option('--batch_size', default=4, type=int)
@click.option('--ndf', default=64, type=int)
@click.option('--which_direction', default='AtoB')
@click.option('--workers', default=15, type=int)
@click.option('--saving_freq', default=1, type=int)
@click.option('--phase', default='train', type=str)
@click.option('--style_a', default='SYNTHIA-SEQS-{}-SPRING-CUT', type=str)
@click.option('--style_b', default='SYNTHIA-SEQS-{}-NIGHT-CUT', type=str)
@click.option('--synthia_seqs', default=['02', '05', '06'], type=list)
@click.option('--pool_size', default=50, type=int)
@click.option('--opt_choice', default='scale_width')
def main(
    name, root_dir, load_size, fine_size, epochs, batch_size, ndf, pool_size, synthia_seqs,
    which_direction, workers, saving_freq, style_a, style_b, phase, opt_choice):
    
    print('=======> Prepare DataLoader')
    dataset_args = {
        'root_dir': '../../../shared/SYNTHIA/', 'style_A': style_a, 'style_B': style_b,
        'load_size': load_size, 'fine_size': fine_size, 'phase': phase,
        'opt_choice': opt_choice, 'synthia_seqs': synthia_seqs
    }
    loader_args = {'num_workers': workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        dataset=ImageDataset(**dataset_args),
        batch_size=batch_size, **loader_args
    )
    print('=======> Prepare Model')
    print('-------Model: {}'.format(name))
    net = GANModel(
        root_dir=root_dir, input_nc=3, output_nc=3, ndf=ndf, 
        which_direction=which_direction, pool_size=pool_size,
        saving_freq=saving_freq, use_sigmoid=True, name=name, 
        batch_size=batch_size, bg_weight=0.4
    )
    print('=======> Start Training')
    net.train(data_loader=train_loader, epochs=epochs)

if __name__ == '__main__':
    main()
