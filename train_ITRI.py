import time
from net_ITRIGAN import *
from data.dataset_ITRI import ImageDataset
import click

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--name', default='softsharing102_ITRI_fxseg_resume_1')
@click.option('--root_dir', default='../gan-for-synthia/data')
@click.option('--load_size', default=(286, 286), type=tuple)
@click.option('--fine_size', default=256, type=int)
@click.option('--epochs', default=200, type=int)
@click.option('--batch_size', default=4, type=int)
@click.option('--ndf', default=64, type=int)
@click.option('--which_direction', default='AtoB')
@click.option('--workers', default=15, type=int)
@click.option('--display_freq', default=4, type=int)
@click.option('--phase', default='train', type=str)
@click.option('--style_a', default='itri_day/JPEGImages', type=str)
@click.option('--style_b', default='itri_night/JPEGImages', type=str)
@click.option('--pool_size', default=50, type=int)
@click.option('--opt_choice', default='scale_width')
@click.option('--saving_freq', default=1, type=int)
def main(
        name, root_dir, load_size, fine_size, epochs, batch_size, ndf, pool_size,
        which_direction, workers, saving_freq, style_a, style_b, phase, opt_choice):

    #Load Path: model path to be load 
    load_path = '../gan-for-synthia/output/softsharing102_ITRI_fxseg_resume/epoch_6/G_A_net.pth'

    print('=======> Prepare DataLoader')
    dataset_args = {
        'root_dir': '../../../shared/2017_CVPR_Dataset/', 'style_A': style_a, 'style_B': style_b,
        'load_size': load_size, 'fine_size': fine_size, 'phase': phase,
        'opt_choice': opt_choice
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
        batch_size=batch_size, model_dict_path=load_path
    )
    print('=======> Start Training')
    net.train(data_loader=train_loader, epochs=epochs)

if __name__ == '__main__':
    main()
