import time
from trainer import *
from network import *
from utils.util import NetConfig
from data.SYNTHIA_loader import ImageDataset
#from data.GTA_loader import ImageDataset
#from data.GTA_loader_crop import ImageDataset
#from data.fake_real_loader import ImageDataset

torch.backends.cudnn.benchmark = True

def main():
    params = NetConfig('config.yaml')
    print('-----------------> preparing DataLoader')
    dataset_args = params.hyperparameters['dataset']
    loader_args = params.hyperparameters['loader']
    batch_size = params.hyperparameters['network']['net']['batch_size']
    name = params.hyperparameters['network']['net']['name']
    discription = params.discription
    save_path = params.save_path
    epochs = params.epochs
    
    train_loader = torch.utils.data.DataLoader(
        dataset=ImageDataset(**dataset_args),
        batch_size=batch_size, **loader_args
    )
    print('-----------------> preparing model: {}'.format(name))
    net = network(params.hyperparameters['network'])
    coach = trainer(net, save_path, name, discription)
    coach.train(data_loader=train_loader, epochs=epochs)
    print('-----------------> start training')

if __name__ == '__main__':
    main()
