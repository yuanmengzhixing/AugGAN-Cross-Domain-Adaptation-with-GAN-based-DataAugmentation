from network import *
class trainer():
    def __init__(self, network, save_path, name, discription):
        super(trainer, self).__init__()
        self.network = network
        self.save_path = save_path
        self.name = name
        self.discription = discription

    def train(self, data_loader, epochs):
        for epoch in range(1, epochs+1):
            print('currently running.....: {}'.format(self.name))
            print('discription: {}'.format(self.discription))
            self.epoch = epoch
            self.network.folder_path = self.save_path.format(self.network.name, epoch)

            progress = tqdm.tqdm(data_loader)
            hist = EpochHistory(length=len(progress))
            for data in progress:
                self.network.set_input(data)
                self.network.optimize_params()
                loss = self.network.current_loss()
                hist.add(loss)
                progress.set_description('Epoch #%d' % self.epoch)
                progress.set_postfix(
                    g_x='%.04f' % loss.get('loss_g_x'), g_y='%.04f' % loss.get('loss_g_y'),
                    seg_x='%.04f' % loss.get('loss_seg_x'), seg_y='%.04f' % loss.get('loss_seg_y'),
                    d_x='%.04f' % loss.get('loss_d_x'), d_y='%.04f' % loss.get('loss_d_y'),
                    ws_x='%.04f' % loss.get('loss_ws_x'), ws_y='%.04f' % loss.get('loss_ws_y'))
                            
            metrics = hist.metric()
            print('---> Epoch# %d summary loss g_x:{loss_g_x:.4f}, g_y:{loss_g_y:.4f}, '
                  'seg_x:{loss_seg_x:.4f}, seg_y:{loss_seg_y:.4f}, d_x:{loss_d_x:.4f}, d_y:{loss_d_y:.4f}, '
                  'ws_x:{loss_ws_x:.4f}, ws_y:{loss_ws_y:.4f} '
                  .format(self.epoch, **metrics))
            
            self.network.sumup_image(self.epoch)
            for k,v in metrics.items():
                self.network.tf_summary.scalar(k,v, self.epoch)
            if self.epoch % 3 == 0:
                self.network.save()
            if epoch > self.network.niter:
                self.network.update_learning_rate()
    
 
