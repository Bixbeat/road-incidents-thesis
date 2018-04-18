import visdom
import torch
import numpy as np
import datetime as dt
import time as t
import matplotlib.pyplot as plt

def plot_pairs(image, label):
    """Takes an image tensor and its reconstruction vars, and
    argmaxed softmax predictions to create a 1x2 comparison plot."""    
    _, (imgplot, lblplot) = plt.subplots(1,2, figsize=(12, 6))
    imgplot.imshow(image, aspect='auto')
    imgplot.grid(color='r', linestyle='dashed', alpha=0.75)

    lblplot.imshow(label, aspect='auto')
    lblplot.grid(color='r', linestyle='dashed', alpha=0.75)                
    
    plt.show(block=False)
    
def encoded_img_and_lbl_to_data(image, predictions, means, sdevs, label_colours):
    """For a given image and label pair, reconstruct both into
    images"""
    if isinstance(image, torch.autograd.variable.Variable):
        image = image.data.cpu()

    coloured_label = colour_lbl(predictions, label_colours)
    restored_img = decode_image(image, means, sdevs)
    return restored_img, coloured_label
    
def colour_lbl(tensor, colours):
    """For a given RGB image, constructs a RGB image map
    using the defined image classes.
    TODO: remove hardcoding, flexible input
    SRC: https://github.com/bfortuner/pytorch_tiramisu/blob/master/tiramisu-pytorch.ipynb
    """
    label_colours = np.array([colours[i] for i in colours])
    temp = tensor.numpy()
    
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,len(label_colours)):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))

    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb

def decode_image(tensor, mean, sdev):
    """For a given normalized image tensor, reconstructs
    the image by undoing the normalization and transforming
    the tensor to and image"""
    transposed_tensor = tensor.numpy().transpose((1, 2, 0))
    unnormed_img = np.array(sdev) * transposed_tensor + np.array(mean)
    image = np.clip(unnormed_img, 0, 1)
    return image

class Visualiser(object):
    def __init__(self):
        self.vis = visdom.Visdom()
        
        self.start_time = None
        self.avg_samples_per_sec = {}
    def vis_plot_loss(self,title):
        loss_window = self.vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='epoch',
                                         ylabel='Loss',
                                         title=title,
                                         legend=['Loss']))
        return loss_window
    
    def vis_create_timetracker(self, current_time):
        self.start_time = current_time
        timetracker = self.vis.text("<b>Starting time:</b> {}".format(self.start_time.replace(microsecond=0)))
        return timetracker

    def vis_img_window(self):
        """UNUSED: initialises a visdom image window"""
        img_window = self.vis.images(np.random.rand(3, 4, 4))
        return img_window


    def custom_update_loss_plot(self, loss_window, loss, color='#ffa500', marker_size=3, title='plot'):
        """For a given visdom window, creates a plot for a single loss trace"""
        x_axis = [i for i,_ in enumerate(loss)]
        y_axis = list(loss)
        
        trace = dict(x=x_axis, y=y_axis, mode="lines+markers", type='custom',
                     marker={'color': color, 'size': marker_size}, name='1st Trace')
        layout = dict(title=title, xaxis=dict(title='epoch'), yaxis=dict(title=title, range=[0,1]))
        
        self.vis._send({'data': [trace], 'layout': layout, 'win': loss_window})        

    def custom_combined_loss_plot(self, loss_window, train_loss, val_loss, size=10, title='<b>Combined loss</b>'):
        """For a given visdom window, creates a plot containing both train & validation traces"""
        x_axis = [i for i,_ in enumerate(train_loss)] # Get number of epochs
        
        train_y_axis = list(train_loss)
        val_y_axis = list(val_loss)
        
        train_trace = dict(x=x_axis, y=train_y_axis, mode="lines", type='custom', name='Train loss',
                     marker={'color': '#0047ab', 'size': size})
        
        val_trace = dict(x=x_axis, y=val_y_axis, mode="lines", type='custom', name='Validation loss',
                     marker={'color': '#ffa500', 'size': size})
        
        layout = dict(title=title, xaxis=dict(title='<b>epoch</b>'), yaxis=dict(title='<b>Loss</b>', range=[0,1]))
        
        self.vis._send({'data': [train_trace, val_trace], 'layout': layout, 'win': loss_window})        

    def update_timer(self, title, time_textbox, current_time, epoch_time, epoch_spd):
        """For a given initialized timebox, updates the timer with
        the latest epoch statistics"""
        time_elapsed = _remove_microseconds(current_time - self.start_time)
        samples_per_sec = 1/epoch_spd.total_seconds()
        if not title in self.avg_samples_per_sec:
            self.avg_samples_per_sec[title] = [samples_per_sec]
        else:
            self.avg_samples_per_sec[title].append(samples_per_sec)
        
        update_text = """
        <h2 style="color:#ffa500 !important; border-bottom: 2px solid #ffa500 !important; text-align: center !important">
            <b>{ttl}</b>
        </h2>
        <h4 style="color:#293E6A !important;"><b>Total time elapsed</b></h4>
        <b>Model start:</b>
        <br>
        {st}
        <br>
        <b>Total runtime:</b>
        <br>
        {tot}
        <h4 style="color:#293E6A !important;"><b>Epoch evaluation</b></h4>
        <b>Epoch duration:</b>
        <br>
        {ep_t}
        <br>
        <b>Epoch samples per second</b>:
        <br>
        {ep_spd}
        <br>
        <b>Avg samples per second</b>:
        <br>
        {avg_spd}
        <br>
        """.format(ttl=title,
                   st=self.start_time.replace(microsecond=0),
                   tot = time_elapsed,
                   ep_t = epoch_time,
                   ep_spd = samples_per_sec,
                   avg_spd = np.mean(self.avg_samples_per_sec[title]))
        self.vis.text(update_text, time_textbox)
        
    def vis_update_images(self, windows, img_content, title):
        """UNUSED: used to plot images to Visdom"""
        img = img_content[0]
        lbl =  img_content[1]
        pred =  img_content[2]
        
        self.vis.images(img, win=windows[0], opts=dict(title=title))
        self.vis.images(lbl, win=windows[1], opts=dict(title=title))
        self.vis.images(pred,win=windows[2], opts=dict(title=title))
        
def _remove_microseconds(time_delta):
    return time_delta - dt.timedelta(microseconds=time_delta.microseconds)
            
        
if __name__ == '__main__':
    visualize_loss = Visualiser()
    
    # Test single plot init
    train_loss = np.array([0.855151, 0.72231, 0.62123])
    train_loss_window = visualize_loss.vis_plot_loss("Train")
    visualize_loss.custom_update_loss_plot(train_loss_window, train_loss)
    
    # Test single plot update
    train_loss = np.append(train_loss,[0.5555])
    visualize_loss.custom_update_loss_plot(train_loss_window, train_loss)
    
    # Test combined plotting
    val_loss = np.array([0.9, 0.7787, 0.64, 0.58])
    visualize_loss.custom_combined_loss_plot(train_loss_window, train_loss, val_loss)
    
    # Test timebox
    epoch_start = dt.datetime.now()
    timebox = visualize_loss.vis_create_timetracker(epoch_start)
    t.sleep(2)
    epoch_end = dt.datetime.now()
    epoch_time = epoch_end-epoch_start
    epoch_spd = epoch_time/10
    visualize_loss.update_timer("Training",timebox, epoch_start, epoch_time, epoch_spd)
    
    
    
