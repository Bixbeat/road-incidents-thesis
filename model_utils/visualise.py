"""Grad-cam by Utku Ozbulak - github.com/utkuozbulak"""

import visdom
import torch
from torchvision.transforms import Normalize, ToPILImage
import numpy as np
from PIL import Image, ImageOps
import datetime as dt
import time as t
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from model_utils.analysis_utils import var_to_cpu

def plot_pairs(image, label):
    """Takes an image tensor and its reconstruction vars, and
    argmaxed softmax predictions to create a 1x2 comparison plot."""
    _, (imgplot, lblplot) = plt.subplots(1, 2, figsize=(12, 6))
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
    the tensor to an image"""
    transposed_tensor = tensor.numpy().transpose((1, 2, 0))
    unnormed_img = np.array(sdev) * transposed_tensor + np.array(mean)
    image = np.clip(unnormed_img, 0, 1)
    return image   

class VisdomVisualiser():
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

    def vis_img_window(self, win_id):
        img_window = self.vis.image(np.random.rand(3, 224, 224), opts=dict(title='CAM window', caption='Class goes here', win=win_id))

    def update_img_window(self, img_window, img, title, caption):
        img_window = self.vis.image(img, opts=dict(win=img_window, title=title, caption=caption))

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
        <b>Epoch batches per second</b>:
        <br>
        {ep_spd}
        <br>
        <b>Avg batches per second</b>:
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
        
    def vis_update_images(self, window, images, labels):
        """UNUSED: used to plot images to Visdom"""
        img = img_content[0]
        lbl =  img_content[1]
        pred =  img_content[2]
        
        self.vis.images(img, win=windows[0], opts=dict(title=title))
        self.vis.images(lbl, win=windows[1], opts=dict(title=title))
        self.vis.images(pred,win=windows[2], opts=dict(title=title))

    def close_all_plots(self):
        self.vis.close(win=None)

def _remove_microseconds(time_delta):
    return time_delta - dt.timedelta(microseconds=time_delta.microseconds)

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        layer = self.model._modules.get(self.target_layer)

        for module_pos, module in self.model._modules.items():
            if module_pos == 'fc':
                x = x.view(x.size(0), -1)            
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, out_img_size, means, sdevs, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target_conv = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target_conv.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target_conv[i, :, :]

        input_image = var_to_cpu(input_image)

        cam_normalized = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam_colourized = colourize_gradient(cam_normalized)[:, :, :3]
        cam_img = Image.fromarray(np.uint8(cam_colourized*255))
        cam_resized = ImageOps.fit(cam_img, (out_img_size, out_img_size))
        target_img = normalized_img_tensor_to_pil(input_image.data[0], means, sdevs)
        cam_overlaid = Image.blend(target_img, cam_resized, 0.6)
        return cam_overlaid

def colourize_gradient(img_array):
    colour = mpl.cm.get_cmap('rainbow')
    coloured_img = colour(img_array)
    return coloured_img

def normalized_img_tensor_to_pil(img_tensor, means, sdevs):
    bands = len(means)
    to_pil = ToPILImage()
    inverse_normalize = Normalize(
        mean =[-means[band]/sdevs[band] for band in range(bands)],
        std=[1/sdevs[band] for band in range(bands)]
    )
    inverse_tensor = inverse_normalize(img_tensor)      
    return to_pil(inverse_tensor)

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