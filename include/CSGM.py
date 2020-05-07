import torch
import numpy as np
if torch.cuda.device_count()==0:
    dtype = torch.FloatTensor
    device = 'cpu'
else:
    dtype = torch.cuda.FloatTensor
    device = 'cuda'
import time


def CSGM(G, latentDim, y, device, num_iter=1600):
    G.eval()
    G.to(device)
    
    loss_per_iter = np.zeros(num_iter)
    
    mse = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(1,latentDim)).to(device)

    z = torch.autograd.Variable(z_init, requires_grad = True)
    optimizer = torch.optim.Adam([{'params': z, 'lr': 0.1}])

    print('Running CSGM:')
    for i in range(num_iter):
        optimizer.zero_grad()
        
        Gz = G(z)
        
        loss = mse(Gz, y)
        loss.backward()
        optimizer.step()
        
        if(i % 100 == 0):
            print('CSGM step %d/%d, objective loss = %.5f' %(i, num_iter, loss.item()))
        
        loss_per_iter[i] = loss.item()
    return z, loss_per_iter



def CSGM2(G, latentDim, y, A, device, num_iter=1600):
    G.eval()
    G.to(device)
    
    mse = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(1,latentDim)).to(device)
    
    loss_per_iter = np.zeros(num_iter)

    z = torch.autograd.Variable(z_init, requires_grad = True)
    optimizer = torch.optim.Adam([{'params': z, 'lr': 0.1}])
    
    print('Running CSGM:')
    for i in range(num_iter):
        optimizer.zero_grad()
        
        Gz = G(z)
        Gz_vec = Gz.reshape(Gz.numel())
        AGz = torch.matmul(A, Gz_vec) # TODO: try nonlinear measurement processes
        
        loss = mse(AGz, y)
        loss.backward()
        optimizer.step()
        
        loss_per_iter[i] = loss.item() 
        if(i % 50 == 0):
            print('CSGM step %d/%d, objective = %.5f' %(i, num_iter, loss_per_iter[i]))
        
        
    return z, loss_per_iter


def CS_DD(net, num_channels, d_image, y, A, device, num_iter = 8000, lr_decay_epoch = 3000, decodetype = 'upsample'):
    # compute the size of (fixed) latent vector and draw it uniformly  
    totalupsample = 2**(len(num_channels)-1)
    w = np.sqrt(int(d_image/3)) # =d_image / out_channels = số chiều của mỗi cạnh ảnh
    width = int(w/(totalupsample))
    height = int(w/(totalupsample))

    shape = [1,num_channels[0], width, height]  
    print("shape of latent code B1: ", shape)

    print("initializing latent code B1...")
    net_input = torch.autograd.Variable(torch.zeros(shape))
    net_input.data.uniform_()
    net_input.data *= 1./10

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    #x_in = net(net_input.type(dtype)).data.clone() #initializing image

    # processing optimization
    if decodetype=='upsample':
        p = [x for x in net.decoder.parameters() ] #list of all weigths
    elif decodetype=='transposeconv':
        p = [x for x in net.convdecoder.parameters() ] #list of all weigths

    optimizer = torch.optim.Adam(p, lr=0.0002)
    mse = torch.nn.MSELoss()
    
    for i in range(num_iter):

        #################
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=0.0002, lr_decay_epoch=lr_decay_epoch,factor=0.7)

        #################
        def closure():
            optimizer.zero_grad()           
            x_np = net(net_input.type(dtype)).to(device)
            x_var = 2*x_np-1
            
            #y_var = x_var
            y_var = torch.matmul(A,x_var.reshape(d_image))
            
            loss = mse(y_var, y) #torch.matmul(A,x_hat)
            loss.backward()
            #mse_wrt_truth[i] = loss.data.cpu().numpy()
            
            return loss

        loss = optimizer.step(closure) 
        if i %100 == 0:
            print ('Iteration %05d   Train loss %f ' % (i, loss.item() ))
        #print ('Iteration %05d   Train loss %f ' % (i, loss.detach().cpu().numpy()), '\r', end='')

    return net, net_input, loss


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500, factor=0.5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('\nLearning rate is set to {}'.format(lr))
        print('\n')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def CS_hybrid2(G, net, net_input, num_channels, d_image, y, A, device, z_0, latentDim, num_iter = 1000, lr_decay_epoch = 300, decodetype = 'upsample'):
    G.eval()
    #z = torch.normal(torch.zeros(1,latentDim)) #.to(config.device)
    z = torch.autograd.Variable(z_0, requires_grad = True).to(device)
    
    
    # collecting all trainable parameters
    alpha_init = torch.zeros(1).to(device)
    beta_init = torch.zeros(1).to(device)
    alpha_init.data[0] = 1
    beta_init.data[0] = 0

    alpha = torch.autograd.Variable(alpha_init, requires_grad=True)
    beta = torch.autograd.Variable(beta_init, requires_grad=True)
    
    if decodetype=='upsample':
        p = [x for x in net.decoder.parameters() ] #list of all weigths
    elif decodetype=='transposeconv':
        p = [x for x in net.convdecoder.parameters() ] #list of all weigths
    
    
    #weight_decay = 0
    #optimizer = torch.optim.Adam(p, lr=0.001)
    
    optimizer_z = torch.optim.Adam(
    [
        {"params": alpha, "lr": 0.005},
        {"params": beta, "lr": 0.005},
        {"params": z, "lr": 0.1}
    ])
    
    optimizer_net = torch.optim.Adam(
    [
        {"params": p, "lr": 0.0002}
    ])
    
    mse = torch.nn.MSELoss()
    loss_per_iter = np.zeros(num_iter)
    
    for i in range(num_iter):

        #################
        if lr_decay_epoch is not 0:
            optimizer_net = exp_lr_scheduler(optimizer_net, i, init_lr=0.0002, lr_decay_epoch=lr_decay_epoch,factor=0.7)

        #################
        
        optimizer_z.zero_grad()
        optimizer_net.zero_grad()
            
        alpha_clamp = alpha.clamp(0,1)
        beta_clamp = beta.clamp(0,1)
            
        x_var = alpha_clamp*G(z) + beta_clamp*(2*net(net_input.type(dtype))-1)
        #y_hat = x_hat
            
        y_var = torch.matmul(A,x_var.reshape(d_image))
        loss = mse(y_var, y) #torch.matmul(A,x_hat)
        loss.backward()
        #mse_wrt_truth[i] = loss.data.cpu().numpy()
        
        optimizer_z.step()
        optimizer_net.step()
        
        loss_per_iter[i] = loss.item() 
        if i % 100 == 0:
            print ('Iteration %04d   Train loss %f ' % (i, loss_per_iter[i] ))

    return net, net_input, z, alpha, beta, loss_per_iter

