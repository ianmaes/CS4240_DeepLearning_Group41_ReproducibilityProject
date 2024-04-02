import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from autoencoder import FullNetwork

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = FullNetwork(params)
    loss, losses, loss_refinement = autoencoder_network.define_loss(torch.tensor(training_data['x'], dtype=torch.float32), torch.tensor(training_data['dx'], dtype=torch.float32), params)
    
    # Define optimizer
    optimizer = optim.Adam(autoencoder_network.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_milestones'], gamma=params['lr_decay'])
    
    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    x_norm = np.mean(val_data['x']**2)
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)
    else:
        sindy_predict_norm_x = np.mean(val_data['ddx']**2)

    validation_losses = []
    sindy_model_terms = [torch.sum(params['coefficient_mask'])]

    print('TRAINING')
    autoencoder_network.train()
    for i in range(params['max_epochs']):
        for j in range(params['epoch_size'] // params['batch_size']):
            batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            optimizer.zero_grad()
            output = autoencoder_network(train_dict['x'])
            loss_val = loss(output, train_dict['x'])
            loss_val.backward()
            optimizer.step()
            
        scheduler.step()
        
        if params['print_progress'] and (i % params['print_frequency'] == 0):
            with torch.no_grad():
                validation_losses.append(print_progress(autoencoder_network, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

        if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
            params['coefficient_mask'] = torch.abs(autoencoder_network.sindy_coefficients) > params['coefficient_threshold']
            validation_dict['coefficient_mask'] = params['coefficient_mask']
            print('THRESHOLDING: %d active coefficients' % torch.sum(params['coefficient_mask']))
            sindy_model_terms.append(torch.sum(params['coefficient_mask']))

    print('REFINEMENT')
    for i_refinement in range(params['refinement_epochs']):
        for j in range(params['epoch_size'] // params['batch_size']):
            batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            optimizer.zero_grad()
            output = autoencoder_network(train_dict['x'])
            loss_val = loss_refinement(output, train_dict['x'])
            loss_val.backward()
            optimizer.step()

        if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
            with torch.no_grad():
                validation_losses.append(print_progress(autoencoder_network, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

    torch.save(autoencoder_network.state_dict(), params['data_path'] + params['save_name'] + '.pth')
    pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))

    with torch.no_grad():
        autoencoder_network.eval()
        final_losses = (losses['decoder'](autoencoder_network(validation_dict['x'])),
                        losses['sindy_x'](autoencoder_network(validation_dict['x'])),
                        losses['sindy_z'](autoencoder_network(validation_dict['x'])),
                        losses['sindy_regularization'](autoencoder_network.sindy_coefficients))

        if params['model_order'] == 1:
            sindy_predict_norm_z = torch.mean(autoencoder_network.dx(validation_dict['x'])**2)
        else:
            sindy_predict_norm_z = torch.mean(autoencoder_network.ddx(validation_dict['x'])**2)
        sindy_coefficients = autoencoder_network.sindy_coefficients.detach().cpu().numpy()

        results_dict = {}
        results_dict['num_epochs'] = i
        results_dict['x_norm'] = x_norm
        results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
        results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z.item()
        results_dict['sindy_coefficients'] = sindy_coefficients
        results_dict['loss_decoder'] = final_losses[0].item()
        results_dict['loss_decoder_sindy'] = final_losses[1].item()
        results_dict['loss_sindy'] = final_losses[2].item()
        results_dict['loss_sindy_regularization'] = final_losses[3].item()
        results_dict['validation_losses'] = np.array(validation_losses)
        results_dict['sindy_model_terms'] = np.array(sindy_model_terms)

    return results_dict

def print_progress(network, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    """
    Print loss function values to keep track of the training progress.

    Arguments:
        network - the PyTorch network
        i - the training iteration
        loss - PyTorch loss function used in training
        losses - tuple of the individual loss functions that make up the total loss
        train_dict - feed dictionary of training data
        validation_dict - feed dictionary of validation data
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.

    Returns:
        Tuple of losses calculated on the validation set.
    """
    network.eval()
    with torch.no_grad():
        output_train = network(train_dict['x'])
        output_val = network(validation_dict['x'])
        training_loss_vals = (loss(output_train, train_dict['x']),) + tuple(losses(output_train, train_dict['x']).values())
        validation_loss_vals = (loss(output_val, validation_dict['x']),) + tuple(losses(output_val, validation_dict['x']).values())
        print("Epoch %d" % i)
        print("   training loss", training_loss_vals)
        print("   validation loss", validation_loss_vals)
        decoder_losses = (losses['decoder'](output_val, validation_dict['x']).item() / x_norm,
                          losses['sindy_x'](output_val, validation_dict['x']).item() / sindy_predict_norm)
        print("decoder loss ratio: %f, decoder SINDy loss ratio: %f" % decoder_losses)
        return validation_loss_vals

def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into PyTorch.

    Arguments:
        data - Dictionary object containing input data 'x',
               along with the first (and possibly second) order time derivatives 'dx' ('ddx').
        params - Dictionary object containing model and training parameters. The relevant
                 parameters are 'model_order' (determines whether the SINDy model predicts first or
                 second order time derivatives), 'sequential_thresholding' (indicates whether or not
                 coefficient thresholding is performed), 'coefficient_mask' (optional if sequential
                 thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
                 model), and 'learning_rate' (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
               in to PyTorch. If None, all examples are used.

    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to PyTorch.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {
        'x': torch.tensor(data['x'][idxs], dtype=torch.float32),
        'dx': torch.tensor(data['dx'][idxs], dtype=torch.float32)
    }
    if params['model_order'] == 2:
        feed_dict['ddx'] = torch.tensor(data['ddx'][idxs], dtype=torch.float32)
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask'] = torch.tensor(params['coefficient_mask'], dtype=torch.float32)
    feed_dict['learning_rate'] = torch.tensor(params['learning_rate'], dtype=torch.float32)
    return feed_dict