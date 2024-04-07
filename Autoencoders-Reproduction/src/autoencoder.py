import torch
import torch.nn as nn
import torch.nn.functional as F

class FullNetwork(nn.Module):
    def __init__(self, params):
        super(FullNetwork, self).__init__()
        self.input_dim = params['input_dim']
        self.latent_dim = params['latent_dim']
        self.poly_order = params['poly_order']
        self.include_sine = params.get('include_sine', False)
        self.library_dim = params['library_dim']
        self.model_order = params['model_order']
        self.activation = params['activation']
        self.sequential_thresholding = params['sequential_thresholding']
        if self.activation == 'relu':
            self.activation = torch.relu
        elif self.activation == 'elu':
            self.activation = torch.elu
        elif self.activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.activation_name = params['activation']
        # Initialize Autoencoder
        if params['activation'] == 'linear':
            self.autoencoder = LinearAutoencoder(self.input_dim, self.latent_dim)
        else:
            self.autoencoder = NonLinearAutoencoder(self.input_dim, self.latent_dim, params['widths'], activation_fn=self.activation)
        
        # Initialize SINDy Coefficients
        self.sindy_coefficients = nn.Parameter(torch.Tensor(self.library_dim, self.latent_dim))
        self._initialize_coefficients(params['coefficient_initialization'], params.get('init_coefficients'))
        self.coefficient_mask =params['coefficient_mask']


    def _initialize_coefficients(self, init_method, init_values=None):
        if init_method == 'xavier':
            nn.init.xavier_uniform_(self.sindy_coefficients)
        elif init_method == 'specified' and init_values is not None:
            self.sindy_coefficients.data = init_values
        elif init_method == 'constant':
            nn.init.constant_(self.sindy_coefficients, 1.0)
        elif init_method == 'normal':
            nn.init.normal_(self.sindy_coefficients)
        else:
            raise ValueError("Unknown coefficient initialization method.")

    def forward(self, x, dx, ddx=None):
    
        z, x_decode = self.autoencoder(x)
        ddz = None
        ddx_decode = None
        # Derivative computation
        if self.model_order == 1:
            dz = z_derivative(x, dx, self.autoencoder.encoder, activation=self.activation_name)
            self.Theta = sindy_library_tf(z, self.latent_dim, self.poly_order, self.include_sine)
        else:
            # print("x:", x, "dx:", dx, "ddx:", ddx)
            dz, ddz = z_derivative_order2(x, dx, ddx, self.autoencoder.encoder, activation=self.activation_name)
            self.Theta = sindy_library_tf_order2(z, dz, self.latent_dim, self.poly_order, self.include_sine)
        Theta = self.Theta
        # Apply SINDy coefficients
        if self.sequential_thresholding:
            self.sindy_coefficients_masked = self.coefficient_mask * self.sindy_coefficients
            sindy_predict = torch.matmul(Theta, self.sindy_coefficients_masked)
        else:
            sindy_predict = torch.matmul(Theta, self.sindy_coefficients)
            

        # Decode derivatives
        if self.model_order == 1:
            dx_decode = z_derivative(z, sindy_predict, self.autoencoder.decoder, activation=self.activation_name)
        else:
            dx_decode, ddx_decode = z_derivative_order2(z, dz, sindy_predict, self.autoencoder.decoder, activation=self.activation_name)

        return x_decode, dx_decode, ddx_decode, z, dz, ddz, Theta, sindy_predict
    
    def define_loss(self, x, dx, ddx=None, params=None):

        x_decode, dx_decode, ddx_decode, z, dz, ddz, Theta, sindy_predict = self.forward(x, dx, ddx)
        # Loss for the decoder

        self.sindy_coefficients_masked = self.coefficient_mask * self.sindy_coefficients
        # Regularization loss

        losses = {}
        losses['decoder'] = torch.mean((x - x_decode)**2)
        if self.model_order == 1:
            losses['sindy_z'] = torch.mean((dz - sindy_predict)**2)
            losses['sindy_x'] = torch.mean((dx - dx_decode)**2)
        else:
            losses['sindy_z'] = torch.mean((ddz - sindy_predict)**2)
            losses['sindy_x'] = torch.mean((ddx - ddx_decode)**2)
        losses['sindy_regularization'] = torch.mean(torch.abs(self.sindy_coefficients_masked))

        total_loss = (params['loss_weight_decoder'] * losses['decoder'] +
                    params['loss_weight_sindy_z'] * losses['sindy_z'] +
                    params['loss_weight_sindy_x'] * losses['sindy_x'] +
                    params['loss_weight_sindy_regularization'] * losses['sindy_regularization'])
        
        loss_refinement = (params['loss_weight_decoder'] * losses['decoder'] +
                    params['loss_weight_sindy_z'] * losses['sindy_z'] +
                    params['loss_weight_sindy_x'] * losses['sindy_x'])

        return total_loss, losses, loss_refinement
    

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_decode = self.decoder(z)
        return z, x_decode
    
class CustomLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation_fn
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, input):
        x = self.linear(input)
        if self.activation is not None:
            x = self.activation(x)
        return x

class NonLinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, widths, activation_fn):
        super(NonLinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(*self._build_layers(input_dim, latent_dim, widths, activation_fn))
        self.decoder = nn.Sequential(*self._build_layers(latent_dim, input_dim, widths[::-1], activation_fn)) 

    def _build_layers(self, input_dim, output_dim, widths, activation_fn):
        layers = []
        last_dim = input_dim
        for width in widths:
            layers.append(CustomLayer(last_dim, width, activation_fn))
            last_dim = width
        layers.append(CustomLayer(last_dim, output_dim, None))  # Include activation on final layer
        return layers
    
    def forward(self, x):
        z = self.encoder(x)
        x_decode = self.decoder(z)
        return z, x_decode
    
def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D torch tensor of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D torch tensor containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.size(0)).to(z.device)]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i] * z[:,j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i] * z[:,j] * z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, dim=1).to(z.device)

def sindy_library_tf_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second-order system in PyTorch.
    """
    z_combined = torch.cat([z, dz], dim=1)
    return sindy_library_tf(z_combined, 2*latent_dim, poly_order, include_sine)

def z_derivative(input, dx, layers, activation):
    dz = dx
    for i, layer in enumerate(layers):
        if i < len(layers) - 1:
            input = layer.linear(input)
            if activation == 'elu':
                dz = torch.where(input < 0, torch.exp(input), torch.ones_like(input)) * layer.linear(dz)
                input = torch.elu(input)
            elif activation == 'relu':
                dz = (input > 0).float() * layer.linear(dz)
                input = torch.relu(input)
            elif activation == 'sigmoid':
                input = torch.sigmoid(input)
                dz = input * (1 - input) * layer.linear(dz)
        else:
            dz = layer.linear(dz)
    return dz

def z_derivative_order2(input, dx, ddx, layers, activation):
    dz = dx
    ddz = ddx
    for i, layer in enumerate(layers):
        if i < len(layers) - 1:
            input = layer(input)

            if activation == 'elu':
                # ELU derivative: exp(input) if input < 0 else 1
                elu_derivative = torch.where(input < 0, torch.exp(input), torch.tensor(1.0))
                elu_double_derivative = torch.where(input < 0, torch.exp(input), torch.tensor(0.0))
                
                dz_prev = layer.linear(dz)
                ddz_prev = layer.linear(ddz)

                # Apply the chain rule for the first and second derivative
                dz = elu_derivative * dz_prev
                ddz = elu_double_derivative * dz_prev * dz_prev + elu_derivative * ddz_prev

                input = torch.elu(input)

            elif activation == 'relu':
                # ReLU derivative: 1 if input > 0 else 0
                relu_derivative = (input > 0).float()
                relu_double_derivative = torch.zeros_like(input) # second derivative of ReLU is 0

                dz_prev = layer.linear(dz)
                ddz_prev = layer.linear(ddz)

                # Apply the chain rule for the first and second derivative
                dz = relu_derivative * dz_prev
                ddz = relu_double_derivative * dz_prev * dz_prev + relu_derivative * ddz_prev

                input = torch.relu(input)

            elif activation == 'sigmoid':
                # Sigmoid derivative: sigmoid(input) * (1 - sigmoid(input))
                sigmoid_derivative = F.sigmoid(input) * (1 - F.sigmoid(input))
                sigmoid_double_derivative = F.sigmoid(input) * (1 - F.sigmoid(input)) * (1 - 2 * F.sigmoid(input))

                dz_prev = layer.linear(dz)
                ddz_prev = layer.linear(ddz)

                # Apply the chain rule for the first and second derivative
                dz = sigmoid_derivative * dz_prev
                ddz = sigmoid_double_derivative * dz_prev * dz_prev + sigmoid_derivative * ddz_prev

                input = torch.sigmoid(input)

            # Add other activation conditions here
        else:
            dz = layer.linear(dz)
            ddz = layer.linear(ddz)

    return dz, ddz