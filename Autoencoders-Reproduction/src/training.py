import sys
sys.path.append("../../src")

from autoencoder import *
import pickle
params = pickle.load(open(r'C:\Users\ianma\Desktop\Study\CS4240_DeepLearning_Group41_ReproducibilityProject\SindyAutoencoders-master\examples\lorenz\model2_params.pkl', 'rb'))

autoencoder_network = FullNetwork(params)

print(autoencoder_network.sindy_coefficients.size())

