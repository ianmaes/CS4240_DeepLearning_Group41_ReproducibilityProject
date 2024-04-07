import pandas as pd
import numpy as np
dataframe = pd.read_pickle(r'C:\Users\ianma\Desktop\Study\Python\CS4240_DeepLearning_Group41_ReproducibilityProject\Autoencoders-Reproduction\examples\experiment_results_202404071405.pkl')
print(np.array(dataframe['validation_losses']))