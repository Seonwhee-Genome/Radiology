TwoPathCNN
==========
 > M. Havaei *et al.* **Brain Tumor Segmentation with Deep Neural Networks** Medical Image Analysis 35 (2017) 18-31
Two-phase training
------------------
| The first training phase : They initially constructed their patches data-set such that all labels were equiprobable.
| The second training phase : They accounted for the un-balanced nature of the data and re-trained only the output layer(i.e. keeping the kernels of all other layers fixed) with a more representative distribution of the labels.

Regularization
--------------
| To obtain good results, they took several forms of regularization.
| 1) They bounded the absolute value of the kernel weights in all layers and applied both L1 and L2 regularization to prevent overfitting. This was done by adding the regularization terms to the negative log-probability.
| 2) They used a validation set for early stopping(i.e. stop training when the validation performance stopped improving) and for tuning hyper-parameters of the model.
| 3) They used Dropout.

Cascaded architectures
----------------------
| After training the TwoPATHCNN with the two-phase stochastic gradient descent procedure, they fixed the parameters of the TwoPATHCNN and included it in the cascaded architecture(the INPUTCascadeCNN, the LOCALCascadeCNN, or the MFCascadeCNN).
.. image:: Cascaded_Architectures.png
