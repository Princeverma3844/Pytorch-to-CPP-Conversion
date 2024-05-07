# pytorch_to_cpp_models
Conversion of  pytorch implementations of model such as VGG16, Densenet and LSTM to cpp.

# For densenet model

1. Download the file Mnist_image.h and Dense_net_weights.h from trained model (Dense_net.ipynb). 
2. Add the downloaded file to the folder containing the files.
3. Include the image file  and weight file  as header in the Dense_net.cpp (c++ code of densenet model).
4. Run the C++ code of DenseNet model.

---
We get the same label number in output of the  C++(Dense_net.cpp) implementation as the trained model (Dense_net.ipynb) on kaggle.

# For VGG16 model

1. Download the file cifar10_image.h and VGG_weights.h from trained model (VGG16.ipynb) . 
2. Add the downloaded file to the folder containing the files.
3. Include the image file  and weight file  as header in the VGG16.cpp (c++ code of VGG16 model).
4. Run the file VGG16.cpp (C++ code of VGG16 model).

---
We get the same label number in output of the  C++(VGG16.cpp) implementation as the trained model (VGG16.ipynb) on kaggle.

# For LSTM model

1. Download the file lstm_weights.h from trained model (LSTM.ipynb). 
2. Add the downloaded file to the folder containing the files.
3. Include the weight file  as header in the LSTM.cpp (c++ code of LSTM model).
4. Run the LSTM.cpp (C++ code of LSTM model).

---

