#include <iostream>
#include <vector>
#include <algorithm>
#include "VGG_weights.h"
#include "cifar10_image.h"

using namespace std;

// Define the Conv2d structure
struct Conv2dParameters {
    int in_channels;
    int out_channels;
    int kernel_size;
    int padding;
    int stride;
    vector<vector<vector<vector<double>>>> weights;  // 4D weights tensor [out_channels][in_channels][kernel_height][kernel_width]
    vector<double> bias;  // Bias for each output channel
};

vector<vector<vector<double>>> addBias(const vector<vector<vector<double>>>& tensor, const vector<double>& bias) {
    vector<vector<vector<double>>> result(tensor.size(), vector<vector<double>>(tensor[0].size(), vector<double>(tensor[0][0].size(), 0.0)));
    for (size_t oc = 0; oc < tensor.size(); ++oc) {
        for (size_t i = 0; i < tensor[0].size(); ++i) {
            for (size_t j = 0; j < tensor[0][0].size(); ++j) {
                result[oc][i][j] = tensor[oc][i][j] + bias[oc];
            }
        }
    }
    return result;
}

// Function to perform convolution
vector<vector<vector<double>>> conv2d(vector<vector<vector<double>>>& input,
                                      Conv2dParameters conv_params) {
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    
    int output_height = (input_height + 2 * conv_params.padding - conv_params.kernel_size) / conv_params.stride + 1;
    int output_width = (input_width + 2 * conv_params.padding - conv_params.kernel_size) / conv_params.stride + 1;
    // Initialize the output tensor with zeros
    vector<vector<vector<double>>> output(conv_params.out_channels,
                                           vector<vector<double>>(output_height, vector<double>(output_width, 0.0)));

    // Perform the convolution operation
    for (int oc = 0; oc < conv_params.out_channels; ++oc) {
        for (int ic = 0; ic < conv_params.in_channels; ++ic) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    for (int m = 0; m < conv_params.kernel_size; ++m) {
                        for (int n = 0; n < conv_params.kernel_size; ++n) {
                            int input_row = i * conv_params.stride + m - conv_params.padding;
                            int input_col = j * conv_params.stride + n - conv_params.padding;
                            if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
                                output[oc][i][j] += input[ic][input_row][input_col] * conv_params.weights[oc][ic][m][n];
                            }
                        }
                    }
                }
            }
        }
    }
    return addBias(output, conv_params.bias);  // Add bias for the output channels
    // return output;
}

// Define the Linear layer (equivalent to nn.Linear in PyTorch)
struct Linear {
    int in_features;
    int out_features;
    vector<vector<double>> weights;  // 2D weights tensor [out_features][in_features]
    vector<double> bias;  // Bias for each output feature
};

// Function to perform linear transformation
vector<double> linear(const vector<double>& input, const Linear& linear_params) {
    vector<double> output(linear_params.out_features, 0.0);

    // Perform linear transformation
    for (int i = 0; i < linear_params.out_features; ++i) {
        for (int j = 0; j < linear_params.in_features; ++j) {
            output[i] += input[j] * linear_params.weights[i][j];
        }
        output[i] += linear_params.bias[i];
    }

    return output;
}

vector<vector<vector<double>>> relu(const vector<vector<vector<double>>>& input) {
    vector<vector<vector<double>>> result(input.size(), vector<vector<double>>(input[0].size(), vector<double>(input[0][0].size(), 0.0)));
    for (size_t oc = 0; oc < input.size(); ++oc) {
        for (size_t i = 0; i < input[0].size(); ++i) {
            for (size_t j = 0; j < input[0][0].size(); ++j) {
                double zero = 0;
                result[oc][i][j] = max(input[oc][i][j], zero);
            }
        }
    }
    return result;
}

// Define the VGG16 model
class VGG16 {
private:
    Conv2dParameters conv1;
    Conv2dParameters conv2;
    Conv2dParameters conv3;
    Linear linear_layer;

public:
    VGG16(int num_classes) {
        // Initialize convolutional layers
        conv1.in_channels = 3;
        conv1.out_channels = 16;
        conv1.kernel_size = 3;
        conv1.weights = conv0_weight;
        conv1.bias = conv0_bias;
        conv1.padding = 1;
        conv1.stride = 1;

        conv2.in_channels = 16;
        conv2.out_channels = 32;
        conv2.kernel_size = 3;
        conv2.weights = conv1_weight;
        conv2.bias = conv1_bias;
        conv2.padding = 1;
        conv2.stride = 1;

        conv3.in_channels = 32;
        conv3.out_channels = 64;
        conv3.kernel_size = 3;
        conv3.weights = conv2_weight;
        conv3.bias = conv2_bias;
        conv3.padding = 1;
        conv3.stride = 1;

        // Initialize linear layer
        linear_layer.in_features = 64 * 8 * 8;  // Assuming input size after convolutions and max pooling
        linear_layer.out_features = num_classes;
        // Initialize weights and bias for linear layer
        linear_layer.weights = linear_weight;
        linear_layer.bias = linear_bias;
    }

    vector<double> forward(vector<vector<vector<double>>>& x) {
        // Apply convolutional layers
        x = conv2d(x, conv1);
        x = relu(x);
        x = conv2d(x, conv2);
        x = relu(x);
        x = max_pool(x);
        x = conv2d(x, conv3);
        x = max_pool(x);

        // Flatten the output for linear layer
        vector<double> flattened_output;
        for (const auto& row : x) {
            for (const auto& col : row) {
                flattened_output.insert(flattened_output.end(), col.begin(), col.end());
            }
        }
        // Apply linear layer
        return linear(flattened_output, linear_layer);
    }

    // Function to perform max pooling
    vector<vector<vector<double>>> max_pool(vector<vector<vector<double>>>& input) {
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int output_height = input_height / 2;
        int output_width = input_width / 2;

        vector<vector<vector<double>>> output(input.size(),
                                             vector<vector<double>>(output_height, vector<double>(output_width, 0.0)));

        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < output_height; ++j) {
                for (size_t k = 0; k < output_width; ++k) {
                    double max_val = max({ input[i][j * 2][k * 2], input[i][j * 2][k * 2 + 1], input[i][j * 2 + 1][k * 2], input[i][j * 2 + 1][k * 2 + 1] });
                    output[i][j][k] = max_val;
                }
            }
        }

        return output;
    }
};

int main() {
    VGG16 model(10);  // Assuming 10 classes for CIFAR-10
    vector<vector<vector<double>>> input = image;

    // Perform forward pass
    vector<double> output = model.forward(input);

    // Print the output dimensions
    
    int img_class = -1;
    double val = -1e9;
    for(int i=0;i<output.size();i++){
        cout<<output[i]<<" ";
        if(output[i] > val){
            val = output[i];
            img_class = i;
        }
    }
    cout<<endl;
    cout<<img_class<<endl;

    return 0;
}
