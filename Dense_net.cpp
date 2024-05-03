#include <bits/stdc++.h>
#include "Dense_net_weights.h"
// #include "conv_fun.h"
#include "Mnist_image.h"

using namespace std;

// Define a ReLU activation function

// Define a structure for holding convolution parameters
struct Conv2dParameters {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    vector<vector<vector<vector<float>>>> weights;  // 4D weights tensor [out_channels][in_channels][kernel_height][kernel_width]
    vector<float> bias;                              // Bias for each output channel
};

// Function to add bias to a 2D tensor
vector<vector<vector<float>>> addBias(const vector<vector<vector<float>>>& tensor, const vector<float>& bias) {
    vector<vector<vector<float>>> result(tensor.size(), vector<vector<float>>(tensor[0].size(), vector<float>(tensor[0][0].size(), 0.0)));
    for (size_t oc = 0; oc < tensor.size(); ++oc) {
        for (size_t i = 0; i < tensor[0].size(); ++i) {
            for (size_t j = 0; j < tensor[0][0].size(); ++j) {
                result[oc][i][j] = tensor[oc][i][j] + bias[oc];
            }
        }
    }
    return result;
}

// Define a 2D convolution operation
vector<vector<vector<float>>> conv2d(const vector<vector<vector<float>>>& input,
                                      const Conv2dParameters& conv_params) {
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = (input_height + 2 * conv_params.padding - conv_params.kernel_size) / conv_params.stride + 1;
    int output_width = (input_width + 2 * conv_params.padding - conv_params.kernel_size) / conv_params.stride + 1;
    
    // Initialize the output tensor with zeros
    vector<vector<vector<float>>> output(conv_params.out_channels,
                                           vector<vector<float>>(output_height, vector<float>(output_width, 0.0)));

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

vector<vector<vector<float>>> relu(const vector<vector<vector<float>>>& input) {
    vector<vector<vector<float>>> result(input.size(), vector<vector<float>>(input[0].size(), vector<float>(input[0][0].size(), 0.0)));
    for (size_t oc = 0; oc < input.size(); ++oc) {
        for (size_t i = 0; i < input[0].size(); ++i) {
            for (size_t j = 0; j < input[0][0].size(); ++j) {
                result[oc][i][j] = max(input[oc][i][j], 0.0f);
            }
        }
    }
    return result;
}

// Define a concatenation operation
vector<vector<vector<float>>> concat(const vector<vector<vector<float>>>& tensor1,
                                      const vector<vector<vector<float>>>& tensor2) {
    vector<vector<vector<float>>> concatenated(tensor1.size() + tensor2.size(),
                                                 vector<vector<float>>(tensor1[0].size(), vector<float>(tensor1[0][0].size(), 0.0)));
    for (size_t oc = 0; oc < tensor1.size(); ++oc) {
        for (size_t i = 0; i < tensor1[0].size(); ++i) {
            for (size_t j = 0; j < tensor1[0][0].size(); ++j) {
                concatenated[oc][i][j] = tensor1[oc][i][j];
            }
        }
    }
    for (size_t oc = 0; oc < tensor2.size(); ++oc) {
        for (size_t i = 0; i < tensor2[0].size(); ++i) {
            for (size_t j = 0; j < tensor2[0][0].size(); ++j) {
                concatenated[tensor1.size() + oc][i][j] = tensor2[oc][i][j];
            }
        }
    }
    return concatenated;
}

// Define a 2D batch normalization operation
vector<vector<vector<float>>> batchNorm2d(const vector<vector<vector<float>>>& input,
                                           const vector<float>& mean,
                                           const vector<float>& variance,
                                           const vector<float>& scale,
                                           const vector<float>& bias,
                                           float eps=1e-5) {
    vector<vector<vector<float>>> result(input.size(), vector<vector<float>>(input[0].size(), vector<float>(input[0][0].size(), 0.0)));
    for (size_t oc = 0; oc < input.size(); ++oc) {
        for (size_t i = 0; i < input[0].size(); ++i) {
            for (size_t j = 0; j < input[0][0].size(); ++j) {
                result[oc][i][j] = (input[oc][i][j] - mean[oc]) / sqrt(variance[oc] + eps) * scale[oc] + bias[oc];
            }
        }
    }
    return result;
}

class ConvBlock {
private:
    bool use_bn;
    bool dense;
    Conv2dParameters conv_params;

public:
    ConvBlock(int in_channels, int out_channels, int kernel_size, vector<vector<vector<vector<float>>>> weights, 
              vector<float> bias, bool use_bn=true, bool dense=true) {

        this->use_bn = use_bn;
        this->dense = dense;
        this->conv_params.in_channels = in_channels;
        this->conv_params.out_channels = out_channels;
        this->conv_params.kernel_size = kernel_size;
        this->conv_params.stride = 1;
        this->conv_params.padding = 1;
        this->conv_params.weights = weights;
        this->conv_params.bias = bias;
        // Initialize weights and bias (not shown here, assumed to be done elsewhere)
    }

    vector<vector<vector<float>>> forward(vector<vector<vector<float>>> input) {
        vector<vector<vector<float>>> output = conv2d(input, this->conv_params);
        output = relu(output);
        if (this->dense) {
            output = concat(input, output);
        }
        return output;
    }
};

float adaptiveAvgPool2d(vector<vector<float>>& input, int output_height, int output_width) {
    int input_height = input.size();
    int input_width = input[0].size();

    float sum = 0;
    for(int i=0;i<input_height;i++){
        for(int j=0;j<input_width;j++){
            sum += input[i][j];
        }
    }

    return sum;
}

class Net {
private:
    bool dense;
    int input_size;
    int input_channels;
    int num_classes;
    vector<int> outs;
    vector<vector<vector<vector<vector<float>>>>> weights = {conv0_weight, conv1_weight, conv2_weight};
    vector<vector<float>> bias = {conv0_bias, conv1_bias, conv2_bias};

public:
    Net(bool dense=true, int input_size=32, int input_channels=1, int num_classes=10, vector<int> outs={8, 16, 32}) {
        this->dense = dense;
        this->input_size = input_size;
        this->input_channels = input_channels;
        this->num_classes = num_classes;
        this->outs = outs;
        this->weights = {conv0_weight, conv1_weight, conv2_weight};
    }

    int forward(vector<vector<vector<float>>> input) {

        for (size_t i = 0; i < this->outs.size(); ++i) {
            ConvBlock conv_block(this->input_channels, this->outs[i], 3,this->weights[i], this->bias[i], true, this->dense);
            input = conv_block.forward(input);
            this->input_channels += this->outs[i];
        }

        // Classifier head
        Conv2dParameters head_params;
        head_params.in_channels = this->input_channels;
        head_params.out_channels = this->num_classes;
        head_params.kernel_size = 1;
        head_params.stride = 1;
        head_params.padding = 0;
        head_params.weights = classifier_head_weight;
        head_params.bias = classifier_head_bias;

        vector<vector<vector<float>>> output = conv2d(input, head_params);
        int img_class = -1;
        float val = -1e9;
        for(int i=0;i<output.size();i++){
            float temp = adaptiveAvgPool2d(output[i], output[0].size(), output[0][0].size());
            cout<<temp<<" ";
            if(temp > val){
                val = temp;
                img_class = i;
            }
        }
        cout<<endl;
        return img_class;
    }
};

int main() {
    // Create an instance of the Net model
    Net model(true, 28, 1, 10, {8, 16, 32});

    // Create a sample input tensor (assuming 3D for simplicity)
    vector<vector<vector<float>>> img = image;

    // Perform forward pass
    int output = model.forward(img);

    // Display output size
    cout<<output<<endl;

    return 0;
}
