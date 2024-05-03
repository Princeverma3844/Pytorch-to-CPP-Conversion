#include <iostream>
#include <vector>
#include <cmath>
#include "lstm_weights.h"  // Include header with weights

using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double tanh(double x) {
    if (x > 20.0) return 1.0;  // Avoid overflow
    else if (x < -20.0) return -1.0;  // Avoid overflow
    else {
        double exp2x = exp(2 * x);
        return (exp2x - 1) / (exp2x + 1);
    }
}

vector<vector<double>> transpose(const vector<vector<double>>& mat) {
    vector<vector<double>> result(mat[0].size(), vector<double>(mat.size()));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

class LSTM {
private:
    int input_length;
    int hidden_length;
    double (*a1)(double);
    double (*a2)(double);
    vector<vector<double>> W_i_f, W_f_f, W_i_i, W_f_i, W_i_c, W_f_c, W_i_o, W_f_o;
    vector<double> bf, bi, bc, bo;

public:
    LSTM(int input_length, int hidden_length, double (*a1)(double), double (*a2)(double))
        : input_length(input_length), hidden_length(hidden_length), a1(a1), a2(a2) {
            init_weights();
        }

    void init_weights() {
        // Initialize weights and biases from header file
        W_i_f = W_i_f_weights;
        W_f_f = W_f_f_weights;
        W_i_i = W_i_i_weights;
        W_f_i = W_f_i_weights;
        W_i_c = W_i_c_weights;
        W_f_c = W_f_c_weights;
        W_i_o = W_i_o_weights;
        W_f_o = W_f_o_weights;

        bf = bf_bias;
        bi = bi_bias;
        bc = bc_bias;
        bo = bo_bias;
    }

    vector<vector<double>> forward(
        vector<vector<double>> x) {
            int sequence_length = x.size();
            vector<vector<double>> hidden_sequence;
            vector<vector<double>> ht, ct;

            vector<vector<double>> h0(sequence_length, vector<double>(hidden_length, 0.0));
            vector<vector<double>> c0(sequence_length, vector<double>(hidden_length, 0.0));

            vector<vector<double>> xt = x;
            vector<vector<double>> temp = mat_mul(h0, W_f_i, false);
            vector<vector<double>> temp1 = add_2d(mat_mul(xt, W_i_i,false), add(mat_mul(h0, W_f_i,false), bi));
            vector<vector<double>> it = apply_activation(add_2d(mat_mul(xt, W_i_i, false), add(mat_mul(h0, W_f_i, false), bi)), a1);
            vector<vector<double>> ft = apply_activation(add_2d(mat_mul(xt, W_i_f, false), add(mat_mul(h0, W_f_f, false), bf)), a1);
            vector<vector<double>> gt = apply_activation(add_2d(mat_mul(xt, W_i_c, false), add(mat_mul(h0, W_f_c, false), bc)), a2);
            vector<vector<double>> ot = apply_activation(add_2d(mat_mul(xt, W_i_o, false), add(mat_mul(h0, W_f_o, false), bo)), a1);

            c0 = add_2d(mul(ft, c0), mul(it, gt));
            h0 = mul(ot, apply_activation(c0, a2));

            hidden_sequence.push_back(h0[0]);

            // Concatenate h0 to hidden_sequence along dimension 0 (rows)
            for (int i = 1; i < h0.size(); ++i) {
                hidden_sequence.push_back(h0[i]);
            }

            // Transpose hidden_sequence
            hidden_sequence = transpose(hidden_sequence);

            return hidden_sequence;
    }

    vector<vector<double>> mul(vector<vector<double>> mat1, vector<vector<double>> mat2) {
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0.0));

        for (int i = 0; i < mat1.size(); i++) {
            for(int j=0;j < mat1[0].size(); j++){
                result[i][j] = mat1[i][j] * mat2[i][j];
            }
        }

        return result;
    }

    vector<vector<double>> mat_mul(vector<vector<double>> mat1, vector<vector<double>> mat2, bool tran = true) {
        if(tran){
            mat1 = transpose(mat1);
        }
        int rows1 = mat1.size();
        int cols1 = mat1[0].size();
        int rows2 = mat2.size();
        int cols2 = mat2[0].size();

        vector<vector<double>> result(rows1, vector<double>(cols2, 0.0));

        for (int i = 0; i < rows1; ++i) {
            for (int j = 0; j < cols2; ++j) {
                for (int k = 0; k < cols1; ++k) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }

        return result;
    }

    vector<vector<double>> add(vector<vector<double>> a, vector<double> b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size(),0));
        
        for(int i=0;i<a.size();i++){
            for(int j=0;j<a[0].size();j++){
                result[i][j] = a[i][j] + b[j];
            }
        }
        return result;
    }

    vector<vector<double>> add_2d(vector<vector<double>> a, vector<vector<double>> b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size(),0));
        for(int i=0;i<a.size();i++){
            for(int j=0;j<a[0].size();j++){
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> apply_activation(vector<vector<double>> input, double (*activation)(double)) {
        vector<vector<double>> result(input.size(), vector<double>(input[0].size(),0));
        for (int i = 0; i < input.size(); ++i) {
            for(int j=0; j< input[0].size(); j++){
                 result[i][j] = activation(input[i][j]);
            }
        }
        return result;
    }
};

class Net {
private:
    LSTM lstm;

public:
    Net(int input_length, int hidden_length, int output_size, double (*a1)(double), double (*a2)(double))
        : lstm(input_length, hidden_length, a1, a2) {}

    vector<double> forward(vector<vector<double>> x) {
        vector<vector<double>> out, _;
        vector<double> lstm_out;
        out = lstm.forward(x);
        out = transpose(out);
        for(int i=0;i<5;i++){
            double sum = 0;
            for(int j=0;j<50;j++){
                sum += (out[i][j])*(fc_weight[0][j]);
            }
            lstm_out.push_back(sum + fc_bias[0]);
        }
        return lstm_out;

    }
};

int main() {
    Net net(5, 50, 5, sigmoid, tanh);

    vector<vector<double>> input_data = {{242},{233},{267},{269},{270}};
    vector<double> output = net.forward(input_data);
    cout<<int(output[4])<<endl;

    return 0;
}
