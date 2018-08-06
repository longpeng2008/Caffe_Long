#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/adapted_weighted_softmax_with_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AdaptedWeightedSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom,top);
}

template <typename Dtype>
void AdaptedWeightedSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top,propagate_down,bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(AdaptedWeightedSoftmaxWithLossLayer);

}  // namespace caffe
