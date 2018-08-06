#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/knowledge_distillation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom,top);
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(bottom,propagate_down,top);
  }

INSTANTIATE_LAYER_GPU_FUNCS(KnowledgeDistillationLayer);

}  // namespace caffe
