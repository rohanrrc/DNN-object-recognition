#include <algorithm>
#include <vector>
#include <omp.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <pthread.h>
// #include "zmmintrin.h"

#define NTHR 4

namespace caffe {


template <typename Dtype> 
struct worker_t {
    Dtype *top_data;
   const Dtype *bottom_data = NULL; 
  int start;
  int end;
   Dtype negative_slope;
   int tid;
};

template <typename Dtype>
void *relu_worker(void *arg)
{
  worker_t<Dtype> *t = static_cast<worker_t<Dtype>*>(arg);
  Dtype *top_data = t->top_data;
  const Dtype *bottom_data = t->bottom_data;

  for(int i = t->start; i < t->end; i++)
  {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + t->negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}


template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  //uncomment for pThread
  // pthread_t *thr = new pthread_t[NTHR];
  // worker_t<Dtype> *tInfo = new worker_t<Dtype>[NTHR];
  //  long id;
  //  for(id=0; id<NTHR; id++){
  //     tInfo[id].top_data = top_data;
  //     tInfo[id].bottom_data = bottom_data;
  //     tInfo[id].negative_slope = negative_slope;
  //     tInfo[id].start = id * count / NTHR;
  //     tInfo[id].end = (id+1)*count / NTHR;
  //     tInfo[id].tid = id;
  //     if (id == NTHR-1) tInfo[id].end  = count;
  //     pthread_create(&thr[id], NULL, relu_worker<Dtype>, (void *) &(tInfo[id]));
  //  }
  // void * status;
  // for(id=0; id<NTHR ; ++id){
  //   pthread_join(thr[id],&status);
  // }
  // delete [] thr;
  // delete [] tInfo;


  // const int chunk_size =count/omp_get_num_threads()
  omp_set_dynamic(0);
  omp_set_num_threads(NTHR);
  #pragma omp parallel for 
  for (int i = 0; i < count; i++) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}


template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    omp_set_num_threads(NTHR);
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
