#include <algorithm>
#include <vector>
#include <omp.h>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <pthread.h>
#include <xmmintrin.h>

#define NTHR 16

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
void relu_worker(void *arg)
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


template<typename Dtype>
struct switch_value {};

template<>
struct switch_value<float>
{
    enum { value = 1 };
};

template<>
struct switch_value<double>
{
    enum { value = 2 };
};

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  //uncomment for pThreads
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
  //     pthread_create(&thr[id], NULL, &relu_worker<Dtype>, (void *) &(tInfo[id]));
  //  }
  // void * status;
  // for(id=0; id<NTHR ; ++id){
  //   pthread_join(thr[id],&status);
  // }
  // delete [] thr; 
  // delete [] tInfo;

  if (sizeof(negative_slope) == 4){
      omp_set_num_threads(NTHR);
      double chunksize = count/omp_get_num_threads();
      #pragma omp parallel for schedule(dynamic, chunksize)
      for (int i = 0; i < count; i+=16) {
        __m128 bottom = _mm_load_ps((const float *) bottom_data+i);
        __m128 zero = _mm_setzero_ps();
        _mm_store_ps((float *)top_data+i, _mm_add_ps(_mm_max_ps( bottom,zero), _mm_mul_ps(_mm_set1_ps(negative_slope), _mm_min_ps (bottom, zero))));

      // __m256 bottom = _mm256_load_ps((const float *) bottom_data+i);
      // __m256 zero = _mm256_setzero_ps();
      // _mm256_store_ps((float *)top_data+i, _mm256_add_ps(_mm256_max_ps( bottom,zero), _mm256_mul_ps(_mm256_set1_ps(negative_slope), _mm256_min_ps (bottom, zero))));
      // top_data[i] = std::max(bottom_data[i], Dtype(0)) + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }else{
      omp_set_num_threads(NTHR);
      #pragma omp parallel for 
      for (int i = 0; i < count; i+=16) {
      // __m128 bottom = _mm_load_ps(bottom_data+i);
      // __m128 zero = _mm_setzero_ps();
      // _mm_store_ps(top_data+i, _mm_add_ps(_mm_max_ps(bottom,zero), _mm_mul_ps(_mm_set1_ps(negative_slope), _mm_min_ps (bottom, zero))));
      // top_data[i] = std::max(bottom_data[i], Dtype(0)) + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
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
    // omp_set_num_threads(NTHR);
    // #pragma omp parallel for
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
// template class ReLULayer<float>; 

}  // namespace caffe
