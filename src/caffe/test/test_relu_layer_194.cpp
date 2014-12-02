// #include <cstring>
// #include <vector>
// #include <sys/time.h>
// #include <time.h>

// #include "gtest/gtest.h"

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/filler.hpp"
// #include "caffe/vision_layers.hpp"

// #include "caffe/test/test_caffe_main.hpp"
// #include "caffe/test/test_gradient_check_util.hpp"

// double timestamp()
// {
//   struct timeval tv;
//   gettimeofday (&tv, 0);
//   return tv.tv_sec + 1e-6*tv.tv_usec;
// }

// namespace caffe {

// template <typename TypeParam>
// class ReluLayerTest : public MultiDeviceTest<TypeParam> {
//   typedef typename TypeParam::Dtype Dtype;

//  protected:
//   ReluLayerTest()
//       : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
//         blob_top_(new Blob<Dtype>()) {
//     Caffe::set_random_seed(1701);
//     // fill the values
//     FillerParameter filler_param;
//     GaussianFiller<Dtype> filler(filler_param);
//     filler.Fill(this->blob_bottom_);
//     blob_bottom_vec_.push_back(blob_bottom_);
//     blob_top_vec_.push_back(blob_top_);
//   }
//   virtual ~ReluLayerTest() { delete blob_bottom_; delete blob_top_; }
//   Blob<Dtype>* const blob_bottom_;
//   Blob<Dtype>* const blob_top_;
//   vector<Blob<Dtype>*> blob_bottom_vec_;
//   vector<Blob<Dtype>*> blob_top_vec_;

//   void TestDropoutForward(const float dropout_ratio) {
//     LayerParameter layer_param;
//     // Fill in the given dropout_ratio, unless it's 0.5, in which case we don't
//     // set it explicitly to test that 0.5 is the default.
//     if (dropout_ratio != 0.5) {
//       layer_param.mutable_dropout_param()->set_dropout_ratio(dropout_ratio);
//     }
//     Caffe::set_phase(Caffe::TRAIN);
//     DropoutLayer<Dtype> layer(layer_param);
//     layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
//     layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
//     // Now, check values
//     const Dtype* bottom_data = this->blob_bottom_->cpu_data();
//     const Dtype* top_data = this->blob_top_->cpu_data();
//     float scale = 1. / (1. - layer_param.dropout_param().dropout_ratio());
//     const int count = this->blob_bottom_->count();
//     // Initialize num_kept to count the number of inputs NOT dropped out.
//     int num_kept = 0;
//     for (int i = 0; i < count; ++i) {
//       if (top_data[i] != 0) {
//         ++num_kept;
//         EXPECT_EQ(top_data[i], bottom_data[i] * scale);
//       }
//     }
//     const Dtype std_error = sqrt(dropout_ratio * (1 - dropout_ratio) / count);
//     // Fail if the number dropped was more than 1.96 * std_error away from the
//     // expected number -- requires 95% confidence that the dropout layer is not
//     // obeying the given dropout_ratio for test failure.
//     const Dtype empirical_dropout_ratio = 1 - num_kept / Dtype(count);
//     EXPECT_NEAR(empirical_dropout_ratio, dropout_ratio, 1.96 * std_error);
//   }
// };

// TYPED_TEST_CASE(ReluLayerTest, TestDtypesAndDevices);



// TYPED_TEST(ReluLayerTest, TestReLU) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ReLULayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   double t0 = timestamp();
//   layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   t0 = timestamp()-t0;
//   printf("timing RELU ----------------------------------------------: %g seconds\n", t0 );
//   // Now, check values
//   const Dtype* bottom_data = this->blob_bottom_->cpu_data();
//   const Dtype* top_data = this->blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_bottom_->count(); ++i) {
//     EXPECT_GE(top_data[i], 0.);
//     EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
//   }
// }

// TYPED_TEST(ReluLayerTest, TestReLUGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ReLULayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
//   checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
//       &(this->blob_top_vec_));
// }

// TYPED_TEST(ReluLayerTest, TestReLUWithNegativeSlope) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   layer_param.ParseFromString("relu_param{negative_slope:0.01}");
//   ReLULayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   // Now, check values
//   const Dtype* bottom_data = this->blob_bottom_->cpu_data();
//   const Dtype* top_data = this->blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_bottom_->count(); ++i) {
//     EXPECT_GE(top_data[i], 0.);
//     EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
//   }
// }

// TYPED_TEST(ReluLayerTest, TestReLUGradientWithNegativeSlope) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   layer_param.ParseFromString("relu_param{negative_slope:0.01}");
//   ReLULayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
//   checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
//       &(this->blob_top_vec_));
// }


// #ifdef USE_CUDNN
// template <typename Dtype>
// class CuDNNReluLayerTest : public ::testing::Test {
//  protected:
//   CuDNNReluLayerTest()
//       : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
//         blob_top_(new Blob<Dtype>()) {
//     Caffe::set_random_seed(1701);
//     // fill the values
//     FillerParameter filler_param;
//     GaussianFiller<Dtype> filler(filler_param);
//     filler.Fill(this->blob_bottom_);
//     blob_bottom_vec_.push_back(blob_bottom_);
//     blob_top_vec_.push_back(blob_top_);
//   }
//   virtual ~CuDNNReluLayerTest() { delete blob_bottom_; delete blob_top_; }
//   Blob<Dtype>* const blob_bottom_;
//   Blob<Dtype>* const blob_top_;
//   vector<Blob<Dtype>*> blob_bottom_vec_;
//   vector<Blob<Dtype>*> blob_top_vec_;
// };

// TYPED_TEST_CASE(CuDNNReluLayerTest, TestDtypes);

// TYPED_TEST(CuDNNReluLayerTest, TestReLUCuDNN) {
//   Caffe::set_mode(Caffe::GPU);
//   LayerParameter layer_param;
//   CuDNNReLULayer<TypeParam> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   // Now, check values
//   const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
//   const TypeParam* top_data = this->blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_bottom_->count(); ++i) {
//     EXPECT_GE(top_data[i], 0.);
//     EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
//   }
// }

// TYPED_TEST(CuDNNReluLayerTest, TestReLUGradientCuDNN) {
//   Caffe::set_mode(Caffe::GPU);
//   LayerParameter layer_param;
//   CuDNNReLULayer<TypeParam> layer(layer_param);
//   GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
//   checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
//       &(this->blob_top_vec_));
// }

// TYPED_TEST(CuDNNReluLayerTest, TestReLUWithNegativeSlopeCuDNN) {
//   Caffe::set_mode(Caffe::GPU);
//   LayerParameter layer_param;
//   layer_param.ParseFromString("relu_param{negative_slope:0.01}");
//   CuDNNReLULayer<TypeParam> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
//   // Now, check values
//   const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
//   const TypeParam* top_data = this->blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_bottom_->count(); ++i) {
//     EXPECT_GE(top_data[i], 0.);
//     EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
//   }
// }

// TYPED_TEST(CuDNNReluLayerTest, TestReLUGradientWithNegativeSlopeCuDNN) {
//   Caffe::set_mode(Caffe::GPU);
//   LayerParameter layer_param;
//   layer_param.ParseFromString("relu_param{negative_slope:0.01}");
//   CuDNNReLULayer<TypeParam> layer(layer_param);
//   GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
//   checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
//       &(this->blob_top_vec_));
// }

// #endif

// }  // namespace caffe
