#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__
  int operator()(const int& x, const int& y) const
  { 
    return a * x + y;
  }
};

void saxpy_fast(float A, thrust::device_vector<int>& X, thrust::device_vector<int> Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

int main(int argc, char** argv)
{
  thrust::device_vector<int> X(10);
  thrust::device_vector<int> Y(10);
  thrust::device_vector<int> Z(10);
  thrust::sequence(X.begin(),X.end());
  thrust::fill(Y.begin(),Y.end(),1);

  //saxpy_fast(0.1,X,Y);
  thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), saxpy_functor(2));

  thrust::copy(Z.begin(),Z.end(),std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}
