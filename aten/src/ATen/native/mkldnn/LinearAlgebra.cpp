#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()
namespace at {
namespace native {
void matmul_common(
    const ideep::tensor &x,
    const ideep::tensor &w,
    const ideep::tensor &bias,
    ideep::tensor &y,
    at::Scalar beta=1,
    at::Scalar alpha=1,
    const ideep::attr_t& attr = ideep::attr_t()) {
  TORCH_CHECK(false, "mkldnn_bmm: ATen not compiled with MKLDNN support");
}


} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
namespace at {
namespace native {

void matmul_common(
    const ideep::tensor &x,
    const ideep::tensor &w,
    const ideep::tensor &bias,
    ideep::tensor &y,
    at::Scalar beta=1,
    at::Scalar alpha=1,
    const ideep::attr_t& attr = ideep::attr_t()) {
  float dst_coeff = alpha.to<float>();
  float sum_coeff = beta.to<float>();
  if (!bias.is_empty()) {
    // DNNL only supports bias in 1xN dims
    // use bias for sum can save tensor memory copy
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.get_dim(0) == 1) {
      ideep::matmul_forward::compute(x, w, bias, y);
      return;
    }
    ideep::direct_copy::compute(bias, y);
  }

  ideep::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr);
}

at::Tensor mkldnn_mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  Tensor result = at::empty({self.sizes()[0], mat2.sizes()[1]}, self.options());
  return mkldnn_mm_out(self, mat2, result);
}

at::Tensor& mkldnn_mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(self.scalar_type() == at::kBFloat16, "mkldnn_mm:  only enabled for bf16 path");
  TORCH_CHECK(mat2.scalar_type() == at::kBFloat16, "mkldnn_mm:  only enabled for bf16 path");
  TORCH_CHECK(result.scalar_type() == at::kBFloat16, "mkldnn_mm:  only enabled for bf16 path");
  TORCH_CHECK(mkldnn_bf16_device_check(),
      "mkldnn_mm: mkldnn_mm bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  const ideep::tensor x = itensor_from_tensor(self);
  const ideep::tensor w = itensor_from_tensor(mat2);
  ideep::tensor y = itensor_from_tensor(result);
  matmul_common(x, w, ideep::tensor(), y);
  return result;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
