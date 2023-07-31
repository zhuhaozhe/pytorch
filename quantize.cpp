#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>

at::Tensor quantize_per_tensor(const at::Tensor& t, float scale, int32_t zp)
{
    at::Tensor out = at::empty_like(t, at::kByte);
    auto in_ptr0 = t.data_ptr<at::BFloat16>();
    auto out_ptr0 = out.data_ptr<uint8_t>();
    auto n = t.numel();
    auto vecsize = at::vec::Vectorized<float>::size();
    auto vec_end = 0;
    long i0 = 0;
    {
        for(; (i0 + vecsize) <static_cast<long>(n); i0+=static_cast<long>(vecsize))
        {
            auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0 + static_cast<long>(i0), vecsize);
            at::vec::Vectorized<float> res_vec1(0);
            at::vec::Vectorized<float> res_vec2(0);
            std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
            auto tmp1 = res_vec1;
            // auto tmp1 = cvt_bf16_to_fp32(tmp0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.round();
            auto tmp7 = (tmp6);
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.0));
            auto tmp9 = at::vec::maximum(tmp7, tmp8);
            auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(255.0));
            auto tmp11 = at::vec::minimum(tmp9, tmp10);
            auto tmp12 = (tmp11);
            auto tmp13 = at::vec::convert_float_to_uint8(tmp12);
            tmp13.store(out_ptr0 + static_cast<long>(i0), vecsize);
        }
        for(; i0<static_cast<long>(n); i0+=static_cast<long>(1))
        {
            auto tmp0 = in_ptr0[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.05);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1.0);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = std::nearbyint(tmp5);
            auto tmp7 = static_cast<float>(tmp6);
            auto tmp8 = static_cast<float>(0.0);
            // auto tmp9 = max_propagate_nan(tmp7, tmp8);
            auto tmp9 = 0;
            if (at::_isnan(tmp7)) {
                tmp9 = tmp7;
            }
            tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
            auto tmp10 = static_cast<float>(255.0);
            auto tmp11 = 0;
            if (at::_isnan(tmp9)) {
                tmp11 = tmp9;
            }
            tmp11 =  tmp9 < tmp10 ? tmp9 : tmp10;
            // auto tmp11 = min_propagate_nan(tmp9, tmp10);
            auto tmp12 = static_cast<float>(tmp11);
            auto tmp13 = static_cast<unsigned char>(tmp12);
            out_ptr0[static_cast<long>(i0)] = tmp13;
        }
    }
    return out;
}
