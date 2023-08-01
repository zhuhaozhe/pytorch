#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>

at::Tensor quantize_per_tensor(const at::Tensor& t, const at::Tensor& scale_, const at::Tensor& zp_)
{
    // t is shape of [M, K] and contiguous tensor
    int64_t M = t.size(0);
    int64_t K = t.size(1);
    at::Tensor out = at::empty_like(t, at::kByte);
    float scale = scale_.item().toFloat();
    int32_t zp = zp_.item().toInt();
    auto in_ptr0 = t.data_ptr<at::BFloat16>();
    auto out_ptr0 = out.data_ptr<uint8_t>();
    auto n = t.numel();
    auto vecsize = at::vec::Vectorized<float>::size();
    int64_t k_block = 64; // k_block should a multiple of vec_size;
    int64_t n_k_block = K / k_block;
    // process block
    int64_t m_offset = 0;
    for (int m = 0; m < M; m++){
        m_offset += m * K;
        for (int k = 0 ; k < k_block; k += vecsize) {
            int64_t offset = m_offset + k;
            auto in_ptr = in_ptr0 + offset;
            auto out_ptr = out_ptr0 + offset;
            auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr, vecsize);
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
            tmp13.store(out_ptr, vecsize);
        }
    }

    // process tail
    m_offset = 0;
    for (int m = 0; m < M; m++){
        // vec tail
        m_offset += m * K;
        int64_t k = n_k_block * k_block;
        for (; k + vecsize < K; k += vecsize) {
            int64_t offset = m_offset + k;
            auto in_ptr = in_ptr0 + offset;
            auto out_ptr = out_ptr0 + offset;
            auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr, vecsize);
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
            tmp13.store(out_ptr, vecsize);
        }
        // scalar tail
        for(; k < K; k++)
        {
            int64_t offset = m_offset + k;
            auto in_ptr = in_ptr0 + offset;
            auto out_ptr = out_ptr0 + offset;            
            auto tmp0 = *in_ptr;
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(scale);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(zp);
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
            out_ptr[0] = tmp13;
        }
    }
    return out;
}
