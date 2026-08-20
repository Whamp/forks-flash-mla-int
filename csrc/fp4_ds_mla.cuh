#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace fp4_ds_mla {

constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kGroupSize = 32;
constexpr int kPackedBytes = kNopeDim / 2;                   // 224
constexpr int kTokenDataBytes = kPackedBytes + kRopeDim * 2; // 352
constexpr int kScaleGroups = kNopeDim / kGroupSize;          // 14
constexpr int kScaleBytes = 16; // 14 UE8M0 scales + 2 zero padding bytes
constexpr int kRowBytes = kTokenDataBytes + kScaleBytes; // 368

__device__ __forceinline__ float decode_e2m1(uint8_t nibble) {
  const int magnitude_code = nibble & 0x7;
  float magnitude;
  if (magnitude_code <= 4) {
    magnitude = 0.5f * magnitude_code;
  } else if (magnitude_code == 5) {
    magnitude = 3.0f;
  } else if (magnitude_code == 6) {
    magnitude = 4.0f;
  } else {
    magnitude = 6.0f;
  }
  return (nibble & 0x8) ? -magnitude : magnitude;
}

__device__ __forceinline__ void
row_pointers(const uint8_t *cache, int64_t block_stride, int block_size,
             int slot, const uint8_t *&token_data, const uint8_t *&scales) {
  const int block = slot / block_size;
  const int position = slot - block * block_size;
  const uint8_t *block_base =
      cache + static_cast<int64_t>(block) * block_stride;
  token_data = block_base + static_cast<int64_t>(position) * kTokenDataBytes;
  scales = block_base + static_cast<int64_t>(block_size) * kTokenDataBytes +
           static_cast<int64_t>(position) * kScaleBytes;
}

// Decode two consecutive semantic dimensions. ``dimension`` must be even.
__device__ __forceinline__ __nv_bfloat162 decode_pair(const uint8_t *token_data,
                                                      const uint8_t *scales,
                                                      int dimension) {
  if (dimension < kNopeDim) {
    const uint8_t packed = token_data[dimension / 2];
    const float scale =
        ldexpf(1.f, static_cast<int>(scales[dimension / kGroupSize]) - 127);
    return __nv_bfloat162(__float2bfloat16(decode_e2m1(packed & 0x0f) * scale),
                          __float2bfloat16(decode_e2m1(packed >> 4) * scale));
  }
  const __nv_bfloat16 *rope =
      reinterpret_cast<const __nv_bfloat16 *>(token_data + kPackedBytes);
  return __nv_bfloat162(rope[dimension - kNopeDim],
                        rope[dimension + 1 - kNopeDim]);
}

} // namespace fp4_ds_mla
