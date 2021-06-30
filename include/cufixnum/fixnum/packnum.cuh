#pragma once

#include "cufixnum/fixnum/word_fixnum.cuh"

namespace cuFIXNUM {
/* This is packed array style high-precision number.
 * It uses mostly same interface as fixnum.
 * Sometimes it's more efficient to run these computations serially.
 *
 * Methods defined here shall also run on host.
 */

#define _DEF_ __device__ __host__

template<int BYTES_, typename digit_ = u32_word_ft>
struct packnum {
public:
  using word_ft = digit_;
  using digit_t = typename digit_::digit_t;
  using packnum_t = packnum;

  static constexpr int BYTES = BYTES_;
  static constexpr int BITS = 8*BYTES;
  static constexpr int SLOT_WIDTH = 1;
  static constexpr int NWORD = BYTES / sizeof(digit_t);

  digit_t data[NWORD];
};

#undef _DEF_

}
