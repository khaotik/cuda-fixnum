#pragma once

#include "cufixnum/functions/modinv.cuh"

namespace cuFIXNUM {

template< typename fixnum_t >
class divexact {
public:
    __device__ divexact(fixnum_t divisor) {
        b = divisor;

        // divisor must be odd
        // TODO: Handle even divisor. Should be easy: just make sure
        // the 2-part of the divisor and dividend are the same and
        // then remove them.
        typename fixnum_t::digit_t b0 = fixnum_t::get(b, 0);
        assert(b0 & 1);

        // Calculate b inverse
        modinv<fixnum_t> minv;
        minv(bi, b, fixnum_t::BITS/2);
    }

    /*
     * q = a / b, assuming b divides a.
     *
     * Source: MCA Algorithm 1.10.
     */
    __device__ void operator()(fixnum_t &q, fixnum_t a) const {
        fixnum_t t, w = fixnum_t::zero();

        // w <- a bi  (mod 2^(NBITS / 2))

        // FIXME: This is wasteful since we only want the bottom half of the
        // result. Could we do something like:
        //
        //   create half_fixnum which is fixnum< FIXNUM_BYTES / 2 > but
        //   with same slot_layout. Then use half_fixnum::mul_lo(w, a, bi)
        //
        fixnum_t::mul_lo(w, a, bi);
        // FIXME: This doesn't work when SLOT_WIDTH = 0
        //w = (fixnum::slot_layout::laneIdx() < fixnum::SLOT_WIDTH / 2) ? w : 0;

        // TODO: Can use the "middle product" to speed this up a
        // bit. See MCA Section 1.4.5.
        // t <- b w (mod 2^NBITS)
        fixnum_t::mul_lo(t, b, w);
        // t <- a - b w (mod 2^NBITS)
        fixnum_t::sub(t, a, t);
        // t <- bi (a - b w) (mod 2^NBITS)
        fixnum_t::mul_lo(t, bi, t);
        // w <- w + bi (a - b w)
        fixnum_t::add(w, w, t);

        q = w;
    }

private:
    // Divisor
    fixnum_t b;
    // 1/b (mod 2^(NBITS/2)) where NBITS := FIXNUM_BITS.  bi is
    // nevertheless treated as an NBITS fixnum, so its hi half must be
    // all zeros.
    fixnum_t bi;
};

} // End namespace cuFIXNUM
