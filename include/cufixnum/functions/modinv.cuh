#pragma once

namespace cuFIXNUM {

/*
 * Calculate the modular inverse.
 * TODO: Only supports moduli of the form 2^k at the moment.
 */
template< typename fixnum_t >
struct modinv {
    /*
     * Return x = 1/b (mod 2^k).  Must have 0 < k <= BITS.
     *
     * Source: MCA Algorithm 1.10.
     *
     * TODO: Calculate this using the multiple inversion trick (MCA 2.5.1)
     */
    __device__ void operator()(fixnum_t &x, fixnum_t b, int k) const {
        typedef typename fixnum_t::word_ft word_ft;
        // b must be odd
        word_ft b0 = fixnum_t::get(b, 0);
        assert(k > 0 && k <= fixnum_t::BITS);

        word_ft binv;
        word_ft::modinv_2exp(binv, b0);
        x = fixnum_t::zero();
        fixnum_t::set(x, binv, 0);
        if (k <= word_ft::BITS) {
            word_ft::rem_2exp(x, x, k);
            return;
        }

        // Hensel lift x from (mod 2^WORD_BITS) to (mod 2^k)
        // FIXME: Double-check this condition on k!
        while (k >>= 1) {
            fixnum_t t;
            // TODO: Make multiplications faster by using the "middle
            // product" (see MCA 1.4.5 and 3.3.2).
            fixnum_t::mul_lo(t, b, x);
            fixnum_t::sub(t, fixnum_t::one(), t);
            fixnum_t::mul_lo(t, t, x);
            fixnum_t::add(x, x, t);
        }
    }
};

} // End namespace cuFIXNUM
