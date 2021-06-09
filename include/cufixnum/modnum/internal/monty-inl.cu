#pragma once

#include "cufixnum/functions/quorem_preinv.cuh"

namespace cuFIXNUM {

namespace internal {

template< typename fixnum_ >
class monty {
public:
    typedef fixnum_ fixnum_t;
    typedef fixnum_ modnum_t;

    __device__ monty(fixnum_t modulus);

    __device__ void add(modnum_t &z, modnum_t x, modnum_t y) const {
        fixnum_t::add(z, x, y);
        if (fixnum_t::cmp(z, mod) >= 0)
            fixnum_t::sub(z, z, mod);
    }

    __device__ void neg(modnum_t &z, modnum_t x) const {
        fixnum_t::sub(z, mod, x);
    }

    __device__ void sub(modnum_t &z, modnum_t x, modnum_t y) const {
        fixnum_t my;
        neg(my, y);
        fixnum_t::add(z, x, my);
        if (fixnum_t::cmp(z, mod) >= 0)
            fixnum_t::sub(z, z, mod);
    }

    /*
     * Return the Montgomery image of one.
     */
    __device__ modnum_t one() const {
        return R_mod;
    }

    /*
     * Return the Montgomery image of one.
     */
    __device__ modnum_t zero() const {
        return fixnum_t::zero();
    }

    // FIXME: Get rid of this hack
    int is_valid;

    // Modulus for Monty arithmetic
    fixnum_t mod;
    // R_mod = 2^fixnum_t::BITS % mod
    modnum_t R_mod;
    // Rsqr = R^2 % mod
    modnum_t Rsqr_mod;

    // TODO: We save this after using it in the constructor; work out
    // how to make it available for later use. For example, it could
    // be used to reduce arguments to modexp prior to the main
    // iteration.
    quorem_preinv<fixnum_t> modrem;

    __device__ void normalise(modnum_t &x, int msb) const;
};


template< typename fixnum_t >
__device__
monty<fixnum_t>::monty(fixnum_t modulus)
: mod(modulus), modrem(modulus)
{
    // mod must be odd > 1 in order to calculate R^-1 mod "mod".
    // FIXME: Handle these errors properly
    if (fixnum_t::two_valuation(modulus) != 0 //fixnum_t::get(modulus, 0) & 1 == 0
            || fixnum_t::cmp(modulus, fixnum_t::one()) == 0) {
        is_valid = 0;
        return;
    }
    is_valid = 1;

    fixnum_t Rsqr_hi, Rsqr_lo;

    // R_mod = R % mod
    modrem(R_mod, fixnum_t::one(), fixnum_t::zero());
    fixnum_t::sqr_wide(Rsqr_hi, Rsqr_lo, R_mod);
    // Rsqr_mod = R^2 % mod
    modrem(Rsqr_mod, Rsqr_hi, Rsqr_lo);
}

/*
 * Let X = x + msb * 2^64.  Then return X -= m if X > m.
 *
 * Assumes X < 2*m, i.e. msb = 0 or 1, and if msb = 1, then x < m.
 */
template< typename fixnum_t >
__device__ void
monty<fixnum_t>::normalise(modnum_t &x, int msb) const {
    typedef typename fixnum_t::word_ft word_ft;
    modnum_t r;
    word_ft br;

    // br = 0 ==> x >= mod
    fixnum_t::sub_br(r, br, x, mod);
    if (msb || word_ft::is_zero(br)) {
        // If the msb was set, then we must have had to borrow.
        assert(!msb || msb == br);
        x = r;
    }
}

} // End namespace internal

} // End namespace cuFIXNUM
