#pragma once

#include "cufixnum/functions/modinv.cuh"
#include "cufixnum/modnum/internal/monty-inl.cu"

namespace cuFIXNUM {

template< typename fixnum_t_ >
class modnum_monty_cios {
public:
    typedef fixnum_t_ fixnum_t;
    typedef fixnum_t modnum_t;

    __device__ modnum_monty_cios(fixnum_t modulus);

    __device__ modnum_t zero() const { return monty.zero(); }
    __device__ modnum_t one() const { return monty.one(); }
    __device__ void add(modnum_t &z, modnum_t x, modnum_t y) const { monty.add(z, x, y); }
    __device__ void sub(modnum_t &z, modnum_t x, modnum_t y) const { monty.sub(z, x, y); }
    __device__ void neg(modnum_t &z, modnum_t x, modnum_t y) const { monty.neg(z, x); }

    /**
     * z <- x * y
     */
    __device__ void mul(modnum_t &z, modnum_t x, modnum_t y) const;

    /**
     * z <- x^2
     */
    __device__ void sqr(modnum_t &z, modnum_t x) const {
        mul(z, x, x);
    }

    // TODO: Might be worth specialising multiplication for this case, since one of
    // the operands is known.
    __device__ void to_modnum(modnum_t &z, fixnum_t x) const {
        mul(z, x, monty.Rsqr_mod);
    }

    // TODO: Might be worth specialising multiplication for this case, since one of
    // the operands is known.
    __device__ void from_modnum(fixnum_t &z, modnum_t x) const {
        mul(z, x, fixnum_t::one());
    }

private:
    typedef typename fixnum_t::word_ft word_ft;
    // TODO: Check whether we can get rid of this declaration
    static constexpr int WIDTH = fixnum_t::SLOT_WIDTH;

    internal::monty<fixnum_t> monty;

    // inv_mod * mod = -1 % 2^digit::BITS.
    word_ft  inv_mod;
};


template< typename fixnum_t >
__device__
modnum_monty_cios<fixnum_t>::modnum_monty_cios(fixnum_t mod)
: monty(mod)
{
    if ( ! monty.is_valid)
        return;

    // TODO: Tidy this up.
    modinv<fixnum_t> minv;
    fixnum_t im;
    minv(im, mod, word_ft::BITS);
    word_ft::neg(inv_mod, im);
    // TODO: Ugh.
    typedef typename fixnum_t::layout layout;
    // TODO: Can we avoid this broadcast?
    inv_mod = layout::shfl(inv_mod, 0);
    assert(1 + inv_mod * layout::shfl(mod, 0) == 0);
}

/*
 * z = x * y (mod) in Monty form.
 *
 * Spliced multiplication/reduction implementation of Montgomery
 * modular multiplication.  Specifically it is the CIOS (coursely
 * integrated operand scanning) splice.
 */
template< typename fixnum_t >
__device__ void
modnum_monty_cios<fixnum_t>::mul(modnum_t &z, modnum_t x, modnum_t y) const
{
    typedef typename fixnum_t::layout layout;
    // FIXME: Fix this hack!
    z = zero();
    if (!monty.is_valid) { return; }

    int L = layout::laneIdx();
    word_ft tmp;
    word_ft::mul_lo(tmp, x, inv_mod);
    word_ft::mul_lo(tmp, tmp, fixnum_t::get(y, 0));
    word_ft cy = word_ft::zero();

    for (int i = 0; i < WIDTH; ++i) {
        word_ft u;
        word_ft xi = fixnum_t::get(x, i);
        word_ft z0 = fixnum_t::get(z, 0);
        word_ft tmpi = fixnum_t::get(tmp, i);

        word_ft::mad_lo(u, z0, inv_mod, tmpi);

        word_ft::mad_lo_cy(z, cy, monty.mod, u, z);
        word_ft::mad_lo_cy(z, cy, y, xi, z);

        assert(L || word_ft::is_zero(z));  // z[0] must be 0
        z = layout::shfl_down0(z, 1); // Shift right one word

        word_ft::add_cy(z, cy, z, cy);

        word_ft::mad_hi_cy(z, cy, monty.mod, u, z);
        word_ft::mad_hi_cy(z, cy, y, xi, z);
    }
    // Resolve carries
    word_ft msw = fixnum_t::top_digit(cy);
    cy = layout::shfl_up0(cy, 1); // left shift by 1
    fixnum_t::add_cy(z, cy, z, cy);
    word_ft::add(msw, msw, cy);
    assert(msw == !!msw); // msw = 0 or 1.

    monty.normalise(z, (int)msw);
}

} // End namespace cuFIXNUM
