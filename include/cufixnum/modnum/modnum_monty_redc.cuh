#pragma once

#include "cufixnum/modnum/internal/monty-inl.cu"

namespace cuFIXNUM {

template< typename fixnum_ >
class modnum_monty_redc {
public:
    typedef fixnum_ fixnum_t;
    typedef fixnum_ modnum_t;

    __device__ modnum_monty_redc(fixnum_t mod)
    : monty(mod) {
        if ( ! monty.is_valid) return;

        modinv<fixnum_t> minv;
        minv(inv_mod, mod, fixnum_t::BITS);
        fixnum_t::neg(inv_mod, inv_mod);
#ifndef NDEBUG
        fixnum_t tmp;
        fixnum_t::mul_lo(tmp, inv_mod, mod);
        fixnum_t::add(tmp, tmp, fixnum_t::one());
        assert(fixnum_t::is_zero(tmp));
#endif
    }

    __device__ modnum_t zero() const { return monty.zero(); }
    __device__ modnum_t one() const { return monty.one(); }
    __device__ void add(modnum_t &z, modnum_t x, modnum_t y) const { monty.add(z, x, y); }
    __device__ void sub(modnum_t &z, modnum_t x, modnum_t y) const { monty.sub(z, x, y); }
    __device__ void neg(modnum_t &z, modnum_t x, modnum_t y) const { monty.neg(z, x); }

    __device__ void sqr(modnum_t &z, modnum_t x) const {
        // FIXME: Fix this hack!
        z = zero();
        if (!monty.is_valid) return;

        modnum_t a_hi, a_lo;
        fixnum_t::sqr_wide(a_hi, a_lo, x);
        redc(z, a_hi, a_lo);
    }

    __device__ void mul(modnum_t &z, modnum_t x, modnum_t y) const {
        // FIXME: Fix this hack!
        z = zero();
        if (!monty.is_valid) return;

        modnum_t a_hi, a_lo;
        fixnum_t::mul_wide(a_hi, a_lo, x, y);
        redc(z, a_hi, a_lo);
    }

    // TODO: Might be worth specialising multiplication for this case, since one of
    // the operands is known.
    __device__ void to_modnum(modnum_t &z, fixnum_t x) const {
        mul(z, x, monty.Rsqr_mod);
    }

    __device__ void from_modnum(fixnum_t &z, modnum_t x) const {
        //mul(z, x, fixnum_t::one());
        redc(z, fixnum_t::zero(), x);
    }

private:
    internal::monty<fixnum_t> monty;
    // inv_mod * mod = -1 % 2^fixnum_t::BITS.
    fixnum_t inv_mod;

    __device__ void redc(fixnum_t &r, fixnum_t a_hi, fixnum_t a_lo) const;
};


template< typename fixnum_t >
__device__ void
modnum_monty_redc<fixnum_t>::redc(fixnum_t &r, fixnum_t a_hi, fixnum_t a_lo) const {
    typedef typename fixnum_t::word_ft word_ft;
    fixnum_t b, s_hi, s_lo;
    word_ft cy, c;

    // FIXME: Fix this hack!
    r = zero();
    if (!monty.is_valid) return;

    fixnum_t::mul_lo(b, a_lo, inv_mod);

    // This section is essentially s = floor(mad_wide(b, mod, a) / R)

    // TODO: Can we employ the trick to avoid a multiplication because we
    // know b = am' (mod R)?
    fixnum_t::mul_wide(s_hi, s_lo, b, monty.mod);
    // TODO: Only want the carry; find a cheaper way to determine that
    // without doing the full addition.
    fixnum_t::add_cy(s_lo, cy, s_lo, a_lo);

    // TODO: The fact that we need to turn cy into a fixnum before using it in
    // arithmetic should be handled more cleanly. Also, this code is already in
    // the private function digit_to_fixnum() in ''warp_fixnum.cu'.
    int L = fixnum_t::layout::laneIdx();
    cy = (L == 0) ? cy : word_ft::zero();

    // TODO: The assert below fails; work out why.
#if 0
    // NB: b = am' (mod R) => a + bm = a + amm' = 2a (mod R). So surely
    // all I need to propagate is the top bit of a_lo?
    fixnum_t top_bit, dummy;
    fixnum_t::lshift(dummy, top_bit, a_lo, 1);
    assert(digit::cmp(cy, top_bit) == 0);
#endif
    fixnum_t::add_cy(r, cy, s_hi, cy);
    fixnum_t::add_cy(r, c, r, a_hi);
    word_ft::add(cy, cy, c);
    assert(cy == !!cy); // cy = 0 or 1.

    monty.normalise(r, cy);
}

} // End namespace cuFIXNUM
