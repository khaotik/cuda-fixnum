#pragma once

#include "cufixnum/functions/quorem_preinv.cuh"
#include "cufixnum/functions/divexact.cuh"
#include "cufixnum/functions/chinese.cuh"
#include "cufixnum/functions/multi_modexp.cuh"
#include "cufixnum/modnum/modnum_monty_cios.cuh"

namespace cuFIXNUM {

template< typename fixnum_t >
class paillier_decrypt_mod;

template< typename fixnum_t >
class paillier_decrypt {
public:
    __device__ paillier_decrypt(fixnum_t p, fixnum_t q)
        : n(prod(p, q))
        , crt(p, q)
        , decrypt_modp(p, n)
        , decrypt_modq(q, n) {  }

    __device__ void operator()(fixnum_t &ptxt, fixnum_t ctxt_hi, fixnum_t ctxt_lo) const;

private:
    // We only need this in the constructor to initialise decrypt_mod[pq], but we
    // store it here because it's the only way to save the computation and pass
    // it to the constructors of decrypt_mod[pq].
    fixnum_t n;

    // Secret key is (p, q).
    paillier_decrypt_mod<fixnum_t> decrypt_modp, decrypt_modq;

    // TODO: crt and decrypt_modq both compute and hold quorem_preinv(q); find a
    // way to share them.
    chinese<fixnum_t> crt;

    // TODO: It is flipping stupid that this is necessary.
    __device__ fixnum_t prod(fixnum_t p, fixnum_t q) {
        fixnum_t n;
        // TODO: These don't work when SLOT_WIDTH = 0
        //assert(fixnum_t::slot_layout::laneIdx() < fixnum_t::SLOT_WIDTH/2 || p == 0);
        //assert(fixnum_t::slot_layout::laneIdx() < fixnum_t::SLOT_WIDTH/2 || q == 0);
        fixnum_t::mul_lo(n, p, q);
        return n;
    }
};

/**
 * Decrypt the ciphertext c = (c_hi, c_lo) and put the resulting plaintext in m.
 *
 * m, c_hi and c_lo must be PLAINTEXT_DIGITS long.
 */
template< typename fixnum_t >
__device__ void
paillier_decrypt<fixnum_t>::operator()(fixnum_t &ptxt, fixnum_t ctxt_hi, fixnum_t ctxt_lo) const
{
    fixnum_t mp, mq;
    decrypt_modp(mp, ctxt_hi, ctxt_lo);
    decrypt_modq(mq, ctxt_hi, ctxt_lo);
    crt(ptxt, mp, mq);
}


template< typename fixnum_t >
class paillier_decrypt_mod {
public:
    __device__ paillier_decrypt_mod(fixnum_t p, fixnum_t n);

    __device__ void operator()(fixnum_t &mp, fixnum_t c_hi, fixnum_t c_lo) const;

private:
    // FIXME: These all have width = WIDTH/2, so this is a waste of
    // space, and (worse) the operations below waste cycles.

    // Precomputation of
    //   L((1 + n)^(p - 1) mod p^2)^-1 (mod p)
    // for CRT, where n = pq is the public key, and L(x) = (x-1)/p.
    fixnum_t h;

    // We only need this in the constructor to initialise mod_p2 and pow, but we
    // store it here because it's the only way to save the computation and pass
    // it to the constructors of mod_p2 and pow.
    fixnum_t p_sqr;

    // Exact division by p
    divexact<fixnum_t> div_p;
    // Remainder after division by p.
    quorem_preinv<fixnum_t> mod_p;
    // Remainder after division by p^2.
    quorem_preinv<fixnum_t> mod_p2;

    // Modexp for x |--> x^(p - 1) (mod p^2)
    typedef modnum_monty_cios<fixnum_t> modnum;
    modexp<modnum> pow;

    // TODO: It is flipping stupid that these are necessary.
    __device__ fixnum_t square(fixnum_t p) {
        fixnum_t p2;
        // TODO: This doesn't work when SLOT_WIDTH = 0
        //assert(fixnum_t::slot_layout::laneIdx() < fixnum_t::SLOT_WIDTH/2 || p == 0);
        fixnum_t::sqr_lo(p2, p);
        return p2;
    }
    __device__ fixnum_t sub1(fixnum_t p) {
        fixnum_t pm1;
        fixnum_t::sub(pm1, p, fixnum_t::one());
        return pm1;
    }
};


template< typename fixnum_t >
__device__
paillier_decrypt_mod<fixnum_t>::paillier_decrypt_mod(fixnum_t p, fixnum_t n)
    : p_sqr(square(p))
    , div_p(p)
    , mod_p(p)
    , mod_p2(p_sqr)
    , pow(p_sqr, sub1(p))
{
    typedef typename fixnum_t::warp_ft warp_ft;
    warp_ft cy;
    fixnum_t t = n;
    cy = fixnum_t::incr_cy(t);
    // n is the product of primes, and 2^(2^k) - 1 has (at least) k factors,
    // hence n is less than 2^FIXNUM_BITS - 1, hence incrementing n shouldn't
    // overflow.
    assert(warp_ft::is_zero(cy));
    // TODO: Check whether reducing t is necessary.
    mod_p2(t, fixnum_t::zero(), t);
    pow(t, t);
    fixnum_t::decr_br(t);
    div_p(t, t);

    // TODO: Make modinv use xgcd and use modinv instead.
    // Use a^(p-2) = 1 (mod p)
    fixnum_t pm2;
    fixnum_t::sub(pm2, p, fixnum_t::two());
    multi_modexp<modnum> minv(p);
    minv(h, t, pm2);
}

/*
 * Decrypt ciphertext (c_hi, c_lo) and put the result in mp.
 *
 * Decryption mod p of c is put in the (bottom half of) mp.
 */
template< typename fixnum_t >
__device__ void
paillier_decrypt_mod<fixnum_t>::operator()(fixnum_t &mp, fixnum_t c_hi, fixnum_t c_lo) const
{
    fixnum_t c, u, hi, lo;
    // mp = c_hi * 2^n + c_lo (mod p^2)  which is nonzero because p != q
    mod_p2(c, c_hi, c_lo);

    pow(u, c);
    fixnum_t::decr_br(u);
    div_p(u, u);
    // Check that the high half of u is now zero.
    // TODO: This doesn't work when SLOT_WIDTH = 0
    //assert(fixnum_t::slot_layout::laneIdx() < fixnum_t::SLOT_WIDTH/2 || u == 0);

    // TODO: make use of the fact that u and h are half-width.
    fixnum_t::mul_wide(hi, lo, u, h);
    assert(fixnum_t::is_zero(hi));
    mod_p(mp, hi, lo);
}

} // End namespace cuFIXNUM
