#pragma once

#include "cufixnum/functions/quorem_preinv.cuh"
#include "cufixnum/functions/multi_modexp.cuh"
#include "cufixnum/modnum/modnum_monty_cios.cuh"

namespace cuFIXNUM {

template< typename fixnum_t >
class chinese {
public:
    __device__ chinese(fixnum_t p, fixnum_t q);

    __device__ void operator()(fixnum_t &m, fixnum_t mp, fixnum_t mq) const;

private:
    // TODO: These all have width = WIDTH/2, so this is a waste of
    // space, and (worse) the operations below waste cycles.
    fixnum_t p, q, c;  // c = p^-1 (mod q)

    quorem_preinv<fixnum_t> mod_q;
};

template< typename fixnum_t >
__device__
chinese<fixnum_t>::chinese(fixnum_t p_, fixnum_t q_)
    : p(p_), q(q_), mod_q(q)
{
    typedef modnum_monty_cios<fixnum_t> modnum;

    // TODO: q is now stored here and in mod_q; need to work out how
    // to share q between them.  Probably best just to provide quorem_preinv
    // with an accessor to the divisor.

    // TODO: Make modinv use xgcd and use modinv instead.
    // Use a^(q-2) = 1 (mod q)
    fixnum_t qm2, two = fixnum_t::two();
    fixnum_t::sub(qm2, q, two);
    multi_modexp<modnum> minv(q);
    minv(c, p, qm2);
}


/*
 * CRT on Mp and Mq.
 *
 * Mp, Mq, p, q must all be WIDTH/2 digits long
 *
 * Source HAC, Note 14.75.
 */
template< typename fixnum_t >
__device__ void
chinese<fixnum_t>::operator()(fixnum_t &m, fixnum_t mp, fixnum_t mq) const
{
    using word_ft = typename fixnum_t::word_ft;
    // u = (mq - mp) * c (mod q)
    fixnum_t u, t, hi, lo;
    word_ft br;
    fixnum_t::sub_br(u, br, mq, mp);

    // TODO: It would be MUCH better to ensure that the mul_wide
    // and mod_q parts of this condition occur on the main
    // execution path to avoid long warp divergence.
    if (br) {
        // Mp > Mq
        // TODO: Can't I get this from u above?  Need a negation
        // function; maybe use "method of complements".
        fixnum_t::sub_br(u, br, mp, mq);
        assert(word_ft::is_zero(br));

        // TODO: Replace mul_wide with the equivalent mul_lo
        //digit_mul(hi, lo, u, c, width/2);
        fixnum_t::mul_wide(hi, lo, u, c);
        assert(word_ft::is_zero(hi));

        t = fixnum_t::zero();
        //quorem_rem(mod_q, t, hi, lo, width/2);
        mod_q(t, hi, lo);

        // TODO: This is a mess.
        if ( ! fixnum_t::is_zero(t)) {
            fixnum_t::sub_br(u, br, q, t);
            assert(word_ft::is_zero(br));
        } else {
            u = t;
        }
    } else {
        // Mp < Mq
        // TODO: Replace mul_wide with the equivalent mul_lo
        //digit_mul(hi, lo, u, c, width/2);
        fixnum_t::mul_wide(hi, lo, u, c);
        assert(word_ft::is_zero(hi));

        u = fixnum_t::zero();
        //quorem_rem(mod_q, u, hi, lo, width/2);
        mod_q(u, hi, lo);
    }
    // TODO: Replace mul_wide with the equivalent mul_lo
    //digit_mul(hi, lo, u, p, width/2);
    fixnum_t::mul_wide(hi, lo, u, p);
    //shfl_up(hi, width/2, width);
    //t = (L < width/2) ? lo : hi;
    assert(word_ft::is_zero(hi));
    t = lo;

    //digit_add(m, mp, t, width);
    fixnum_t::add(m, mp, t);
}

} // End namespace cuFIXNUM
