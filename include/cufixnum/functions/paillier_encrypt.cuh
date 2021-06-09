#pragma once

#include "cufixnum/functions/quorem_preinv.cuh"
#include "cufixnum/functions/multi_modexp.cuh"
#include "cufixnum/modnum/modnum_monty_cios.cuh"

namespace cuFIXNUM {

template< typename fixnum_t >
class paillier_encrypt {
public:
    __device__ paillier_encrypt(fixnum_t n_)
        : n(n_), n_sqr(square(n_)), pwr(n_sqr, n_), mod_n2(n_sqr) { }

    /*
     * NB: In reality, the values r^n should be calculated out-of-band or
     * stock-piled and piped into an encryption function.
     */
    __device__ void operator()(fixnum_t &ctxt, fixnum_t m, fixnum_t r) const {
        // TODO: test this properly
        //assert(fixnum_t::slot_layout::laneIdx() < fixnum_t::SLOT_WIDTH/2 || m == 0);
        fixnum_t::mul_lo(m, m, n);
        fixnum_t::incr_cy(m);
        pwr(r, r);
        fixnum_t c_hi, c_lo;
        fixnum_t::mul_wide(c_hi, c_lo, m, r);
        mod_n2(ctxt, c_hi, c_lo);
    }

private:
    typedef modnum_monty_cios<fixnum_t> modnum_t;

    fixnum_t n;
    fixnum_t n_sqr;
    modexp<modnum_t> pwr;
    quorem_preinv<fixnum_t> mod_n2;

    // TODO: It is flipping stupid that this is necessary.
    __device__ fixnum_t square(fixnum_t n) {
        fixnum_t n2;
        // TODO: test this properly
        //assert(fixnum_t::slot_layout::laneIdx() < fixnum_t::SLOT_WIDTH/2 || n == 0);
        fixnum_t::sqr_lo(n2, n);
        return n2;
    }
};

} // End namespace cuFIXNUM
