#pragma once

namespace cuFIXNUM {

/*
 * Quotient and remainder via long-division.
 *
 * Source: MCA Algo 1.6, HAC Algo 14.20.
 *
 * TODO: Implement Svoboda divisor preconditioning (using
 * Newton-Raphson iteration to calculate floor(beta^(n+1)/div)) (see
 * MCA Algo 1.7).
 */
template< typename fixnum_t >
class quorem {
    static constexpr int WIDTH = fixnum_t::SLOT_WIDTH;
    typedef typename fixnum_t::word_ft word_ft;

public:
    __device__ void operator()(
        fixnum_t &q, fixnum_t &r,
        fixnum_t A, fixnum_t div) const;

    // TODO: These functions obviously belong somewhere else. The need
    // to be available to both quorem (here) and quorem_preinv.
    static __device__ int normalise_divisor(fixnum_t &div);
    static __device__ fixnum_t normalise_dividend(fixnum_t &u, int k);
    static __device__ fixnum_t normalise_dividend(fixnum_t &u_hi, fixnum_t &u_lo, int k);
    static __device__ void quorem_with_candidate_quotient(
        fixnum_t &quo, fixnum_t &rem,
        fixnum_t A_hi, fixnum_t A_lo, fixnum_t div, fixnum_t q);
};

template< typename fixnum_t >
__device__ int
quorem<fixnum_t>::normalise_divisor(fixnum_t &div) {
    static constexpr int BITS = fixnum_t::BITS;
    int lz = BITS - (fixnum_t::msb(div) + 1);
    fixnum_t overflow;
    fixnum_t::lshift(div, overflow, div, lz);
    assert(fixnum_t::is_zero(overflow));
    return lz;
}

// TODO: Ideally the algos would be written to incorporate the
// normalisation factor, rather than "physically" normalising the
// dividend.
template< typename fixnum_t >
__device__ fixnum_t
quorem<fixnum_t>::normalise_dividend(fixnum_t &u, int k) {
    fixnum_t overflow;
    fixnum_t::lshift(u, overflow, u, k);
    return overflow;
}

// TODO: Ideally the algos would be written to incorporate the
// normalisation factor, rather than "physically" normalising the
// dividend.
template< typename fixnum_t >
__device__ fixnum_t
quorem<fixnum_t>::normalise_dividend(fixnum_t &u_hi, fixnum_t &u_lo, int k) {
    fixnum_t hi_part, middle_part;
    fixnum_t::lshift(u_hi, hi_part, u_hi, k);
    fixnum_t::lshift(u_lo, middle_part, u_lo, k);
    word_ft cy;
    fixnum_t::add_cy(u_hi, cy, u_hi, middle_part);
    assert(word_ft::is_zero(cy));
    return hi_part;
}

template< typename fixnum_t >
__device__ void
quorem<fixnum_t>::quorem_with_candidate_quotient(
    fixnum_t &quo, fixnum_t &rem,
    fixnum_t A_hi, fixnum_t A_lo, fixnum_t div, fixnum_t q)
{
    fixnum_t hi, lo, r, t, msw;
    word_ft br;
    int L = fixnum_t::layout::laneIdx();

    // (hi, lo) = q*d
    fixnum_t::mul_wide(hi, lo, q, div);

    // (msw, r) = A - q*d
    fixnum_t::sub_br(r, br, A_lo, lo);
    fixnum_t::sub_br(msw, t, A_hi, hi);
    assert(word_ft::is_zero(t));  // A_hi >= hi

    // TODO: Could skip these two lines if we could pass br to the last
    // sub_br above as a "borrow in".
    // Make br into a fixnum_t
    br = (L == 0) ? br : word_ft::zero(); // digit to fixnum_t
    fixnum_t::sub_br(msw, t, msw, br);
    assert(word_ft::is_zero(t));  // msw >= br
    assert((L == 0 && word_ft::cmp(msw, 4) < 0)
           || word_ft::is_zero(msw)); // msw < 4 (TODO: possibly should have msw < 3)
    // Broadcast
    msw = fixnum_t::layout::shfl(msw, 0);

    // NB: Could call incr_cy in the loops instead; as is, it will
    // incur an extra add_cy even when msw is 0 and r < d.
    word_ft q_inc = word_ft::zero();
    while ( ! word_ft::is_zero(msw)) {
        fixnum_t::sub_br(r, br, r, div);
        word_ft::sub(msw, msw, br);
        word_ft::incr(q_inc);
    }
    fixnum_t::sub_br(t, br, r, div);
    while (word_ft::is_zero(br)) {
        r = t;
        word_ft::incr(q_inc);
        fixnum_t::sub_br(t, br, r, div);
    }
    // TODO: Replace loops above with something like the one below,
    // which will reduce warp divergence a bit.
#if 0
    fixnum_t tmp, q_inc;
    while (1) {
        br = fixnum_t::sub_br(tmp, r, div);
        if (msw == 0 && br == 1)
            break;
        msr -= br;
        ++q_inc;
        r = tmp;
    }
#endif

    q_inc = (L == 0) ? q_inc : word_ft::zero();
    fixnum_t::add(q, q, q_inc);

    quo = q;
    rem = r;
}

#if 0
template< typename fixnum_t >
__device__ void
quorem<fixnum_t>::operator()(
    fixnum_t &q_hi, fixnum_t &q_lo, fixnum_t &r,
    fixnum_t A_hi, fixnum_t A_lo, fixnum_t div) const
{
    int k = normalise_divisor(div);
    fixnum_t t = normalise_dividend(A_hi, A_lo, k);
    assert(t == 0); // dividend too big.

    fixnum_t r_hi;
    (*this)(q_hi, r_hi, A_hi, div);

    // FIXME WRONG! r_hi is not a good enough candidate quotient!
    // Do div2by1 of (r_hi, A_lo) by div using that r_hi < div.
    // r_hi is now the candidate quotient
    fixnum_t qq = r_hi;
    if (fixnum_t::cmp(A_lo, div) > 0)
        fixnum_t::incr_cy(qq);

    quorem_with_candidate_quotient(q_lo, r, r_hi, A_lo, div, qq);

    digit lo_bits = fixnum_t::rshift(r, r, k);
    assert(lo_bits == 0);
}
#endif

// TODO: Implement a specifically *parallel* algorithm for division,
// such as those of Takahashi.
template< typename fixnum_t >
__device__ void
quorem<fixnum_t>::operator()(
    fixnum_t &q, fixnum_t &r, fixnum_t A, fixnum_t div) const
{
    int n = fixnum_t::most_sig_dig(div) + 1;
    assert(n >= 0); // division by zero.

    word_ft div_msw = fixnum_t::get(div, n - 1);

    // TODO: Factor out the normalisation code.
    int k = word_ft::clz(div_msw); // guaranteed to be >= 0, since div_msw != 0

    // div is normalised when its msw is >= 2^(WORD_BITS - 1),
    // i.e. when its highest bit is on, i.e. when the number of
    // leading zeros of msw is 0.
    if (k > 0) {
        fixnum_t h;
        // Normalise div by shifting it to the left.
        fixnum_t::lshift(div, h, div, k);
        assert(fixnum_t::is_zero(h));
        fixnum_t::lshift(A, h, A, k);
        // FIXME: We should be able to handle this case.
        assert(fixnum_t::is_zero(h));  // FIXME: check if h == 0 using cmp() and zero()
        word_ft::lshift(div_msw, div_msw, k);
    }

    int m = fixnum_t::most_sig_dig(A) - n + 1;
    // FIXME: Just return div in this case
    assert(m >= 0); // dividend too small

    // TODO: Work out if we can just incorporate the normalisation factor k
    // into the subsequent algorithm, rather than actually modifying div and A.

    q = r = fixnum_t::zero();

    // Set q_m
    word_ft qj;
    fixnum_t dj, tmp;
    // TODO: Urgh.
    typedef typename fixnum_t::layout layout;
    dj = layout::shfl_up0(div, m);
    word_ft br;
    fixnum_t::sub_br(tmp, br, A, dj);
    if (br) qj = fixnum_t::zero(); // dj > A
    else { qj = fixnum_t::one(); A = tmp; }

    fixnum_t::set(q, qj, m);

    word_ft dinv = internal::quorem_reciprocal(div_msw);
    for (int j = m - 1; j >= 0; --j) {
        word_ft a_hi, a_lo, hi, dummy;

        // (q_hi, q_lo) = floor((a_{n+j} B + a_{n+j-1}) / div_msw)
        // TODO: a_{n+j} is a_{n+j-1} from the previous iteration; hence I
        // should be able to get away with just one call to get() per
        // iteration.
        // TODO: Could normalise A on the fly here, one word at a time.
        a_hi = fixnum_t::get(A, n + j);
        a_lo = fixnum_t::get(A, n + j - 1);

        // TODO: uquorem_wide has a bad branch at the start which will
        // cause trouble when div_msw < a_hi is not universally true
        // across the warp. Need to investigate ways to alleviate that.
        word_ft::quorem_wide(qj, dummy, a_hi, a_lo, div_msw, dinv);

        dj = layout::shfl_up0(div, j);
        hi = fixnum_t::mul_digit(tmp, qj, dj);
        assert(word_ft::is_zero(hi));

        int iters = 0;
        fixnum_t AA;
        while (1) {
            fixnum_t::sub_br(AA, br, A, tmp);
            if (!br)
                break;
            fixnum_t::sub_br(tmp, br, tmp, dj);
            assert(word_ft::is_zero(br));
            --qj;
            ++iters;
        }
        A = AA;
        assert(iters <= 2); // MCA, Proof of Theorem 1.3.
        fixnum_t::set(q, qj, j);
    }
    // Denormalise A to produce r.
    fixnum_t::rshift(r, tmp, A, k);
    assert(fixnum_t::is_zero(tmp)); // Above division should be exact.
}

} // End namespace cuFIXNUM
