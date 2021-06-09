#pragma once

#include "cufixnum/fixnum/slot_layout.cuh"
#include "cufixnum/fixnum/word_fixnum.cuh"

namespace cuFIXNUM {

/*
 * This is an archetypal implementation of a fixnum instruction
 * set. It defines the de facto interface for such implementations.
 *
 * All methods are defined for the device. It is someone else's
 * problem to get the data onto the device.
 */
template< int BYTES_, typename digit_ = u32_word_ft >
class warp_fixnum {
public:
    // NB: Language convention: Call something a 'digit' when it is constant
    // across the slot, and call it a 'fixnum' when it can vary between lanes in
    // the slot. Similarly, prefix a function call with 'word_ft::' when the
    // arguments are interpreted component-wise, and with 'warp_ft::' when
    // they're interpreted "across the slot".
    using word_ft = digit_;
    using digit_t = typename digit_::digit_t;
    using warp_ft = warp_fixnum;

    static constexpr int BYTES = BYTES_;
    static constexpr int BITS = 8 * BYTES;
    static constexpr int SLOT_WIDTH = BYTES / word_ft::BYTES;
    typedef slot_layout<word_ft, SLOT_WIDTH> layout;

    static_assert(BYTES > 0,
                  "Fixnum bytes must be positive.");
    static_assert(BYTES % word_ft::BYTES == 0,
                  "Fixnum digit size must divide fixnum bytes.");
    // TODO: Specialise std::is_integral for fixnum_u32?
    //static_assert(std::is_integral< digit >::value,
    //        "digit must be integral.");
    struct packed_t {
      digit_t data[SLOT_WIDTH];
    };

private:
    word_ft x;

    // TODO: These should be private
public:
    __device__ __forceinline__
    operator word_ft () const { return x; }

    __device__ __forceinline__
    operator word_ft &() { return x; }

public:
    __device__ __forceinline__
    warp_fixnum() { }

    // TODO: Shouldn't this be equivalent to the digit_to_fixnum() function
    // below?
    __device__ __forceinline__
    warp_fixnum(word_ft z) : x(z) { }

    /***************************
     * Representation functions.
     */

    /*
     * Set r using bytes, interpreting bytes as a base-256 unsigned
     * integer. Return the number of bytes used. If nbytes >
     * BYTES, then the last nbytes - BYTES are ignored.
     *
     * NB: Normally we would expect from_bytes to be exclusively a
     * device function, but it's the same for the host, so we leave it
     * in.
     */
    __host__ __device__ static int from_bytes(uint8_t *r, const uint8_t *bytes, int nbytes) {
        int n = min(nbytes, BYTES);
        memcpy(r, bytes, n);
        memset(r + n, 0, BYTES - n);
        return n;
    }

    /*
     * Set bytes using r, converting r to a base-256 unsigned
     * integer. Return the number of bytes written. If nbytes <
     * BYTES, then the last BYTES - nbytes are ignored.
     *
     * NB: Normally we would expect from_bytes to be exclusively a
     * device function, but it's the same for the host, so we leave it
     * in.
     */
    __host__ __device__ static int to_bytes(uint8_t *bytes, int nbytes, const uint8_t *r) {
        int n = min(nbytes, BYTES);
        memcpy(bytes, r, n);
        return n;
    }
    __device__ static warp_ft unpack(const packed_t arr) {
      return warp_ft(arr.data[layout::laneIdx()]);
    }
    __device__ void pack(packed_t &arr) {
      arr.data[layout::laneIdx()] = x;
    } 

    /*
     * Return word at index idx.
     */
    __device__ static word_ft get(word_ft var, int idx) {
        return layout::shfl(var, idx);
    }

    /*
     * Set var word at index idx to be x.
     */
    __device__ static void set(warp_ft &var, word_ft x, int idx) {
        var = (layout::laneIdx() == idx) ? (warp_ft)x : var;
    }

    /*
     * Return word in most significant place. Might be zero.
     */
    __device__ static word_ft top_digit(warp_ft var) {
        return layout::shfl(var, layout::toplaneIdx);
    }

    /*
     * Return digit in the least significant place. Might be zero.
     *
     * TODO: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static word_ft bottom_digit(warp_ft var) {
        return layout::shfl(var, 0);
    }

    /***********************
     * Arithmetic functions.
     */

    // TODO: Handle carry in
    // TODO: A more consistent syntax might be
    // fixnum add(fixnum a, fixnum b)
    // fixnum add_cc(fixnum a, fixnum b, int &cy_out)
    // fixnum addc(fixnum a, fixnum b, int cy_in)
    // fixnum addc_cc(fixnum a, fixnum b, int cy_in, int &cy_out)
    __device__ static void add_cy(warp_ft &r, word_ft &cy_hi, warp_ft a, warp_ft b) {
        word_ft cy;
        word_ft::add_cy(r, cy, a, b);
        // r propagates carries iff r = FIXNUM_MAX
        word_ft r_cy = effective_carries(cy_hi, word_ft::is_max(r), cy);
        word_ft::add(r, r, r_cy);
    }

    __device__ static void add(warp_ft &r, warp_ft a, warp_ft b) {
        word_ft cy;
        add_cy(r, cy, a, b);
    }

    // TODO: Handle borrow in
    __device__ static void sub_br(warp_ft &r, word_ft &br_hi, warp_ft a, warp_ft b) {
        word_ft br;
        word_ft::sub_br(r, br, a, b);
        // r propagates borrows iff r = FIXNUM_MIN
        word_ft r_br = effective_carries(br_hi, word_ft::is_min(r), br);
        word_ft::sub(r, r, r_br);
    }

    __device__ static void sub(warp_ft &r, warp_ft a, warp_ft b) {
        word_ft br;
        sub_br(r, br, a, b);
    }

    __device__ static warp_ft zero() {
        return word_ft::zero();
    }

    __device__ static warp_ft one() {
        return word_ft(layout::laneIdx() == 0);
    }

    __device__ static warp_ft two() {
        return word_ft(layout::laneIdx() == 0 ? 2 : 0);
    }

    __device__ static int is_zero(warp_ft a) {
        return nonzero_mask(a) == 0;
    }

    __device__ static word_ft incr_cy(warp_ft &r) {
        word_ft cy;
        add_cy(r, cy, r, one());
        return cy;
    }

    __device__ static word_ft decr_br(warp_ft &r) {
        word_ft br;
        sub_br(r, br, r, one());
        return br;
    }

    __device__ static void neg(warp_ft &r, warp_ft a) {
        sub(r, zero(), a);
    }

    /*
     * r = a * u, where a is interpreted as a single word, and u a
     * full fixnum. a should be constant across the slot for the
     * result to make sense.
     *
     * TODO: Can this be refactored with mad_cy?
     * TODO: Come up with a better name for this function. It's
     * scalar multiplication in the vspace of polynomials...
     */
    __device__ static word_ft mul_digit(warp_ft &r, word_ft a, warp_ft u) {
        warp_ft hi, lo;
        word_ft cy, cy_hi;

        word_ft::mul_wide(hi, lo, a, u);
        cy_hi = top_digit(hi);
        hi = layout::shfl_up0(hi, 1);
        add_cy(lo, cy, lo, hi);

        return cy_hi + cy;
    }

    /*
     * r = lo_half(a * b)
     *
     * The "lo_half" is the product modulo 2^(8*BYTES),
     * i.e. the same size as the inputs.
     */
    __device__ static void mul_lo(warp_ft &r, warp_ft a, warp_ft b) {
        // TODO: Implement specific mul_lo function.
        word_ft cy = word_ft::zero();

        r = zero();
        for (int i = layout::WIDTH - 1; i >= 0; --i) {
            word_ft aa = layout::shfl(a, i);

            word_ft::mad_hi_cy(r, cy, aa, b, r);
            // TODO: Could use rotate here, which is slightly
            // cheaper than shfl_up0...
            r = layout::shfl_up0(r, 1);
            cy = layout::shfl_up0(cy, 1);
            word_ft::mad_lo_cy(r, cy, aa, b, r);
        }
        cy = layout::shfl_up0(cy, 1);
        add(r, r, cy);
    }

    /*
     * (s, r) = a * b
     *
     * r is the "lo half" (see mul_lo above) and s is the
     * corresponding "hi half".
     */
    __device__ static void mul_wide(warp_ft &ss, warp_ft &rr, warp_ft a, warp_ft b) {
        int L = layout::laneIdx();

        warp_ft r, s;
        r = warp_ft::zero();
        s = warp_ft::zero();
        word_ft cy = word_ft::zero();

        warp_ft ai = get(a, 0);
        word_ft::mul_lo(s, ai, b);
        r = L == 0 ? s : r;  // r[0] = s[0];
        s = layout::shfl_down0(s, 1);
        word_ft::mad_hi_cy(s, cy, ai, b, s);

        for (int i = 1; i < layout::WIDTH; ++i) {
            warp_ft ai = get(a, i);
            word_ft::mad_lo_cc(s, ai, b, s);

            warp_ft s0 = get(s, 0);
            r = (L == i) ? s0 : r; // r[i] = s[0]
            s = layout::shfl_down0(s, 1);

            // TODO: Investigate whether deferring this carry resolution until
            // after the loop improves performance much.
            word_ft::addc_cc(s, s, cy);  // add carry from prev digit
            word_ft::addc(cy, 0, 0);     // cy = CC.CF
            word_ft::mad_hi_cy(s, cy, ai, b, s);
        }
        cy = layout::shfl_up0(cy, 1);
        add(s, s, cy);
        rr = r;
        ss = s;
    }

    __device__ static void mul_hi(warp_ft &s, warp_ft a, warp_ft b) {
        // TODO: Implement specific mul_hi function.
        warp_ft r;
        mul_wide(s, r, a, b);
    }

    /*
     * Adapt "rediagonalisation" trick described in Figure 4 of Ozturk,
     * Guilford, Gopal (2013) "Large Integer Squaring on Intel
     * Architecture Processors".
     *
     * TODO: This function is only definitively faster than mul_wide when WIDTH
     * is 32 (but in that case it's ~50% faster).
     */
    __device__ static void
    sqr_wide_(warp_ft &ss, warp_ft &rr, warp_ft a)
    {
        constexpr int W = layout::WIDTH;
        int L = layout::laneIdx();

        warp_ft r, s;
        r = warp_ft::zero();
        s = warp_ft::zero();
        warp_ft diag_lo = warp_ft::zero();
        word_ft cy = word_ft::zero();

        for (int i = 0; i < W / 2; ++i) {
            warp_ft a1, a2, s0;
            int lpi = L + i;
            // TODO: Explain how on Earth these formulae pick out the correct
            // terms for the squaring.
            // NB: Could achieve the same with iterative shuffle's; the expressions
            // would be clearer, but the shuffles would (presumably) be more expensive.
            a1 = get(a, lpi < W ? i : lpi - W/2);
            a2 = get(a, lpi < W ? lpi : W/2 + i);

            assert(L != 0 || word_ft::cmp(a1,a2)==0); // a1 = a2 when L == 0

            warp_ft hi, lo;
            word_ft::mul_wide(hi, lo, a1, a2);

            // TODO: These two (almost identical) blocks cause lots of pipeline
            // stalls; need to find a way to reduce their data dependencies.
            word_ft::add_cyio(s, cy, s, lo);
            lo = get(lo, 0);
            diag_lo = (L == 2*i) ? lo : diag_lo;
            s0 = get(s, 0);
            r = (L == 2*i) ? s0 : r; // r[2i] = s[0]
            s = layout::shfl_down0(s, 1);

            word_ft::add_cyio(s, cy, s, hi);
            hi = get(hi, 0);
            diag_lo = (L == 2*i + 1) ? hi : diag_lo;
            s0 = get(s, 0);
            r = (L == 2*i + 1) ? s0 : r; // r[2i+1] = s[0]
            s = layout::shfl_down0(s, 1);
        }

        // TODO: All these carries and borrows into s should be accumulated into
        // one call.
        add(s, s, cy);

        warp_ft overflow;
        lshift_small(s, s, 1);  // s *= 2
        lshift_small(r, overflow, r, 1);  // r *= 2
        add_cy(s, cy, s, overflow); // really a logior, since s was just lshifted.
        assert(word_ft::is_zero(cy));

        // Doubling r above means we've doubled the diagonal terms, though they
        // shouldn't be. Compensate by subtracting a copy of them here.
        word_ft br;
        sub_br(r, br, r, diag_lo);
        br = (L == 0) ? br : word_ft::zero();
        sub(s, s, br);

        // TODO: This is wasteful, since the odd lane lo's are discarded as are
        // the even lane hi's.
        warp_ft lo, hi, ai = get(a, W/2 + L/2);
        word_ft::mul_lo(lo, ai, ai);
        word_ft::mul_hi(hi, ai, ai);
        warp_ft diag_hi = L & 1 ? hi : lo;

        add(s, s, diag_hi);

        rr = r;
        ss = s;
    }

    __device__ __forceinline__ static void
    sqr_wide(warp_ft &ss, warp_ft &rr, warp_ft a) {
        // Width below which the general multiplication function is used instead
        // of this one. TODO: 16 is very high; need to work out why we're not
        // doing better on smaller widths.
        constexpr int SQUARING_WIDTH_THRESHOLD = 16;
        if (layout::WIDTH < SQUARING_WIDTH_THRESHOLD)
            mul_wide(ss, rr, a, a);
        else
            sqr_wide_(ss, rr, a);
    }

    __device__ static void sqr_lo(warp_ft &r, warp_ft a) {
        // TODO: Implement specific sqr_lo function.
        warp_ft s;
        sqr_wide(s, r, a);
    }

    __device__ static void sqr_hi(warp_ft &s, warp_ft a) {
        // TODO: Implement specific sqr_hi function.
        warp_ft r;
        sqr_wide(s, r, a);
    }

    /*
     * Return a mask of width bits whose ith bit is set if and only if
     * the ith digit of r is nonzero. In particular, result is zero
     * iff r is zero.
     */
    __device__ static uint32_t nonzero_mask(warp_ft r) {
        return layout::ballot( ! word_ft::is_zero(r));
    }

    /*
     * Return -1, 0, or 1, depending on whether x is less than, equal
     * to, or greater than y.
     */
    __device__ static int cmp(warp_ft x, warp_ft y) {
        warp_ft r;
        word_ft br;
        sub_br(r, br, x, y);
        // r != 0 iff x != y. If x != y, then br != 0 => x < y.
        return nonzero_mask(r) ? (br ? -1 : 1) : 0;
    }

    /*
     * Return the index of the most significant digit of x, or -1 if x is
     * zero.
     */
    __device__ static int most_sig_dig(warp_ft x) {
        // FIXME: Should be able to get this value from limits or numeric_limits
        // or whatever.
        enum { UINT32_BITS = 8 * sizeof(uint32_t) };
        static_assert(UINT32_BITS == 32, "uint32_t isn't 32 bits");

        uint32_t a = nonzero_mask(x);
        return UINT32_BITS - (internal::clz(a) + 1);
    }

    /*
     * Return the index of the most significant bit of x, or -1 if x is
     * zero.
     *
     * TODO: Give this function a better name; maybe floor_log2()?
     */
    __device__ static int msb(warp_ft x) {
        int b = most_sig_dig(x);
        if (b < 0) return b;
        word_ft y = layout::shfl(x, b);
        // TODO: These two lines are basically the same as most_sig_dig();
        // refactor.
        int c = word_ft::clz(y);
        return word_ft::BITS - (c + 1) + word_ft::BITS * b;
    }

    /*
     * Return the 2-valuation of x, i.e. the integer k >= 0 such that
     * 2^k divides x but 2^(k+1) does not divide x.  Depending on the
     * representation, can think of this as CTZ(x) ("Count Trailing
     * Zeros").  The 2-valuation of zero is *ahem* warp_ft::BITS.
     *
     * TODO: Refactor common code between here, msb() and
     * most_sig_dig(). Perhaps write msb in terms of two_valuation?
     *
     * FIXME: Pretty sure this function is broken; e.g. if x is 0 but width <
     * warpSize, the answer is wrong.
     */
    __device__ static int two_valuation(warp_ft x) {
        uint32_t a = nonzero_mask(x);
        int b = internal::ctz(a), c = 0;
        if (b < SLOT_WIDTH) {
            word_ft y = layout::shfl(x, b);
            c = word_ft::ctz(y);
        } else
            b = SLOT_WIDTH;
        return c + b * word_ft::BITS;
    }

    __device__
    static void
    lshift_small(warp_ft &y, warp_ft &overflow, warp_ft x, int b) {
        assert(b >= 0);
        assert(b <= word_ft::BITS);
        int L = layout::laneIdx();

        warp_ft cy;
        word_ft::lshift(y, cy, x, b);
        overflow = top_digit(cy);
        overflow = (L == 0) ? overflow : warp_ft::zero();
        cy = layout::shfl_up0(cy, 1);
        word_ft::add(y, y, cy); // logior
    }

    __device__
    static void
    lshift_small(warp_ft &y, warp_ft x, int b) {
        assert(b >= 0);
        assert(b <= word_ft::BITS);

        warp_ft cy;
        word_ft::lshift(y, cy, x, b);
        cy = layout::shfl_up0(cy, 1);
        word_ft::add(y, y, cy); // logior
    }

    /*
     * Set y to be x shifted by b bits to the left; effectively
     * multiply by 2^b. Return the top b bits of x in overflow.
     *
     * FIXME: Currently assumes that fixnum is unsigned.
     *
     * TODO: Think of better names for these functions. Something like
     * mul_2exp.
     *
     * TODO: Could improve performance significantly by using the funnel shift
     * instruction: https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
     */
    __device__
    static void
    lshift(warp_ft &y, warp_ft &overflow, warp_ft x, int b) {
        assert(b >= 0);
        assert(b <= BITS);
        int q = b / word_ft::BITS, r = b % word_ft::BITS;

        y = layout::rotate_up(x, q);
        // Hi bits of y[i] (=overflow) become the lo bits of y[(i+1) % width]
        word_ft::lshift(y, overflow, y, r);
        overflow = layout::rotate_up(overflow, 1);
        // TODO: This was "y |= overflow"; any advantage to using logior?
        word_ft::add(y, y, overflow);

        warp_ft t;
        int L = layout::laneIdx();
        word_ft::set_if(overflow, y, L <= q);  // Kill high (q-1) words of y;
        word_ft::rem_2exp(t, overflow, r);     // Kill high BITS - r bits of overflow[q]
        set(overflow, t, q);
        word_ft::set_if(y, y, L >= q);         // Kill low q words of y;
        word_ft::rshift(t, y, r);              // Kill low r bits of y[q]
        word_ft::lshift(t, t, r);
        set(y, t, q);
    }

    __device__
    static void
    lshift(warp_ft &y, warp_ft x, int b) {
        assert(b >= 0);
        assert(b <= BITS);
        int q = b / word_ft::BITS, r = b % word_ft::BITS;

        y = layout::shfl_up0(x, q);
        lshift_small(y, y, r);
    }

    /*
     * Set y to be x shifted by b bits to the right; effectively
     * divide by 2^b. Return the bottom b bits of x.
     *
     * TODO: Think of better names for these functions. Something like
     * mul_2exp.
     */
    __device__
    static void
    rshift(warp_ft &y, warp_ft &underflow, warp_ft x, int b) {
        lshift(underflow, y, x, BITS - b);
    }

    __device__
    static void
    rshift(warp_ft &y, warp_ft x, int b) {
        warp_ft underflow;
        rshift(y, underflow, x, b);
    }

private:
    __device__
    static void
    digit_to_fixnum(word_ft &c) {
        int L = layout::laneIdx();
        // TODO: Try without branching?  c &= -(digit)(L == 0);
        c = (L == 0) ? c : word_ft::zero();
    }

    __device__
    static word_ft
    effective_carries(word_ft &cy_hi, int propagate, int cy) {
        int L = layout::laneIdx();
        uint32_t allcarries, p, g;

        g = layout::ballot(cy);              // carry generate
        p = layout::ballot(propagate);       // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        // NB: There is no way to unify these two expressions to remove the
        // conditional. The conditional should be optimised away though, since
        // WIDTH is a compile-time constant.
        cy_hi = (layout::WIDTH == WARPSIZE) // detect hi overflow
            ? (allcarries < g)
            : ((allcarries >> layout::WIDTH) & 1);
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        return (allcarries >> L) & 1;
    }
};

} // End namespace cuFIXNUM
