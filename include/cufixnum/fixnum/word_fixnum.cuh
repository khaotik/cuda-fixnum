#pragma once

#include "cufixnum/fixnum/internal/primitives-inl.cu"

namespace cuFIXNUM {

template<typename T>
class word_fixnum {
public:
    using digit_t = T;
    typedef word_fixnum word_ft;

    static constexpr int BYTES = sizeof(T);
    static constexpr int BITS = BYTES * 8;

private:
    digit_t x;

    // TODO: These should be private
public:
    __device__ __forceinline__
    operator digit_t () const { return x; }

    __device__ __forceinline__
    operator digit_t &() { return x; }

public:
    __device__ __forceinline__
    word_fixnum() { }

    __device__ __forceinline__
    word_fixnum(digit_t z) : x(z) { }

    __device__ __forceinline__
    static void
    set_if(word_ft &s, word_ft a, int cond) {
        s = a & -(digit_t)cond;
    }

    // TODO: Implement/use something like numeric_limits<T>::max() for this
    // and most_negative().
    // FIXME: These two functions assume that T is unsigned.
    __device__ __forceinline__
    static constexpr word_ft
    most_positive() { return ~(word_ft)0; }

    __device__ __forceinline__
    static constexpr word_ft
    most_negative() { return zero(); };

    __device__ __forceinline__
    static constexpr word_ft
    zero() { return (word_ft)0; }

    __device__ __forceinline__
    static constexpr word_ft
    one() { return (word_ft)1; }

    __device__ __forceinline__
    static constexpr word_ft
    two() { return (word_ft)2; }

    __device__ __forceinline__
    static void
    add(word_ft &s, word_ft a, word_ft b) {
        s = a + b;
    }

    // TODO: this function does not follow the convention of later '*_cy'
    // functions of accumulating the carry into cy.
    __device__ __forceinline__
    static void
    add_cy(word_ft &s, digit_t &cy, word_ft a, word_ft b) {
        s = a + b;
        cy = s < a;
    }

    __device__ __forceinline__
    static void
    add_cyio(word_ft &s, digit_t &cy, word_ft a, word_ft b) {
        s = a + cy;
        cy = s < a;
        s += b;
        cy |= s < b;
    }

    __device__ __forceinline__
    static void
    add_cc(word_ft &s, word_ft a, word_ft b) {
        internal::add_cc(s, a, b);
    }

    __device__ __forceinline__
    static void
    addc(word_ft &s, word_ft a, word_ft b) {
        internal::addc(s, a, b);
    }

    __device__ __forceinline__
    static void
    addc_cc(word_ft &s, word_ft a, word_ft b) {
        internal::addc_cc(s, a, b);
    }

    __device__ __forceinline__
    static void
    incr(word_ft &s) {
        ++s;
    }

    __device__ __forceinline__
    static void
    sub(word_ft &d, word_ft a, word_ft b) {
        d = a - b;
    }

    __device__ __forceinline__
    static void
    sub_br(word_ft &d, digit_t &br, word_ft a, word_ft b) {
        d = a - b;
        br = d > a;
    }

    __device__ __forceinline__
    static void
    neg(word_ft &ma, word_ft a) {
        ma = -a;
    }

    __device__ __forceinline__
    static void
    mul_lo(word_ft &lo, word_ft a, word_ft b) {
        lo = a * b;
    }

    // hi * 2^32 + lo = a * b
    __device__ __forceinline__
    static void
    mul_hi(word_ft &hi, word_ft a, word_ft b) {
        internal::mul_hi(hi, a, b);
    }

    // hi * 2^32 + lo = a * b
    __device__ __forceinline__
    static void
    mul_wide(word_ft &hi, word_ft &lo, word_ft a, word_ft b) {
        internal::mul_wide(hi, lo, a, b);
    }

    // (hi, lo) = a * b + c
    __device__ __forceinline__
    static void
    mad_wide(word_ft &hi, word_ft &lo, word_ft a, word_ft b, word_ft c) {
        internal::mad_wide(hi, lo, a, b, c);
    }

    // lo = a * b + c (mod 2^32)
    __device__ __forceinline__
    static void
    mad_lo(word_ft &lo, word_ft a, word_ft b, word_ft c) {
        internal::mad_lo(lo, a, b, c);
    }

    // as above but increment cy by the mad carry
    __device__ __forceinline__
    static void
    mad_lo_cy(word_ft &lo, word_ft &cy, word_ft a, word_ft b, word_ft c) {
        internal::mad_lo_cc(lo, a, b, c);
        internal::addc(cy, cy, 0);
    }

    __device__ __forceinline__
    static void
    mad_hi(word_ft &hi, word_ft a, word_ft b, word_ft c) {
        internal::mad_hi(hi, a, b, c);
    }

    // as above but increment cy by the mad carry
    __device__ __forceinline__
    static void
    mad_hi_cy(word_ft &hi, word_ft &cy, word_ft a, word_ft b, word_ft c) {
        internal::mad_hi_cc(hi, a, b, c);
        internal::addc(cy, cy, 0);
    }

    // TODO: There are weird and only included for mul_wide
    __device__ __forceinline__
    static void
    mad_lo_cc(word_ft &lo, word_ft a, word_ft b, word_ft c) {
        internal::mad_lo_cc(lo, a, b, c);
    }

    // Returns the reciprocal for d.
    __device__ __forceinline__
    static word_ft
    quorem(word_ft &q, word_ft &r, word_ft n, word_ft d) {
        return quorem_wide(q, r, zero(), n, d);
    }

    // Accepts a reciprocal for d.
    __device__ __forceinline__
    static void
    quorem(word_ft &q, word_ft &r, word_ft n, word_ft d, word_ft v) {
        quorem_wide(q, r, zero(), n, d, v);
    }

    // Returns the reciprocal for d.
    // NB: returns q = r = fixnum::MAX if n_hi > d.
    __device__ __forceinline__
    static word_ft
    quorem_wide(word_ft &q, word_ft &r, word_ft n_hi, word_ft n_lo, word_ft d) {
        return internal::quorem_wide(q, r, n_hi, n_lo, d);
    }

    // Accepts a reciprocal for d.
    // NB: returns q = r = fixnum::MAX if n_hi > d.
    __device__ __forceinline__
    static void
    quorem_wide(word_ft &q, word_ft &r, word_ft n_hi, word_ft n_lo, word_ft d, word_ft v) {
        internal::quorem_wide(q, r, n_hi, n_lo, d, v);
    }

    __device__ __forceinline__
    static void
    rem_2exp(word_ft &r, word_ft n, unsigned k) {
        unsigned kp = BITS - k;
        r = (n << kp) >> kp;
    }

    /*
     * Count Leading Zeroes in x.
     *
     * TODO: This is not an intrinsic quality of a digit_t, so probably shouldn't
     * be in the interface.
     */
    __device__ __forceinline__
    static int
    clz(word_ft x) {
        return internal::clz(x);
    }

    /*
     * Count Trailing Zeroes in x.
     *
     * TODO: This is not an intrinsic quality of a digit_t, so probably shouldn't
     * be in the interface.
     */
    __device__ __forceinline__
    static int
    ctz(word_ft x) {
        return internal::ctz(x);
    }

    __device__ __forceinline__
    static int
    cmp(word_ft a, word_ft b) {
        // TODO: There is probably a PTX instruction for this.
        int br = (a - b) > a;
        return br ? -br : (a != b);
    }

    __device__ __forceinline__
    static int
    is_max(word_ft a) { return a == most_positive(); }

    __device__ __forceinline__
    static int
    is_min(word_ft a) { return a == most_negative(); }

    __device__ __forceinline__
    static int
    is_zero(word_ft a) { return a == zero(); }

    __device__ __forceinline__
    static void
    min(word_ft &m, word_ft a, word_ft b) {
        internal::min(m, a, b);
    }

    __device__ __forceinline__
    static void
    max(word_ft &m, word_ft a, word_ft b) {
        internal::max(m, a, b);
    }

    __device__ __forceinline__
    static void
    lshift(word_ft &z, word_ft x, unsigned b) {
        z = x << b;
    }

    __device__ __forceinline__
    static void
    lshift(word_ft &z, word_ft &overflow, word_ft x, unsigned b) {
        internal::lshift(overflow, z, 0, x, b);
    }

    __device__ __forceinline__
    static void
    rshift(word_ft &z, word_ft x, unsigned b) {
        z = x >> b;
    }

    __device__ __forceinline__
    static void
    rshift(word_ft &z, word_ft &underflow, word_ft x, unsigned b) {
        internal::rshift(z, underflow, x, 0, b);
    }

    /*
     * Return 1/b (mod 2^BITS) where b is odd.
     *
     * Source: MCA, Section 2.5.
     */
    __device__ __forceinline__
    static void
    modinv_2exp(word_ft &x, word_ft b) {
        internal::modinv_2exp(x, b);
    }

    /*
     * Return 1 if x = 2^n for some n, 0 otherwise. (Caveat: Returns 1 for x = 0
     * which is not a binary power.)
     *
     * FIXME: This doesn't belong here.
     */
    template< typename uint_type >
    __device__ __forceinline__
    static int
    is_binary_power(uint_type x) {
        //static_assert(std::is_unsigned<uint_type>::value == true,
        //              "template type must be unsigned");
        return ! (x & (x - 1));
    }

    /*
     * y >= x such that y = 2^n for some n. NB: This really is "inclusive"
     * next, i.e. if x is a binary power we just return it.
     *
     * FIXME: This doesn't belong here.
     */
    __device__ __forceinline__
    static word_ft
    next_binary_power(word_ft x) {
        return is_binary_power(x)
            ? x
            : (word_ft)((digit_t)1 << (BITS - clz(x)));
    }
};

typedef word_fixnum<std::uint32_t> u32_word_ft;
typedef word_fixnum<std::uint64_t> u64_word_ft;

} // End namespace cuFIXNUM
