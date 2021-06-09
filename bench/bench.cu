#include <cstdio>
#include <cstring>
#include <cassert>

#include "cufixnum/fixnum/warp_fixnum.cuh"
#include "cufixnum/array/fixnum_array.cuh"
#include "cufixnum/functions/modexp.cuh"
#include "cufixnum/functions/multi_modexp.cuh"
#include "cufixnum/modnum/modnum_monty_redc.cuh"
#include "cufixnum/modnum/modnum_monty_cios.cuh"

using namespace std;
using namespace cuFIXNUM;

template< typename fixnum_t >
struct mul_lo {
    __device__ void operator()(fixnum_t &r, fixnum_t a) {
        fixnum_t s;
        fixnum_t::mul_lo(s, a, a);
        r = s;
    }
};

template< typename fixnum_t >
struct mul_wide {
    __device__ void operator()(fixnum_t &r, fixnum_t a) {
        fixnum_t rr, ss;
        fixnum_t::mul_wide(ss, rr, a, a);
        r = ss;
    }
};

template< typename fixnum_t >
struct sqr_wide {
    __device__ void operator()(fixnum_t &r, fixnum_t a) {
        fixnum_t rr, ss;
        fixnum_t::sqr_wide(ss, rr, a);
        r = ss;
    }
};

template< typename modnum >
struct my_modexp {
    typedef typename modnum::fixnum_t fixnum_t;

    __device__ void operator()(fixnum_t &z, fixnum_t x) {
        modexp<modnum> me(x, x);
        fixnum_t zz;
        me(zz, x);
        z = zz;
    };
};

template< typename modnum >
struct my_multi_modexp {
    typedef typename modnum::fixnum_t fixnum_t;

    __device__ void operator()(fixnum_t &z, fixnum_t x) {
        multi_modexp<modnum> mme(x);
        fixnum_t zz;
        mme(zz, x, x);
        z = zz;
    };
};

template< int fn_bytes, typename word_fixnum, template <typename> class Func >
void bench(int nelts) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    if (nelts == 0) {
        puts(" -*-  nelts == 0; skipping...  -*-");
        return;
    }

    uint8_t *input = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i)
        input[i] = (i * 17 + 11) % 256;

    fixnum_array_t *res, *in;
    in = fixnum_array_t::create(input, fn_bytes * nelts, fn_bytes);
    res = fixnum_array_t::create(nelts);

    // warm up
    fixnum_array_t::template map<Func>(res, in);

    clock_t c = clock();
    fixnum_array_t::template map<Func>(res, in);
    c = clock() - c;

    double secinv = (double)CLOCKS_PER_SEC / c;
    double total_MiB = fixnum_t::BYTES * (double)nelts / (1 << 20);
    printf(" %4d   %3d    %6.1f   %7.3f  %12.1f\n",
           fixnum_t::BITS, fixnum_t::word_ft::BITS, total_MiB,
           1/secinv, nelts * 1e-3 * secinv);

    delete in;
    delete res;
    delete[] input;
}

template< template <typename> class Func >
void bench_func(const char *fn_name, int nelts) {
    printf("Function: %s, #elts: %de3\n", fn_name, (int)(nelts * 1e-3));
    printf("fixnum digit  total data   time       Kops/s\n");
    printf(" bits  bits     (MiB)    (seconds)\n");
    bench<4, u32_word_ft, Func>(nelts);
    bench<8, u32_word_ft, Func>(nelts);
    bench<16, u32_word_ft, Func>(nelts);
    bench<32, u32_word_ft, Func>(nelts);
    bench<64, u32_word_ft, Func>(nelts);
    bench<128, u32_word_ft, Func>(nelts);
    puts("");

    bench<8, u64_word_ft, Func>(nelts);
    bench<16, u64_word_ft, Func>(nelts);
    bench<32, u64_word_ft, Func>(nelts);
    bench<64, u64_word_ft, Func>(nelts);
    bench<128, u64_word_ft, Func>(nelts);
    bench<256, u64_word_ft, Func>(nelts);
    puts("");
}

template< typename fixnum_t >
using modexp_redc = my_modexp< modnum_monty_redc<fixnum_t> >;

template< typename fixnum_t >
using modexp_cios = my_modexp< modnum_monty_cios<fixnum_t> >;

template< typename fixnum_t >
using multi_modexp_redc = my_multi_modexp< modnum_monty_redc<fixnum_t> >;

template< typename fixnum_t >
using multi_modexp_cios = my_multi_modexp< modnum_monty_cios<fixnum_t> >;

int main(int argc, char *argv[]) {
    long m = 1;
    if (argc > 1)
        m = atol(argv[1]);
    m = std::max(m, 1000L);

    bench_func<mul_lo>("mul_lo", m);
    puts("");
    bench_func<mul_wide>("mul_wide", m);
    puts("");
    bench_func<sqr_wide>("sqr_wide", m);
    puts("");
    bench_func<modexp_redc>("modexp redc", m / 100);
    puts("");
    bench_func<modexp_cios>("modexp cios", m / 100);
    puts("");

    bench_func<modexp_redc>("multi modexp redc", m / 100);
    puts("");
    bench_func<modexp_cios>("multi modexp cios", m / 100);
    puts("");

    return 0;
}
