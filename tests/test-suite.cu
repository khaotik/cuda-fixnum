#include <gtest/gtest.h>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <fstream>
#include <string>
#include <sstream>

#include "cufixnum/array/fixnum_array.cuh"
#include "cufixnum/fixnum/word_fixnum.cuh"
#include "cufixnum/fixnum/warp_fixnum.cuh"
#include "cufixnum/modnum/modnum_monty_cios.cuh"
#include "cufixnum/modnum/modnum_monty_redc.cuh"
#include "cufixnum/functions/modexp.cuh"
#include "cufixnum/functions/paillier_encrypt.cuh"
#include "cufixnum/functions/paillier_decrypt.cuh"

using namespace std;
using namespace cuFIXNUM;

typedef vector<uint8_t> byte_array;

void die_if(bool p, const string &msg) {
    if (p) {
        cerr << "Error: " << msg << endl;
        abort();
    }
}

int
arrays_are_equal(
    const uint8_t *expected, size_t expected_len,
    const uint8_t *actual, size_t actual_len)
{
    if (expected_len > actual_len)
        return actual_len;
    size_t i;
    for (i = 0; i < expected_len; ++i) {
        if (expected[i] != actual[i])
            return i;
    }
    for (; i < actual_len; ++i) {
        if (actual[i] != 0)
            return i;
    }
    return -1;
}


template< typename fixnum_ >
struct TypedPrimitives : public ::testing::Test {
    typedef fixnum_ fixnum_t;

    TypedPrimitives() {}
};

typedef ::testing::Types<
    warp_fixnum<4, u32_word_ft>,
    warp_fixnum<8, u32_word_ft>,
    warp_fixnum<16, u32_word_ft>,
    warp_fixnum<32, u32_word_ft>,
    warp_fixnum<64, u32_word_ft>,
    warp_fixnum<128, u32_word_ft>,

    warp_fixnum<8, u64_word_ft>,
    warp_fixnum<16, u64_word_ft>,
    warp_fixnum<32, u64_word_ft>,
    warp_fixnum<64, u64_word_ft>,
    warp_fixnum<128, u64_word_ft>,
    warp_fixnum<256, u64_word_ft>
> FixnumImplTypes;

TYPED_TEST_CASE(TypedPrimitives, FixnumImplTypes);

void read_into(ifstream &file, uint8_t *buf, size_t nbytes) {
    file.read(reinterpret_cast<char *>(buf), nbytes);
    die_if( ! file.good(), "Read error.");
    die_if(static_cast<size_t>(file.gcount()) != nbytes, "Expected more data.");
}

uint32_t read_int(ifstream &file) {
    uint32_t res;
    file.read(reinterpret_cast<char*>(&res), sizeof(res));
    return res;
}

template<typename fixnum_t>
void read_tcases(
        vector<byte_array> &res,
        fixnum_array<fixnum_t> *&xs,
        const string &fname,
        int nargs) {
    static constexpr int fixnum_bytes = fixnum_t::BYTES;
    ifstream file(fname + "_" + std::to_string(fixnum_bytes));
    die_if( ! file.good(), "Couldn't open file.");

    uint32_t fn_bytes, vec_len, noutvecs;
    fn_bytes = read_int(file);
    vec_len = read_int(file);
    noutvecs = read_int(file);

    stringstream ss;
    ss << "Inconsistent reporting of fixnum bytes. "
       << "Expected " << fixnum_bytes << " got " << fn_bytes << ".";
    die_if(fixnum_bytes != fn_bytes, ss.str());

    size_t nbytes = fixnum_bytes * vec_len;
    uint8_t *buf = new uint8_t[nbytes];

    read_into(file, buf, nbytes);
    xs = fixnum_array<fixnum_t>::create(buf, nbytes, fixnum_bytes);

    // ninvecs = number of input combinations
    uint32_t ninvecs = 1;
    for (int i = 1; i < nargs; ++i)
        ninvecs *= vec_len;
    res.reserve(noutvecs * ninvecs);
    for (uint32_t i = 0; i < ninvecs; ++i) {
        for (uint32_t j = 0; j < noutvecs; ++j) {
            read_into(file, buf, nbytes);
            res.emplace_back(buf, buf + nbytes);
        }
    }

    delete[] buf;
}

template< typename fixnum_t, typename tcase_iter >
void check_result(
    tcase_iter &tcase, uint32_t vec_len,
    initializer_list<const fixnum_array<fixnum_t> *> args,
    int skip = 1,
    uint32_t nvecs = 1)
{
    static constexpr int fixnum_bytes = fixnum_t::BYTES;
    size_t total_vec_len = vec_len * nvecs;
    size_t nbytes = fixnum_bytes * total_vec_len;
    // TODO: The fixnum_arrays are in managed memory; there isn't really any
    // point to copying them into buf.
    byte_array buf(nbytes);

    int arg_idx = 0;
    for (auto arg : args) {
        auto buf_iter = buf.begin();
        for (uint32_t i = 0; i < nvecs; ++i) {
            std::copy(tcase->begin(), tcase->end(), buf_iter);
            buf_iter += fixnum_bytes*vec_len;
            tcase += skip;
        }
        int r = arrays_are_equal(buf.data(), nbytes, arg->get_ptr(), nbytes);
        EXPECT_TRUE(r < 0) << "failed near byte " << r << " in argument " << arg_idx;
        ++arg_idx;
    }
}

template< typename fixnum_t >
struct add_cy {
    __device__ void operator()(fixnum_t &r, fixnum_t &cy, fixnum_t a, fixnum_t b) {
        typedef typename fixnum_t::word_ft word_ft;
        word_ft c;
        fixnum_t::add_cy(r, c, a, b);
        // TODO: This is like digit_to_fixnum
        cy = (fixnum_t::layout::laneIdx() == 0) ? c : word_ft::zero();
    }
};

TYPED_TEST(TypedPrimitives, add_cy) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *cys, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/add_cy", 2);
    int vec_len = xs->length();
    res = fixnum_array_t::create(vec_len);
    cys = fixnum_array_t::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *ys = xs->rotate(i);
        fixnum_array_t::template map<add_cy>(res, cys, xs, ys);
        check_result(tcase, vec_len, {res, cys});
        delete ys;
    }
    delete res;
    delete cys;
    delete xs;
}


template< typename fixnum_t >
struct sub_br {
    __device__ void operator()(fixnum_t &r, fixnum_t &br, fixnum_t a, fixnum_t b) {
        typedef typename fixnum_t::word_ft word_ft;
        word_ft bb;
        fixnum_t::sub_br(r, bb, a, b);
        br = (fixnum_t::layout::laneIdx() == 0) ? bb : word_ft::zero();
    }
};

TYPED_TEST(TypedPrimitives, sub_br) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *brs, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/sub_br", 2);
    int vec_len = xs->length();
    res = fixnum_array_t::create(vec_len);
    brs = fixnum_array_t::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *ys = xs->rotate(i);
        fixnum_array_t::template map<sub_br>(res, brs, xs, ys);
        check_result(tcase, vec_len, {res, brs});
        delete ys;
    }
    delete res;
    delete brs;
    delete xs;
}

template< typename fixnum_t >
struct mul_lo {
    __device__ void operator()(fixnum_t &r, fixnum_t a, fixnum_t b) {
        fixnum_t rr;
        fixnum_t::mul_lo(rr, a, b);
        r = rr;
    }
};

TYPED_TEST(TypedPrimitives, mul_lo) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/mul_wide", 2);
    int vec_len = xs->length();
    res = fixnum_array_t::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *ys = xs->rotate(i);
        fixnum_array_t::template map<mul_lo>(res, xs, ys);
        check_result(tcase, vec_len, {res}, 2);
        delete ys;
    }
    delete res;
    delete xs;
}

template< typename fixnum_t >
struct mul_hi {
    __device__ void operator()(fixnum_t &r, fixnum_t a, fixnum_t b) {
        fixnum_t rr;
        fixnum_t::mul_hi(rr, a, b);
        r = rr;
    }
};

TYPED_TEST(TypedPrimitives, mul_hi) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/mul_wide", 2);
    int vec_len = xs->length();
    res = fixnum_array_t::create(vec_len);

    auto tcase = tcases.begin() + 1;
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *ys = xs->rotate(i);
        fixnum_array_t::template map<mul_hi>(res, xs, ys);
        check_result(tcase, vec_len, {res}, 2);
        delete ys;
    }
    delete res;
    delete xs;
}

template< typename fixnum_t >
struct mul_wide {
    __device__ void operator()(fixnum_t &s, fixnum_t &r, fixnum_t a, fixnum_t b) {
        fixnum_t rr, ss;
        fixnum_t::mul_wide(ss, rr, a, b);
        s = ss;
        r = rr;
    }
};

TYPED_TEST(TypedPrimitives, mul_wide) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *his, *los, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/mul_wide", 2);
    int vec_len = xs->length();
    his = fixnum_array_t::create(vec_len);
    los = fixnum_array_t::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *ys = xs->rotate(i);
        fixnum_array_t::template map<mul_wide>(his, los, xs, ys);
        check_result(tcase, vec_len, {los, his});
        delete ys;
    }
    delete his;
    delete los;
    delete xs;
}

template< typename fixnum_t >
struct sqr_lo {
    __device__ void operator()(fixnum_t &r, fixnum_t a) {
        fixnum_t rr;
        fixnum_t::sqr_lo(rr, a);
        r = rr;
    }
};

TYPED_TEST(TypedPrimitives, sqr_lo) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/sqr_wide", 1);
    int vec_len = xs->length();
    res = fixnum_array_t::create(vec_len);

    fixnum_array_t::template map<sqr_lo>(res, xs);
    auto tcase = tcases.begin();
    check_result(tcase, vec_len, {res}, 2);

    delete res;
    delete xs;
}

template< typename fixnum_t >
struct sqr_hi {
    __device__ void operator()(fixnum_t &r, fixnum_t a) {
        fixnum_t rr;
        fixnum_t::sqr_hi(rr, a);
        r = rr;
    }
};

TYPED_TEST(TypedPrimitives, sqr_hi) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/sqr_wide", 1);
    int vec_len = xs->length();
    res = fixnum_array_t::create(vec_len);

    fixnum_array_t::template map<sqr_hi>(res, xs);
    auto tcase = tcases.begin() + 1;
    check_result(tcase, vec_len, {res}, 2);

    delete res;
    delete xs;
}

template< typename fixnum_t >
struct sqr_wide {
    __device__ void operator()(fixnum_t &s, fixnum_t &r, fixnum_t a) {
        fixnum_t rr, ss;
        fixnum_t::sqr_wide(ss, rr, a);
        s = ss;
        r = rr;
    }
};

TYPED_TEST(TypedPrimitives, sqr_wide) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *his, *los, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/sqr_wide", 1);
    int vec_len = xs->length();
    his = fixnum_array_t::create(vec_len);
    los = fixnum_array_t::create(vec_len);

    fixnum_array_t::template map<sqr_wide>(his, los, xs);
    auto tcase = tcases.begin();
    check_result(tcase, vec_len, {los, his});

    delete his;
    delete los;
    delete xs;
}

template< typename modnum_t >
struct my_modexp {
    typedef typename modnum_t::fixnum_t fixnum_t;

    __device__ void operator()(fixnum_t &z, fixnum_t x, fixnum_t e, fixnum_t m) {
        modexp<modnum_t> me(m, e);
        fixnum_t zz;
        me(zz, x);
        z = zz;
    };
};

// TODO: Refactor the modexp tests; need to fix check_result().
template< typename fixnum_t >
using modexp_redc = my_modexp< modnum_monty_redc<fixnum_t> >;

TYPED_TEST(TypedPrimitives, modexp_redc) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *input, *xs, *zs;
    vector<byte_array> tcases;

    read_tcases(tcases, input, "tests/modexp", 3);
    int vec_len = input->length();
    int vec_len_sqr = vec_len * vec_len;

    res = fixnum_array_t::create(vec_len_sqr);
    xs = input->repeat(vec_len);
    zs = input->rotations(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *tmp = input->rotate(i);
        fixnum_array_t *ys = tmp->repeat(vec_len);

        fixnum_array_t::template map<modexp_redc>(res, xs, ys, zs);
        check_result(tcase, vec_len, {res}, 1, vec_len);

        delete ys;
        delete tmp;
    }
    delete res;
    delete input;
    delete xs;
    delete zs;
}

template< typename fixnum_t >
using modexp_cios = my_modexp< modnum_monty_cios<fixnum_t> >;

TYPED_TEST(TypedPrimitives, modexp_cios) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *input, *xs, *zs;
    vector<byte_array> tcases;

    read_tcases(tcases, input, "tests/modexp", 3);
    int vec_len = input->length();
    int vec_len_sqr = vec_len * vec_len;

    res = fixnum_array_t::create(vec_len_sqr);
    xs = input->repeat(vec_len);
    zs = input->rotations(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *tmp = input->rotate(i);
        fixnum_array_t *ys = tmp->repeat(vec_len);

        fixnum_array_t::template map<modexp_cios>(res, xs, ys, zs);
        check_result(tcase, vec_len, {res}, 1, vec_len);

        delete ys;
        delete tmp;
    }
    delete res;
    delete input;
    delete xs;
    delete zs;
}


template< typename modnum_t >
struct my_multi_modexp {
    typedef typename modnum_t::fixnum_t fixnum_t;

    __device__ void operator()(fixnum_t &z, fixnum_t x, fixnum_t e, fixnum_t m) {
        multi_modexp<modnum_t> mme(m);
        fixnum_t zz;
        mme(zz, x, e);
        z = zz;
    };
};

template< typename fixnum_t >
using multi_modexp_redc = my_multi_modexp< modnum_monty_redc<fixnum_t> >;

TYPED_TEST(TypedPrimitives, multi_modexp_redc) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *input, *xs, *zs;
    vector<byte_array> tcases;

    read_tcases(tcases, input, "tests/modexp", 3);
    int vec_len = input->length();
    int vec_len_sqr = vec_len * vec_len;

    res = fixnum_array_t::create(vec_len_sqr);
    xs = input->repeat(vec_len);
    zs = input->rotations(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *tmp = input->rotate(i);
        fixnum_array_t *ys = tmp->repeat(vec_len);

        fixnum_array_t::template map<multi_modexp_redc>(res, xs, ys, zs);
        check_result(tcase, vec_len, {res}, 1, vec_len);

        delete ys;
        delete tmp;
    }
    delete res;
    delete input;
    delete xs;
    delete zs;
}

template< typename fixnum_t >
using multi_modexp_cios = my_multi_modexp< modnum_monty_cios<fixnum_t> >;

TYPED_TEST(TypedPrimitives, multi_modexp_cios) {
    typedef typename TestFixture::fixnum_t fixnum_t;
    typedef fixnum_array<fixnum_t> fixnum_array_t;

    fixnum_array_t *res, *input, *xs, *zs;
    vector<byte_array> tcases;

    read_tcases(tcases, input, "tests/modexp", 3);
    int vec_len = input->length();
    int vec_len_sqr = vec_len * vec_len;

    res = fixnum_array_t::create(vec_len_sqr);
    xs = input->repeat(vec_len);
    zs = input->rotations(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array_t *tmp = input->rotate(i);
        fixnum_array_t *ys = tmp->repeat(vec_len);

        fixnum_array_t::template map<multi_modexp_cios>(res, xs, ys, zs);
        check_result(tcase, vec_len, {res}, 1, vec_len);

        delete ys;
        delete tmp;
    }
    delete res;
    delete input;
    delete xs;
    delete zs;
}

template< typename fixnum_t >
struct pencrypt {
    __device__ void operator()(fixnum_t &z, fixnum_t p, fixnum_t q, fixnum_t r, fixnum_t m) {
        fixnum_t n, zz;
        fixnum_t::mul_lo(n, p, q);
        paillier_encrypt<fixnum_t> enc(n);
        enc(zz, m, r);
        z = zz;
    };
};

template< typename fixnum_t >
struct pdecrypt {
    __device__ void operator()(fixnum_t &z, fixnum_t ct, fixnum_t p, fixnum_t q, fixnum_t r, fixnum_t m) {
        if (fixnum_t::cmp(p, q) == 0
              || fixnum_t::cmp(r, p) == 0
              || fixnum_t::cmp(r, q) == 0) {
            z = fixnum_t::zero();
            return;
        }
        paillier_decrypt<fixnum_t> dec(p, q);
        fixnum_t n, zz;
        dec(zz, fixnum_t::zero(), ct);
        fixnum_t::mul_lo(n, p, q);
        quorem_preinv<fixnum_t> qr(n);
        qr(m, fixnum_t::zero(), m);

        // z = (z != m)
        z = fixnum_t::word_ft( !! fixnum_t::cmp(zz, m));
    };
};

TYPED_TEST(TypedPrimitives, paillier) {
    typedef typename TestFixture::fixnum_t fixnum_t;

    typedef fixnum_t ctx_t;
    // TODO: BYTES/2 only works when BYTES > 4
    //typedef default_fixnum<ctxt::BYTES/2, typename ctxt::word_tp> ptxt;
    typedef fixnum_t ptx_t;

    typedef fixnum_array<ctx_t> ctx_array_t;
    typedef fixnum_array<ptx_t> ptx_array_t;

    ctx_array_t *ct, *pt, *p;
    vector<byte_array> tcases;
    read_tcases(tcases, p, "tests/paillier_encrypt", 4);

    int vec_len = p->length();
    ct = ctx_array_t::create(vec_len);
    pt = ptx_array_t::create(vec_len);

    // TODO: Parallelise these tests similar to modexp above.
    ctx_array_t *zeros = ctx_array_t::create(vec_len, 0);
    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        ctx_array_t *q = p->rotate(i);
        for (int j = 0; j < vec_len; ++j) {
            ctx_array_t *r = p->rotate(j);
            for (int k = 0; k < vec_len; ++k) {
                ctx_array_t *m = p->rotate(k);

                ctx_array_t::template map<pencrypt>(ct, p, q, r, m);
                check_result(tcase, vec_len, {ct});

                ptx_array_t::template map<pdecrypt>(pt, ct, p, q, r, m);

                size_t nbytes = vec_len * ctx_t::BYTES;
                const uint8_t *zptr = reinterpret_cast<const uint8_t *>(zeros->get_ptr());
                const uint8_t *ptptr = reinterpret_cast<const uint8_t *>(pt->get_ptr());
                EXPECT_TRUE(arrays_are_equal(zptr, nbytes, ptptr, nbytes));

                delete m;
            }
            delete r;
        }
        delete q;
    }

    delete p;
    delete ct;
    delete zeros;
}

int main(int argc, char *argv[]) {
    int r;

    testing::InitGoogleTest(&argc, argv);
    r = RUN_ALL_TESTS();
    return r;
}
