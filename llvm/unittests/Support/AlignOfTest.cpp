//===- llvm/unittest/Support/AlignOfTest.cpp - Alignment utility tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Disable warnings about questionable type definitions.
// We're testing that even questionable types work with the alignment utilities.
#ifdef _MSC_VER
#pragma warning(disable:4584)
#endif

// Define some fixed alignment types to use in these tests.
#if __cplusplus == 201103L || __has_feature(cxx_alignas)
typedef char alignas(1) A1;
typedef char alignas(2) A2;
typedef char alignas(4) A4;
typedef char alignas(8) A8;
#elif defined(__clang__) || defined(__GNUC__)
typedef char A1 __attribute__((aligned(1)));
typedef char A2 __attribute__((aligned(2)));
typedef char A4 __attribute__((aligned(4)));
typedef char A8 __attribute__((aligned(8)));
#elif defined(_MSC_VER)
typedef __declspec(align(1)) char A1;
typedef __declspec(align(2)) char A2;
typedef __declspec(align(4)) char A4;
typedef __declspec(align(8)) char A8;
#else
# error No supported align as directive.
#endif

// Wrap the forced aligned types in structs to hack around compiler bugs.
struct SA1 { A1 a; };
struct SA2 { A2 a; };
struct SA4 { A4 a; };
struct SA8 { A8 a; };

struct S1 {};
struct S2 { char a; };
struct S3 { int x; };
struct S4 { double y; };
struct S5 { A1 a1; A2 a2; A4 a4; A8 a8; };
struct S6 { double f(); };
struct D1 : S1 {};
struct D2 : S6 { float g(); };
struct D3 : S2 {};
struct D4 : S2 { int x; };
struct D5 : S3 { char c; };
struct D6 : S2, S3 {};
struct D7 : S1, S3 {};
struct D8 : S1, D4, D5 { double x[2]; };
struct D9 : S1, D1 { S1 s1; };
struct V1 { virtual ~V1(); };
struct V2 { int x; virtual ~V2(); };
struct V3 : V1 { virtual ~V3(); };
struct V4 : virtual V2 { int y; virtual ~V4(); };
struct V5 : V4, V3 { double z; virtual ~V5(); };
struct V6 : S1 { virtual ~V6(); };
struct V7 : virtual V2, virtual V6 { virtual ~V7(); };
struct V8 : V5, virtual V6, V7 { double zz; virtual ~V8(); };

// Ensure alignment is a compile-time constant.
char LLVM_ATTRIBUTE_UNUSED test_arr1
  [AlignOf<char>::Alignment > 0]
  [AlignOf<short>::Alignment > 0]
  [AlignOf<int>::Alignment > 0]
  [AlignOf<long>::Alignment > 0]
  [AlignOf<long long>::Alignment > 0]
  [AlignOf<float>::Alignment > 0]
  [AlignOf<double>::Alignment > 0]
  [AlignOf<long double>::Alignment > 0]
  [AlignOf<void *>::Alignment > 0]
  [AlignOf<int *>::Alignment > 0]
  [AlignOf<double (*)(double)>::Alignment > 0]
  [AlignOf<double (S6::*)()>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr2
  [AlignOf<A1>::Alignment > 0]
  [AlignOf<A2>::Alignment > 0]
  [AlignOf<A4>::Alignment > 0]
  [AlignOf<A8>::Alignment > 0]
  [AlignOf<SA1>::Alignment > 0]
  [AlignOf<SA2>::Alignment > 0]
  [AlignOf<SA4>::Alignment > 0]
  [AlignOf<SA8>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr3
  [AlignOf<S1>::Alignment > 0]
  [AlignOf<S2>::Alignment > 0]
  [AlignOf<S3>::Alignment > 0]
  [AlignOf<S4>::Alignment > 0]
  [AlignOf<S5>::Alignment > 0]
  [AlignOf<S6>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr4
  [AlignOf<D1>::Alignment > 0]
  [AlignOf<D2>::Alignment > 0]
  [AlignOf<D3>::Alignment > 0]
  [AlignOf<D4>::Alignment > 0]
  [AlignOf<D5>::Alignment > 0]
  [AlignOf<D6>::Alignment > 0]
  [AlignOf<D7>::Alignment > 0]
  [AlignOf<D8>::Alignment > 0]
  [AlignOf<D9>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr5
  [AlignOf<V1>::Alignment > 0]
  [AlignOf<V2>::Alignment > 0]
  [AlignOf<V3>::Alignment > 0]
  [AlignOf<V4>::Alignment > 0]
  [AlignOf<V5>::Alignment > 0]
  [AlignOf<V6>::Alignment > 0]
  [AlignOf<V7>::Alignment > 0]
  [AlignOf<V8>::Alignment > 0];

TEST(AlignOfTest, BasicAlignmentInvariants) {
  // For a very strange reason, many compilers do not support this. Both Clang
  // and GCC fail to align these properly.
  EXPECT_EQ(1u, alignOf<A1>());
#if 0
  EXPECT_EQ(2u, alignOf<A2>());
  EXPECT_EQ(4u, alignOf<A4>());
  EXPECT_EQ(8u, alignOf<A8>());
#endif

  // But once wrapped in structs, the alignment is correctly managed.
  EXPECT_LE(1u, alignOf<SA1>());
  EXPECT_LE(2u, alignOf<SA2>());
  EXPECT_LE(4u, alignOf<SA4>());
  EXPECT_LE(8u, alignOf<SA8>());

  EXPECT_EQ(1u, alignOf<char>());
  EXPECT_LE(alignOf<char>(),   alignOf<short>());
  EXPECT_LE(alignOf<short>(),  alignOf<int>());
  EXPECT_LE(alignOf<int>(),    alignOf<long>());
  EXPECT_LE(alignOf<long>(),   alignOf<long long>());
  EXPECT_LE(alignOf<char>(),   alignOf<float>());
  EXPECT_LE(alignOf<float>(),  alignOf<double>());
  EXPECT_LE(alignOf<char>(),   alignOf<long double>());
  EXPECT_LE(alignOf<char>(),   alignOf<void *>());
  EXPECT_EQ(alignOf<void *>(), alignOf<int *>());
  EXPECT_LE(alignOf<char>(),   alignOf<S1>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S2>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S3>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S4>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S5>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S6>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D1>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D2>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D3>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D4>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D5>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D6>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D7>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D8>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D9>());
  EXPECT_LE(alignOf<S1>(),     alignOf<V1>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V2>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V3>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V4>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V5>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V6>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V7>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V8>());
}

TEST(AlignOfTest, BasicAlignedArray) {
  // Note: this code exclusively uses the struct-wrapped arbitrarily aligned
  // types because of the bugs mentioned above where GCC and Clang both
  // disregard the arbitrary alignment specifier until the type is used to
  // declare a member of a struct.
  EXPECT_LE(1u, alignOf<AlignedCharArray<SA1>::union_type>());
  EXPECT_LE(2u, alignOf<AlignedCharArray<SA2>::union_type>());
  EXPECT_LE(4u, alignOf<AlignedCharArray<SA4>::union_type>());
  EXPECT_LE(8u, alignOf<AlignedCharArray<SA8>::union_type>());

  EXPECT_LE(1u, sizeof(AlignedCharArray<SA1>::union_type));
  EXPECT_LE(2u, sizeof(AlignedCharArray<SA2>::union_type));
  EXPECT_LE(4u, sizeof(AlignedCharArray<SA4>::union_type));
  EXPECT_LE(8u, sizeof(AlignedCharArray<SA8>::union_type));

  EXPECT_EQ(1u, (alignOf<AlignedCharArray<SA1>::union_type>()));
  EXPECT_EQ(2u, (alignOf<AlignedCharArray<SA1, SA2>::union_type>()));
  EXPECT_EQ(4u, (alignOf<AlignedCharArray<SA1, SA2, SA4>::union_type>()));
  EXPECT_EQ(8u, (alignOf<AlignedCharArray<SA1, SA2, SA4, SA8>::union_type>()));

  EXPECT_EQ(1u, sizeof(AlignedCharArray<SA1>::union_type));
  EXPECT_EQ(2u, sizeof(AlignedCharArray<SA1, SA2>::union_type));
  EXPECT_EQ(4u, sizeof(AlignedCharArray<SA1, SA2, SA4>::union_type));
  EXPECT_EQ(8u, sizeof(AlignedCharArray<SA1, SA2, SA4, SA8>::union_type));

  EXPECT_EQ(1u, (alignOf<AlignedCharArray<SA1[1]>::union_type>()));
  EXPECT_EQ(2u, (alignOf<AlignedCharArray<SA1[2], SA2[1]>::union_type>()));
  EXPECT_EQ(4u, (alignOf<AlignedCharArray<SA1[42], SA2[55],
                                          SA4[13]>::union_type>()));
  EXPECT_EQ(8u, (alignOf<AlignedCharArray<SA1[2], SA2[1],
                                          SA4, SA8>::union_type>()));

  EXPECT_EQ(1u,  sizeof(AlignedCharArray<SA1[1]>::union_type));
  EXPECT_EQ(2u,  sizeof(AlignedCharArray<SA1[2], SA2[1]>::union_type));
  EXPECT_EQ(4u,  sizeof(AlignedCharArray<SA1[3], SA2[2], SA4>::union_type));
  EXPECT_EQ(16u, sizeof(AlignedCharArray<SA1, SA2[3],
                                         SA4[3], SA8>::union_type));

  // For other tests we simply assert that the alignment of the union mathes
  // that of the fundamental type and hope that we have any weird type
  // productions that would trigger bugs.
  EXPECT_EQ(alignOf<char>(), alignOf<AlignedCharArray<char>::union_type>());
  EXPECT_EQ(alignOf<short>(), alignOf<AlignedCharArray<short>::union_type>());
  EXPECT_EQ(alignOf<int>(), alignOf<AlignedCharArray<int>::union_type>());
  EXPECT_EQ(alignOf<long>(), alignOf<AlignedCharArray<long>::union_type>());
  EXPECT_EQ(alignOf<long long>(),
            alignOf<AlignedCharArray<long long>::union_type>());
  EXPECT_EQ(alignOf<float>(), alignOf<AlignedCharArray<float>::union_type>());
  EXPECT_EQ(alignOf<double>(), alignOf<AlignedCharArray<double>::union_type>());
  EXPECT_EQ(alignOf<long double>(),
            alignOf<AlignedCharArray<long double>::union_type>());
  EXPECT_EQ(alignOf<void *>(), alignOf<AlignedCharArray<void *>::union_type>());
  EXPECT_EQ(alignOf<int *>(), alignOf<AlignedCharArray<int *>::union_type>());
  EXPECT_EQ(alignOf<double (*)(double)>(),
            alignOf<AlignedCharArray<double (*)(double)>::union_type>());
  EXPECT_EQ(alignOf<double (S6::*)()>(),
            alignOf<AlignedCharArray<double (S6::*)()>::union_type>());
  EXPECT_EQ(alignOf<S1>(), alignOf<AlignedCharArray<S1>::union_type>());
  EXPECT_EQ(alignOf<S2>(), alignOf<AlignedCharArray<S2>::union_type>());
  EXPECT_EQ(alignOf<S3>(), alignOf<AlignedCharArray<S3>::union_type>());
  EXPECT_EQ(alignOf<S4>(), alignOf<AlignedCharArray<S4>::union_type>());
  EXPECT_EQ(alignOf<S5>(), alignOf<AlignedCharArray<S5>::union_type>());
  EXPECT_EQ(alignOf<S6>(), alignOf<AlignedCharArray<S6>::union_type>());
  EXPECT_EQ(alignOf<D1>(), alignOf<AlignedCharArray<D1>::union_type>());
  EXPECT_EQ(alignOf<D2>(), alignOf<AlignedCharArray<D2>::union_type>());
  EXPECT_EQ(alignOf<D3>(), alignOf<AlignedCharArray<D3>::union_type>());
  EXPECT_EQ(alignOf<D4>(), alignOf<AlignedCharArray<D4>::union_type>());
  EXPECT_EQ(alignOf<D5>(), alignOf<AlignedCharArray<D5>::union_type>());
  EXPECT_EQ(alignOf<D6>(), alignOf<AlignedCharArray<D6>::union_type>());
  EXPECT_EQ(alignOf<D7>(), alignOf<AlignedCharArray<D7>::union_type>());
  EXPECT_EQ(alignOf<D8>(), alignOf<AlignedCharArray<D8>::union_type>());
  EXPECT_EQ(alignOf<D9>(), alignOf<AlignedCharArray<D9>::union_type>());
  EXPECT_EQ(alignOf<V1>(), alignOf<AlignedCharArray<V1>::union_type>());
  EXPECT_EQ(alignOf<V2>(), alignOf<AlignedCharArray<V2>::union_type>());
  EXPECT_EQ(alignOf<V3>(), alignOf<AlignedCharArray<V3>::union_type>());
  EXPECT_EQ(alignOf<V4>(), alignOf<AlignedCharArray<V4>::union_type>());
  EXPECT_EQ(alignOf<V5>(), alignOf<AlignedCharArray<V5>::union_type>());
  EXPECT_EQ(alignOf<V6>(), alignOf<AlignedCharArray<V6>::union_type>());
  EXPECT_EQ(alignOf<V7>(), alignOf<AlignedCharArray<V7>::union_type>());

  // Some versions of MSVC get this wrong somewhat disturbingly. The failure
  // appears to be benign: alignOf<V8>() produces a preposterous value: 12
#ifndef _MSC_VER
  EXPECT_EQ(alignOf<V8>(), alignOf<AlignedCharArray<V8>::union_type>());
#endif

  EXPECT_EQ(sizeof(char), sizeof(AlignedCharArray<char>::union_type));
  EXPECT_EQ(sizeof(char[1]), sizeof(AlignedCharArray<char[1]>::union_type));
  EXPECT_EQ(sizeof(char[2]), sizeof(AlignedCharArray<char[2]>::union_type));
  EXPECT_EQ(sizeof(char[3]), sizeof(AlignedCharArray<char[3]>::union_type));
  EXPECT_EQ(sizeof(char[4]), sizeof(AlignedCharArray<char[4]>::union_type));
  EXPECT_EQ(sizeof(char[5]), sizeof(AlignedCharArray<char[5]>::union_type));
  EXPECT_EQ(sizeof(char[8]), sizeof(AlignedCharArray<char[8]>::union_type));
  EXPECT_EQ(sizeof(char[13]), sizeof(AlignedCharArray<char[13]>::union_type));
  EXPECT_EQ(sizeof(char[16]), sizeof(AlignedCharArray<char[16]>::union_type));
  EXPECT_EQ(sizeof(char[21]), sizeof(AlignedCharArray<char[21]>::union_type));
  EXPECT_EQ(sizeof(char[32]), sizeof(AlignedCharArray<char[32]>::union_type));
  EXPECT_EQ(sizeof(short), sizeof(AlignedCharArray<short>::union_type));
  EXPECT_EQ(sizeof(int), sizeof(AlignedCharArray<int>::union_type));
  EXPECT_EQ(sizeof(long), sizeof(AlignedCharArray<long>::union_type));
  EXPECT_EQ(sizeof(long long),
            sizeof(AlignedCharArray<long long>::union_type));
  EXPECT_EQ(sizeof(float), sizeof(AlignedCharArray<float>::union_type));
  EXPECT_EQ(sizeof(double), sizeof(AlignedCharArray<double>::union_type));
  EXPECT_EQ(sizeof(long double),
            sizeof(AlignedCharArray<long double>::union_type));
  EXPECT_EQ(sizeof(void *), sizeof(AlignedCharArray<void *>::union_type));
  EXPECT_EQ(sizeof(int *), sizeof(AlignedCharArray<int *>::union_type));
  EXPECT_EQ(sizeof(double (*)(double)),
            sizeof(AlignedCharArray<double (*)(double)>::union_type));
  EXPECT_EQ(sizeof(double (S6::*)()),
            sizeof(AlignedCharArray<double (S6::*)()>::union_type));
  EXPECT_EQ(sizeof(S1), sizeof(AlignedCharArray<S1>::union_type));
  EXPECT_EQ(sizeof(S2), sizeof(AlignedCharArray<S2>::union_type));
  EXPECT_EQ(sizeof(S3), sizeof(AlignedCharArray<S3>::union_type));
  EXPECT_EQ(sizeof(S4), sizeof(AlignedCharArray<S4>::union_type));
  EXPECT_EQ(sizeof(S5), sizeof(AlignedCharArray<S5>::union_type));
  EXPECT_EQ(sizeof(S6), sizeof(AlignedCharArray<S6>::union_type));
  EXPECT_EQ(sizeof(D1), sizeof(AlignedCharArray<D1>::union_type));
  EXPECT_EQ(sizeof(D2), sizeof(AlignedCharArray<D2>::union_type));
  EXPECT_EQ(sizeof(D3), sizeof(AlignedCharArray<D3>::union_type));
  EXPECT_EQ(sizeof(D4), sizeof(AlignedCharArray<D4>::union_type));
  EXPECT_EQ(sizeof(D5), sizeof(AlignedCharArray<D5>::union_type));
  EXPECT_EQ(sizeof(D6), sizeof(AlignedCharArray<D6>::union_type));
  EXPECT_EQ(sizeof(D7), sizeof(AlignedCharArray<D7>::union_type));
  EXPECT_EQ(sizeof(D8), sizeof(AlignedCharArray<D8>::union_type));
  EXPECT_EQ(sizeof(D9), sizeof(AlignedCharArray<D9>::union_type));
  EXPECT_EQ(sizeof(D9[1]), sizeof(AlignedCharArray<D9[1]>::union_type));
  EXPECT_EQ(sizeof(D9[2]), sizeof(AlignedCharArray<D9[2]>::union_type));
  EXPECT_EQ(sizeof(D9[3]), sizeof(AlignedCharArray<D9[3]>::union_type));
  EXPECT_EQ(sizeof(D9[4]), sizeof(AlignedCharArray<D9[4]>::union_type));
  EXPECT_EQ(sizeof(D9[5]), sizeof(AlignedCharArray<D9[5]>::union_type));
  EXPECT_EQ(sizeof(D9[8]), sizeof(AlignedCharArray<D9[8]>::union_type));
  EXPECT_EQ(sizeof(D9[13]), sizeof(AlignedCharArray<D9[13]>::union_type));
  EXPECT_EQ(sizeof(D9[16]), sizeof(AlignedCharArray<D9[16]>::union_type));
  EXPECT_EQ(sizeof(D9[21]), sizeof(AlignedCharArray<D9[21]>::union_type));
  EXPECT_EQ(sizeof(D9[32]), sizeof(AlignedCharArray<D9[32]>::union_type));
  EXPECT_EQ(sizeof(V1), sizeof(AlignedCharArray<V1>::union_type));
  EXPECT_EQ(sizeof(V2), sizeof(AlignedCharArray<V2>::union_type));
  EXPECT_EQ(sizeof(V3), sizeof(AlignedCharArray<V3>::union_type));
  EXPECT_EQ(sizeof(V4), sizeof(AlignedCharArray<V4>::union_type));
  EXPECT_EQ(sizeof(V5), sizeof(AlignedCharArray<V5>::union_type));
  EXPECT_EQ(sizeof(V6), sizeof(AlignedCharArray<V6>::union_type));
  EXPECT_EQ(sizeof(V7), sizeof(AlignedCharArray<V7>::union_type));

  // Some versions of MSVC also get this wrong. The failure again appears to be
  // benign: sizeof(V8) is only 52 bytes, but our array reserves 56.
#ifndef _MSC_VER
  EXPECT_EQ(sizeof(V8), sizeof(AlignedCharArray<V8>::union_type));
#endif
}

}
