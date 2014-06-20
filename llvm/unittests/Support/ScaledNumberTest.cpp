//===- llvm/unittest/Support/ScaledNumberTest.cpp - ScaledPair tests -----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ScaledNumber.h"

#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::ScaledNumbers;

namespace {

template <class UIntT> struct ScaledPair {
  UIntT D;
  int S;
  ScaledPair(const std::pair<UIntT, int16_t> &F) : D(F.first), S(F.second) {}
  ScaledPair(UIntT D, int S) : D(D), S(S) {}

  bool operator==(const ScaledPair<UIntT> &X) const {
    return D == X.D && S == X.S;
  }
};
template <class UIntT>
bool operator==(const std::pair<UIntT, int16_t> &L,
                const ScaledPair<UIntT> &R) {
  return ScaledPair<UIntT>(L) == R;
}
template <class UIntT>
void PrintTo(const ScaledPair<UIntT> &F, ::std::ostream *os) {
  *os << F.D << "*2^" << F.S;
}

typedef ScaledPair<uint32_t> SP32;
typedef ScaledPair<uint64_t> SP64;

TEST(ScaledNumberHelpersTest, getRounded) {
  EXPECT_EQ(getRounded<uint32_t>(0, 0, false), SP32(0, 0));
  EXPECT_EQ(getRounded<uint32_t>(0, 0, true), SP32(1, 0));
  EXPECT_EQ(getRounded<uint32_t>(20, 21, true), SP32(21, 21));
  EXPECT_EQ(getRounded<uint32_t>(UINT32_MAX, 0, false), SP32(UINT32_MAX, 0));
  EXPECT_EQ(getRounded<uint32_t>(UINT32_MAX, 0, true), SP32(1 << 31, 1));

  EXPECT_EQ(getRounded<uint64_t>(0, 0, false), SP64(0, 0));
  EXPECT_EQ(getRounded<uint64_t>(0, 0, true), SP64(1, 0));
  EXPECT_EQ(getRounded<uint64_t>(20, 21, true), SP64(21, 21));
  EXPECT_EQ(getRounded<uint64_t>(UINT32_MAX, 0, false), SP64(UINT32_MAX, 0));
  EXPECT_EQ(getRounded<uint64_t>(UINT32_MAX, 0, true),
            SP64(UINT64_C(1) << 32, 0));
  EXPECT_EQ(getRounded<uint64_t>(UINT64_MAX, 0, false), SP64(UINT64_MAX, 0));
  EXPECT_EQ(getRounded<uint64_t>(UINT64_MAX, 0, true),
            SP64(UINT64_C(1) << 63, 1));
}
}
