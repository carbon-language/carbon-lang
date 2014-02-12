//===-- sanitizer_bitvector_test.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// Tests for sanitizer_bitvector.h.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_bitvector.h"

#include "sanitizer_test_utils.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <vector>
#include <set>

using namespace __sanitizer;
using namespace std;

template <class BV>
void TestBitVector(uptr expected_size) {
  BV bv;
  EXPECT_EQ(expected_size, BV::kSize);
  bv.clear();
  EXPECT_TRUE(bv.empty());
  bv.setBit(13);
  EXPECT_FALSE(bv.empty());
  EXPECT_FALSE(bv.getBit(12));
  EXPECT_FALSE(bv.getBit(14));
  EXPECT_TRUE(bv.getBit(13));
  bv.clearBit(13);
  EXPECT_FALSE(bv.getBit(13));

  // test random bits
  bv.clear();
  set<uptr> s;
  for (uptr it = 0; it < 1000; it++) {
    uptr bit = ((uptr)my_rand() % bv.size());
    EXPECT_EQ(bv.getBit(bit), s.count(bit) == 1);
    switch (my_rand() % 2) {
      case 0:
        bv.setBit(bit);
        s.insert(bit);
        break;
      case 1:
        bv.clearBit(bit);
        s.erase(bit);
        break;
    }
    EXPECT_EQ(bv.getBit(bit), s.count(bit) == 1);
  }

  // test getAndClearFirstOne.
  vector<uptr>bits(bv.size());
  for (uptr it = 0; it < 30; it++) {
    // iota
    for (size_t j = 0; j < bits.size(); j++) bits[j] = j;
    random_shuffle(bits.begin(), bits.end());
    uptr n_bits = ((uptr)my_rand() % bv.size()) + 1;
    EXPECT_TRUE(n_bits > 0 && n_bits <= bv.size());
    bv.clear();
    set<uptr> s(bits.begin(), bits.begin() + n_bits);
    for (uptr i = 0; i < n_bits; i++) {
      bv.setBit(bits[i]);
      s.insert(bits[i]);
    }
    while (!bv.empty()) {
      uptr idx = bv.getAndClearFirstOne();
      EXPECT_TRUE(s.erase(idx));
    }
    EXPECT_TRUE(s.empty());
  }
}

TEST(SanitizerCommon, BasicBitVector) {
  TestBitVector<BasicBitVector<> >(SANITIZER_WORDSIZE);
}

TEST(SanitizerCommon, TwoLevelBitVector) {
  uptr ws = SANITIZER_WORDSIZE;
  TestBitVector<TwoLevelBitVector<> >(ws * ws);
  TestBitVector<TwoLevelBitVector<2> >(ws * ws * 2);
  TestBitVector<TwoLevelBitVector<3> >(ws * ws * 3);
}
