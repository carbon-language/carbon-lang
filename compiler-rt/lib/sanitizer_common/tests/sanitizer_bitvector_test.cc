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
  BV bv, bv1;
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
        EXPECT_EQ(bv.setBit(bit), s.insert(bit).second);
        break;
      case 1:
        size_t old_size = s.size();
        s.erase(bit);
        EXPECT_EQ(bv.clearBit(bit), old_size > s.size());
        break;
    }
    EXPECT_EQ(bv.getBit(bit), s.count(bit) == 1);
  }

  vector<uptr>bits(bv.size());
  // Test setUnion, intersectsWith, and getAndClearFirstOne.
  for (uptr it = 0; it < 30; it++) {
    // iota
    for (size_t j = 0; j < bits.size(); j++) bits[j] = j;
    random_shuffle(bits.begin(), bits.end());
    set<uptr> s, s1;
    bv.clear();
    bv1.clear();
    uptr n_bits = ((uptr)my_rand() % bv.size()) + 1;
    uptr n_bits1 = (uptr)my_rand() % (bv.size() / 2);
    EXPECT_TRUE(n_bits > 0 && n_bits <= bv.size());
    EXPECT_TRUE(n_bits1 < bv.size() / 2);
    for (uptr i = 0; i < n_bits; i++) {
      bv.setBit(bits[i]);
      s.insert(bits[i]);
    }
    for (uptr i = 0; i < n_bits1; i++) {
      bv1.setBit(bits[bv.size() / 2 + i]);
      s1.insert(bits[bv.size() / 2 + i]);
    }

    vector<uptr> vec;
    set_intersection(s.begin(), s.end(), s1.begin(), s1.end(),
                     back_insert_iterator<vector<uptr> >(vec));
    EXPECT_EQ(bv.intersectsWith(bv1), !vec.empty());

    size_t old_size = s.size();
    s.insert(s1.begin(), s1.end());
    EXPECT_EQ(bv.setUnion(bv1), old_size != s.size());
    if (0)
      printf("union %zd %zd: %zd => %zd;  added %zd; intersection: %zd\n",
             n_bits, n_bits1, old_size, s.size(), s.size() - old_size,
             vec.size());
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
  TestBitVector<TwoLevelBitVector<3, BasicBitVector<u16> > >
      (16 * 16 * 3);
}
