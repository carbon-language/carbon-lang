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


// Check the 'bv' == 's' and that the indexes go in increasing order.
template <class BV>
static void CheckBV(const BV &bv, const set<uptr> &s) {
  BV t;
  t.copyFrom(bv);
  set<uptr> t_s(s);
  uptr last_idx = bv.size();
  while (!t.empty()) {
    uptr idx = t.getAndClearFirstOne();
    if (last_idx != bv.size())
      EXPECT_LT(last_idx, idx);
    last_idx = idx;
    EXPECT_TRUE(t_s.erase(idx));
  }
  EXPECT_TRUE(t_s.empty());
}

template <class BV>
void Print(const BV &bv) {
  BV t;
  t.copyFrom(bv);
  while (!t.empty()) {
    uptr idx = t.getAndClearFirstOne();
    fprintf(stderr, "%zd ", idx);
  }
  fprintf(stderr, "\n");
}

void Print(const set<uptr> &s) {
  for (set<uptr>::reverse_iterator it = s.rbegin(); it != s.rend(); ++it) {
    fprintf(stderr, "%zd ", *it);
  }
  fprintf(stderr, "\n");
}

template <class BV>
void TestBitVector(uptr expected_size) {
  BV bv, bv1, t_bv;
  EXPECT_EQ(expected_size, BV::kSize);
  bv.clear();
  EXPECT_TRUE(bv.empty());
  bv.setBit(5);
  EXPECT_FALSE(bv.empty());
  EXPECT_FALSE(bv.getBit(4));
  EXPECT_FALSE(bv.getBit(6));
  EXPECT_TRUE(bv.getBit(5));
  bv.clearBit(5);
  EXPECT_FALSE(bv.getBit(5));

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
  // Test setUnion, setIntersection, intersectsWith, and getAndClearFirstOne.
  for (uptr it = 0; it < 30; it++) {
    // iota
    for (size_t j = 0; j < bits.size(); j++) bits[j] = j;
    random_shuffle(bits.begin(), bits.end());
    set<uptr> s, s1, t_s;
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
    CheckBV(bv, s);
    for (uptr i = 0; i < n_bits1; i++) {
      bv1.setBit(bits[bv.size() / 2 + i]);
      s1.insert(bits[bv.size() / 2 + i]);
    }
    CheckBV(bv1, s1);

    vector<uptr> vec;
    set_intersection(s.begin(), s.end(), s1.begin(), s1.end(),
                     back_insert_iterator<vector<uptr> >(vec));
    EXPECT_EQ(bv.intersectsWith(bv1), !vec.empty());

    // setUnion
    t_s = s;
    t_bv.copyFrom(bv);
    t_s.insert(s1.begin(), s1.end());
    EXPECT_EQ(t_bv.setUnion(bv1), s.size() != t_s.size());
    CheckBV(t_bv, t_s);

    // setIntersection
    t_s = set<uptr>(vec.begin(), vec.end());
    t_bv.copyFrom(bv);
    EXPECT_EQ(t_bv.setIntersection(bv1), s.size() != t_s.size());
    CheckBV(t_bv, t_s);
  }
}

TEST(SanitizerCommon, BasicBitVector) {
  TestBitVector<BasicBitVector<u8> >(8);
  TestBitVector<BasicBitVector<u16> >(16);
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
