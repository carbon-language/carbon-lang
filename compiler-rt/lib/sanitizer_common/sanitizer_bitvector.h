//===-- sanitizer_bitvector.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Specializer BitVector implementation.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_BITVECTOR_H
#define SANITIZER_BITVECTOR_H

#include "sanitizer_common.h"

namespace __sanitizer {

// Fixed size bit vector based on a single basic integer.
template <class basic_int_t = uptr>
class BasicBitVector {
 public:
  enum SizeEnum { kSize = sizeof(basic_int_t) * 8 };
  uptr size() const { return kSize; }
  // No CTOR.
  void clear() { bits_ = 0; }
  bool empty() const { return bits_ == 0; }
  void setBit(uptr idx) { bits_ |= mask(idx); }
  void clearBit(uptr idx) { bits_ &= ~mask(idx); }
  bool getBit(uptr idx) const { return bits_ & mask(idx); }
  uptr getAndClearFirstOne() {
    CHECK(!empty());
    // FIXME: change to LeastSignificantSetBitIndex?
    uptr idx = MostSignificantSetBitIndex(bits_);
    clearBit(idx);
    return idx;
  }

 private:
  basic_int_t mask(uptr idx) const {
    CHECK_LE(idx, size());
    return (basic_int_t)1UL << idx;
  }
  basic_int_t bits_;
};

// Fixed size bit vector of (kLevel1Size*BV::kSize**2) bits.
// The implementation is optimized for a sparse bit vector, i.e. the one
// that has few set bits.
template <uptr kLevel1Size = 1, class BV = BasicBitVector<> >
class TwoLevelBitVector {
  // This is essentially a 2-level bit vector.
  // Set bit in the first level BV indicates that there are set bits
  // in the corresponding BV of the second level.
  // This structure allows O(kLevel1Size) time for clear() and empty(),
  // as well fast handling of sparse BVs.
 public:
  enum SizeEnum { kSize = BV::kSize * BV::kSize * kLevel1Size };
  // No CTOR.
  uptr size() const { return kSize; }
  void clear() {
    for (uptr i = 0; i < kLevel1Size; i++)
      l1_[i].clear();
  }
  bool empty() {
    for (uptr i = 0; i < kLevel1Size; i++)
      if (!l1_[i].empty())
        return false;
    return true;
  }
  void setBit(uptr idx) {
    check(idx);
    uptr i0 = idx0(idx);
    uptr i1 = idx1(idx);
    uptr i2 = idx2(idx);
    if (!l1_[i0].getBit(i1)) {
      l1_[i0].setBit(i1);
      l2_[i0][i1].clear();
    }
    l2_[i0][i1].setBit(i2);
    // Printf("%s: %zd => %zd %zd %zd\n", __FUNCTION__, idx, i0, i1, i2);
  }
  void clearBit(uptr idx) {
    check(idx);
    uptr i0 = idx0(idx);
    uptr i1 = idx1(idx);
    uptr i2 = idx2(idx);
    if (l1_[i0].getBit(i1)) {
      l2_[i0][i1].clearBit(i2);
      if (l2_[i0][i1].empty())
        l1_[i0].clearBit(i1);
    }
  }
  bool getBit(uptr idx) {
    check(idx);
    uptr i0 = idx0(idx);
    uptr i1 = idx1(idx);
    uptr i2 = idx2(idx);
    // Printf("%s: %zd => %zd %zd %zd\n", __FUNCTION__, idx, i0, i1, i2);
    return l1_[i0].getBit(i1) && l2_[i0][i1].getBit(i2);
  }
  uptr getAndClearFirstOne() {
    for (uptr i0 = 0; i0 < kLevel1Size; i0++) {
      if (l1_[i0].empty()) continue;
      uptr i1 = l1_[i0].getAndClearFirstOne();
      uptr i2 = l2_[i0][i1].getAndClearFirstOne();
      if (!l2_[i0][i1].empty())
        l1_[i0].setBit(i1);
      uptr res = i0 * BV::kSize * BV::kSize + i1 * BV::kSize + i2;
      // Printf("getAndClearFirstOne: %zd %zd %zd => %zd\n", i0, i1, i2, res);
      return res;
    }
    CHECK(0);
    return 0;
  }

 private:
  void check(uptr idx) { CHECK_LE(idx, size()); }
  uptr idx0(uptr idx) {
    uptr res = idx / (BV::kSize * BV::kSize);
    CHECK_LE(res, kLevel1Size);
    return res;
  }
  uptr idx1(uptr idx) {
    uptr res = (idx / BV::kSize) % BV::kSize;
    CHECK_LE(res, BV::kSize);
    return res;
  }
  uptr idx2(uptr idx) {
    uptr res = idx % BV::kSize;
    CHECK_LE(res, BV::kSize);
    return res;
  }

  BV l1_[kLevel1Size];
  BV l2_[kLevel1Size][BV::kSize];
};

} // namespace __sanitizer

#endif // SANITIZER_BITVECTOR_H
