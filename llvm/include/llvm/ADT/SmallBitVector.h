//===- llvm/ADT/SmallBitVector.h - 'Normally small' bit vectors -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallBitVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLBITVECTOR_H
#define LLVM_ADT_SMALLBITVECTOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {

/// SmallBitVector - This is a 'bitvector' (really, a variable-sized bit array),
/// optimized for the case when the array is small.  It contains one
/// pointer-sized field, which is directly used as a plain collection of bits
/// when possible, or as a pointer to a larger heap-allocated array when
/// necessary.  This allows normal "small" cases to be fast without losing
/// generality for large inputs.
///
class SmallBitVector {
  // TODO: In "large" mode, a pointer to a BitVector is used, leading to an
  // unnecessary level of indirection. It would be more efficient to use a
  // pointer to memory containing size, allocation size, and the array of bits.
  PointerIntPair<BitVector *, 1, uintptr_t> X;

  // The number of bits in this class.
  static const size_t NumBaseBits = sizeof(uintptr_t) * CHAR_BIT;

  // One bit is used to discriminate between small and large mode. The
  // remaining bits are used for the small-mode representation.
  static const size_t SmallNumRawBits = NumBaseBits - 1;

  // A few more bits are used to store the size of the bit set in small mode.
  // Theoretically this is a ceil-log2. These bits are encoded in the most
  // significant bits of the raw bits.
  static const size_t SmallNumSizeBits = (NumBaseBits == 32 ? 5 :
                                          NumBaseBits == 64 ? 6 :
                                          SmallNumRawBits);

  // The remaining bits are used to store the actual set in small mode.
  static const size_t SmallNumDataBits = SmallNumRawBits - SmallNumSizeBits;

  bool isSmall() const {
    return X.getInt();
  }

  void switchToSmall(uintptr_t NewSmallBits, size_t NewSize) {
    X.setInt(true);
    setSmallSize(NewSize);
    setSmallBits(NewSmallBits);
  }

  void switchToLarge(BitVector *BV) {
    X.setInt(false);
    X.setPointer(BV);
  }

  // Return all the bits used for the "small" representation; this includes
  // bits for the size as well as the element bits.
  uintptr_t getSmallRawBits() const {
    return reinterpret_cast<uintptr_t>(X.getPointer()) >> 1;
  }

  void setSmallRawBits(uintptr_t NewRawBits) {
    return X.setPointer(reinterpret_cast<BitVector *>(NewRawBits << 1));
  }

  // Return the size.
  size_t getSmallSize() const {
    return getSmallRawBits() >> SmallNumDataBits;
  }

  void setSmallSize(size_t Size) {
    setSmallRawBits(getSmallBits() | (Size << SmallNumDataBits));
  }

  // Return the element bits.
  uintptr_t getSmallBits() const {
    return getSmallRawBits() & ~(~uintptr_t(0) << SmallNumDataBits);
  }

  void setSmallBits(uintptr_t NewBits) {
    setSmallRawBits((getSmallRawBits() & (~uintptr_t(0) << SmallNumDataBits)) |
                    (NewBits & ~(~uintptr_t(0) << getSmallSize())));
  }

public:
  /// SmallBitVector default ctor - Creates an empty bitvector.
  SmallBitVector() : X(0, 1) {}

  /// SmallBitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to the specified value.
  explicit SmallBitVector(unsigned s, bool t = false) : X(0, 1) {
    if (s <= SmallNumRawBits)
      switchToSmall(t ? ~uintptr_t(0) : 0, s);
    else
      switchToLarge(new BitVector(s, t));
  }

  /// SmallBitVector copy ctor.
  SmallBitVector(const SmallBitVector &RHS) {
    if (RHS.isSmall())
      X = RHS.X;
    else
      switchToLarge(new BitVector(*RHS.X.getPointer()));
  }

  ~SmallBitVector() {
    if (!isSmall())
      delete X.getPointer();
  }

  /// empty - Tests whether there are no bits in this bitvector.
  bool empty() const {
    return isSmall() ? getSmallSize() == 0 : X.getPointer()->empty();
  }

  /// size - Returns the number of bits in this bitvector.
  size_t size() const {
    return isSmall() ? getSmallSize() : X.getPointer()->size();
  }

  /// count - Returns the number of bits which are set.
  unsigned count() const {
    if (isSmall()) {
      uintptr_t Bits = getSmallBits();
      if (sizeof(uintptr_t) * CHAR_BIT == 32)
        return CountPopulation_32(Bits);
      if (sizeof(uintptr_t) * CHAR_BIT == 64)
        return CountPopulation_64(Bits);
      assert(0 && "Unsupported!");
    }
    return X.getPointer()->count();
  }

  /// any - Returns true if any bit is set.
  bool any() const {
    if (isSmall())
      return getSmallBits() != 0;
    return X.getPointer()->any();
  }

  /// none - Returns true if none of the bits are set.
  bool none() const {
    if (isSmall())
      return getSmallBits() == 0;
    return X.getPointer()->none();
  }

  /// find_first - Returns the index of the first set bit, -1 if none
  /// of the bits are set.
  int find_first() const {
    if (isSmall()) {
      uintptr_t Bits = getSmallBits();
      if (sizeof(uintptr_t) * CHAR_BIT == 32)
        return CountTrailingZeros_32(Bits);
      if (sizeof(uintptr_t) * CHAR_BIT == 64)
        return CountTrailingZeros_64(Bits);
      assert(0 && "Unsupported!");
    }
    return X.getPointer()->find_first();
  }

  /// find_next - Returns the index of the next set bit following the
  /// "Prev" bit. Returns -1 if the next set bit is not found.
  int find_next(unsigned Prev) const {
    if (isSmall()) {
      uintptr_t Bits = getSmallBits();
      // Mask off previous bits.
      Bits &= ~uintptr_t(0) << Prev;
      if (sizeof(uintptr_t) * CHAR_BIT == 32)
        return CountTrailingZeros_32(Bits);
      if (sizeof(uintptr_t) * CHAR_BIT == 64)
        return CountTrailingZeros_64(Bits);
      assert(0 && "Unsupported!");
    }
    return X.getPointer()->find_next(Prev);
  }

  /// clear - Clear all bits.
  void clear() {
    if (!isSmall())
      delete X.getPointer();
    switchToSmall(0, 0);
  }

  /// resize - Grow or shrink the bitvector.
  void resize(unsigned N, bool t = false) {
    if (!isSmall()) {
      X.getPointer()->resize(N, t);
    } else if (getSmallSize() >= N) {
      setSmallSize(N);
      setSmallBits(getSmallBits());
    } else {
      BitVector *BV = new BitVector(N, t);
      uintptr_t OldBits = getSmallBits();
      for (size_t i = 0, e = getSmallSize(); i != e; ++i)
        (*BV)[i] = (OldBits >> i) & 1;
      switchToLarge(BV);
    }
  }

  void reserve(unsigned N) {
    if (isSmall()) {
      if (N > SmallNumDataBits) {
        uintptr_t OldBits = getSmallRawBits();
        size_t SmallSize = getSmallSize();
        BitVector *BV = new BitVector(SmallSize);
        for (size_t i = 0; i < SmallSize; ++i)
          if ((OldBits >> i) & 1)
            BV->set(i);
        BV->reserve(N);
        switchToLarge(BV);
      }
    } else {
      X.getPointer()->reserve(N);
    }
  }

  // Set, reset, flip
  SmallBitVector &set() {
    if (isSmall())
      setSmallBits(~uintptr_t(0));
    else
      X.getPointer()->set();
    return *this;
  }

  SmallBitVector &set(unsigned Idx) {
    if (isSmall())
      setSmallBits(getSmallBits() | (uintptr_t(1) << Idx));
    else
      X.getPointer()->set(Idx);
    return *this;
  }

  SmallBitVector &reset() {
    if (isSmall())
      setSmallBits(0);
    else
      X.getPointer()->reset();
    return *this;
  }

  SmallBitVector &reset(unsigned Idx) {
    if (isSmall())
      setSmallBits(getSmallBits() & ~(uintptr_t(1) << Idx));
    else
      X.getPointer()->reset(Idx);
    return *this;
  }

  SmallBitVector &flip() {
    if (isSmall())
      setSmallBits(~getSmallBits());
    else
      X.getPointer()->flip();
    return *this;
  }

  SmallBitVector &flip(unsigned Idx) {
    if (isSmall())
      setSmallBits(getSmallBits() ^ (uintptr_t(1) << Idx));
    else
      X.getPointer()->flip(Idx);
    return *this;
  }

  // No argument flip.
  SmallBitVector operator~() const {
    return SmallBitVector(*this).flip();
  }

  // Indexing.
  // TODO: Add an index operator which returns a "reference" (proxy class).
  bool operator[](unsigned Idx) const {
    assert(Idx < size() && "Out-of-bounds Bit access.");
    if (isSmall())
      return ((getSmallBits() >> Idx) & 1) != 0;
    return X.getPointer()->operator[](Idx);
  }

  bool test(unsigned Idx) const {
    return (*this)[Idx];
  }

  // Comparison operators.
  bool operator==(const SmallBitVector &RHS) const {
    if (size() != RHS.size())
      return false;
    if (isSmall())
      return getSmallBits() == RHS.getSmallBits();
    else
      return *X.getPointer() == *RHS.X.getPointer();
  }

  bool operator!=(const SmallBitVector &RHS) const {
    return !(*this == RHS);
  }

  // Intersection, union, disjoint union.
  BitVector &operator&=(const SmallBitVector &RHS); // TODO: implement

  BitVector &operator|=(const SmallBitVector &RHS); // TODO: implement

  BitVector &operator^=(const SmallBitVector &RHS); // TODO: implement

  // Assignment operator.
  const SmallBitVector &operator=(const SmallBitVector &RHS) {
    if (isSmall()) {
      if (RHS.isSmall())
        X = RHS.X;
      else
        switchToLarge(new BitVector(*RHS.X.getPointer()));
    } else {
      if (!RHS.isSmall())
        *X.getPointer() = *RHS.X.getPointer();
      else {
        delete X.getPointer();
        X = RHS.X;
      }
    }
    return *this;
  }

  void swap(SmallBitVector &RHS) {
    std::swap(X, RHS.X);
  }
};

inline SmallBitVector
operator&(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  SmallBitVector Result(LHS);
  Result &= RHS;
  return Result;
}

inline SmallBitVector
operator|(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  SmallBitVector Result(LHS);
  Result |= RHS;
  return Result;
}

inline SmallBitVector
operator^(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  SmallBitVector Result(LHS);
  Result ^= RHS;
  return Result;
}

} // End llvm namespace

namespace std {
  /// Implement std::swap in terms of BitVector swap.
  inline void
  swap(llvm::SmallBitVector &LHS, llvm::SmallBitVector &RHS) {
    LHS.swap(RHS);
  }
}

#endif
