//===- llvm/Support/KnownBits.h - Stores known zeros/ones -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a class for representing known zeros and ones used by
// computeKnownBits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_KNOWNBITS_H
#define LLVM_SUPPORT_KNOWNBITS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"

namespace llvm {

// Struct for tracking the known zeros and ones of a value.
struct KnownBits {
  APInt Zero;
  APInt One;

private:
  // Internal constructor for creating a KnownBits from two APInts.
  KnownBits(APInt Zero, APInt One)
      : Zero(std::move(Zero)), One(std::move(One)) {}

public:
  // Default construct Zero and One.
  KnownBits() {}

  /// Create a known bits object of BitWidth bits initialized to unknown.
  KnownBits(unsigned BitWidth) : Zero(BitWidth, 0), One(BitWidth, 0) {}

  /// Get the bit width of this value.
  unsigned getBitWidth() const {
    assert(Zero.getBitWidth() == One.getBitWidth() &&
           "Zero and One should have the same width!");
    return Zero.getBitWidth();
  }

  /// Returns true if there is conflicting information.
  bool hasConflict() const { return Zero.intersects(One); }

  /// Returns true if we know the value of all bits.
  bool isConstant() const {
    assert(!hasConflict() && "KnownBits conflict!");
    return Zero.countPopulation() + One.countPopulation() == getBitWidth();
  }

  /// Returns the value when all bits have a known value. This just returns One
  /// with a protective assertion.
  const APInt &getConstant() const {
    assert(isConstant() && "Can only get value when all bits are known");
    return One;
  }

  /// Returns true if we don't know any bits.
  bool isUnknown() const { return Zero.isNullValue() && One.isNullValue(); }

  /// Resets the known state of all bits.
  void resetAll() {
    Zero.clearAllBits();
    One.clearAllBits();
  }

  /// Returns true if value is all zero.
  bool isZero() const {
    assert(!hasConflict() && "KnownBits conflict!");
    return Zero.isAllOnesValue();
  }

  /// Returns true if value is all one bits.
  bool isAllOnes() const {
    assert(!hasConflict() && "KnownBits conflict!");
    return One.isAllOnesValue();
  }

  /// Make all bits known to be zero and discard any previous information.
  void setAllZero() {
    Zero.setAllBits();
    One.clearAllBits();
  }

  /// Make all bits known to be one and discard any previous information.
  void setAllOnes() {
    Zero.clearAllBits();
    One.setAllBits();
  }

  /// Returns true if this value is known to be negative.
  bool isNegative() const { return One.isSignBitSet(); }

  /// Returns true if this value is known to be non-negative.
  bool isNonNegative() const { return Zero.isSignBitSet(); }

  /// Returns true if this value is known to be non-zero.
  bool isNonZero() const { return !One.isNullValue(); }

  /// Returns true if this value is known to be positive.
  bool isStrictlyPositive() const { return Zero.isSignBitSet() && !One.isNullValue(); }

  /// Make this value negative.
  void makeNegative() {
    One.setSignBit();
  }

  /// Make this value non-negative.
  void makeNonNegative() {
    Zero.setSignBit();
  }

  /// Return the minimal unsigned value possible given these KnownBits.
  APInt getMinValue() const {
    // Assume that all bits that aren't known-ones are zeros.
    return One;
  }

  /// Return the minimal signed value possible given these KnownBits.
  APInt getSignedMinValue() const {
    // Assume that all bits that aren't known-ones are zeros.
    APInt Min = One;
    // Sign bit is unknown.
    if (Zero.isSignBitClear())
      Min.setSignBit();
    return Min;
  }

  /// Return the maximal unsigned value possible given these KnownBits.
  APInt getMaxValue() const {
    // Assume that all bits that aren't known-zeros are ones.
    return ~Zero;
  }

  /// Return the maximal signed value possible given these KnownBits.
  APInt getSignedMaxValue() const {
    // Assume that all bits that aren't known-zeros are ones.
    APInt Max = ~Zero;
    // Sign bit is unknown.
    if (One.isSignBitClear())
      Max.clearSignBit();
    return Max;
  }

  /// Return known bits for a truncation of the value we're tracking.
  KnownBits trunc(unsigned BitWidth) const {
    return KnownBits(Zero.trunc(BitWidth), One.trunc(BitWidth));
  }

  /// Return known bits for an "any" extension of the value we're tracking,
  /// where we don't know anything about the extended bits.
  KnownBits anyext(unsigned BitWidth) const {
    return KnownBits(Zero.zext(BitWidth), One.zext(BitWidth));
  }

  /// Return known bits for a zero extension of the value we're tracking.
  KnownBits zext(unsigned BitWidth) const {
    unsigned OldBitWidth = getBitWidth();
    APInt NewZero = Zero.zext(BitWidth);
    NewZero.setBitsFrom(OldBitWidth);
    return KnownBits(NewZero, One.zext(BitWidth));
  }

  /// Return known bits for a sign extension of the value we're tracking.
  KnownBits sext(unsigned BitWidth) const {
    return KnownBits(Zero.sext(BitWidth), One.sext(BitWidth));
  }

  /// Return known bits for an "any" extension or truncation of the value we're
  /// tracking.
  KnownBits anyextOrTrunc(unsigned BitWidth) const {
    if (BitWidth > getBitWidth())
      return anyext(BitWidth);
    if (BitWidth < getBitWidth())
      return trunc(BitWidth);
    return *this;
  }

  /// Return known bits for a zero extension or truncation of the value we're
  /// tracking.
  KnownBits zextOrTrunc(unsigned BitWidth) const {
    if (BitWidth > getBitWidth())
      return zext(BitWidth);
    if (BitWidth < getBitWidth())
      return trunc(BitWidth);
    return *this;
  }

  /// Return known bits for a sign extension or truncation of the value we're
  /// tracking.
  KnownBits sextOrTrunc(unsigned BitWidth) const {
    if (BitWidth > getBitWidth())
      return sext(BitWidth);
    if (BitWidth < getBitWidth())
      return trunc(BitWidth);
    return *this;
  }

  /// Return known bits for a in-register sign extension of the value we're
  /// tracking.
  KnownBits sextInReg(unsigned SrcBitWidth) const;

  /// Insert the bits from a smaller known bits starting at bitPosition.
  void insertBits(const KnownBits &SubBits, unsigned BitPosition) {
    Zero.insertBits(SubBits.Zero, BitPosition);
    One.insertBits(SubBits.One, BitPosition);
  }

  /// Return a subset of the known bits from [bitPosition,bitPosition+numBits).
  KnownBits extractBits(unsigned NumBits, unsigned BitPosition) const {
    return KnownBits(Zero.extractBits(NumBits, BitPosition),
                     One.extractBits(NumBits, BitPosition));
  }

  /// Return KnownBits based on this, but updated given that the underlying
  /// value is known to be greater than or equal to Val.
  KnownBits makeGE(const APInt &Val) const;

  /// Returns the minimum number of trailing zero bits.
  unsigned countMinTrailingZeros() const {
    return Zero.countTrailingOnes();
  }

  /// Returns the minimum number of trailing one bits.
  unsigned countMinTrailingOnes() const {
    return One.countTrailingOnes();
  }

  /// Returns the minimum number of leading zero bits.
  unsigned countMinLeadingZeros() const {
    return Zero.countLeadingOnes();
  }

  /// Returns the minimum number of leading one bits.
  unsigned countMinLeadingOnes() const {
    return One.countLeadingOnes();
  }

  /// Returns the number of times the sign bit is replicated into the other
  /// bits.
  unsigned countMinSignBits() const {
    if (isNonNegative())
      return countMinLeadingZeros();
    if (isNegative())
      return countMinLeadingOnes();
    return 0;
  }

  /// Returns the maximum number of trailing zero bits possible.
  unsigned countMaxTrailingZeros() const {
    return One.countTrailingZeros();
  }

  /// Returns the maximum number of trailing one bits possible.
  unsigned countMaxTrailingOnes() const {
    return Zero.countTrailingZeros();
  }

  /// Returns the maximum number of leading zero bits possible.
  unsigned countMaxLeadingZeros() const {
    return One.countLeadingZeros();
  }

  /// Returns the maximum number of leading one bits possible.
  unsigned countMaxLeadingOnes() const {
    return Zero.countLeadingZeros();
  }

  /// Returns the number of bits known to be one.
  unsigned countMinPopulation() const {
    return One.countPopulation();
  }

  /// Returns the maximum number of bits that could be one.
  unsigned countMaxPopulation() const {
    return getBitWidth() - Zero.countPopulation();
  }

  /// Create known bits from a known constant.
  static KnownBits makeConstant(const APInt &C) {
    return KnownBits(~C, C);
  }

  /// Compute known bits common to LHS and RHS.
  static KnownBits commonBits(const KnownBits &LHS, const KnownBits &RHS) {
    return KnownBits(LHS.Zero & RHS.Zero, LHS.One & RHS.One);
  }

  /// Return true if LHS and RHS have no common bits set.
  static bool haveNoCommonBitsSet(const KnownBits &LHS, const KnownBits &RHS) {
    return (LHS.Zero | RHS.Zero).isAllOnesValue();
  }

  /// Compute known bits resulting from adding LHS, RHS and a 1-bit Carry.
  static KnownBits computeForAddCarry(
      const KnownBits &LHS, const KnownBits &RHS, const KnownBits &Carry);

  /// Compute known bits resulting from adding LHS and RHS.
  static KnownBits computeForAddSub(bool Add, bool NSW, const KnownBits &LHS,
                                    KnownBits RHS);

  /// Compute known bits resulting from multiplying LHS and RHS.
  static KnownBits mul(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits from sign-extended multiply-hi.
  static KnownBits mulhs(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits from zero-extended multiply-hi.
  static KnownBits mulhu(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for udiv(LHS, RHS).
  static KnownBits udiv(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for urem(LHS, RHS).
  static KnownBits urem(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for srem(LHS, RHS).
  static KnownBits srem(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for umax(LHS, RHS).
  static KnownBits umax(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for umin(LHS, RHS).
  static KnownBits umin(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for smax(LHS, RHS).
  static KnownBits smax(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for smin(LHS, RHS).
  static KnownBits smin(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for shl(LHS, RHS).
  /// NOTE: RHS (shift amount) bitwidth doesn't need to be the same as LHS.
  static KnownBits shl(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for lshr(LHS, RHS).
  /// NOTE: RHS (shift amount) bitwidth doesn't need to be the same as LHS.
  static KnownBits lshr(const KnownBits &LHS, const KnownBits &RHS);

  /// Compute known bits for ashr(LHS, RHS).
  /// NOTE: RHS (shift amount) bitwidth doesn't need to be the same as LHS.
  static KnownBits ashr(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_EQ result.
  static Optional<bool> eq(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_NE result.
  static Optional<bool> ne(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_UGT result.
  static Optional<bool> ugt(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_UGE result.
  static Optional<bool> uge(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_ULT result.
  static Optional<bool> ult(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_ULE result.
  static Optional<bool> ule(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_SGT result.
  static Optional<bool> sgt(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_SGE result.
  static Optional<bool> sge(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_SLT result.
  static Optional<bool> slt(const KnownBits &LHS, const KnownBits &RHS);

  /// Determine if these known bits always give the same ICMP_SLE result.
  static Optional<bool> sle(const KnownBits &LHS, const KnownBits &RHS);

  /// Update known bits based on ANDing with RHS.
  KnownBits &operator&=(const KnownBits &RHS);

  /// Update known bits based on ORing with RHS.
  KnownBits &operator|=(const KnownBits &RHS);

  /// Update known bits based on XORing with RHS.
  KnownBits &operator^=(const KnownBits &RHS);

  /// Compute known bits for the absolute value.
  KnownBits abs(bool IntMinIsPoison = false) const;

  KnownBits byteSwap() {
    return KnownBits(Zero.byteSwap(), One.byteSwap());
  }

  KnownBits reverseBits() {
    return KnownBits(Zero.reverseBits(), One.reverseBits());
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

inline KnownBits operator&(KnownBits LHS, const KnownBits &RHS) {
  LHS &= RHS;
  return LHS;
}

inline KnownBits operator&(const KnownBits &LHS, KnownBits &&RHS) {
  RHS &= LHS;
  return std::move(RHS);
}

inline KnownBits operator|(KnownBits LHS, const KnownBits &RHS) {
  LHS |= RHS;
  return LHS;
}

inline KnownBits operator|(const KnownBits &LHS, KnownBits &&RHS) {
  RHS |= LHS;
  return std::move(RHS);
}

inline KnownBits operator^(KnownBits LHS, const KnownBits &RHS) {
  LHS ^= RHS;
  return LHS;
}

inline KnownBits operator^(const KnownBits &LHS, KnownBits &&RHS) {
  RHS ^= LHS;
  return std::move(RHS);
}

} // end namespace llvm

#endif
