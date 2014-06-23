//===- llvm/Support/ScaledNumber.h - Support for scaled numbers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions (and a class) useful for working with scaled
// numbers -- in particular, pairs of integers where one represents digits and
// another represents a scale.  The functions are helpers and live in the
// namespace ScaledNumbers.  The class ScaledNumber is useful for modelling
// certain cost metrics that need simple, integer-like semantics that are easy
// to reason about.
//
// These might remind you of soft-floats.  If you want one of those, you're in
// the wrong place.  Look at include/llvm/ADT/APFloat.h instead.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SCALEDNUMBER_H
#define LLVM_SUPPORT_SCALEDNUMBER_H

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace llvm {
namespace ScaledNumbers {

/// \brief Get the width of a number.
template <class DigitsT> inline int getWidth() { return sizeof(DigitsT) * 8; }

/// \brief Conditionally round up a scaled number.
///
/// Given \c Digits and \c Scale, round up iff \c ShouldRound is \c true.
/// Always returns \c Scale unless there's an overflow, in which case it
/// returns \c 1+Scale.
///
/// \pre adding 1 to \c Scale will not overflow INT16_MAX.
template <class DigitsT>
inline std::pair<DigitsT, int16_t> getRounded(DigitsT Digits, int16_t Scale,
                                              bool ShouldRound) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  if (ShouldRound)
    if (!++Digits)
      // Overflow.
      return std::make_pair(DigitsT(1) << (getWidth<DigitsT>() - 1), Scale + 1);
  return std::make_pair(Digits, Scale);
}

/// \brief Convenience helper for 32-bit rounding.
inline std::pair<uint32_t, int16_t> getRounded32(uint32_t Digits, int16_t Scale,
                                                 bool ShouldRound) {
  return getRounded(Digits, Scale, ShouldRound);
}

/// \brief Convenience helper for 64-bit rounding.
inline std::pair<uint64_t, int16_t> getRounded64(uint64_t Digits, int16_t Scale,
                                                 bool ShouldRound) {
  return getRounded(Digits, Scale, ShouldRound);
}

/// \brief Adjust a 64-bit scaled number down to the appropriate width.
///
/// \pre Adding 64 to \c Scale will not overflow INT16_MAX.
template <class DigitsT>
inline std::pair<DigitsT, int16_t> getAdjusted(uint64_t Digits,
                                               int16_t Scale = 0) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  const int Width = getWidth<DigitsT>();
  if (Width == 64 || Digits <= std::numeric_limits<DigitsT>::max())
    return std::make_pair(Digits, Scale);

  // Shift right and round.
  int Shift = 64 - Width - countLeadingZeros(Digits);
  return getRounded<DigitsT>(Digits >> Shift, Scale + Shift,
                             Digits & (UINT64_C(1) << (Shift - 1)));
}

/// \brief Convenience helper for adjusting to 32 bits.
inline std::pair<uint32_t, int16_t> getAdjusted32(uint64_t Digits,
                                                  int16_t Scale = 0) {
  return getAdjusted<uint32_t>(Digits, Scale);
}

/// \brief Convenience helper for adjusting to 64 bits.
inline std::pair<uint64_t, int16_t> getAdjusted64(uint64_t Digits,
                                                  int16_t Scale = 0) {
  return getAdjusted<uint64_t>(Digits, Scale);
}

/// \brief Multiply two 64-bit integers to create a 64-bit scaled number.
///
/// Implemented with four 64-bit integer multiplies.
std::pair<uint64_t, int16_t> multiply64(uint64_t LHS, uint64_t RHS);

/// \brief Multiply two 32-bit integers to create a 32-bit scaled number.
///
/// Implemented with one 64-bit integer multiply.
template <class DigitsT>
inline std::pair<DigitsT, int16_t> getProduct(DigitsT LHS, DigitsT RHS) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  if (getWidth<DigitsT>() <= 32 || (LHS <= UINT32_MAX && RHS <= UINT32_MAX))
    return getAdjusted<DigitsT>(uint64_t(LHS) * RHS);

  return multiply64(LHS, RHS);
}

/// \brief Convenience helper for 32-bit product.
inline std::pair<uint32_t, int16_t> getProduct32(uint32_t LHS, uint32_t RHS) {
  return getProduct(LHS, RHS);
}

/// \brief Convenience helper for 64-bit product.
inline std::pair<uint64_t, int16_t> getProduct64(uint64_t LHS, uint64_t RHS) {
  return getProduct(LHS, RHS);
}

/// \brief Divide two 64-bit integers to create a 64-bit scaled number.
///
/// Implemented with long division.
///
/// \pre \c Dividend and \c Divisor are non-zero.
std::pair<uint64_t, int16_t> divide64(uint64_t Dividend, uint64_t Divisor);

/// \brief Divide two 32-bit integers to create a 32-bit scaled number.
///
/// Implemented with one 64-bit integer divide/remainder pair.
///
/// \pre \c Dividend and \c Divisor are non-zero.
std::pair<uint32_t, int16_t> divide32(uint32_t Dividend, uint32_t Divisor);

/// \brief Divide two 32-bit numbers to create a 32-bit scaled number.
///
/// Implemented with one 64-bit integer divide/remainder pair.
///
/// Returns \c (DigitsT_MAX, INT16_MAX) for divide-by-zero (0 for 0/0).
template <class DigitsT>
std::pair<DigitsT, int16_t> getQuotient(DigitsT Dividend, DigitsT Divisor) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");
  static_assert(sizeof(DigitsT) == 4 || sizeof(DigitsT) == 8,
                "expected 32-bit or 64-bit digits");

  // Check for zero.
  if (!Dividend)
    return std::make_pair(0, 0);
  if (!Divisor)
    return std::make_pair(std::numeric_limits<DigitsT>::max(), INT16_MAX);

  if (getWidth<DigitsT>() == 64)
    return divide64(Dividend, Divisor);
  return divide32(Dividend, Divisor);
}

/// \brief Convenience helper for 32-bit quotient.
inline std::pair<uint32_t, int16_t> getQuotient32(uint32_t Dividend,
                                                  uint32_t Divisor) {
  return getQuotient(Dividend, Divisor);
}

/// \brief Convenience helper for 64-bit quotient.
inline std::pair<uint64_t, int16_t> getQuotient64(uint64_t Dividend,
                                                  uint64_t Divisor) {
  return getQuotient(Dividend, Divisor);
}

/// \brief Implementation of getLg() and friends.
///
/// Returns the rounded lg of \c Digits*2^Scale and an int specifying whether
/// this was rounded up (1), down (-1), or exact (0).
///
/// Returns \c INT32_MIN when \c Digits is zero.
template <class DigitsT>
inline std::pair<int32_t, int> getLgImpl(DigitsT Digits, int16_t Scale) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  if (!Digits)
    return std::make_pair(INT32_MIN, 0);

  // Get the floor of the lg of Digits.
  int32_t LocalFloor = sizeof(Digits) * 8 - countLeadingZeros(Digits) - 1;

  // Get the actual floor.
  int32_t Floor = Scale + LocalFloor;
  if (Digits == UINT64_C(1) << LocalFloor)
    return std::make_pair(Floor, 0);

  // Round based on the next digit.
  assert(LocalFloor >= 1);
  bool Round = Digits & UINT64_C(1) << (LocalFloor - 1);
  return std::make_pair(Floor + Round, Round ? 1 : -1);
}

/// \brief Get the lg (rounded) of a scaled number.
///
/// Get the lg of \c Digits*2^Scale.
///
/// Returns \c INT32_MIN when \c Digits is zero.
template <class DigitsT> int32_t getLg(DigitsT Digits, int16_t Scale) {
  return getLgImpl(Digits, Scale).first;
}

/// \brief Get the lg floor of a scaled number.
///
/// Get the floor of the lg of \c Digits*2^Scale.
///
/// Returns \c INT32_MIN when \c Digits is zero.
template <class DigitsT> int32_t getLgFloor(DigitsT Digits, int16_t Scale) {
  auto Lg = getLgImpl(Digits, Scale);
  return Lg.first - (Lg.second > 0);
}

/// \brief Get the lg ceiling of a scaled number.
///
/// Get the ceiling of the lg of \c Digits*2^Scale.
///
/// Returns \c INT32_MIN when \c Digits is zero.
template <class DigitsT> int32_t getLgCeiling(DigitsT Digits, int16_t Scale) {
  auto Lg = getLgImpl(Digits, Scale);
  return Lg.first + (Lg.second < 0);
}

/// \brief Implementation for comparing scaled numbers.
///
/// Compare two 64-bit numbers with different scales.  Given that the scale of
/// \c L is higher than that of \c R by \c ScaleDiff, compare them.  Return -1,
/// 1, and 0 for less than, greater than, and equal, respectively.
///
/// \pre 0 <= ScaleDiff < 64.
int compareImpl(uint64_t L, uint64_t R, int ScaleDiff);

/// \brief Compare two scaled numbers.
///
/// Compare two scaled numbers.  Returns 0 for equal, -1 for less than, and 1
/// for greater than.
template <class DigitsT>
int compare(DigitsT LDigits, int16_t LScale, DigitsT RDigits, int16_t RScale) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  // Check for zero.
  if (!LDigits)
    return RDigits ? -1 : 0;
  if (!RDigits)
    return 1;

  // Check for the scale.  Use getLgFloor to be sure that the scale difference
  // is always lower than 64.
  int32_t lgL = getLgFloor(LDigits, LScale), lgR = getLgFloor(RDigits, RScale);
  if (lgL != lgR)
    return lgL < lgR ? -1 : 1;

  // Compare digits.
  if (LScale < RScale)
    return compareImpl(LDigits, RDigits, RScale - LScale);

  return -compareImpl(RDigits, LDigits, LScale - RScale);
}

/// \brief Match scales of two numbers.
///
/// Given two scaled numbers, match up their scales.  Change the digits and
/// scales in place.  Shift the digits as necessary to form equivalent numbers,
/// losing precision only when necessary.
///
/// If the output value of \c LDigits (\c RDigits) is \c 0, the output value of
/// \c LScale (\c RScale) is unspecified.
///
/// As a convenience, returns the matching scale.  If the output value of one
/// number is zero, returns the scale of the other.  If both are zero, which
/// scale is returned is unspecifed.
template <class DigitsT>
int16_t matchScales(DigitsT &LDigits, int16_t &LScale, DigitsT &RDigits,
                    int16_t &RScale) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  if (LScale < RScale)
    // Swap arguments.
    return matchScales(RDigits, RScale, LDigits, LScale);
  if (!LDigits)
    return RScale;
  if (!RDigits || LScale == RScale)
    return LScale;

  // Now LScale > RScale.  Get the difference.
  int32_t ScaleDiff = int32_t(LScale) - RScale;
  if (ScaleDiff >= 2 * getWidth<DigitsT>()) {
    // Don't bother shifting.  RDigits will get zero-ed out anyway.
    RDigits = 0;
    return LScale;
  }

  // Shift LDigits left as much as possible, then shift RDigits right.
  int32_t ShiftL = std::min<int32_t>(countLeadingZeros(LDigits), ScaleDiff);
  assert(ShiftL < getWidth<DigitsT>() && "can't shift more than width");

  int32_t ShiftR = ScaleDiff - ShiftL;
  if (ShiftR >= getWidth<DigitsT>()) {
    // Don't bother shifting.  RDigits will get zero-ed out anyway.
    RDigits = 0;
    return LScale;
  }

  LDigits <<= ShiftL;
  RDigits >>= ShiftR;

  LScale -= ShiftL;
  RScale += ShiftR;
  assert(LScale == RScale && "scales should match");
  return LScale;
}

/// \brief Get the sum of two scaled numbers.
///
/// Get the sum of two scaled numbers with as much precision as possible.
///
/// \pre Adding 1 to \c LScale (or \c RScale) will not overflow INT16_MAX.
template <class DigitsT>
std::pair<DigitsT, int16_t> getSum(DigitsT LDigits, int16_t LScale,
                                   DigitsT RDigits, int16_t RScale) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  // Check inputs up front.  This is only relevent if addition overflows, but
  // testing here should catch more bugs.
  assert(LScale < INT16_MAX && "scale too large");
  assert(RScale < INT16_MAX && "scale too large");

  // Normalize digits to match scales.
  int16_t Scale = matchScales(LDigits, LScale, RDigits, RScale);

  // Compute sum.
  DigitsT Sum = LDigits + RDigits;
  if (Sum >= RDigits)
    return std::make_pair(Sum, Scale);

  // Adjust sum after arithmetic overflow.
  DigitsT HighBit = DigitsT(1) << (getWidth<DigitsT>() - 1);
  return std::make_pair(HighBit | Sum >> 1, Scale + 1);
}

/// \brief Convenience helper for 32-bit sum.
inline std::pair<uint32_t, int16_t> getSum32(uint32_t LDigits, int16_t LScale,
                                             uint32_t RDigits, int16_t RScale) {
  return getSum(LDigits, LScale, RDigits, RScale);
}

/// \brief Convenience helper for 64-bit sum.
inline std::pair<uint64_t, int16_t> getSum64(uint64_t LDigits, int16_t LScale,
                                             uint64_t RDigits, int16_t RScale) {
  return getSum(LDigits, LScale, RDigits, RScale);
}

/// \brief Get the difference of two scaled numbers.
///
/// Get LHS minus RHS with as much precision as possible.
///
/// Returns \c (0, 0) if the RHS is larger than the LHS.
template <class DigitsT>
std::pair<DigitsT, int16_t> getDifference(DigitsT LDigits, int16_t LScale,
                                          DigitsT RDigits, int16_t RScale) {
  static_assert(!std::numeric_limits<DigitsT>::is_signed, "expected unsigned");

  // Normalize digits to match scales.
  const DigitsT SavedRDigits = RDigits;
  const int16_t SavedRScale = RScale;
  matchScales(LDigits, LScale, RDigits, RScale);

  // Compute difference.
  if (LDigits <= RDigits)
    return std::make_pair(0, 0);
  if (RDigits || !SavedRDigits)
    return std::make_pair(LDigits - RDigits, LScale);

  // Check if RDigits just barely lost its last bit.  E.g., for 32-bit:
  //
  //   1*2^32 - 1*2^0 == 0xffffffff != 1*2^32
  const auto RLgFloor = getLgFloor(SavedRDigits, SavedRScale);
  if (!compare(LDigits, LScale, DigitsT(1), RLgFloor + getWidth<DigitsT>()))
    return std::make_pair(std::numeric_limits<DigitsT>::max(), RLgFloor);

  return std::make_pair(LDigits, LScale);
}

/// \brief Convenience helper for 32-bit difference.
inline std::pair<uint32_t, int16_t> getDifference32(uint32_t LDigits,
                                                    int16_t LScale,
                                                    uint32_t RDigits,
                                                    int16_t RScale) {
  return getDifference(LDigits, LScale, RDigits, RScale);
}

/// \brief Convenience helper for 64-bit difference.
inline std::pair<uint64_t, int16_t> getDifference64(uint64_t LDigits,
                                                    int16_t LScale,
                                                    uint64_t RDigits,
                                                    int16_t RScale) {
  return getDifference(LDigits, LScale, RDigits, RScale);
}

} // end namespace ScaledNumbers
} // end namespace llvm

#endif
