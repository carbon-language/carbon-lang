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

} // end namespace ScaledNumbers
} // end namespace llvm

#endif

