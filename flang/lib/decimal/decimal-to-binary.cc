// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "big-radix-floating-point.h"
#include "binary-floating-point.h"
#include "decimal.h"
#include "../common/bit-population-count.h"
#include "../common/leading-zero-bit-count.h"
#include <cinttypes>
#include <ctype.h>

namespace Fortran::decimal {

template<int PREC, int LOG10RADIX>
bool BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ParseNumber(
    const char *&p) {
  SetToZero();
  while (*p == ' ') {
    ++p;
  }
  const char *q{p};
  isNegative_ = *q == '-';
  if (*q == '-' || *q == '+') {
    ++q;
  }
  const char *start{q};
  bool inexact{false};
  while (*q == '0') {
    ++q;
  }
  for (; *q >= '0' && *q <= '9' && !IsFull(); ++q) {
    MultiplyBy<10>(*q - '0');
  }
  if (IsFull()) {
    for (; *q >= '0' && *q <= '9'; ++q) {
      if (*q != '0') {
        inexact = true;
      }
      ++exponent_;
    }
  }
  if (*q == '.') {
    for (++q; *q >= '0' && *q <= '9' && !IsFull(); ++q) {
      MultiplyBy<10>(*q - '0');
      --exponent_;
    }
    if (IsFull()) {
      for (; *q >= '0' && *q <= '9'; ++q) {
        if (*q != '0') {
          inexact = true;
        }
      }
    }
  }
  if (q == start || (q == start + 1 && *start == '.')) {
    // No digits?  No bueno.
    return false;
  }

  switch (*q) {
  case 'e':
  case 'E':
  case 'd':
  case 'D':
  case 'q':
  case 'Q': {
    bool negExpo{*++q == '-'};
    if (*q == '-' || *q == '+') {
      ++q;
    }
    if (*q >= '0' && *q <= '9') {
      int expo{0};
      for (int j{0}; j < 8 && *q >= '0' && *q <= '9'; ++j) {
        expo = 10 * expo + *q++ - '0';
      }
      if (negExpo) {
        exponent_ -= expo;
      } else {
        exponent_ += expo;
      }
    }
  } break;
  default: break;
  }
  p = q;
  return true;
}

// This local utility class represents an unrounded nonnegative
// binary floating-point value with an unbiased (i.e., signed)
// binary exponent, an integer value (not a fraction) with an implied
// binary point to its *right*, and the usual three guard/round/sticky
// bits for rounding.
template<int PREC> class IntermediateFloat {
public:
  static constexpr int precision{PREC};
  using IntType = HostUnsignedIntType<precision>;
  static constexpr IntType topBit{static_cast<IntType>(1) << (precision - 1)};
  static constexpr IntType mask{topBit + (topBit - 1)};

  IntermediateFloat() {}
  IntermediateFloat(const IntermediateFloat &) = default;

  // Assumes that exponent_ is valid on entry, and may increment it.
  template<typename UINT> void SetTo(UINT n) {
    static constexpr int nBits{CHAR_BIT * sizeof n};
    if constexpr (precision >= nBits) {
      value_ = n;
      guard_ = 0;
    } else {
      int shift{nBits - common::LeadingZeroBitCount(n) - precision};
      if (shift <= 0) {
        value_ = n;
        guard_ = 0;
      } else {
        value_ = n >> shift;
        exponent_ += shift;
        if (shift <= precision) {
          guard_ = (n << (precision - shift)) & mask;
        } else {
          bool sticky{
              (n & ((static_cast<UINT>(1) << (shift - precision)) - 1)) != 0};
          guard_ = ((n >> (shift - precision)) & mask) | sticky;
        }
      }
    }
  }

  void ShiftDown() {
    guard_ = (guard_ & 1) | (guard_ >> 1) | ((value_ & 1) << (precision - 1));
    value_ >>= 1;
    ++exponent_;
  }

  // Multiply by 2 and add an incoming carry (pmk: simplify now that it's 2)
  template<int N> void MultiplyAndAdd(int carry = 0) {
    static_assert(N > 1 && N < 16);
    HostUnsignedIntType<precision + 4> v{value_}, g{guard_};
    g *= N;
    guard_ = (guard_ & 1) | (g & mask);
    g >>= precision;
    v *= N;
    v += g + carry;
    value_ = v & mask;
    for (v >>= precision; v > 0; v >>= 1) {
      ShiftDown();
      value_ |= static_cast<IntType>(v & 1) << (precision - 1);
    }
  }

  bool IsFull() const { return value_ >= topBit; }
  bool IsEmpty() const { return value_ == 0; }
  void AdjustExponent(int by) { exponent_ += by; }
  void SetGuard(int g) {
    guard_ = g;
    guard_ = ((guard_ & 6) << (precision - 3)) | (guard_ & 1);
  }

  ConversionToBinaryResult<PREC> ToBinary(
      bool isNegative = false, bool rounding = true) const;

private:
  IntType value_{0}, guard_{0};
  int exponent_{0};
};

template<int PREC>
ConversionToBinaryResult<PREC> IntermediateFloat<PREC>::ToBinary(
    bool isNegative, bool rounding) const {
  using Binary = BinaryFloatingPointNumber<PREC>;
  // Create a fraction with a binary point to the left of the integer
  // value_, and bias the exponent.
  IntType fraction{value_};
  IntType guard{guard_};
  int expo{exponent_ + Binary::exponentBias + (precision - 1)};
  while (expo < 1 && (fraction > 0 || guard > topBit)) {
    guard = (guard & 1) | (guard >> 1) | ((fraction & 1) << (precision - 1));
    fraction >>= 1;
    ++expo;
  }
  int flags{0};
  if (guard != 0) {
    flags |= Inexact;
    if (!rounding) {
      guard = 0;
    }
  }
  if (fraction == 0 && guard <= topBit) {
    return {Binary{}, static_cast<enum BinaryConversionFlags>(flags)};
  }
  // The value is nonzero; normalize it.
  while (fraction < topBit && expo > 1) {
    --expo;
    fraction = 2 * fraction + ((guard & topBit) >> (precision - 2));
    guard = (guard & 1) | ((guard & (topBit - 2)) << 1);
  }
  // Apply rounding
  if (guard > topBit || (guard == topBit && (fraction & 1))) {
    // round fraction up
    if (fraction == mask) {
      // rounding causes a carry
      ++expo;
      fraction = topBit;
    } else {
      ++fraction;
    }
  }
  if (expo == 1 && fraction < topBit) {
    expo = 0;  // subnormal
  }
  if (expo >= Binary::maxExponent) {
    expo = Binary::maxExponent;  // Inf
    flags |= Overflow;
    fraction = 0;
  }
  using Raw = typename Binary::RawType;
  Raw raw = static_cast<Raw>(isNegative) << (Binary::bits - 1);
  raw |= static_cast<Raw>(expo) << Binary::significandBits;
  if constexpr (Binary::implicitMSB) {
    fraction &= ~topBit;
  }
  raw |= fraction;
  return {Binary(raw), static_cast<enum BinaryConversionFlags>(flags)};
}

template<int PREC, int LOG10RADIX>
ConversionToBinaryResult<PREC>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToBinary(bool rounding) {
  using Binary = BinaryFloatingPointNumber<PREC>;
  // *this holds a multi-precision integer value in a radix of a large power
  // of ten.  Its radix point is defined to be to the right of its digits,
  // and "exponent_" is the power of ten by which it is to be scaled.
  if (IsZero()) {
    if (isNegative_) {
      using Raw = typename Binary::RawType;
      Raw negZero{static_cast<Raw>(1) << (Binary::bits - 1)};
      return {Binary{negZero}, static_cast<enum BinaryConversionFlags>(0)};
    } else {
      return {Binary{}, static_cast<enum BinaryConversionFlags>(0)};
    }
  }

  Normalize();
  IntermediateFloat<PREC> f;

  // Align the decimal exponent to be a multiple of log10(radix) so
  // that the digits can be viewed as having an effective radix point.
  if (int align{exponent_ % log10Radix}) {
    int adjust{align < 0 ? log10Radix + align : align};
    exponent_ -= adjust;
    f.AdjustExponent(adjust);
    digitLimit_ = maxDigits;
    for (; adjust >= 4; adjust -= 4) {
      MultiplyBy<(5 * 5 * 5 * 5)>();
    }
    for (; adjust > 0; --adjust) {
      MultiplyBy<5>();
    }
  }

  if (exponent_ > 0) {
    int adjust{exponent_};
    f.AdjustExponent(adjust);
    digitLimit_ = maxDigits;
    exponent_ -= adjust;
    for (; adjust >= 4; adjust -= 4) {
      MultiplyBy<(5 * 5 * 5 * 5)>();
    }
    for (; adjust > 0; --adjust) {
      MultiplyBy<5>();
    }
  }

  // Isolate the integer part, if any, into a single digit.
  while (exponent_ > (1 - digits_) * log10Radix) {
    int shift{common::BitsNeededFor(digit_[digits_ - 1])};
    if (shift > log10Radix) {
      shift = log10Radix;
    }
    DivideByPowerOfTwo(shift);
    f.AdjustExponent(shift);
    RemoveLeadingZeroDigits();
  }

  // Transfer the single digit of the integer part (if any) to
  // constitute the initial integer part (not fraction!) of the
  // binary result.
  if (exponent_ == (1 - digits_) * log10Radix) {
    f.SetTo(digit_[--digits_]);
    if (f.IsFull()) {
      return f.ToBinary(isNegative_, rounding);
    }
  }

  // Shift the radix (& decimal) point to the *left* of the remaining
  // digits, turning them into a fraction, by augmenting the decimal exponent.
  exponent_ += digits_ * log10Radix;

  // Convert the remaining fraction into bits of the
  // resulting floating-point value until it is normalized.
  if (!IsZero() && f.IsEmpty() && exponent_ < 0) {
    // fast-forward
    while (true) {
      digitLimit_ = digits_;
      std::uint32_t carry = MultiplyWithoutNormalization<512>();
      RemoveLeastOrderZeroDigits();
      f.AdjustExponent(-9);
      if (carry != 0) {
        digit_[digits_++] = carry;
        exponent_ += log10Radix;
        if (exponent_ >= 0) {
          break;
        }
      }
    }
  }
  while (!f.IsFull() && !IsZero()) {
    f.AdjustExponent(-1);
    digitLimit_ = digits_;
    std::uint32_t carry = MultiplyWithoutNormalization<2>();
    RemoveLeastOrderZeroDigits();
    if (carry != 0 && exponent_ < 0) {
      digit_[digits_++] = carry;
      exponent_ += log10Radix;
      f.template MultiplyAndAdd<2>(0);
    } else {
      f.template MultiplyAndAdd<2>(carry);
    }
  }
  if (!IsZero()) {
    // Get the next two bits for use as guard & round bits, then
    // set the sticky bit if anything else is left.
    int guard{MultiplyBy<4>() * 2};
    if (!IsZero()) {
      guard |= 1;
    }
    f.SetGuard(guard);
  }
  return f.ToBinary(isNegative_, rounding);
}

template<int PREC, int LOG10RADIX>
ConversionToBinaryResult<PREC>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToBinary(
    const char *&p, bool rounding) {
  if (ParseNumber(p)) {
    return ConvertToBinary(rounding);
  } else {
    // Could not parse a decimal floating-point number.  p has been
    // advanced over any leading spaces.
    using Binary = BinaryFloatingPointNumber<PREC>;
    using Raw = typename Binary::RawType;
    static constexpr Raw inf{
        static_cast<Raw>(Binary::maxExponent) << Binary::significandBits};
    static constexpr Raw nan{
        inf | (static_cast<Raw>(1) << (Binary::significandBits - 2))};
    static constexpr Raw negInf{
        inf | (static_cast<Raw>(1) << (Binary::bits - 1))};
    if (toupper(p[0]) == 'N' && toupper(p[1]) == 'A' && toupper(p[2]) == 'N') {
      // NaN
      p += 3;
      return {Binary{nan}, static_cast<enum BinaryConversionFlags>(0)};
    } else {
      // Try to parse Inf, maybe with a sign
      const char *q{p};
      bool isNegative{*q == '-'};
      if (*q == '-' || *q == '+') {
        ++q;
      }
      if (toupper(q[0]) == 'I' && toupper(q[1]) == 'N' &&
          toupper(q[2]) == 'F') {
        return {Binary(isNegative ? negInf : inf),
            static_cast<enum BinaryConversionFlags>(0)};
      } else {
        // Invalid input
        return {Binary{nan}, Invalid};
      }
    }
  }
}

template<int PREC>
ConversionToBinaryResult<PREC> ConvertToBinary(const char *&p, bool rounding) {
  BigRadixFloatingPointNumber<PREC> n;
  return n.ConvertToBinary(p, rounding);
}

template ConversionToBinaryResult<8> ConvertToBinary<8>(
    const char *&, bool rounding);
template ConversionToBinaryResult<11> ConvertToBinary<11>(
    const char *&, bool rounding);
template ConversionToBinaryResult<24> ConvertToBinary<24>(
    const char *&, bool rounding);
template ConversionToBinaryResult<53> ConvertToBinary<53>(
    const char *&, bool rounding);
template ConversionToBinaryResult<64> ConvertToBinary<64>(
    const char *&, bool rounding);
template ConversionToBinaryResult<112> ConvertToBinary<112>(
    const char *&, bool rounding);

}
