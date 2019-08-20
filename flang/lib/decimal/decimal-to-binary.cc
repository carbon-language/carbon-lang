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
#include <cstring>
#include <ctype.h>

namespace Fortran::decimal {

template<int PREC, int LOG10RADIX>
bool BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ParseNumber(
    const char *&p, bool &inexact) {
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
  while (*q == '0') {
    ++q;
  }
  const char *firstDigit{q};
  for (; *q >= '0' && *q <= '9'; ++q) {
  }
  const char *point{*q == '.' ? q : nullptr};
  if (point) {
    for (++q; *q >= '0' && *q <= '9'; ++q) {
    }
  }
  if (q == start || (q == start + 1 && *start == '.')) {
    return false;  // require at least one digit
  }
  auto times{radix};
  const char *d{q};
  if (point != nullptr) {
    while (d > firstDigit && d[-1] == '0') {
      --d;
    }
    if (d[-1] == '.') {
      point = nullptr;
      --d;
    }
  }
  if (point == nullptr) {
    while (d > firstDigit && d[-1] == '0') {
      --d;
      ++exponent_;
    }
  }
  if (d == firstDigit) {
    exponent_ = 0;
  }
  if (point != nullptr) {
    exponent_ -= static_cast<int>(d - point - 1);
  }
  const char *limit{firstDigit + maxDigits * log10Radix + (point != nullptr)};
  if (d > limit) {
    inexact = true;
    while (d-- > limit) {
      if (*d == '.') {
        point = nullptr;
        --limit;
      } else if (point == nullptr) {
        ++exponent_;
      }
    }
  }
  while (d-- > firstDigit) {
    if (*d != '.') {
      if (times == radix) {
        digit_[digits_++] = *d - '0';
        times = 10;
      } else {
        digit_[digits_ - 1] += times * (*d - '0');
        times *= 10;
      }
    }
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
// binary point to its *right*, and some guard bits for rounding.
template<int PREC> class IntermediateFloat {
public:
  static constexpr int precision{PREC};
  using IntType = HostUnsignedIntType<precision>;
  static constexpr IntType topBit{static_cast<IntType>(1) << (precision - 1)};
  static constexpr IntType mask{topBit + (topBit - 1)};

  IntermediateFloat() {}
  IntermediateFloat(const IntermediateFloat &) = default;

  // Assumes that exponent_ is valid on entry, and may increment it.
  // Returns the number of guard_ bits that also been determined.
  template<typename UINT> bool SetTo(UINT n) {
    static constexpr int nBits{CHAR_BIT * sizeof n};
    if constexpr (precision >= nBits) {
      value_ = n;
      guard_ = 0;
      return 0;
    } else {
      int shift{common::BitsNeededFor(n) - precision};
      if (shift <= 0) {
        value_ = n;
        guard_ = 0;
        return 0;
      } else {
        value_ = n >> shift;
        exponent_ += shift;
        n <<= nBits - shift;
        guard_ = (n >> (nBits - precision)) | ((n << precision) != 0);
        return shift;
      }
    }
  }

  void ShiftIn(int bit = 0) { value_ = value_ + value_ + bit; }
  bool IsFull() const { return value_ >= topBit; }
  void AdjustExponent(int by) { exponent_ += by; }
  void SetGuard(int g) {
    guard_ |= (static_cast<IntType>(g & 6) << (precision - 3)) | (g & 1);
  }

  ConversionToBinaryResult<PREC> ToBinary(
      bool isNegative, FortranRounding) const;

private:
  IntType value_{0}, guard_{0};  // TODO pmk revert to 3-bit guard?
  int exponent_{0};
};

template<int PREC>
ConversionToBinaryResult<PREC> IntermediateFloat<PREC>::ToBinary(
    bool isNegative, FortranRounding rounding) const {
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
  int flags{Exact};
  if (guard != 0) {
    flags |= Inexact;
  }
  if (fraction == 0 && guard <= topBit) {
    return {Binary{}, static_cast<enum ConversionResultFlags>(flags)};
  }
  // The value is nonzero; normalize it.
  while (fraction < topBit && expo > 1) {
    --expo;
    fraction = 2 * fraction + ((guard & topBit) >> (precision - 2));
    guard = (guard & 1) | ((guard & (topBit - 2)) << 1);
  }
  // Apply rounding
  bool incr{false};
  switch (rounding) {
  case RoundNearest:
  case RoundDefault:
    incr = guard > topBit || (guard == topBit && (fraction & 1));
    break;
  case RoundUp: incr = guard > 0 && !isNegative; break;
  case RoundDown: incr = guard > 0 && isNegative; break;
  case RoundToZero: break;
  case RoundCompatible: incr = guard >= topBit; break;
  }
  if (incr) {
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
  return {Binary(raw), static_cast<enum ConversionResultFlags>(flags)};
}

template<int PREC, int LOG10RADIX>
ConversionToBinaryResult<PREC>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToBinary() {
  using Binary = BinaryFloatingPointNumber<PREC>;
  // On entry, *this holds a multi-precision integer value in a radix of a
  // large power of ten.  Its radix point is defined to be to the right of its
  // digits, and "exponent_" is the power of ten by which it is to be scaled.
  Normalize();
  if (digits_ == 0) {
    if (isNegative_) {
      using Raw = typename Binary::RawType;
      Raw negZero{static_cast<Raw>(1) << (Binary::bits - 1)};
      return {Binary{negZero}};
    } else {
      return {Binary{}};
    }
  }
  // The value is not zero.
  // Shift our perspective on the radix (& decimal) point so that
  // it sits to the *left* of the digits.
  exponent_ += digits_ * log10Radix;
  // Apply any negative decimal exponent by multiplication
  // by a power of two, adjusting the binary exponent to compensate.
  IntermediateFloat<PREC> f;
  while (exponent_ < log10Radix) {
    f.AdjustExponent(-9);
    digitLimit_ = digits_;
    int carry{MultiplyWithoutNormalization<512>()};
    RemoveLeastOrderZeroDigits();
    if (carry != 0) {
      digit_[digits_++] = carry;
      exponent_ += log10Radix;
    }
  }
  // Apply any positive decimal exponent greater than
  // is needed to treat the topmost digit as an integer
  // part by multiplying by 10 or 10000 repeatedly.
  while (exponent_ > log10Radix) {
    digitLimit_ = digits_;
    int carry;
    if (exponent_ >= log10Radix + 4) {
      exponent_ -= 4;
      carry = MultiplyWithoutNormalization<(5 * 5 * 5 * 5)>();
      f.AdjustExponent(4);
    } else {
      --exponent_;
      carry = MultiplyWithoutNormalization<5>();
      f.AdjustExponent(1);
    }
    RemoveLeastOrderZeroDigits();
    if (carry != 0) {
      digit_[digits_++] = carry;
      exponent_ += log10Radix;
    }
  }
  // So exponent_ is now log10Radix, meaning that the
  // MSD can be taken as an integer part and transferred
  // to the binary result.
  int guardShift{f.SetTo(digit_[--digits_])};
  // Transfer additional bits until the result is normal.
  digitLimit_ = digits_;
  while (!f.IsFull()) {
    f.AdjustExponent(-1);
    std::uint32_t carry = MultiplyWithoutNormalization<2>();
    f.ShiftIn(carry);
  }
  // Get the next few bits for rounding.  Allow for some guard bits
  // that may have already been set in f.SetTo() above.
  int guard{0};
  if (guardShift == 0) {
    guard = MultiplyWithoutNormalization<4>();
  } else if (guardShift == 1) {
    guard = MultiplyWithoutNormalization<2>();
  }
  guard = guard + guard + !IsZero();
  f.SetGuard(guard);
  return f.ToBinary(isNegative_, rounding_);
}

template<int PREC, int LOG10RADIX>
ConversionToBinaryResult<PREC>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToBinary(const char *&p) {
  bool inexact{false};
  if (ParseNumber(p, inexact)) {
    auto result{ConvertToBinary()};
    if (inexact) {
      result.flags =
          static_cast<enum ConversionResultFlags>(result.flags | Inexact);
    }
    return result;
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
      return {Binary{nan}};
    } else {
      // Try to parse Inf, maybe with a sign
      const char *q{p};
      bool isNegative{*q == '-'};
      if (*q == '-' || *q == '+') {
        ++q;
      }
      if (toupper(q[0]) == 'I' && toupper(q[1]) == 'N' &&
          toupper(q[2]) == 'F') {
        p = q + 3;
        return {Binary(isNegative ? negInf : inf)};
      } else {
        // Invalid input
        return {Binary{nan}, Invalid};
      }
    }
  }
}

template<int PREC>
ConversionToBinaryResult<PREC> ConvertToBinary(
    const char *&p, enum FortranRounding rounding) {
  return BigRadixFloatingPointNumber<PREC>{rounding}.ConvertToBinary(p);
}

template ConversionToBinaryResult<8> ConvertToBinary<8>(
    const char *&, enum FortranRounding);
template ConversionToBinaryResult<11> ConvertToBinary<11>(
    const char *&, enum FortranRounding);
template ConversionToBinaryResult<24> ConvertToBinary<24>(
    const char *&, enum FortranRounding);
template ConversionToBinaryResult<53> ConvertToBinary<53>(
    const char *&, enum FortranRounding);
template ConversionToBinaryResult<64> ConvertToBinary<64>(
    const char *&, enum FortranRounding);
template ConversionToBinaryResult<112> ConvertToBinary<112>(
    const char *&, enum FortranRounding);

extern "C" {
enum ConversionResultFlags ConvertDecimalToFloat(
    const char **p, float *f, enum FortranRounding rounding) {
  auto result{Fortran::decimal::ConvertToBinary<24>(*p, rounding)};
  std::memcpy(reinterpret_cast<void *>(f),
      reinterpret_cast<const void *>(&result.binary), sizeof *f);
  return result.flags;
}
enum ConversionResultFlags ConvertDecimalToDouble(
    const char **p, double *d, enum FortranRounding rounding) {
  auto result{Fortran::decimal::ConvertToBinary<53>(*p, rounding)};
  std::memcpy(reinterpret_cast<void *>(d),
      reinterpret_cast<const void *>(&result.binary), sizeof *d);
  return result.flags;
}
#if __x86_64__
enum ConversionResultFlags ConvertDecimalToLongDouble(
    const char **p, long double *ld, enum FortranRounding rounding) {
  auto result{Fortran::decimal::ConvertToBinary<64>(*p, rounding)};
  std::memcpy(reinterpret_cast<void *>(ld),
      reinterpret_cast<const void *>(&result.binary), sizeof *ld);
  return result.flags;
}
#endif
}
}
