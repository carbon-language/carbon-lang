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

#ifndef BIG_RADIX_FLOATING_POINT_H_
#define BIG_RADIX_FLOATING_POINT_H_

// This is a helper class for use in floating-point conversions
// to and from decimal representations.  It holds a multiple-precision
// integer value using digits of a radix that is a large even power of ten.
// The digits are accompanied by a signed exponent that denotes multiplication
// by a power of ten.
//
// The operations supported by this class are limited to those required
// for conversions between binary and decimal representations; it is not
// a general-purpose facility.

#include "binary-floating-point.h"
#include "decimal.h"
#include "../common/bit-population-count.h"
#include "../common/leading-zero-bit-count.h"
#include <cinttypes>
#include <limits>
#include <type_traits>

namespace Fortran::decimal {

static constexpr std::uint64_t TenToThe(int power) {
  return power <= 0 ? 1 : 10 * TenToThe(power - 1);
}

// 10**(LOG10RADIX + 3) must be < 2**wordbits, and LOG10RADIX must be
// even, so that pairs of decimal digits do not straddle Digits.
// So LOG10RADIX must be 16 or 6.
template<int PREC, int LOG10RADIX = 16> class BigRadixFloatingPointNumber {
private:
  using Real = BinaryFloatingPointNumber<PREC>;

  static constexpr int log10Radix{LOG10RADIX};
  static constexpr std::uint64_t uint64Radix{TenToThe(log10Radix)};
  static constexpr int minDigitBits{64 - common::LeadingZeroBitCount(uint64Radix)};
  using Digit = HostUnsignedIntType<minDigitBits>;
  static constexpr Digit radix{uint64Radix};
  static_assert(radix < std::numeric_limits<Digit>::max() / 1000,
      "radix is somehow too big");
  static_assert(radix > std::numeric_limits<Digit>::max() / 10000,
      "radix is somehow too small");

  // The base-2 logarithm of the least significant bit that can arise
  // in a subnormal IEEE floating-point number.
  static constexpr int minLog2AnyBit{
      -static_cast<int>(Real::exponentBias) - Real::precision};
  static constexpr int maxDigits{3 - minLog2AnyBit / log10Radix};

public:
  BigRadixFloatingPointNumber() {}

  // Converts a binary floating point value, the key step
  // in binary-to-decimal conversion.
  explicit BigRadixFloatingPointNumber(Real);

  BigRadixFloatingPointNumber &SetToZero() {
    isNegative_ = false;
    digits_ = 0;
    exponent_ = 0;
    return *this;
  }

  BigRadixFloatingPointNumber &Clamp(int decimals) {
    if (decimals >= maxDigits * log10Radix) {
      digitLimit_ = maxDigits;
    } else {
      digitLimit_ = (decimals + log10Radix - 1) / log10Radix;
    }
    return *this;
  }

  // Reads a character representation of a floating-point value into
  // this decimal floating-point representation.  The reference argument
  // is a pointer that is left pointing to the first character that wasn't
  // included.
  bool ParseNumber(const char *&);

  // Converts decimal floating-point to binary.
  ConversionToBinaryResult<PREC> ConvertToBinary(bool roundToNearest = true);

  // Parses and converts to binary.  Also handles "NaN" & "Inf".
  ConversionToBinaryResult<PREC> ConvertToBinary(
      const char *&, bool roundToNearest = true);

  // Formats a decimal floating-point number.
  ConversionToDecimalResult ConvertToDecimal(
      char *, std::size_t, int flags) const;

  // Discard decimal digits not needed to distinguish this value
  // from the decimal encodings of two others (viz., the nearest binary
  // floating-point numbers immediately below and above this one).
  void Minimize(
      BigRadixFloatingPointNumber &&less, BigRadixFloatingPointNumber &&more);

private:
  BigRadixFloatingPointNumber(const BigRadixFloatingPointNumber &that)
    : digits_{that.digits_}, exponent_{that.exponent_}, isNegative_{
                                                            that.isNegative_} {
    for (int j{0}; j < digits_; ++j) {
      digit_[j] = that.digit_[j];
    }
  }

  bool IsZero() const {
    for (int j{0}; j < digits_; ++j) {
      if (digit_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  bool IsOdd() const { return digits_ > 0 && (digit_[0] & 1); }

  // Predicate: true when 10*value would cause a carry.
  // (When this happens during decimal-to-binary conversion,
  // there are more digits in the input string than can be
  // represented precisely.)
  bool IsFull() const {
    return digits_ == digitLimit_ && 10 * digit_[digits_ - 1] >= radix;
  }

  // Set to an unsigned integer value.
  // Returns any remainder (usually zero).
  template<typename UINT> UINT SetTo(UINT n) {
    static_assert(
        std::is_same_v<UINT, __uint128_t> || std::is_unsigned_v<UINT>);
    SetToZero();
    while (n != 0) {
      auto q{n / 10};
      if (n != 10 * q) {
        break;
      }
      ++exponent_;
      n = q;
    }
    if constexpr (sizeof n < sizeof(Digit)) {
      if (n != 0) {
        digit_[digits_++] = n;
      }
      return 0;
    } else {
      while (n != 0 && digits_ < digitLimit_) {
        auto q{n / radix};
        digit_[digits_++] = n - radix * q;
        n = q;
      }
      return n;
    }
  }

  int RemoveLeastOrderZeroDigits() {
    int remove{0};
    if (digits_ > 0 && digit_[0] == 0) {
      while (remove < digits_ && digit_[remove] == 0) {
        ++remove;
      }
      if (remove >= digits_) {
        digits_ = 0;
      } else if (remove > 0) {
        for (int j{0}; j + remove < digits_; ++j) {
          digit_[j] = digit_[j + remove];
        }
        digits_ -= remove;
      }
    }
    return remove;
  }

  void RemoveLeadingZeroDigits() {
    while (digits_ > 0 && digit_[digits_ - 1] == 0) {
      --digits_;
    }
  }

  void Normalize() {
    RemoveLeadingZeroDigits();
    exponent_ += RemoveLeastOrderZeroDigits() * log10Radix;
  }

  // This limited divisibility test only works for even divisors of the radix,
  // which is fine since it's only used with 2 and 5.
  template<int N> bool IsDivisibleBy() const {
    static_assert(N > 1 && radix % N == 0, "bad modulus");
    return digits_ == 0 || (digit_[0] % N) == 0;
  }

  template<unsigned DIVISOR> int DivideBy() {
    Digit remainder{0};
    for (int j{digits_ - 1}; j >= 0; --j) {
      // N.B. Because DIVISOR is a constant, these operations should be cheap.
      Digit q{digit_[j] / DIVISOR};
      Digit nrem{digit_[j] - DIVISOR * q};
      digit_[j] = q + (radix / DIVISOR) * remainder;
      remainder = nrem;
    }
    return remainder;
  }

  int DivideByPowerOfTwo(int twoPow) {  // twoPow <= LOG10RADIX
    int remainder{0};
    for (int j{digits_ - 1}; j >= 0; --j) {
      Digit q{digit_[j] >> twoPow};
      int nrem = digit_[j] - (q << twoPow);
      digit_[j] = q + (radix >> twoPow) * remainder;
      remainder = nrem;
    }
    return remainder;
  }

  int AddCarry(int position = 0, int carry = 1) {
    for (; position < digits_; ++position) {
      Digit v{digit_[position] + carry};
      if (v < radix) {
        digit_[position] = v;
        return 0;
      }
      digit_[position] = v - radix;
      carry = 1;
    }
    if (digits_ < digitLimit_) {
      digit_[digits_++] = carry;
      return 0;
    }
    Normalize();
    if (digits_ < digitLimit_) {
      digit_[digits_++] = carry;
      return 0;
    }
    return carry;
  }

  template<int N> int MultiplyByHelper(int carry = 0) {
    for (int j{0}; j < digits_; ++j) {
      digit_[j] = N * digit_[j] + carry;
      carry = digit_[j] / radix;  // N.B. radix is constant, this is fast
      digit_[j] -= carry * radix;
    }
    return carry;
  }

  template<int N> int MultiplyBy(int carry = 0) {
    if (int newCarry{MultiplyByHelper<N>(carry)}) {
      return AddCarry(digits_, newCarry);
    } else {
      return 0;
    }
  }

  template<int N> int MultiplyWithoutNormalization() {
    int carry{MultiplyByHelper<N>(0)};
    if (carry > 0 && digits_ < digitLimit_) {
      digit_[digits_++] = carry;
      carry = 0;
    }
    return carry;
  }

  void LoseLeastSignificantDigit() {
    if (digits_ >= 2) {
      Digit LSD{digit_[0]};
      for (int j{0}; j < digits_ - 1; ++j) {
        digit_[j] = digit_[j + 1];
      }
      digit_[digits_ - 1] = 0;
      exponent_ += log10Radix;
      if (LSD > radix / 2 || (digit_[0] % 2 != 0 && LSD == radix / 2)) {
        // round up, ties to even
        for (int j{0}; ++digit_[j] == radix; ++j) {
          digit_[j] = 0;
        }
      }
    }
  }

  template<int N> void MultiplyByRounded() {
    if (int carry{MultiplyBy<N>()}) {
      LoseLeastSignificantDigit();
      digit_[digits_ - 1] += carry;
    }
  }

  // Adds another number and then divides by two.
  // Assumes same exponent and sign.
  // Returns true when the the result has effectively been rounded down.
  bool Mean(const BigRadixFloatingPointNumber &);

  Digit digit_[maxDigits];  // in little-endian order: digit_[0] is LSD
  int digits_{0};  // # of elements in digit_[] array; zero when zero
  int digitLimit_{maxDigits};  // precision clamp
  int exponent_{0};  // signed power of ten
  bool isNegative_{false};
};
}
#endif
