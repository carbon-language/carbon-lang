// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_DECIMAL_H_
#define FORTRAN_EVALUATE_DECIMAL_H_

#include "common.h"
#include "integer.h"
#include "leading-zero-bit-count.h"
#include "real.h"
#include <cinttypes>
#include <limits>
#include <ostream>

// This is a helper class for use in floating-point conversions
// to and from decimal representations.  It holds a multiple-precision
// integer value using digits in radix that is a large power of ten.
// (A radix of 10**18 (one quintillion) is the largest that could be used
// because this radix is the largest power of ten such that 10 times that
// value will fit in an unsigned 64-bit binary integer; a radix of 10**8,
// however, runs faster since unsigned 32-bit division by a constant can be
// performed cheaply.)  The digits are accompanied by a signed exponent
// that denotes multiplication by a power of ten.
//
// The operations supported by this class are limited to those required
// for conversions between binary and decimal representations; it is not
// a general-purpose facility.

namespace Fortran::evaluate::value {

static constexpr std::uint64_t TenToThe(int power) {
  return power <= 0 ? 1 : 10 * TenToThe(power - 1);
}

// The default radix is 10**8 (100,000,000) for best
// performance.
template<typename REAL, int LOG10RADIX = 8> class Decimal {
private:
  using Real = REAL;
  static constexpr int log10Radix{LOG10RADIX};
  static constexpr std::uint64_t uint64Radix{TenToThe(log10Radix)};
  static constexpr int minDigitBits{64 - LeadingZeroBitCount(uint64Radix)};
  using Digit = HostUnsignedInt<minDigitBits>;
  static constexpr Digit radix{uint64Radix};
  static_assert(radix < std::numeric_limits<Digit>::max() / 10,
      "radix is too big somehow");
  static_assert(radix > std::numeric_limits<Digit>::max() / 100,
      "radix is too small somehow");

  // The base-2 logarithm of the least significant bit that can arise
  // in a subnormal IEEE floating-point number.  Padded up to make room
  // for scaling for alignment in decimal->binary conversion.
  static constexpr int minLog2AnyBit{
      -static_cast<int>(Real::exponentBias) - Real::precision};
  static constexpr int maxDigits{3 - minLog2AnyBit / log10Radix};

public:
  Decimal() {}

  Decimal &SetToZero() {
    isNegative_ = false;
    digits_ = 0;
    exponent_ = 0;
    return *this;
  }

  Decimal &FromReal(const Real &);

  // Convert a character representation of a floating-point value to
  // the underlying Real type.  The reference argument is a pointer that
  // is left pointing to the first character that wasn't included.
  ValueWithRealFlags<Real> ToReal(
      const char *&, Rounding rounding = defaultRounding);

  // ToString() emits the mathematically exact decimal representation
  // in scientific notation.  ToMinimalString() emits the shortest
  // decimal representation that reads back to the original value.
  std::string ToString(int maxDigits = 1000000) const;
  std::string ToMinimalString(
      const Real &, Rounding rounding = defaultRounding) const;

private:
  std::ostream &Dump(std::ostream &) const;

  bool IsZero() const {
    for (int j{0}; j < digits_; ++j) {
      if (digit_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  // Predicate: true when 10*value would cause a carry
  bool IsFull() const {
    return digits_ == digitLimit_ && 10 * digit_[digits_ - 1] >= radix;
  }

  // Sets this value to that of an Integer<> instance.
  // Returns any remainder (usually zero).
  template<typename INT> INT SetTo(INT n) {
    SetToZero();
    while (!n.IsZero()) {
      auto qr{n.DivideUnsigned(10)};
      if (!qr.remainder.IsZero()) {
        break;
      }
      ++exponent_;
      n = qr.quotient;
    }
    if constexpr (INT::bits < minDigitBits) {
      // n is necessarily small enough to fit into a digit
      if (!n.IsZero()) {
        digit_[digits_++] = n.ToUInt64();
      }
      return {};
    } else {
      while (!n.IsZero() && digits_ < digitLimit_) {
        auto qr{n.DivideUnsigned(radix)};
        digit_[digits_++] = qr.remainder.ToUInt64();
        n = qr.quotient;
      }
      return n;
    }
  }

  int RemoveLeastOrderZeroDigits() {
    int remove{0};
    while (remove < digits_ && digit_[remove] == 0) {
      ++remove;
    }
    if (remove >= digits_) {
      digits_ = 0;
      return remove;
    }
    if (remove > 0) {
      for (int j{0}; j + remove < digits_; ++j) {
        digit_[j] = digit_[j + remove];
      }
      digits_ -= remove;
    }
    return remove;
  }

  void Normalize() {
    while (digits_ > 0 && digit_[digits_ - 1] == 0) {
      --digits_;
    }
    exponent_ += RemoveLeastOrderZeroDigits() * log10Radix;
  }

  // This limited divisibility test only works for even divisors of the radix,
  // which is fine since it's only used with 2 and 5.
  template<int N> bool IsDivisibleBy() const {
    static_assert(N > 1 && radix % N == 0, "bad modulus");
    return digits_ == 0 || (digit_[0] % N) == 0;
  }

  template<int DIVISOR> int DivideBy() {
    int remainder{0};
    for (int j{digits_ - 1}; j >= 0; --j) {
      // N.B. Because DIVISOR is a constant, these operations should be cheap.
      int nrem = digit_[j] % DIVISOR;
      digit_[j] /= DIVISOR;
      digit_[j] += (radix / DIVISOR) * remainder;
      remainder = nrem;
    }
    return remainder;
  }

  template<int N> int MultiplyBy(int carry = 0) {
    for (int j{0}; j < digits_; ++j) {
      digit_[j] = N * digit_[j] + carry;
      carry = digit_[j] / radix;  // N.B. radix is constant, this is fast
      digit_[j] %= radix;
    }
    if (carry != 0) {
      if (digits_ < digitLimit_) {
        digit_[digits_++] = carry;
        carry = 0;
      }
    }
    return carry;
  }

  Digit digit_[maxDigits];  // in little-endian order
  int digits_{0};  // significant elements in digit_[] array
  int digitLimit_{maxDigits};  // clamp
  int exponent_{0};  // signed power of ten
  bool isNegative_{false};
};

extern template class Decimal<Real<Integer<16>, 11>>;
extern template class Decimal<Real<Integer<32>, 24>>;
extern template class Decimal<Real<Integer<64>, 53>>;
extern template class Decimal<Real<Integer<80>, 64, false>>;
extern template class Decimal<Real<Integer<128>, 112>>;
}
#endif  // FORTRAN_EVALUATE_DECIMAL_H_
