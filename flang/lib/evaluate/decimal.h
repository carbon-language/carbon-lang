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

#include "real.h"
#include <cinttypes>
#include <limits>
#include <ostream>

// This is a helper class for use in floating-point conversions
// to and from decimal representations.  It holds a multiple-precision
// integer value using digits in base 10**18 (one quintillion).  This
// radix is the largest power of ten such that 10 times that value will
// fit in an unsigned 64-bit binary integer.  It is accompanied by a
// signed exponent that denotes multiplication by a power of ten.
//
// The operations supported by this class are limited to those required
// for conversions between binary and decimal representations; it is not
// a general-purpose facility.

namespace Fortran::evaluate::value {

template<typename REAL> class Decimal {
private:
  using Digit = std::uint64_t;
  using Real = REAL;

  // 10 * quintillion must not overflow a 64-bit unsigned integer
  static constexpr int log10Quintillion{18};
  static constexpr Digit quintillion{
      static_cast<Digit>(1000000) * 1000000 * 1000000};
  static_assert(quintillion < std::numeric_limits<Digit>::max() / 10,
      "10**18 is too big somehow");
  static_assert(quintillion > std::numeric_limits<Digit>::max() / 100,
      "10**18 is too small somehow");

  // The base-2 logarithm of the least significant bit that can arise
  // in a subnormal IEEE floating-point number.
  static constexpr int minLog2AnyBit{
      -static_cast<int>(Real::exponentBias) - Real::precision};
  static constexpr int maxDigits{2 - minLog2AnyBit / log10Quintillion};

public:
  Decimal() {}

  void SetToZero() {
    isNegative_ = false;
    digits_ = 0;
    first_ = 0;
    exponent_ = 0;
  }
  void FromReal(const Real &);
  Real ToReal(const char *&);  // arg left pointing to first unparsed char
  std::string ToString(int maxDigits = 1000000) const;

private:
  std::ostream &Dump(std::ostream &) const;

  bool IsZero() const {
    for (int j{first_}; j < digits_; ++j) {
      if (digit_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  bool IsFull() const {
    return digits_ == digitLimit_ && 10 * digit_[digits_ - 1] >= quintillion;
  }

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
    while (!n.IsZero() && digits_ < digitLimit_) {
      auto qr{n.DivideUnsigned(quintillion)};
      digit_[digits_++] = qr.remainder.ToUInt64();
      if (digits_ == first_ + 1 && digit_[first_] == 0) {
        ++first_;
      }
      n = qr.quotient;
    }
    return n;
  }

  int RemoveLeastOrderZeroDigits() {
    int removed{0};
    while (first_ < digits_ && digit_[first_] == 0) {
      ++first_;
      ++removed;
    }
    if (first_ == digits_) {
      first_ = digits_ = 0;
    }
    return removed;
  }

  void Normalize() {
    while (digits_ > 0 && digit_[digits_ - 1] == 0) {
      --digits_;
    }
    exponent_ += RemoveLeastOrderZeroDigits() * log10Quintillion;
  }

  // This limited divisibility test only works for even divisors of 10**18,
  // which is fine since it's only used with 2 and 5.
  template<int N> bool IsDivisibleBy() const {
    static_assert(N > 1 && quintillion % N == 0, "bad modulus");
    return digits_ == first_ || (digit_[first_] % N) == 0;
  }

  template<int N> int DivideBy() {
    int remainder{0};
    for (int j{digits_ - 1}; j >= 0; --j) {
      if (j < first_) {
        if (remainder == 0) {
          break;
        }
        first_ = j;
      }
      int nrem = digit_[j] % N;
      digit_[j] /= N;
      digit_[j] += (quintillion / N) * remainder;
      remainder = nrem;
    }
    return remainder;
  }

  template<int N> int MultiplyBy(int carry = 0) {
    for (int j{first_}; j < digits_; ++j) {
      digit_[j] = N * digit_[j] + carry;
      carry = digit_[j] / quintillion;
      digit_[j] %= quintillion;
      if (j == first_ && digit_[j] == 0) {
        ++first_;
      }
    }
    if (carry != 0) {
      if (digits_ < digitLimit_) {
        digit_[digits_++] = carry;
        carry = 0;
      }
    }
    return carry;
  }

  Digit digit_[maxDigits];  // base-quintillion digits in little-endian order
  int digits_{0};  // significant elements in digit_[] array
  int first_{0};  // digits below this are all zero
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
