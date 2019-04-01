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

#include "decimal.h"
#include "integer.h"
#include "leading-zero-bit-count.h"
#include "../common/bit-population-count.h"
#include "../common/idioms.h"

namespace Fortran::evaluate::value {

template<typename REAL, int LOG10RADIX>
std::ostream &Decimal<REAL, LOG10RADIX>::Dump(std::ostream &o) const {
  if (isNegative_) {
    o << '-';
  }
  for (int j{digits_ - 1}; j >= 0; --j) {
    o << ' ' << digit_[j];
  }
  return o << " e" << exponent_ << '\n';
}

template<typename REAL, int LOG10RADIX>
auto Decimal<REAL, LOG10RADIX>::FromReal(const REAL &x) -> Decimal & {
  if (x.IsNegative()) {
    FromReal(x.Negate());
    isNegative_ = true;
    return *this;
  }
  if (x.IsZero()) {
    return SetToZero();
  }
  int twoPow{x.UnbiasedExponent()};
  twoPow -= Real::bits - 1;
  if (!Real::implicitMSB) {
    ++twoPow;
  }
  int lshift{x.exponentBits};
  if (twoPow <= -lshift) {
    twoPow += lshift;
    lshift = 0;
  } else if (twoPow < 0) {
    lshift += twoPow;
    twoPow = 0;
  }
  using Word = typename Real::Word;
  Word word{Word::ConvertUnsigned(x.GetFraction()).value};
  SetTo(word.SHIFTL(lshift));

  for (; twoPow > 0 && IsDivisibleBy<5>(); --twoPow) {
    DivideBy<5>();
    Normalize();
    ++exponent_;
  }

  // Scale by factors of 8, then by 2.
  static constexpr int log2FastForward{3};
  static constexpr int fastForward{1 << log2FastForward};
  for (; twoPow >= log2FastForward; twoPow -= log2FastForward) {
    MultiplyBy<fastForward>();
  }
  for (; twoPow > 0; --twoPow) {
    MultiplyBy<2>();
  }
  for (; twoPow <= -log2FastForward && IsDivisibleBy<fastForward>();
       twoPow += log2FastForward) {
    DivideBy<fastForward>();
    Normalize();
  }
  for (; twoPow < 0 && IsDivisibleBy<2>(); ++twoPow) {
    DivideBy<2>();
    Normalize();
  }
  for (; twoPow < 0; ++twoPow) {
    MultiplyBy<5>();
    --exponent_;
  }
  return *this;
}

// Local utility class: represents an unrounded binary
// floating-point value with an unbiased (i.e., signed)
// binary exponent.
template<typename REAL> class IntermediateFloat {
public:
  using Real = REAL;
  using Word = typename Real::Word;

  template<typename UINT> void SetTo(UINT n) {
    static constexpr int nBits{CHAR_BIT * sizeof n};
    if constexpr (Word::bits >= nBits) {
      word_ = n;
    } else {
      int shift{nBits - LeadingZeroBitCount(n) - Word::bits};
      if (shift <= 0) {
        word_ = n;
      } else {
        word_ = n >> shift;
        exponent_ += shift;
        bool sticky{n << (nBits - shift) != 0};
        if (sticky) {
          word_ = word_.IOR(Word{1});
        }
      }
    }
  }

  void MultiplyAndAdd(std::uint32_t n, std::uint32_t plus = 0) {
    auto product{word_.MultiplyUnsigned(Word{n})};
    if (plus != 0) {
      auto sum{product.lower.AddUnsigned(Word{plus})};
      product.lower = sum.value;
      if (sum.carry) {
        product.upper = product.upper.AddUnsigned(1).value;
      }
    }
    bool sticky{false};
    while (!product.upper.IsZero()) {
      sticky |= product.lower.BTEST(0);
      product.lower = product.lower.SHIFTRWithFill(product.upper, 1);
      product.upper = product.upper.SHIFTR(1);
      ++exponent_;
    }
    word_ = product.lower;
    if (sticky) {
      word_ = word_.IOR(Word{1});
    }
  }

  bool IsZero() const { return word_.IsZero(); }

  bool IsFull() const { return word_.IsNegative(); }

  std::ostream &Dump(std::ostream &) const;

  void AdjustExponent(int by) { exponent_ += by; }

  ValueWithRealFlags<Real> ToReal(
      bool isNegative = false, Rounding rounding = defaultRounding) const;

private:
  Word word_{0};
  int exponent_{0};
};

template<typename REAL>
std::ostream &IntermediateFloat<REAL>::Dump(std::ostream &o) const {
  return o << "0x" << word_.Hexadecimal() << " *2**" << exponent_;
}

template<typename REAL> REAL MakePowerOfTwo(int exponent) {
  auto raw{typename REAL::Word{exponent}.SHIFTL(REAL::significandBits)};
  if (!REAL::implicitMSB) {
    raw = raw.IBSET(REAL::significandBits - 1);
  }
  return REAL{raw};
}

template<typename REAL>
ValueWithRealFlags<REAL> IntermediateFloat<REAL>::ToReal(
    bool isNegative, Rounding rounding) const {
  if (word_.IsNegative()) {
    // word_ represents an unsigned quantity, so shift it down if the MSB is set
    IntermediateFloat shifted;
    Word sticky{word_.IAND(Word{1})};
    shifted.word_ = word_.SHIFTR(1).IOR(sticky);
    shifted.exponent_ = exponent_ + 1;
    return shifted.ToReal(isNegative, rounding);
  }
  ValueWithRealFlags<Real> result;
  if (isNegative) {
    result = Real::FromInteger(word_.Negate().value, rounding);
  } else {
    result = Real::FromInteger(word_, rounding);
  }
  int expo{exponent_};
  while (expo + Real::exponentBias < 1) {
    Real twoPow{MakePowerOfTwo<Real>(1)};  // min normal value
    result.value = result.value.Multiply(twoPow).AccumulateFlags(result.flags);
    expo += Real::exponentBias - 1;
  }
  while (expo + Real::exponentBias >= Real::maxExponent) {
    Real twoPow{MakePowerOfTwo<Real>(Real::maxExponent - 1)};
    result.value = result.value.Multiply(twoPow).AccumulateFlags(result.flags);
    expo += Real::maxExponent - 1 - Real::exponentBias;
  }
  Real twoPow{MakePowerOfTwo<Real>(expo + Real::exponentBias)};
  result.value = result.value.Multiply(twoPow).AccumulateFlags(result.flags);
  return result;
}

template<typename REAL, int LOG10RADIX>
ValueWithRealFlags<REAL> Decimal<REAL, LOG10RADIX>::ToReal(
    const char *&p, Rounding rounding) {
  while (*p == ' ') {
    ++p;
  }
  SetToZero();
  digitLimit_ = maxDigits - 1;
  isNegative_ = *p == '-';
  if (*p == '-' || *p == '+') {
    ++p;
  }

  while (*p == '0') {
    ++p;
  }
  bool decimalPoint{false};
  for (; *p != '\0'; ++p) {
    char c{*p};
    if (c == '.') {
      if (decimalPoint) {
        break;
      }
      decimalPoint = true;
    } else if (c < '0' || c > '9') {
      break;
    } else if (IsFull()) {
      if (!decimalPoint) {
        ++exponent_;
      }
    } else {
      int carry{MultiplyBy<10>(c - '0')};
      CHECK(carry == 0);
      if (decimalPoint) {
        --exponent_;
      }
    }
  }

  switch (*p) {
  case 'e':
  case 'E':
  case 'd':
  case 'D':
  case 'q':
  case 'Q':
    bool negExpo{*++p == '-'};
    if (negExpo || *p == '+') {
      ++p;
    }
    char *q;
    long expoVal{std::strtol(p, &q, 10)};
    p = const_cast<const char *>(q);
    if (negExpo) {
      exponent_ -= expoVal;
    } else {
      exponent_ += expoVal;
    }
  }

  if (IsZero()) {
    ValueWithRealFlags<Real> result;
    if (isNegative_) {
      result.value = Real{}.Negate();  // -0.0
    }
    return result;
  }

  // At this point, *this holds a multi-precision integer value in a radix
  // of a large power of ten.  Its radix point is defined to be to the right
  // of its digits, and "exponent_" is the power of ten by which it is to
  // be scaled.

  IntermediateFloat<Real> f;

  // Avoid needless rounding by scaling the value down by a multiple of two
  // to make it odd.
  Normalize();
  while (digits_ > 1 && (digit_[0] & 1) == 0) {
    f.AdjustExponent(1);
    DivideBy<2>();
  }
  if (digits_ == 1) {
    int shift{common::TrailingZeroBitCount(digit_[0])};
    f.AdjustExponent(shift);
    digit_[0] >>= shift;
  }
  Normalize();

  if (exponent_ < 0) {
    // If the number were to be represented in decimal and scaled,
    // there would be decimal digits to the right of the decimal point.
    // Align that decimal exponent to be a multiple of log10(radix) so
    // that the digits can be viewed as having an effective radix point.
    int align{-exponent_ % log10Radix};
    if (align > 0) {
      digitLimit_ = maxDigits;
      for (; align < log10Radix; ++align) {
        --exponent_;
        f.AdjustExponent(1);
        int carry{MultiplyBy<5>()};
        CHECK(carry == 0);
      }
    }
  }

  // Transfer the integer part, if any, to the floating-point
  // result.  The most significant digit can be moved directly;
  // lesser-order digits require transfer of carries.
  if (exponent_ >= -(digits_ - 1) * log10Radix) {
    f.SetTo(digit_[--digits_]);
    while (exponent_ > -digits_ * log10Radix) {
      digitLimit_ = digits_;
      int carry{MultiplyBy<10>()};
      f.MultiplyAndAdd(10, carry);
      --exponent_;
    }
  }

  // Shift the decimal point up above the remaining
  // digits.  If exponent_ remains negative after this
  // adjustment, additional digits will be created
  // in higher order positions as carries take place.
  // Once exponent_ is zero, the carries will then be
  // appended to the floating-point result.
  exponent_ += digits_ * log10Radix;

  // Convert the remaining fraction into bits of the
  // resulting floating-point value until we run out of
  // room.
  while (!f.IsFull() && !IsZero()) {
    f.AdjustExponent(-1);
    digitLimit_ = digits_;
    std::uint32_t carry = MultiplyBy<2>();
    RemoveLeastOrderZeroDigits();
    if (carry != 0) {
      if (exponent_ < 0) {
        exponent_ += log10Radix;
        digit_[digits_++] = carry;
        carry = 0;
      }
    }
    f.MultiplyAndAdd(2, carry);
  }

  return f.ToReal(isNegative_, rounding);
}

template<typename REAL, int LOG10RADIX>
std::string Decimal<REAL, LOG10RADIX>::ToString(int maxDigits) const {
  std::string result;
  if (isNegative_) {
    result += '-';
  }
  if (IsZero()) {
    result += "0.";
  } else {
    std::string d{std::to_string(digit_[digits_ - 1])};
    for (int j{digits_ - 2}; j >= 0; --j) {
      auto part{std::to_string(digit_[j])};
      unsigned zeroes = log10Radix - part.size();
      d += std::string(zeroes, '0');
      d += part;
    }
    int dn = d.size();
    result += d[0];
    result += '.';
    if (dn > maxDigits) {
      result += d.substr(1, maxDigits - 1);
    } else {
      result += d.substr(1);
    }
    while (result.back() == '0') {
      result.pop_back();
    }
    if (exponent_ + dn - 1 != 0) {
      result += 'e';
      result += std::to_string(exponent_ + dn - 1);
    }
  }
  return result;
}

template<typename REAL, int LOG10RADIX>
std::string Decimal<REAL, LOG10RADIX>::ToMinimalString(
    const Real &x, Rounding rounding) const {
  for (int digits{1};; ++digits) {
    std::string result{ToString(digits)};
    const char *p{result.data()};
    ValueWithRealFlags<Real> readBack{Decimal{}.ToReal(p, rounding)};
    if (x.Compare(readBack.value) == Relation::Equal) {
      return result;
    }
  }
}

template class Decimal<Real<Integer<16>, 11>>;
template class Decimal<Real<Integer<16>, 8>>;
template class Decimal<Real<Integer<32>, 24>>;
template class Decimal<Real<Integer<64>, 53>>;
template class Decimal<Real<Integer<80>, 64, false>>;
template class Decimal<Real<Integer<128>, 112>>;
}
