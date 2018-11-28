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

#include "decimal.h"
#include "integer.h"
#include "leading-zero-bit-count.h"
#include "../common/bit-population-count.h"
#include "../common/idioms.h"

namespace Fortran::evaluate::value {

static constexpr std::ostream *debug{nullptr};

template<typename REAL>
std::ostream &Decimal<REAL>::Dump(std::ostream &o) const {
  if (isNegative_) {
    o << '-';
  }
  for (int j{digits_ - 1}; j >= 0; --j) {
    o << ' ' << digit_[j];
  }
  return o << " e" << exponent_ << '\n';
}

template<typename REAL> void Decimal<REAL>::FromReal(const REAL &x) {
  if (x.IsNegative()) {
    FromReal(x.Negate());
    isNegative_ = true;
    return;
  }
  if (debug) {
    *debug << "FromReal(" << x.DumpHexadecimal() << ") bits " << Real::bits
           << '\n';
  }
  if (x.IsZero()) {
    return;
  }
  int twoPow{x.UnbiasedExponent()};
  twoPow -= Real::bits - 1;
  if (debug) {
    *debug << "initial twoPow " << twoPow << '\n';
  }
  int lshift{x.exponentBits};
  if (twoPow <= -lshift) {
    twoPow += lshift;
    lshift = 0;
  } else if (twoPow < 0) {
    lshift += twoPow;
    twoPow = 0;
  }
  if (debug) {
    *debug << "second twoPow " << twoPow << ", lshift " << lshift << '\n';
  }
  using Word = typename Real::Word;
  Word word{Word::ConvertUnsigned(x.GetFraction()).value};
  SetTo(word.SHIFTL(lshift));
  if (debug) {
    Dump(*debug);
  }

  for (; twoPow > 0 && IsDivisibleBy<5>(); --twoPow) {
    DivideBy<5>();
    Normalize();
    ++exponent_;
    if (debug) {
      Dump(*debug << "/5 ");
    }
  }

  // Scale by factors of 8, then by 2.
  static constexpr int log2FastForward{3};
  static constexpr int fastForward{1 << log2FastForward};
  for (; twoPow >= log2FastForward; twoPow -= log2FastForward) {
    MultiplyBy<fastForward>();
    if (debug) {
      Dump(*debug << '*' << fastForward << ' ');
    }
  }
  for (; twoPow > 0; --twoPow) {
    MultiplyBy<2>();
    if (debug) {
      Dump(*debug << "*2 ");
    }
  }
  for (; twoPow <= -log2FastForward && IsDivisibleBy<fastForward>();
       twoPow += log2FastForward) {
    DivideBy<fastForward>();
    Normalize();
    if (debug) {
      Dump(*debug << '/' << fastForward << ' ');
    }
  }
  for (; twoPow < 0 && IsDivisibleBy<2>(); ++twoPow) {
    DivideBy<2>();
    Normalize();
    if (debug) {
      Dump(*debug << "/2 ");
    }
  }
  for (; twoPow < 0; ++twoPow) {
    MultiplyBy<5>();
    --exponent_;
    if (debug) {
      Dump(*debug << "*5 ");
    }
  }
}

// Represents an unrounded binary floating-point
// value with an unbiased (signed) binary exponent.
template<typename REAL> class IntermediateFloat {
public:
  using Real = REAL;
  using Word = typename Real::Word;

  void SetTo(std::uint64_t n) {
    if constexpr (Word::bits >= 8 * sizeof n) {
      word_ = n;
    } else {
      int shift{64 - LeadingZeroBitCount(n) - Word::bits};
      if (shift <= 0) {
        word_ = n;
      } else {
        word_ = n >> shift;
        exponent_ += shift;
        bool sticky{n << (64 - shift) != 0};
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
      product.lower = product.lower.DSHIFTR(product.upper, 1);
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
      bool isNegative = false, Rounding rounding = Rounding::TiesToEven) const;

private:
  Word word_{0};
  int exponent_{0};
};

template<typename REAL>
std::ostream &IntermediateFloat<REAL>::Dump(std::ostream &o) const {
  return o << "0x" << word_.Hexadecimal() << " *2**" << exponent_;
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
    if (debug) {
      shifted.Dump(*debug << "IntermediateFloat::ToReal: shifted: ") << '\n';
    }
    return shifted.ToReal(isNegative, rounding);
  }
  ValueWithRealFlags<Real> result;
  if (isNegative) {
    result = Real::FromInteger(word_.Negate().value, rounding);
  } else {
    result = Real::FromInteger(word_, rounding);
  }
  if (debug) {
    *debug << "IntermediateFloat::ToReal: after FromInteger: "
           << result.value.DumpHexadecimal() << " * 2**" << exponent_ << '\n';
  }
  int expo{exponent_};
  while (expo + Real::exponentBias < 1) {
    Real twoPow{Word{1}.SHIFTL(Real::significandBits)};  // min normal value
    result.value = result.value.Multiply(twoPow).AccumulateFlags(result.flags);
    expo += Real::exponentBias - 1;
    if (debug) {
      *debug << "IntermediateFloat::ToReal: reduced: "
             << result.value.DumpHexadecimal() << " * 2**" << expo << '\n';
    }
  }
  while (expo + Real::exponentBias >= Real::maxExponent) {
    Real twoPow{Word{Real::maxExponent - 1}.SHIFTL(Real::significandBits)};
    result.value = result.value.Multiply(twoPow).AccumulateFlags(result.flags);
    expo += Real::maxExponent - 1 - Real::exponentBias;
    if (debug) {
      *debug << "IntermediateFloat::ToReal: magnified: "
             << result.value.DumpHexadecimal() << " * 2**" << expo << '\n';
    }
  }
  Real twoPow{Word{expo + Real::exponentBias}.SHIFTL(Real::significandBits)};
  if (debug) {
    *debug << "IntermediateFloat::ToReal: twoPow: " << twoPow.DumpHexadecimal()
           << '\n';
  }
  result.value = result.value.Multiply(twoPow).AccumulateFlags(result.flags);
  return result;
}

template<typename REAL>
ValueWithRealFlags<REAL> Decimal<REAL>::ToReal(
    const char *&p, Rounding rounding) {
  if (debug) {
    *debug << "ToReal('" << p << "')\n";
  }
  while (*p == ' ') {
    ++p;
  }
  SetToZero();
  digitLimit_ = maxDigits;
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
    if (debug) Dump(*debug << "ToReal in loop, p at '" << p << "'\n'");
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

  if (debug) {
    Dump(*debug << "ToReal start, p at '" << p << "'\n");
  }
  if (IsZero()) {
    ValueWithRealFlags<Real> result;
    if (isNegative_) {
      result.value = Real{}.Negate();  // -0.0
    }
    return result;
  }

  // At this point, *this holds a multi-precision base-quintillion
  // integer with its radix point to the right of its digits,
  // and "exponent_" is the power of ten by which it is to be scaled.

  IntermediateFloat<Real> f;

  // Avoid needless rounding by scaling the value down by a multiple of two
  // to make it odd.
  while (digits_ > 0 && (digit_[0] & 1) == 0) {
    f.AdjustExponent(1);
    DivideBy<2>();
  }
  Normalize();
  if (debug) {
    Dump(f.Dump(*debug << "made odd ") << '\n');
  }

  if (exponent_ < 0) {
    // If the number were to be represented in decimal and scaled,
    // there would be decimal digits to the right of the decimal point.
    // Align that decimal exponent to be a multiple of log10(quintillion)
    // so that the base-quintillion digits can be viewed as having an
    // effective radix point that's meaningful.
    int align{-exponent_ % log10Quintillion};
    if (align > 0) {
      for (; align < log10Quintillion; ++align) {
        --exponent_;
        MultiplyBy<5>();
        f.AdjustExponent(1);
      }
      if (debug) {
        Dump(f.Dump(*debug << "aligned ") << '\n');
      }
    }
  }

  // Transfer the integer part, if any, to the floating-point
  // result.  The most significant digit can be moved directly;
  // lesser-order digits require transfer of carries.
  if (exponent_ >= -(digits_ - 1) * log10Quintillion) {
    if (debug) {
      Dump(f.Dump(*debug << "converting integer part ") << '\n');
    }
    f.SetTo(digit_[--digits_]);
    if (debug) {
      Dump(f.Dump(*debug << "after top digit ") << '\n');
    }
    while (exponent_ > -digits_ * log10Quintillion) {
      digitLimit_ = digits_;
      int carry{MultiplyBy<10>()};
      f.MultiplyAndAdd(10, carry);
      --exponent_;
      if (debug) {
        Dump(f.Dump(*debug << "foor of loop after carry " << carry << ": ")
            << '\n');
      }
    }
  }

  // Shift the decimal point up above the remaining
  // digits.  If exponent_ remains negative after this
  // adjustment, additional digits will be created
  // in higher order positions as carries take place.
  // Once exponent_ is zero, the carries will then be
  // appended to the floating-point result.
  exponent_ += digits_ * log10Quintillion;
  if (debug) {
    Dump(f.Dump(*debug << "after converting integer part ") << '\n');
  }

  // Convert the remaining fraction into bits of the
  // resulting floating-point value until we run out of
  // room.
  while (!f.IsFull() && !IsZero()) {
    if (debug) {
      Dump(f.Dump(*debug << "step ") << '\n');
    }
    f.AdjustExponent(-1);
    digitLimit_ = digits_;
    std::uint32_t carry = MultiplyBy<2>();
    RemoveLeastOrderZeroDigits();
    if (carry != 0) {
      if (exponent_ < 0) {
        exponent_ += log10Quintillion;
        digit_[digits_++] = carry;
        carry = 0;
      }
    }
    f.MultiplyAndAdd(2, carry);
  }
  if (debug) {
    Dump(f.Dump(*debug << "after converting fraction ") << '\n');
  }

  return f.ToReal(isNegative_, rounding);
}

template<typename REAL>
std::string Decimal<REAL>::ToString(int maxDigits) const {
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
      unsigned zeroes = log10Quintillion - part.size();
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

template class Decimal<Real<Integer<16>, 11>>;
template class Decimal<Real<Integer<32>, 24>>;
template class Decimal<Real<Integer<64>, 53>>;
template class Decimal<Real<Integer<80>, 64, false>>;
template class Decimal<Real<Integer<128>, 112>>;
}
