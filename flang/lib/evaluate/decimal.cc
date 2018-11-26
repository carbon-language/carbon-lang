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

namespace Fortran::evaluate::value {

static std::ostream *debug{nullptr};

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

template<typename REAL> void Decimal<REAL>::FromReal(const Real &x) {
  if (x.IsNegative()) {
    FromReal(x.Negate());
    isNegative_ = true;
    return;
  }
  if (debug) {
    *debug << "FromReal(" << x.DumpHexadecimal() << ")\n";
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
  SetTo(x.GetFraction().SHIFTL(lshift));

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

  void SetTo(Word n) {
    word_ = n;
    exponent_ = 0;
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
    }
    word_ = product.lower;
    if (sticky) {
      word_ = word_.IOR(word_.MASKR(1));
    }
  }

  bool IsZero() const { return word_.IsZero(); }

  bool IsFull() const { return word_.IsNegative(); }

  std::ostream &Dump(std::ostream &) const;

  void AdjustExponent(int by = -1) { exponent_ += by; }

  Real ToReal(bool isNegative = false) const;

private:
  Word word_{0};
  int exponent_{0};
};

template<typename REAL>
std::ostream &IntermediateFloat<REAL>::Dump(std::ostream &o) const {
  return o << "0x" << word_.Hexadecimal() << " *2**" << exponent_;
}

template<typename REAL>
REAL IntermediateFloat<REAL>::ToReal(bool isNegative) const {
  if (word_.IsNegative()) {
    IntermediateFloat shifted;
    shifted.word_ =
        word_.SHIFTR(1).IOR(word_.IAND(word_.MASKR(1)));  // sticky bit
    shifted.exponent_ = exponent_ + 1;
    return shifted.ToReal(false);
  }
  Real result = Real::FromInteger(word_).value;
  int expo{exponent_};
  while (expo + Real::exponentBias < 1) {
    Real twoPow{Word{1}.SHIFTL(Real::significandBits)};
    result = result.Multiply(twoPow).value;
    expo += Real::exponentBias - 1;
  }
  while (expo + Real::exponentBias >= Real::maxExponent) {
    Real twoPow{Word{Real::maxExponent - 1}.SHIFTL(Real::significandBits)};
    result = result.Multiply(twoPow).value;
    expo += Real::maxExponent - 1 - Real::exponentBias;
  }
  Real twoPow{Word{expo + Real::exponentBias}.SHIFTL(Real::significandBits)};
  if (isNegative) {
    twoPow = twoPow.Negate();
  }
  return result.Multiply(twoPow).value;
}

template<typename REAL> REAL Decimal<REAL>::ToReal(const char *&p) {
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
      MultiplyBy<10>(c - '0');
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
    ++p;
    bool negExpo{*p == '-'};
    if (*p == '+' || *p == '-') {
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
    Dump(*debug << "ToReal start: ");
  }
  if (IsZero()) {
    return Real{}.Negate();  // -0.0
  }

  if (exponent_ < 0) {
    // There are decimal digits to the right of the decimal point.
    // Align that decimal point on a quintillion radix point by
    // appending zero-valued decimal digits.
    int align{-exponent_ % log10Quintillion};
    if (align > 0) {
      for (; align < log10Quintillion; ++align) {
        --exponent_;
        MultiplyBy<10>();
      }
      if (debug) {
        Dump(*debug << "aligned\n");
      }
    }
  }

  IntermediateFloat<Real> f;

  // Transfer the integer part, if any, to the floating-point
  // result.  The most significant digit can be moved directly;
  // lesser-order digits require transfer of carries.
  if (exponent_ >= -(digits_ - 1) * log10Quintillion) {
    f.SetTo(digit_[--digits_]);
    while (exponent_ > -digits_ * log10Quintillion) {
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
  exponent_ += digits_ * log10Quintillion;
  if (debug) {
    *debug << "after converting integer part ";
    f.Dump(*debug) << '\n';
    Dump(*debug);
  }

  // Convert the remaining fraction into bits of the
  // resulting floating-point value until we run out of
  // room.
  while (!f.IsFull() && !IsZero()) {
    if (debug) {
      f.Dump(*debug << "step: ") << '\n';
      Dump(*debug);
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
    f.Dump(*debug << "after converting fraction ") << '\n';
    Dump(*debug);
  }

  return f.ToReal(isNegative_);
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
