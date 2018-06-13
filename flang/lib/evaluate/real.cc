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

#include "real.h"

namespace Fortran::evaluate::value {

template<typename W, int P, bool IM>
Relation Real<W, P, IM>::Compare(const Real &y) const {
  if (IsNotANumber() || y.IsNotANumber()) {  // NaN vs x, x vs NaN
    return Relation::Unordered;
  } else if (IsInfinite()) {
    if (y.IsInfinite()) {
      if (IsNegative()) {  // -Inf vs +/-Inf
        return y.IsNegative() ? Relation::Equal : Relation::Less;
      } else {  // +Inf vs +/-Inf
        return y.IsNegative() ? Relation::Greater : Relation::Equal;
      }
    } else {  // +/-Inf vs finite
      return IsNegative() ? Relation::Less : Relation::Greater;
    }
  } else if (y.IsInfinite()) {  // finite vs +/-Inf
    return y.IsNegative() ? Relation::Greater : Relation::Less;
  } else {  // two finite numbers
    bool isNegative{IsNegative()};
    if (isNegative != y.IsNegative()) {
      if (word_.IOR(y.word_).IBCLR(bits - 1).IsZero()) {
        return Relation::Equal;  // +/-0.0 == -/+0.0
      } else {
        return isNegative ? Relation::Less : Relation::Greater;
      }
    } else {
      // same sign
      Ordering order{CompareUnsigned(Exponent(), y.Exponent())};
      if (order == Ordering::Equal) {
        order = GetSignificand().CompareUnsigned(y.GetSignificand());
      }
      if (isNegative) {
        order = Reverse(order);
      }
      return RelationFromOrdering(order);
    }
  }
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Add(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value.word_ = NaNWord();  // NaN + x -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
    return result;
  }
  bool isNegative{IsNegative()};
  bool yIsNegative{y.IsNegative()};
  if (IsInfinite()) {
    if (y.IsInfinite()) {
      if (isNegative == yIsNegative) {
        result.value = *this;  // +/-Inf + +/-Inf -> +/-Inf
      } else {
        result.value.word_ = NaNWord();  // +/-Inf + -/+Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      }
    } else {
      result.value = *this;  // +/-Inf + x -> +/-Inf
    }
    return result;
  }
  if (y.IsInfinite()) {
    result.value = y;  // x + +/-Inf -> +/-Inf
    return result;
  }
  std::uint64_t exponent{Exponent()};
  std::uint64_t yExponent{y.Exponent()};
  if (exponent < yExponent) {
    // y is larger in magnitude; simplify by reversing operands
    return y.Add(*this, rounding);
  }
  if (exponent == yExponent && isNegative != yIsNegative) {
    Ordering order{GetSignificand().CompareUnsigned(y.GetSignificand())};
    if (order == Ordering::Less) {
      // Same exponent, opposite signs, and y is larger in magnitude
      return y.Add(*this, rounding);
    }
    if (order == Ordering::Equal) {
      // x + (-x) -> +0.0 unless rounding is directed downwards
      if (rounding == Rounding::Down) {
        result.value.word_ = result.value.word_.IBSET(bits - 1);  // -0.0
      }
      return result;
    }
  }
  // Our exponent is greater than y's, or the exponents match and y is not
  // of the opposite sign and greater magnitude.  So (x+y) will have the
  // same sign as x.
  Fraction yFraction{y.GetFraction()};
  Fraction fraction{GetFraction()};
  int rshift = exponent - yExponent;
  if (exponent > 0 && yExponent == 0) {
    --rshift;  // correct overshift when only y is denormal
  }
  RoundingBits roundingBits{yFraction, rshift};
  yFraction = yFraction.SHIFTR(rshift);
  bool carry{false};
  if (isNegative != yIsNegative) {
    // Opposite signs: subtract via addition of two's complement of y and
    // the rounding bits.
    yFraction = yFraction.NOT();
    carry = roundingBits.Negate();
  }
  auto sum = fraction.AddUnsigned(yFraction, carry);
  fraction = sum.value;
  if (isNegative == yIsNegative && sum.carry) {
    roundingBits.ShiftRight(sum.value.BTEST(0));
    fraction = fraction.SHIFTR(1).IBSET(fraction.bits - 1);
    ++exponent;
  }
  NormalizeAndRound(
      result, isNegative, exponent, fraction, rounding, roundingBits);
  return result;
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Multiply(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value.word_ = NaNWord();  // NaN * x -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool isNegative{IsNegative() != y.IsNegative()};
    if (IsInfinite() || y.IsInfinite()) {
      if (IsZero() || y.IsZero()) {
        result.value.word_ = NaNWord();  // 0 * Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else {
        result.value.word_ = InfinityWord(isNegative);
      }
    } else {
      auto product = GetFraction().MultiplyUnsigned(y.GetFraction());
      std::int64_t exponent{CombineExponents(y, false)};
      if (exponent < 1) {
        int rshift = 1 - exponent;
        exponent = 1;
        bool sticky{false};
        if (rshift >= product.upper.bits + product.lower.bits) {
          sticky = !product.lower.IsZero() || !product.upper.IsZero();
        } else if (rshift >= product.lower.bits) {
          sticky = !product.lower.IsZero() ||
              !product.upper
                   .IAND(product.upper.MASKR(rshift - product.lower.bits))
                   .IsZero();
        } else {
          sticky = !product.lower.IAND(product.lower.MASKR(rshift)).IsZero();
        }
        product.lower = product.lower.DSHIFTR(product.upper, rshift);
        product.upper = product.upper.SHIFTR(rshift);
        if (sticky) {
          product.lower = product.lower.IBSET(0);
        }
      }
      int leadz{product.upper.LEADZ()};
      if (leadz >= product.upper.bits) {
        leadz += product.lower.LEADZ();
      }
      int lshift{leadz};
      if (lshift > exponent - 1) {
        lshift = exponent - 1;
      }
      exponent -= lshift;
      product.upper = product.upper.DSHIFTL(product.lower, lshift);
      product.lower = product.lower.SHIFTL(lshift);
      RoundingBits roundingBits{product.lower, product.lower.bits};
      NormalizeAndRound(
          result, isNegative, exponent, product.upper, rounding, roundingBits);
    }
  }
  return result;
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Divide(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value.word_ = NaNWord();  // NaN / x -> NaN, x / NaN -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool isNegative{IsNegative() != y.IsNegative()};
    if (IsInfinite()) {
      if (y.IsInfinite()) {
        result.value.word_ = NaNWord();  // Inf/Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else {  // Inf/x -> Inf,  Inf/0 -> Inf
        result.value.word_ = InfinityWord(isNegative);
      }
    } else if (y.IsZero()) {
      if (IsZero()) {  // 0/0 -> NaN
        result.value.word_ = NaNWord();
        result.flags.set(RealFlag::InvalidArgument);
      } else {  // x/0 -> Inf, Inf/0 -> Inf
        result.value.word_ = InfinityWord(isNegative);
        result.flags.set(RealFlag::DivideByZero);
      }
    } else if (IsZero() || y.IsInfinite()) {  // 0/x, x/Inf -> 0
      if (isNegative) {
        result.value.word_ = result.value.word_.IBSET(bits - 1);
      }
    } else {
      // dividend and divisor are both finite and nonzero numbers
      Fraction top{GetFraction()}, divisor{y.GetFraction()};
      std::int64_t exponent{CombineExponents(y, true)};
      Fraction quotient;
      bool msb{false};
      if (!top.BTEST(top.bits - 1) || !divisor.BTEST(divisor.bits - 1)) {
        // One or two denormals
        int topLshift{top.LEADZ()};
        top = top.SHIFTL(topLshift);
        int divisorLshift{divisor.LEADZ()};
        divisor = divisor.SHIFTL(divisorLshift);
        exponent += divisorLshift - topLshift;
      }
      for (int j{1}; j <= quotient.bits; ++j) {
        if (NextQuotientBit(top, msb, divisor)) {
          quotient = quotient.IBSET(quotient.bits - j);
        }
      }
      bool guard{NextQuotientBit(top, msb, divisor)};
      bool round{NextQuotientBit(top, msb, divisor)};
      bool sticky{msb || !top.IsZero()};
      RoundingBits roundingBits{guard, round, sticky};
      if (exponent < 1) {
        std::int64_t rshift{1 - exponent};
        for (; rshift > 0; --rshift) {
          roundingBits.ShiftRight(quotient.BTEST(0));
          quotient = quotient.SHIFTR(1);
        }
        exponent = 1;
      }
      NormalizeAndRound(
          result, isNegative, exponent, quotient, rounding, roundingBits);
    }
  }
  return result;
}

template<typename W, int P, bool IM>
RealFlags Real<W, P, IM>::Normalize(bool negative, std::uint64_t exponent,
    const Fraction &fraction, Rounding rounding, RoundingBits *roundingBits) {
  std::uint64_t lshift = fraction.LEADZ();
  if (lshift < fraction.bits) {
    if (lshift < exponent) {
      exponent -= lshift;
    } else if (exponent > 0) {
      lshift = exponent - 1;
      exponent = 0;
    } else if (lshift == 0) {
      exponent = 1;
    } else {
      lshift = 0;
    }
  }
  if (exponent >= maxExponent) {
    if (rounding == Rounding::TiesToEven ||
        rounding == Rounding::TiesAwayFromZero ||
        (rounding == Rounding::Up && !negative) ||
        (rounding == Rounding::Down && negative)) {
      word_ = Word{maxExponent}.SHIFTL(significandBits);  // Inf
    } else {
      // directed rounding: round to largest finite value rather than infinity
      // (x86 does this, not sure whether it's standard or not)
      word_ = Word{word_.MASKR(word_.bits - 1)}.IBCLR(significandBits);
    }
    if (negative) {
      word_ = word_.IBSET(bits - 1);
    }
    RealFlags flags{RealFlag::Overflow};
    if (!fraction.IsZero()) {
      flags.set(RealFlag::Inexact);
    }
    return flags;
  }
  if (lshift >= fraction.bits) {
    // +/-0.0
    word_ = Word{};
    exponent = 0;
  } else {
    word_ = Word::ConvertUnsigned(fraction).value;
    if (lshift > 0) {
      word_ = word_.SHIFTL(lshift);
      if (roundingBits != nullptr) {
        for (; lshift > 0; --lshift) {
          if (roundingBits->ShiftLeft()) {
            word_ = word_.IBSET(lshift - 1);
          }
        }
      }
    }
  }
  if (implicitMSB) {
    word_ = word_.IBCLR(significandBits);
  }
  word_ = word_.IOR(Word{exponent}.SHIFTL(significandBits));
  if (negative) {
    word_ = word_.IBSET(bits - 1);
  }
  return {};
}

template<typename W, int P, bool IM>
RealFlags Real<W, P, IM>::Round(Rounding rounding, const RoundingBits &bits) {
  std::uint64_t exponent{Exponent()};
  RealFlags flags;
  if (!bits.Zero()) {
    flags.set(RealFlag::Inexact);
  }
  if (exponent < maxExponent &&
      bits.MustRound(rounding, IsNegative(), word_.BTEST(0) /* is odd */)) {
    typename Fraction::ValueWithCarry sum{
        GetFraction().AddUnsigned(Fraction{}, true)};
    if (sum.carry) {
      // The fraction was all ones before rounding; sum.value is now zero
      if (++exponent < maxExponent) {
        sum.value = sum.value.IBSET(precision - 1);
      } else {
        // rounded away to an infinity
        flags.set(RealFlag::Overflow);
      }
    }
    flags |= Normalize(IsNegative(), exponent, sum.value);
  }
  return flags;
}

template<typename W, int P, bool IM>
void Real<W, P, IM>::NormalizeAndRound(ValueWithRealFlags<Real> &result,
    bool isNegative, std::uint64_t exponent, const Fraction &fraction,
    Rounding rounding, RoundingBits roundingBits) {
  result.flags |= result.value.Normalize(
      isNegative, exponent, fraction, rounding, &roundingBits);
  std::uint64_t normalizedExponent{result.value.Exponent()};
  result.flags |= result.value.Round(rounding, roundingBits);
  if (result.flags.test(RealFlag::Inexact) && normalizedExponent == 0) {
    result.flags.set(RealFlag::Underflow);
  }
}

template class Real<Integer<16>, 11>;
template class Real<Integer<32>, 24>;
template class Real<Integer<64>, 53>;
template class Real<Integer<80>, 64, false>;
template class Real<Integer<128>, 112>;

}  // namespace Fortran::evaluate::value
