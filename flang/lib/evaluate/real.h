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

#ifndef FORTRAN_EVALUATE_REAL_H_
#define FORTRAN_EVALUATE_REAL_H_

#include "common.h"
#include "integer.h"
#include <cinttypes>
#include <limits>

namespace Fortran::evaluate {

// Models IEEE-754 floating-point numbers.  The first argument to this
// class template must be (or look like) an instance of Integer.
template<typename WORD, int PRECISION, bool IMPLICIT_MSB = true> class Real {
public:
  using Word = WORD;
  static constexpr int bits{Word::bits};
  static constexpr int precision{PRECISION};
  static constexpr bool implicitMSB{IMPLICIT_MSB};
  static constexpr int significandBits{precision - implicitMSB};
  static constexpr int exponentBits{bits - significandBits - 1 /*sign*/};
  static_assert(precision > 0);
  static_assert(exponentBits > 1);
  static constexpr std::uint64_t maxExponent{(1 << exponentBits) - 1};
  static constexpr std::uint64_t exponentBias{maxExponent / 2};

  constexpr Real() {}  // +0.0
  constexpr Real(const Real &) = default;
  constexpr Real(const Word &bits) : word_{bits} {}
  constexpr Real &operator=(const Real &) = default;

  template<typename INT>
  static constexpr ValueWithRealFlags<Real> ConvertSigned(
      const INT &n, Rounding rounding = Rounding::TiesToEven) {
    bool isNegative{n.IsNegative()};
    INT absN{n};
    if (isNegative) {
      absN = n.Negate().value;  // overflow is safe to ignore
    }
    int leadz{absN.LEADZ()};
    if (leadz >= absN.bits) {
      return {};  // all bits zero -> +0.0
    }
    ValueWithRealFlags<Real> result;
    std::uint64_t exponent{exponentBias + absN.bits - leadz - 1};
    int bitsNeeded{absN.bits - (leadz + implicitMSB)};
    int bitsLost{bitsNeeded - significandBits};
    if (bitsLost <= 0) {
      Fraction fraction{Fraction::Convert(absN).value};
      result.flags |= result.value.Normalize(
          isNegative, exponent, fraction.SHIFTL(-bitsLost));
    } else {
      Fraction fraction{Fraction::Convert(absN.SHIFTR(bitsLost)).value};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
      RoundingBits roundingBits{absN, bitsLost};
      result.flags |= result.value.Round(rounding, roundingBits);
    }
    return result;
  }

  // TODO conversion from (or to?) (other) real types
  // TODO AINT/ANINT, CEILING, FLOOR, DIM, MAX, MIN, DPROD, FRACTION
  // HUGE, INT/NINT, MAXEXPONENT, MINEXPONENT, NEAREST, OUT_OF_RANGE,
  // PRECISION, HUGE, TINY, RRSPACING/SPACING, SCALE, SET_EXPONENT, SIGN

  constexpr Word RawBits() const { return word_; }
  constexpr std::uint64_t Exponent() const {
    return word_.IBITS(significandBits, exponentBits).ToUInt64();
  }
  constexpr bool IsNegative() const {
    return !IsNotANumber() && word_.BTEST(bits - 1);
  }
  constexpr bool IsNotANumber() const {
    return Exponent() == maxExponent && !GetSignificand().IsZero();
  }
  constexpr bool IsQuietNaN() const {
    return Exponent() == maxExponent &&
        GetSignificand().BTEST(significandBits - 1);
  }
  constexpr bool IsSignalingNaN() const {
    return IsNotANumber() && !GetSignificand().BTEST(significandBits - 1);
  }
  constexpr bool IsInfinite() const {
    return Exponent() == maxExponent && GetSignificand().IsZero();
  }
  constexpr bool IsZero() const {
    return Exponent() == 0 && GetSignificand().IsZero();
  }
  constexpr bool IsDenormal() const {
    return Exponent() == 0 && !GetSignificand().IsZero();
  }

  constexpr Real ABS() const {  // non-arithmetic, no flags returned
    Real result;
    result.word_ = word_.IBCLR(bits - 1);
    return result;
  }

  constexpr Real Negate() const {
    Real result;
    result.word_ = word_.IEOR(word_.MASKL(1));
    return result;
  }

  constexpr DefaultIntrinsicInteger EXPONENT() const {
    std::uint64_t exponent{Exponent()};
    if (exponent == maxExponent) {
      return DefaultIntrinsicInteger::HUGE();
    } else {
      return {static_cast<std::int64_t>(exponent - exponentBias)};
    }
  }

  static constexpr Real EPSILON() {
    Real epsilon;
    epsilon.Normalize(false, exponentBias - precision, Fraction::MASKL(1));
    return epsilon;
  }

  template<typename INT> constexpr ValueWithRealFlags<INT> ToInteger() const {
    bool isNegative{IsNegative()};
    std::uint64_t exponent{Exponent()};
    Fraction fraction{GetFraction()};
    ValueWithRealFlags<INT> result;
    if (exponent == maxExponent && !fraction.IsZero()) {  // NaN
      result.flags.set(RealFlag::InvalidArgument);
      result.value = result.value.HUGE();
    } else if (exponent >= maxExponent ||  // +/-Inf
        exponent >= exponentBias + result.value.bits) {  // too big
      if (isNegative) {
        result.value = result.value.MASKL(1);
      } else {
        result.value = result.value.HUGE();
      }
      result.flags.set(RealFlag::Overflow);
    } else if (exponent < exponentBias) {  // |x| < 1.0 -> 0
      if (!fraction.IsZero()) {
        result.flags.set(RealFlag::Underflow);
        result.flags.set(RealFlag::Inexact);
      }
    } else {
      // finite number |x| >= 1.0
      constexpr std::uint64_t noShiftExponent{exponentBias + precision - 1};
      if (exponent < noShiftExponent) {
        int rshift = noShiftExponent - exponent;
        if (!fraction.IBITS(0, rshift).IsZero()) {
          result.flags.set(RealFlag::Inexact);
        }
        auto truncated = result.value.Convert(fraction.SHIFTR(rshift));
        if (truncated.overflow) {
          result.flags.set(RealFlag::Overflow);
        } else {
          result.value = truncated.value;
        }
      } else {
        int lshift = exponent - noShiftExponent;
        if (lshift + precision >= result.value.bits) {
          result.flags.set(RealFlag::Overflow);
        } else {
          result.value = result.value.Convert(fraction).value.SHIFTL(lshift);
        }
      }
      if (result.flags.test(RealFlag::Overflow)) {
        result.value = result.value.HUGE();
      } else if (isNegative) {
        auto negated = result.value.Negate();
        if (negated.overflow) {
          result.flags.set(RealFlag::Overflow);
          result.value = result.value.HUGE();
        } else {
          result.value = negated.value;
        }
      }
    }
    return result;
  }

  constexpr Relation Compare(const Real &y) const {
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

  constexpr ValueWithRealFlags<Real> Add(
      const Real &y, Rounding rounding = Rounding::TiesToEven) const {
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
          result.value.Normalize(true, 0, Fraction{}, rounding);  // -0.0
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
    result.flags |= result.value.Normalize(
        isNegative, exponent, fraction, rounding, &roundingBits);
    result.flags |= result.value.Round(rounding, roundingBits);
    return result;
  }

  constexpr ValueWithRealFlags<Real> Subtract(
      const Real &y, Rounding rounding = Rounding::TiesToEven) const {
    return Add(y.Negate(), rounding);
  }

  constexpr ValueWithRealFlags<Real> Multiply(
      const Real &y, Rounding rounding = Rounding::TiesToEven) const {
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
          result.value.Normalize(isNegative, maxExponent, Fraction{});
        }
      } else {
        auto product = GetFraction().MultiplyUnsigned(y.GetFraction());
        std::int64_t exponent = Exponent(), yExponent = y.Exponent();
        // A zero exponent field value has the same weight as 1.
        exponent += !exponent;
        yExponent += !yExponent;
        exponent += yExponent;
        exponent -= exponentBias;
        ++exponent;
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
        result.flags |= result.value.Normalize(
            isNegative, exponent, product.upper, rounding, &roundingBits);
        result.flags |= result.value.Round(rounding, roundingBits);
        if (result.flags.test(RealFlag::Inexact) &&
            result.value.Exponent() == 0) {
          result.flags.set(RealFlag::Underflow);
        }
      }
    }
    return result;
  }

  constexpr ValueWithRealFlags<Real> Divide(
      const Real &y, Rounding rounding = Rounding::TiesToEven) const {
    ValueWithRealFlags<Real> result;
    if (IsNotANumber() || y.IsNotANumber()) {
      result.value.word_ = NaNWord();  // NaN / x -> NaN, x / NaN -> NaN
      result.flags.set(RealFlag::InvalidArgument);
    } else {
      bool isNegative{IsNegative() != y.IsNegative()};
      if (IsInfinite()) {
        if (y.IsInfinite() || y.IsZero()) {
          result.value.word_ = NaNWord();  // Inf/Inf -> NaN, Inf/0 -> Nan
          result.flags.set(RealFlag::InvalidArgument);
        } else {
          result.value.Normalize(isNegative, maxExponent, Fraction{});
        }
      } else if (y.IsInfinite()) {
        result.value.word_ = NaNWord();  // x/Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else {
        auto qr = GetFraction().DivideUnsigned(y.GetFraction());
        if (qr.divisionByZero) {
          result.value.Normalize(isNegative, maxExponent, Fraction{});
          result.flags.set(RealFlag::DivideByZero);
        } else {
          // To round, double the remainder and compare it to the divisor.
          auto doubled = qr.remainder.AddUnsigned(qr.remainder);
          Ordering drcmp{doubled.value.CompareUnsigned(y.GetFraction())};
          RoundingBits roundingBits{
              drcmp != Ordering::Less, drcmp != Ordering::Equal};
          std::uint64_t exponent{Exponent() - y.Exponent() + exponentBias};
          result.flags |=
              result.value.Normalize(isNegative, exponent, qr.quotient);
          result.flags |= result.value.Round(rounding, roundingBits);
        }
      }
    }
    return result;
  }

private:
  using Fraction = Integer<precision>;  // all bits made explicit
  using Significand = Integer<significandBits>;  // no implicit bit

  class RoundingBits {
  public:
    constexpr RoundingBits(
        bool guard = false, bool round = false, bool sticky = false)
      : guard_{guard}, round_{round}, sticky_{sticky} {}

    template<typename FRACTION>
    constexpr RoundingBits(const FRACTION &fraction, int rshift) {
      if (rshift > 0 && rshift < fraction.bits + 1) {
        guard_ = fraction.BTEST(rshift - 1);
      }
      if (rshift > 1 && rshift < fraction.bits + 2) {
        round_ = fraction.BTEST(rshift - 2);
      }
      if (rshift > 2) {
        if (rshift >= fraction.bits + 2) {
          sticky_ = !fraction.IsZero();
        } else {
          auto mask = fraction.MASKR(rshift - 2);
          sticky_ = !fraction.IAND(mask).IsZero();
        }
      }
    }

    constexpr bool Zero() const { return !(guard_ | round_ | sticky_); }

    constexpr bool Negate() {
      bool carry{!sticky_};
      if (carry) {
        carry = !round_;
      } else {
        round_ = !round_;
      }
      if (carry) {
        carry = !guard_;
      } else {
        guard_ = !guard_;
      }
      return carry;
    }

    constexpr bool ShiftLeft() {
      bool oldGuard{guard_};
      guard_ = round_;
      round_ = sticky_;
      return oldGuard;
    }

    constexpr void ShiftRight(bool newGuard) {
      sticky_ |= round_;
      round_ = guard_;
      guard_ = newGuard;
    }

    // Determines whether a value should be rounded by increasing its
    // fraction, given a rounding mode and a summary of the lost bits.
    constexpr bool MustRound(Rounding rounding, const Real &real) const {
      bool round{false};  // to dodge bogus g++ warning about missing return
      switch (rounding) {
      case Rounding::TiesToEven:
        round = guard_ && (round_ | sticky_ | real.RawBits().BTEST(0));
        break;
      case Rounding::ToZero: break;
      case Rounding::Down: round = real.IsNegative() && !Zero(); break;
      case Rounding::Up: round = !real.IsNegative() && !Zero(); break;
      case Rounding::TiesAwayFromZero: round = guard_; break;
      }
      return round;
    }

  private:
    bool guard_{false};
    bool round_{false};
    bool sticky_{false};
  };

  constexpr Significand GetSignificand() const {
    return Significand::Convert(word_).value;
  }

  constexpr Fraction GetFraction() const {
    Fraction result{Fraction::Convert(word_).value};
    if constexpr (!implicitMSB) {
      return result;
    } else {
      std::uint64_t exponent{Exponent()};
      if (exponent > 0 && exponent < maxExponent) {
        return result.IBSET(significandBits);
      } else {
        return result.IBCLR(significandBits);
      }
    }
  }

  // TODO: Configurable NaN representations
  static constexpr Word NaNWord() {
    return Word{maxExponent}
        .SHIFTL(significandBits)
        .IBSET(significandBits - 1)
        .IBSET(significandBits - 2);
  }

  constexpr RealFlags Normalize(bool negative, std::uint64_t exponent,
      const Fraction &fraction, Rounding rounding = Rounding::TiesToEven,
      RoundingBits *roundingBits = nullptr) {
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
    if (fraction.BTEST(fraction.bits - 1)) {
      // fraction is normalized
      word_ = Word::Convert(fraction).value;
      if (exponent == 0) {
        exponent = 1;
      }
    } else {
      std::uint64_t lshift = fraction.LEADZ();
      if (lshift >= fraction.bits) {
        // +/-0.0
        word_ = Word{};
        exponent = 0;
      } else {
        word_ = Word::Convert(fraction).value;
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

  // Rounds a result, if necessary.
  RealFlags Round(Rounding rounding, const RoundingBits &bits) {
    std::uint64_t exponent{Exponent()};
    RealFlags flags;
    if (!bits.Zero()) {
      flags.set(RealFlag::Inexact);
    }
    if (exponent < maxExponent && bits.MustRound(rounding, *this)) {
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

  Word word_{};  // an Integer<>
};

using RealKind2 = Real<Integer<16>, 11>;
extern template class Real<Integer<16>, 11>;

using RealKind4 = Real<Integer<32>, 24>;
extern template class Real<Integer<32>, 24>;

using RealKind8 = Real<Integer<64>, 53>;
extern template class Real<Integer<64>, 53>;

using RealKind10 = Real<Integer<80>, 64, false>;  // 80387 extended precision
extern template class Real<Integer<80>, 64, false>;

using RealKind16 = Real<Integer<128>, 112>;
extern template class Real<Integer<128>, 112>;

// N.B. No "double-double" support.

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_REAL_H_
