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
#include "rounding-bits.h"
#include <cinttypes>
#include <limits>
#include <ostream>
#include <string>

// Some environments, viz. clang on Darwin, allow the macro HUGE
// to leak out of <math.h> even when it is never directly included.
#undef HUGE

namespace Fortran::evaluate::value {

// Models IEEE binary floating-point numbers (IEEE 754-2008,
// ISO/IEC/IEEE 60559.2011).  The first argument to this
// class template must be (or look like) an instance of Integer<>;
// the second specifies the number of effective bits in the fraction;
// the third, if true, indicates that the most significant position of the
// fraction is an implicit bit whose value is assumed to be 1 in a finite
// normal number.
template<typename WORD, int PRECISION, bool IMPLICIT_MSB = true> class Real {
public:
  using Word = WORD;
  static constexpr int precision{PRECISION};
  using Fraction = Integer<precision>;  // all bits made explicit

  static constexpr int bits{Word::bits};
  static constexpr bool implicitMSB{IMPLICIT_MSB};
  static constexpr int significandBits{precision - implicitMSB};
  static constexpr int exponentBits{bits - significandBits - 1 /*sign*/};
  static_assert(precision > 0);
  static_assert(exponentBits > 1);
  static constexpr std::uint64_t maxExponent{(1 << exponentBits) - 1};
  static constexpr std::uint64_t exponentBias{maxExponent / 2};

  // Decimal precision of a binary floating-point representation is
  // actually the same as the base-5 precision, as factors of two
  // can be accommodated by the binary exponent.
  // log(2)/log(5) = 0.430+ in any base.
  // Calculate "precision*0.43" with integer arithmetic so as to be constexpr.
  static constexpr int decimalDigits{(precision * 43) / 100};

  // Associates a decimal exponent with an integral value for formatting.
  // TODO pmk remove
  struct ScaledDecimal {
    bool negative{false};
    Word integer;  // unsigned
    int decimalExponent{0};  // Exxx
  };

  template<typename W, int P, bool I> friend class Real;

  constexpr Real() {}  // +0.0
  constexpr Real(const Real &) = default;
  constexpr Real(const Word &bits) : word_{bits} {}
  constexpr Real &operator=(const Real &) = default;
  constexpr Real &operator=(Real &&) = default;

  // TODO ANINT, CEILING, FLOOR, DIM, MAX, MIN, DPROD, FRACTION
  // HUGE, INT/NINT, MAXEXPONENT, MINEXPONENT, NEAREST, OUT_OF_RANGE,
  // PRECISION, HUGE, TINY, RRSPACING/SPACING, SCALE, SET_EXPONENT, SIGN

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
    return {word_.IBCLR(bits - 1)};
  }

  constexpr Real Negate() const { return {word_.IEOR(word_.MASKL(1))}; }

  Relation Compare(const Real &) const;
  ValueWithRealFlags<Real> Add(
      const Real &, Rounding rounding = defaultRounding) const;
  ValueWithRealFlags<Real> Subtract(
      const Real &y, Rounding rounding = defaultRounding) const {
    return Add(y.Negate(), rounding);
  }
  ValueWithRealFlags<Real> Multiply(
      const Real &, Rounding rounding = defaultRounding) const;
  ValueWithRealFlags<Real> Divide(
      const Real &, Rounding rounding = defaultRounding) const;

  // SQRT(x**2 + y**2) but computed so as to avoid spurious overflow
  // TODO: needed for CABS
  ValueWithRealFlags<Real> HYPOT(
      const Real &, Rounding rounding = defaultRounding) const;

  template<typename INT> constexpr INT EXPONENT() const {
    std::uint64_t exponent{Exponent()};
    if (exponent == maxExponent) {
      return INT::HUGE();
    } else {
      int result = exponent - exponentBias;
      if (IsDenormal()) {
        ++result;
      }
      return {result};
    }
  }

  static constexpr Real EPSILON() {
    Real epsilon;
    epsilon.Normalize(false, exponentBias - precision, Fraction::MASKL(1));
    return epsilon;
  }

  constexpr Real FlushDenormalToZero() const {
    if (IsDenormal()) {
      return Real{};
    }
    return *this;
  }

  // TODO: Configurable NotANumber representations
  static constexpr Real NotANumber() {
    return {Word{maxExponent}
                .SHIFTL(significandBits)
                .IBSET(significandBits - 1)
                .IBSET(significandBits - 2)};
  }

  static constexpr Real Infinity(bool negative) {
    Word infinity{maxExponent};
    infinity = infinity.SHIFTL(significandBits);
    if (negative) {
      infinity = infinity.IBSET(infinity.bits - 1);
    }
    return {infinity};
  }

  template<typename INT>
  static ValueWithRealFlags<Real> FromInteger(
      const INT &n, Rounding rounding = defaultRounding) {
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
      Fraction fraction{Fraction::ConvertUnsigned(absN).value};
      result.flags |= result.value.Normalize(
          isNegative, exponent, fraction.SHIFTL(-bitsLost));
    } else {
      Fraction fraction{Fraction::ConvertUnsigned(absN.SHIFTR(bitsLost)).value};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
      RoundingBits roundingBits{absN, bitsLost};
      result.flags |= result.value.Round(rounding, roundingBits);
    }
    return result;
  }

  // Truncation to integer in same real format.
  constexpr ValueWithRealFlags<Real> AINT() const {
    ValueWithRealFlags<Real> result{*this};
    if (IsNotANumber()) {
      result.flags.set(RealFlag::InvalidArgument);
      result.value = NotANumber();
    } else if (IsInfinite()) {
      result.flags.set(RealFlag::Overflow);
    } else {
      std::uint64_t exponent{Exponent()};
      if (exponent < exponentBias) {  // |x| < 1.0
        result.value.Normalize(IsNegative(), 0, Fraction{});  // +/-0.0
      } else {
        constexpr std::uint64_t noClipExponent{exponentBias + precision - 1};
        if (int clip = noClipExponent - exponent; clip > 0) {
          result.value.word_ = result.value.word_.IAND(Word::MASKR(clip).NOT());
        }
      }
    }
    return result;
  }

  template<typename INT> constexpr ValueWithRealFlags<INT> ToInteger() const {
    ValueWithRealFlags<INT> result;
    if (IsNotANumber()) {
      result.flags.set(RealFlag::InvalidArgument);
      result.value = result.value.HUGE();
      return result;
    }
    bool isNegative{IsNegative()};
    std::uint64_t exponent{Exponent()};
    Fraction fraction{GetFraction()};
    if (exponent >= maxExponent ||  // +/-Inf
        exponent >= exponentBias + result.value.bits) {  // too big
      if (isNegative) {
        result.value = result.value.MASKL(1);  // most negative integer value
      } else {
        result.value = result.value.HUGE();  // most positive integer value
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
        auto truncated{result.value.ConvertUnsigned(fraction.SHIFTR(rshift))};
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
          result.value =
              result.value.ConvertUnsigned(fraction).value.SHIFTL(lshift);
        }
      }
      if (result.flags.test(RealFlag::Overflow)) {
        result.value = result.value.HUGE();
      } else if (isNegative) {
        auto negated{result.value.Negate()};
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

  template<typename A>
  static ValueWithRealFlags<Real> Convert(
      const A &x, Rounding rounding = defaultRounding) {
    bool isNegative{x.IsNegative()};
    A absX{x};
    if (isNegative) {
      absX = x.Negate();
    }
    ValueWithRealFlags<Real> result;
    std::uint64_t exponent{exponentBias + x.Exponent() - A::exponentBias};
    int bitsLost{A::precision - precision};
    typename A::Fraction xFraction{x.GetFraction()};
    if (bitsLost <= 0) {
      Fraction fraction{Fraction::ConvertUnsigned(xFraction).value};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
    } else {
      Fraction fraction{
          Fraction::ConvertUnsigned(xFraction.SHIFTR(bitsLost)).value};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
      RoundingBits roundingBits{xFraction, bitsLost};
      result.flags |= result.value.Round(rounding, roundingBits);
    }
    return result;
  }

  // Represents the number as "J*(10**K)" where J and K are integers.
  ValueWithRealFlags<ScaledDecimal> AsScaledDecimal(
      Rounding rounding = defaultRounding) const;

  constexpr Word RawBits() const { return word_; }

  // Extracts "raw" biased exponent field.
  constexpr std::uint64_t Exponent() const {
    return word_.IBITS(significandBits, exponentBits).ToUInt64();
  }

  // Extracts unbiased exponent value.
  constexpr int UnbiasedExponent() const {
    int exponent =
        word_.IBITS(significandBits, exponentBits).ToUInt64() - exponentBias;
    if (IsDenormal()) {
      ++exponent;
    }
    return exponent;
  }

  // Extracts the fraction; any implied bit is made explicit.
  constexpr Fraction GetFraction() const {
    Fraction result{Fraction::ConvertUnsigned(word_).value};
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

  static ValueWithRealFlags<Real> Read(
      const char *&, Rounding rounding = defaultRounding);
  std::string DumpHexadecimal() const;

  // Emits a character representation for an equivalent Fortran constant
  // or parenthesized constant expression that produces this value.
  std::ostream &AsFortran(std::ostream &, int kind) const;

private:
  using Significand = Integer<significandBits>;  // no implicit bit

  constexpr Significand GetSignificand() const {
    return Significand::ConvertUnsigned(word_).value;
  }

  constexpr std::int64_t CombineExponents(const Real &y, bool forDivide) const {
    std::int64_t exponent = Exponent(), yExponent = y.Exponent();
    // A zero exponent field value has the same weight as 1.
    exponent += !exponent;
    yExponent += !yExponent;
    if (forDivide) {
      exponent += exponentBias - yExponent;
    } else {
      exponent += yExponent - exponentBias + 1;
    }
    return exponent;
  }

  static constexpr bool NextQuotientBit(
      Fraction &top, bool &msb, const Fraction &divisor) {
    bool greaterOrEqual{msb || top.CompareUnsigned(divisor) != Ordering::Less};
    if (greaterOrEqual) {
      top = top.SubtractSigned(divisor).value;
    }
    auto doubled{top.AddUnsigned(top)};
    top = doubled.value;
    msb = doubled.carry;
    return greaterOrEqual;
  }

  // Normalizes and marshals the fields of a floating-point number in place.
  // The value is a number, and a zero fraction means a zero value (i.e.,
  // a maximal exponent and zero fraction doesn't signify infinity, although
  // this member function will detect overflow and encode infinities).
  RealFlags Normalize(bool negative, std::uint64_t exponent,
      const Fraction &fraction, Rounding rounding = defaultRounding,
      RoundingBits *roundingBits = nullptr);

  // Rounds a result, if necessary, in place.
  RealFlags Round(Rounding, const RoundingBits &, bool multiply = false);

  static void NormalizeAndRound(ValueWithRealFlags<Real> &result,
      bool isNegative, std::uint64_t exponent, const Fraction &, Rounding,
      RoundingBits, bool multiply = false);

  Word word_{};  // an Integer<>
};

extern template class Real<Integer<16>, 11>;
extern template class Real<Integer<32>, 24>;
extern template class Real<Integer<64>, 53>;
extern template class Real<Integer<80>, 64, false>;  // 80387 extended precision
extern template class Real<Integer<128>, 112>;
// N.B. No "double-double" support.
}
#endif  // FORTRAN_EVALUATE_REAL_H_
