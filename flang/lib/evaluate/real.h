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

// Model IEEE-754 floating-point numbers.  The exponent range is that of a
// full int, and all significands are explicitly normalized.
// The class template parameter specifies the total number of bits in the
// significand, including the explicit greatest-order bit.
template<int PRECISION> class Real {
public:
  static constexpr int precision{PRECISION};
  static_assert(precision > 0);

  constexpr Real() {}  // +0.0
  constexpr Real(const Real &) = default;
  constexpr Real(std::int64_t n)
    : significand_{n}, exponent_{precision}, negative_{n < 0} {
    if (negative_) {
      significand_ = significand_.Negate().value;  // overflow is safe to ignore
    }
    Normalize();
  }

  // TODO: Change to FromInteger(), return flags
  template<int b, int pb, typename p, typename dp, bool le>
  constexpr Real(const Integer<b, pb, p, dp, le> &n,
      Rounding rounding = Rounding::TiesToEven)
    : negative_{n.IsNegative()} {
    if (negative_) {
      n = n.Negate().value;
    }
    if (n.bits <= precision) {
      exponent_ = precision;
      significand_.Convert(n);
      Normalize();
    } else {
      int lshift{n.LEADZ()};
      exponent_ = n.bits - lshift;
      int rshift{n.bits - (lshift + precision)};
      if (rshift <= 0) {
        significand_.Convert(n);
        significand_ = significand_.SHIFTL(precision - exponent_);
      } else {
        RoundingBits roundingBits;
        roundingBits.round = n.BTEST(rshift - 1);
        roundingBits.guard = !n.SHIFTL(n.bits - rshift).IsZero();
        significand_.Convert(n.SHIFTR(rshift));
        Round(rounding, roundingBits);
      }
    }
  }

  // TODO conversion from (or to?) other real types
  // TODO AINT/ANINT, CEILING, FLOOR, DIM, MAX, MIN, DPROD, FRACTION
  // HUGE, INT/NINT, MAXEXPONENT, MINEXPONENT, NEAREST, OUT_OF_RANGE,
  // PRECISION, HUGE, TINY, RRSPACING/SPACING, SCALE, SET_EXPONENT, SIGN

  constexpr Real &operator=(const Real &) = default;

  constexpr bool IsANumber() const { return !notANumber_; }
  constexpr bool IsNotANumber() const { return notANumber_; }
  constexpr bool IsNegative() const { return negative_ && !notANumber_; }
  constexpr bool IsFinite() const { return !infinite_ && !notANumber_; }
  constexpr bool IsInfinite() const { return infinite_ && !notANumber_; }
  constexpr bool IsZero() const {
    return !notANumber_ && significand_.IsZero();
  }

  constexpr Real ABS() const {  // non-arithmetic, no flags
    Real result{*this};
    result.negative_ = false;
    return result;
  }

  constexpr DefaultIntrinsicInteger EXPONENT() const {
    if (notANumber_ || infinite_) {
      return DefaultIntrinsicInteger::HUGE();
    } else {
      return {std::int64_t{exponent_}};
    }
  }

  static constexpr Real EPSILON() {
    Real epsilon;
    epsilon.exponent_ = -precision;
    epsilon.significand_.IBSET(precision - 1);
    return epsilon;
  }

  template<typename INT> constexpr ValueWithRealFlags<INT> ToInteger() const {
    ValueWithRealFlags<INT> result;
    if (notANumber_) {
      result.flags |= RealFlag::InvalidArgument;
      result.value = result.value.HUGE();
    } else if (infinite_ || exponent_ >= result.value.bits) {
      if (negative_) {
        result.value = result.value.MASKL(1);
      } else {
        result.value = result.value.HUGE();
      }
      result.flags = RealFlag::Overflow;
    } else {
      if (exponent_ > 0) {
        result.value =
            INT::Convert(significand_.SHIFTR(result.value.bits - exponent_));
      }
      if (negative_) {
        auto negated = result.value.Negate();
        if (result.overflow) {
          result.flags |= RealFlag::Overflow;
          result.value = result.value.HUGE();
        } else {
          result.value = negated.value;
        }
      }
    }
    return result;
  }

  constexpr Relation Compare(const Real &y) const {
    if (notANumber_ || y.notANumber_) {
      return Relation::Unordered;
    } else if (infinite_) {
      if (y.infinite_) {
        if (negative_) {
          return y.negative_ ? Relation::Equal : Relation::Less;
        } else {
          return y.negative_ ? Relation::Greater : Relation::Equal;
        }
      } else {
        return negative_ ? Relation::Less : Relation::Greater;
      }
    } else if (y.infinite_) {
      return y.negative_ ? Relation::Greater : Relation::Less;
    } else {
      // two finite numbers
      if (exponent_ == y.exponent_) {
        Ordering order{significand_.CompareUnsigned(y.significand_)};
        if (order == Ordering::Equal) {
          if (negative_ == y.negative_ ||
              (exponent_ == 0 && significand_.IsZero())) {
            // Ignore signs on zeros, +0.0 == -0.0
            return Relation::Equal;
          } else {
            // finite nonzero numbers, same exponent & significand
            return negative_ ? Relation::Less : Relation::Greater;
          }
        } else {
          // finite numbers, same exponent, distinct significands
          if (negative_ != y.negative_) {
            return negative_ ? Relation::Less : Relation::Greater;
          } else {
            return RelationFromOrdering(order);
          }
        }
      } else {
        // not same exponent
        if (negative_ != y.negative_) {
          return negative_ ? Relation::Less : Relation::Greater;
        } else {
          return exponent_ < y.exponent_ ? Relation::Less : Relation::Greater;
        }
      }
    }
  }

  constexpr ValueWithRealFlags<Real> Add(
      const Real &y, Rounding rounding) const {
    ValueWithRealFlags<Real> result;
    if (notANumber_ || y.notANumber_) {
      result.value.notANumber_ = true;  // NaN + x -> NaN
      result.flags = RealFlag::InvalidArgument;
      return result;
    }
    if (infinite_ || y.infinite_) {
      if (negative_ == y.negative_) {
        result.value.infinite_ = true;  // +/-Inf + +/-Inf -> +/-Inf
        result.value.negative_ = negative_;
      } else {
        result.value.notANumber_ = true;  // +/-Inf + -/+Inf -> NaN
        result.flags = RealFlag::InvalidArgument;
      }
      return result;
    }
    if (exponent_ < y.exponent_) {
      // y is larger; simplify by reversing
      return y.Add(*this, rounding);
    }
    if (exponent_ == y.exponent_ && negative_ != y.negative_ &&
        significand_.CompareUnsigned(y.significand_) == Ordering::Less) {
      // Same exponent, opposite signs, and y is larger.
      result = y.Add(*this, rounding);
      result.value.negative_ ^= true;
      return result;
    }
    // exponent is greater than or equal to y's
    result.value = y;
    result.value.exponent_ = exponent_;
    result.value.negative_ = negative_;
    RoundingBits roundingBits{
        result.value.ShiftSignificandRight(exponent_ - y.exponent_)};
    if (negative_ != y.negative_) {
      typename Significand::ValueWithOverflow negated{
          result.value.significand_.Negate()};
      if (negated.overflow) {
        // y had only its MSB set.  Result is our significand, less its MSB.
        result.value.significand_ = significand_.IBCLR(precision - 1);
      } else {
        typename Significand::ValueWithCarry diff{
            significand_.AddUnsigned(negated.value)};
        result.value.significand_ = diff.value;
      }
    } else {
      typename Significand::ValueWithCarry sum{
          significand_.AddUnsigned(result.value.significand_)};
      if (sum.carry) {
        roundingBits.guard |= roundingBits.round;
        roundingBits.round = sum.value.BTEST(0);
        result.value.significand_ = sum.value.SHIFTR(1).IBSET(precision - 1);
        ++result.value.exponent_;
      } else {
        result.value.significand_ = sum.value;
      }
    }
    result.value.Round(rounding, roundingBits);
    result.flags |= result.value.Normalize();
    return result;
  }

  constexpr ValueWithRealFlags<Real> Subtract(
      const Real &y, Rounding rounding) const {
    Real minusy{y};
    minusy.negative_ ^= true;
    return Add(minusy, rounding);
  }

  constexpr ValueWithRealFlags<Real> Multiply(
      const Real &y, Rounding rounding) const {
    ValueWithRealFlags<Real> result;
    if (notANumber_ || y.notANumber_) {
      result.value.notANumber_ = true;  // NaN * x -> NaN
      result.flags = RealFlag::InvalidArgument;
      return result;
    }
    result.value.negative_ = negative_ != y.negative_;
    if (infinite_ || y.infinite_) {
      result.value.infinite_ = true;
      return result;
    }
    typename Significand::Product product{
        significand_.MultiplyUnsigned(y.significand_)};
    result.value.exponent_ = exponent_ + y.exponent_ - 1;
    result.value.significand_ = product.upper;
    RoundingBits roundingBits;
    roundingBits.round = product.lower.BTEST(precision - 1);
    roundingBits.guard = !product.lower.IBCLR(precision - 1).IsZero();
    result.value.Round(rounding, roundingBits);
    result.flags |= result.value.Normalize();
    return result;
  }

  constexpr ValueWithRealFlags<Real> Divide(
      const Real &y, Rounding rounding) const {
    ValueWithRealFlags<Real> result;
    if (notANumber_ || y.notANumber_) {
      result.value.notANumber_ = true;  // NaN * x -> NaN
      result.flags = RealFlag::InvalidArgument;
      return result;
    }
    if (infinite_ || y.infinite_) {
      result.value.infinite_ = true;
      result.value.negative_ = negative_ != y.negative_;
      return result;
    }
    result.value.negative_ = negative_ != y.negative_;
    typename Significand::QuotientWithRemainder divided{
        significand_.DivideUnsigned(y.significand_)};
    if (divided.divisionByZero) {
      result.value.infinite_ = true;
      result.flags |= RealFlag::DivideByZero;
      return result;
    }
    result.value.exponent_ = exponent_ - y.exponent_ + 1;
    result.value.significand_ = divided.quotient;
    // To round, double the remainder and compare it to the divisor.
    RoundingBits roundingBits;
    typename Significand::ValueWithCarry doubledRem{
        divided.remainder.AddUnsigned(divided.remainder)};
    Ordering drcmp{doubledRem.value.CompareUnsigned(y.significand_)};
    roundingBits.round = drcmp != Ordering::Less;
    roundingBits.guard = drcmp != Ordering::Equal;
    result.value.Round(rounding, roundingBits);
    result.flags |= result.value.Normalize();
    return result;
  }

private:
  using Significand = Integer<precision>;
  static constexpr int maxExponent{std::numeric_limits<int>::max() / 2};
  static constexpr int minExponent{-maxExponent};

  struct RoundingBits {
    RoundingBits() {}
    RoundingBits(const RoundingBits &) = default;
    RoundingBits &operator=(const RoundingBits &) = default;
    bool round{false};
    bool guard{false};  // a/k/a "sticky" bit
  };

  // All values are normalized on output and assumed normal on input.
  // Returns flag bits.
  int Normalize() {
    if (notANumber_) {
      return RealFlag::InvalidArgument;
    } else if (infinite_) {
      return RealFlag::Ok;
    } else {
      int shift{significand_.LEADZ()};
      if (shift >= precision) {
        exponent_ = 0;  // +/-0.0
        return RealFlag::Ok;
      } else {
        exponent_ -= shift;
        if (exponent_ < minExponent) {
          exponent_ = 0;
          significand_ = Significand{};
          return RealFlag::Underflow;
        } else if (exponent_ > maxExponent) {
          infinite_ = true;
          return RealFlag::Overflow;
        } else {
          if (shift > 0) {
            significand_ = significand_.SHIFTL(shift);
          }
          return RealFlag::Ok;
        }
      }
    }
  }

  constexpr bool MustRound(Rounding rounding, const RoundingBits &bits) const {
    bool round{false};  // to dodge bogus g++ warning about missing return
    switch (rounding) {
    case Rounding::TiesToEven:
      round = bits.round && !bits.guard && significand_.BTEST(0);
      break;
    case Rounding::ToZero: break;
    case Rounding::Down: round = negative_ && (bits.round || bits.guard); break;
    case Rounding::Up: round = !negative_ && (bits.round || bits.guard); break;
    case Rounding::TiesAwayFromZero: round = bits.round && !bits.guard; break;
    }
    return round;
  }

  void Round(Rounding rounding, const RoundingBits &bits) {
    if (MustRound(rounding, bits)) {
      typename Significand::ValueWithCarry sum{
          significand_.AddUnsigned(Significand{}, true)};
      if (sum.carry) {
        // significand was all ones, and we rounded
        ++exponent_;
        significand_ = sum.value.SHIFTR(1).IBSET(precision - 1);
      } else {
        significand_ = sum.value;
      }
    }
  }

  RoundingBits ShiftSignificandRight(int places) {
    RoundingBits result;
    if (places > significand_.bits) {
      result.guard = !significand_.IsZero();
      significand_ = Significand{};
    } else if (places > 0) {
      if (places > 1) {
        result.guard = significand_.TRAILZ() + 1 < places;
      }
      result.round = significand_.BTEST(places - 1);
      significand_ = significand_.SHIFTR(places);
    }
    return result;
  }

  Significand significand_{};  // all bits explicit
  int exponent_{0};  // unbiased; 1.0 has exponent 1
  bool negative_{false};
  bool infinite_{false};
  bool notANumber_{false};
};

extern template class Real<11>;
extern template class Real<24>;
extern template class Real<53>;
extern template class Real<112>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_REAL_H_
