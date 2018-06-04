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
template<int SIGNIFICAND_BITS> class Real {
public:
  static constexpr int significandBits{SIGNIFICAND_BITS};
  static_assert(significandBits > 0);

  struct RealResult {
    Real value;
    int flags{RealFlag::Ok};
  };

  constexpr Real() {}  // +0.0
  constexpr Real(const Real &) = default;
  constexpr Real(std::int64_t n)
      : significand_{n}, exponent_{64}, negative_{n < 0} {
    if (negative_) {
      significand_ = significand_.Negate().value;  // overflow is safe to ignore
    }
    Normalize();
  }

  constexpr bool IsNegative() const { return negative_ && !notANumber_; }
  constexpr bool IsFinite() const { return !infinite_ && !notANumber_; }
  constexpr bool IsInfinite() const { return infinite_ && !notANumber_; }
  constexpr bool IsANumber() const { return !notANumber_; }
  constexpr bool IsNotANumber() const { return notANumber_; }
  constexpr bool IsZero() const { return !notANumber_ && significand_.IsZero(); }

  constexpr Real &operator=(const Real &) = default;

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

  constexpr RealResult Add(const Real &y, Rounding rounding) const {
    RealResult result;
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
    if (exponent_ == y.exponent_ &&
        negative_ != y.negative_ &&
        significand_.CompareUnsigned(y.significand_) == Ordering::Less) {
      // Same exponent, opposite signs, and y is larger.
      result = y.Add(*this, rounding);
      result.negative_ ^= true;
      return result;
    }
    // exponent is greater than or equal to y's
    result.value = y;
    result.value.exponent_ = exponent_;
    result.value.negative_ = negative_;
    RoundingBits roundingBits{result.value.ShiftSignificandRight(exponent_ - y.exponent_)};
    if (negative_ != y.negative_) {
      ValueWithOverflow negated{result.value.significand_.Negate()};
      if (negated.overflow) {
        // y had only its MSB set.  Result is our significand, less its MSB.
        result.value.significand_ = significand_.IBCLR(significandBits - 1);
      } else {
        ValueWithCarry diff{significand_.AddUnsigned(negated.value)};
        result.value.significand_ = negated.value;
      }
    } else {
      ValueWithCarry sum{significand_.AddUnsigned(result.value.significand_)};
      if (sum.carry) {
        roundingBits.guard |= roundingBits.round;
        roundingBits.round = sum.value.BTEST(0);
        result.value.significand_ = sum.value.SHIFTR(1).IBSET(significandBits - 1);
        ++result.value.exponent_;
      } else {
        result.value.significand_ = sum.value;
      }
    }
    result.value.Round(rounding, roundingBits);
    result.flags |= result.value.Normalize();
    return result;
  }

  constexpr RealResult Subtract(const Real &y, Rounding rounding) const {
    Real minusy{y};
    minusy.negative_ ^= true;
    return Add(minusy, rounding);
  }

  constexpr RealResult Multiply(const Real &y, Rounding rounding) const {
    RealResult result;
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
  }

private:
  using Significand = Integer<significandBits>;
  using DoubleSignificand = Integer<2 * significandBits>;
  static constexpr int maxExponent{std::numeric_limits<int>::max() / 2};
  static constexpr int minExponent{-maxExponent};

  struct RoundingBits {
    RoundingBits() {}
    RoundingBits(const RoundingBits &) = default;
    RoundingBits &operator(const RoundingBits &) = default;
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
      if (shift >= significandBits) {
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
    switch (rounding) {
    case Rounding::TiesToEven:
      return bits.round && !bits.guard && significand_.BTEST(0);
    case Rounding::ToZero:
      return false;
    case Rounding::Down:
      return negative_ && (bits.round || bits.guard);
    case Rounding::Up:
      return !negative_ && (bits.round || bits.guard);
    case Rounding::TiesAwayFromZero:
      return bits.round && !bits.guard;
    }
  }

  void Round(Rounding rounding, const RoundingBits &bits) {
    if (MustRound(rounding, bits)) {
      ValueWithCarry sum{significand_.AddUnsigned(Significand{}, true)};
      if (sum.carry) {
        // significand was all ones, and we rounded
        ++exponent_;
        significand_ = sum.value.SHIFTR(1).IBSET(significandBits - 1);
      } else {
        significand_ = sum.value;
      }
    }
  }

  RoundingBits ShiftSignificandRight(int places) {
    RoundingBits result;
    if (places > bits) {
      result.guard = !significand_.IsZero();
      significand_.Clear();
    } else if (places > 0) {
      if (places > 1) {
        result.guard = significand_.TRAILZ() + 1 < places;
      }
      result.round = significand_.BTEST(places - 1);
      significand_ = significand_.SHIFTR(places);
    }
    return result;
  }

  Significand significand_{};
  int exponent_{0};  // unbiased; 1.0 has exponent 1
  bool negative_{false};
  bool infinite_{false};
  bool notANumber_{false};
};

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_REAL_H_
