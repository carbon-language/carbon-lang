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

namespace Fortran::evaluate {

// The class template parameter specifies the total number of bits in the
// significand, including any bit that might be implicit in a machine
// representation.
template<int SIGNIFICAND_BITS> class Real {
public:
  static constexpr int significandBits{SIGNIFICAND_BITS};
  static_assert(significandBits > 0);

  constexpr Real() {}  // +0.0
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

private:
  using Significand = Integer<significandBits>;
  using DoubleSignificand = Integer<2 * significandBits>;

  // All values are normalized on output and assumed normal on input.
  void Normalize() {
    if (!notANumber_ && !infinite_) {
      int shift{significand_.LEADZ()};
      if (shift >= significandBits) {
        exponent_ = 0;  // +/-0.0
      } else {
        exponent_ -= shift;
        significand_ = significand_.SHIFTL(shift);
      }
    }
  }

  Significand significand_{};
  int exponent_{0};  // unbiased; 1.0 has exponent 1
  bool negative_{false};
  bool infinite_{false};
  bool notANumber_{false};
};

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_REAL_H_
