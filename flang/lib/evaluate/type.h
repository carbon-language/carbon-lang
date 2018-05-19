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

#ifndef FORTRAN_EVALUATE_TYPE_H_
#define FORTRAN_EVALUATE_TYPE_H_

#include <cinttypes>
#include <cstddef>

namespace Fortran::evaluate {

class IntrinsicType {
public:
  enum class Classification { Integer, Real, Complex, Character, Logical };

  // Default REAL just has to be IEEE-754 single precision today.
  // It occupies one numeric storage unit.  The default INTEGER and
  // default LOGICAL intrinsic types also have to occupy one numeric
  // storage unit, so their kinds are forced.  Default COMPLEX occupies
  // two numeric storage unit.
  using KindLenCType = std::int32_t;
  static constexpr KindLenCType defaultRealKind{4};  // IEEE-754 single
  static constexpr KindLenCType defaultIntegerKind{defaultRealKind};
  static constexpr KindLenCType kindLenIntegerKind{defaultIntegerKind};
  static constexpr KindLenCType defaultLogicalKind{defaultIntegerKind};

  static constexpr IntrinsicType IntrinsicTypeParameterType() {
    return IntrinsicType{Classification::Integer, kindLenIntegerKind};
  }

  IntrinsicType() = delete;
  constexpr IntrinsicType(Classification c, KindLenCType kind,
			  KindLenCType len = 1)
    : classification_{c}, kind_{kind}, len_{len} {}

  // Defaulted kinds.
  constexpr explicit IntrinsicType(Classification c)
    : classification_{c}, kind_{-1} /* overridden immediately */ {
    switch (c) {
    case Classification::Integer: kind_ = defaultIntegerKind; break;
    case Classification::Real: kind_ = defaultRealKind; break;
    case Classification::Complex: kind_ = 2 * defaultRealKind; break;
    case Classification::Character: kind_ = 1; break;
    case Classification::Logical: kind_ = defaultLogicalKind; break;
    }
  }
  constexpr IntrinsicType(const IntrinsicType &) = default;
  constexpr IntrinsicType &operator=(const IntrinsicType &) = default;

  constexpr Classification classification() const { return classification_; }
  constexpr KindLenCType kind() const { return kind_; }
  constexpr KindLenCType len() const { return len_; }

  // Not necessarily the size of an aligned allocation of runtime memory.
  constexpr std::size_t MinSizeInBytes() const {
    std::size_t n = kind_;
    if (classification_ == Classification::Character) {
      n *= len_;
    }
    return n;
  }

private:
  Classification classification_;
  KindLenCType kind_;
  KindLenCType len_{1};  // valid only for CHARACTER
};

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TYPE_H_
