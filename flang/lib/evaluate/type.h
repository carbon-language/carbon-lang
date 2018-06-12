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

// These definitions map Fortran's intrinsic types to their value
// representation types in the evaluation library for ease of template
// programming.

#include "integer.h"
#include "logical.h"
#include "real.h"

namespace Fortran::evaluate::type {

enum class Classification { Integer, Real, Complex, Character, Logical };

template<int KIND> struct Integer {
  static constexpr Classification classification{Classification::Integer};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  using ValueType = value::Integer<8 * kind>;
};

template<int KIND> struct Real;
template<> struct Real<2> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{2};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<value::Integer<16>, 11>;
};
template<> struct Real<4> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{4};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<value::Integer<32>, 24>;
};
template<> struct Real<8> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{8};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<value::Integer<64>, 53>;
};
template<> struct Real<10> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{10};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<value::Integer<80>, 64, false>;
};
template<> struct Real<16> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{16};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<value::Integer<128>, 112>;
};

#if 0  // TODO
template<int KIND> struct Complex {
  static constexpr Classification classification{Classification::Complex};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  using ValueType = value::Complex<8 * kind>;
};
#endif

template<int KIND> struct Logical {
  static constexpr Classification classification{Classification::Logical};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  using ValueType = value::Logical<8 * kind>;
};

#if 0  // TODO
template<int KIND> struct Character {
  static constexpr Classification classification{Classification::Character};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{true};
  using ValueType = value::Character<8 * kind>;
};
#endif

// Default REAL just simply has to be IEEE-754 single precision today.
// It occupies one numeric storage unit by definition.  The default INTEGER
// and default LOGICAL intrinsic types also have to occupy one numeric
// storage unit, so their kinds are also forced.  Default COMPLEX occupies
// two numeric storage units.

using DefaultReal = Real<4>;
using DefaultInteger = Integer<DefaultReal::kind>;
using IntrinsicTypeParameterType = DefaultInteger;
#if 0  // TODO
using DefaultComplex = Complex<2 * DefaultReal::kind>;
#endif
using DefaultLogical = Logical<DefaultReal::kind>;
#if 0  // TODO
using DefaultCharacter = Character<1>;
#endif

}  // namespace Fortran::evaluate::type
#endif  // FORTRAN_EVALUATE_TYPE_H_
