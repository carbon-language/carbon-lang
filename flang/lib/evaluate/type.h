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

// These definitions map Fortran's intrinsic types, characterized by byte
// sizes encoded in KIND type parameter values, to their value representation
// types in the evaluation library, which are parameterized in terms of
// total bit width and real precision.

#include "complex.h"
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
  using ValueType = value::Real<typename Integer<kind>::ValueType, 11>;
};
template<> struct Real<4> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{4};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<typename Integer<kind>::ValueType, 24>;
};
template<> struct Real<8> {
  static constexpr Classification classification{Classification::Real};
  static constexpr int kind{8};
  static constexpr bool hasLen{false};
  using ValueType = value::Real<typename Integer<kind>::ValueType, 53>;
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
  using ValueType = value::Real<typename Integer<kind>::ValueType, 112>;
};

template<int KIND> struct Complex {
  static constexpr Classification classification{Classification::Complex};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  using ValueType = value::Complex<typename Real<(8 * kind / 2)>::ValueType>;
};

template<int KIND> struct Logical {
  static constexpr Classification classification{Classification::Logical};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  using ValueType = value::Logical<8 * kind>;
};

template<int KIND> struct Character {
  static constexpr Classification classification{Classification::Character};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{true};
  using ValueType = std::uint8_t[kind];  // TODO: ?
};

// Default REAL just simply has to be IEEE-754 single precision today.
// It occupies one numeric storage unit by definition.  The default INTEGER
// and default LOGICAL intrinsic types also have to occupy one numeric
// storage unit, so their kinds are also forced.  Default COMPLEX occupies
// two numeric storage units.

using DefaultReal = Real<4>;
using DefaultInteger = Integer<DefaultReal::kind>;
using IntrinsicTypeParameterType = DefaultInteger;
using DefaultComplex = Complex<2 * DefaultReal::kind>;
using DefaultLogical = Logical<DefaultReal::kind>;
using DefaultCharacter = Character<1>;

}  // namespace Fortran::evaluate::type
#endif  // FORTRAN_EVALUATE_TYPE_H_
