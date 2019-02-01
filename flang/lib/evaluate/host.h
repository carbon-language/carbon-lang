// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
//
#ifndef FORTRAN_EVALUATE_HOST_H_
#define FORTRAN_EVALUATE_HOST_H_

// Define a compile-time mapping between Fortran intrinsic types and host
// hardware types if possible. The purpose is to avoid having to do any kind of
// assumption on whether a "float" matches the Scalar<Type<TypeCategory::Real,
// 4>> outside of this header. The main tools are HostTypeExists<T> and
// HostType<T>. HostTypeExists<T>() will return true if and only if a host
// hardware type maps to Fortran intrinsic type T. Then HostType<T> can be used
// to safely refer to this hardware type.

#include "type.h"
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace Fortran::evaluate {
namespace Host {

struct UnsupportedType {};

template<typename FTN_T> struct HostTypeHelper {
  using Type = UnsupportedType;
};
template<typename FTN_T> using HostType = typename HostTypeHelper<FTN_T>::Type;

template<typename... T> constexpr inline bool HostTypeExists() {
  return (... && (!std::is_same_v<HostType<T>, UnsupportedType>));
}

template<typename FTN_T>
inline constexpr Scalar<FTN_T> CastHostToFortran(const HostType<FTN_T> &x) {
  if constexpr (FTN_T::category == TypeCategory::Complex &&
      sizeof(Scalar<FTN_T>) != sizeof(HostType<FTN_T>)) {
    // X87 is usually padded to 12 or 16bytes. Need to cast piecewise for
    // complex
    return Scalar<FTN_T>{CastHostToFortran<typename FTN_T::Part>(std::real(x)),
        CastHostToFortran<typename FTN_T::Part>(std::imag(x))};
  } else {
    return *reinterpret_cast<const Scalar<FTN_T> *>(&x);
  }
}

template<typename FTN_T>
inline constexpr HostType<FTN_T> CastFortranToHost(const Scalar<FTN_T> &x) {
  if constexpr (FTN_T::category == TypeCategory::Complex &&
      sizeof(Scalar<FTN_T>) != sizeof(HostType<FTN_T>)) {
    // X87 is usually padded to 12 or 16bytes. Need to cast piecewise for
    // complex
    return HostType<FTN_T>{CastFortranToHost<typename FTN_T::Part>(x.REAL()),
        CastFortranToHost<typename FTN_T::Part>(x.AIMAG())};
  } else {
    return *reinterpret_cast<const HostType<FTN_T> *>(&x);
  }
}

template<typename T> struct BiggerOrSameHostTypeHelper {
  using Type =
      std::conditional_t<HostTypeExists<T>(), HostType<T>, UnsupportedType>;
  using FortranType = T;
};

template<typename FTN_T>
using BiggerOrSameHostType = typename BiggerOrSameHostTypeHelper<FTN_T>::Type;
template<typename FTN_T>
using BiggerOrSameFortanTypeSupportedOnHost =
    typename BiggerOrSameHostTypeHelper<FTN_T>::FortranType;

template<typename... T> constexpr inline bool BiggerOrSameHostTypeExists() {
  return (... && (!std::is_same_v<BiggerOrSameHostType<T>, UnsupportedType>));
}

// Defining the actual mapping
template<> struct HostTypeHelper<Type<TypeCategory::Integer, 1>> {
  using Type = std::int8_t;
};

template<> struct HostTypeHelper<Type<TypeCategory::Integer, 2>> {
  using Type = std::int16_t;
};

template<> struct HostTypeHelper<Type<TypeCategory::Integer, 4>> {
  using Type = std::int32_t;
};

template<> struct HostTypeHelper<Type<TypeCategory::Integer, 8>> {
  using Type = std::int64_t;
};

// no int 128bit

// no float 16bits

template<> struct HostTypeHelper<Type<TypeCategory::Real, 4>> {
  // IEE 754 64bits
  using Type = std::conditional_t<sizeof(float) == 4 &&
          std::numeric_limits<float>::is_iec559,
      float, UnsupportedType>;
};

template<> struct HostTypeHelper<Type<TypeCategory::Real, 8>> {
  // IEE 754 64bits
  using Type = std::conditional_t<sizeof(double) == 8 &&
          std::numeric_limits<double>::is_iec559,
      double, UnsupportedType>;
};

template<> struct HostTypeHelper<Type<TypeCategory::Real, 10>> {
  // X87 80bits
  using Type = std::conditional_t<sizeof(long double) >= 10 &&
          std::numeric_limits<long double>::digits == 64 &&
          std::numeric_limits<long double>::max_exponent == 16384,
      long double, UnsupportedType>;
};

template<> struct HostTypeHelper<Type<TypeCategory::Real, 16>> {
  // IEE 754 128bits
  using Type = std::conditional_t<sizeof(long double) == 16 &&
          std::numeric_limits<long double>::digits == 113 &&
          std::numeric_limits<long double>::max_exponent == 16384,
      long double, UnsupportedType>;
};

template<int KIND> struct HostTypeHelper<Type<TypeCategory::Complex, KIND>> {
  using RealT = Fortran::evaluate::Type<TypeCategory::Real, KIND>;
  using Type = std::conditional_t<HostTypeExists<RealT>(),
      std::complex<HostType<RealT>>, UnsupportedType>;
};

template<int KIND> struct HostTypeHelper<Type<TypeCategory::Character, KIND>> {
  using Type = std::conditional_t<KIND <= 8, std::uint8_t, UnsupportedType>;
};

template<> struct HostTypeHelper<Type<TypeCategory::Character, 1>> {
  using Type = std::string;
};

template<> struct HostTypeHelper<Type<TypeCategory::Character, 2>> {
  using Type = std::u16string;
};

// Utility to find "bigger" types that exist on host. By bigger, it is meant
// that the bigger type can represent all the values of the smaller types without
// information loss.
template<TypeCategory cat, int KIND> struct NextBiggerReal {
  using Type = void;
};
template<TypeCategory cat> struct NextBiggerReal<cat, 2> {
  using Type = Fortran::evaluate::Type<cat, 3>;
};
template<TypeCategory cat> struct NextBiggerReal<cat, 3> {
  using Type = Fortran::evaluate::Type<cat, 4>;
};
template<TypeCategory cat> struct NextBiggerReal<cat, 4> {
  using Type = Fortran::evaluate::Type<cat, 8>;
};

template<int KIND>
struct BiggerOrSameHostTypeHelper<Type<TypeCategory::Real, KIND>> {
  using T = Fortran::evaluate::Type<TypeCategory::Real, KIND>;
  using NextT = typename NextBiggerReal<TypeCategory::Real, KIND>::Type;
  using Type = std::conditional_t<HostTypeExists<T>(), HostType<T>,
      typename BiggerOrSameHostTypeHelper<NextT>::Type>;
  using FortranType = std::conditional_t<HostTypeExists<T>(), T,
      typename BiggerOrSameHostTypeHelper<NextT>::FortranType>;
};

template<int KIND>
struct BiggerOrSameHostTypeHelper<Type<TypeCategory::Complex, KIND>> {
  using T = Fortran::evaluate::Type<TypeCategory::Complex, KIND>;
  using NextT = typename NextBiggerReal<TypeCategory::Complex, KIND>::Type;
  using Type = std::conditional_t<HostTypeExists<T>(), HostType<T>,
      typename BiggerOrSameHostTypeHelper<NextT>::Type>;
  using FortranType = std::conditional_t<HostTypeExists<T>(), T,
      typename BiggerOrSameHostTypeHelper<NextT>::FortranType>;
};
}
}

#endif  // FORTRAN_EVALUATE_HOST_H_
