// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

// This file defines host runtime functions that can be used for folding
// intrinsic functions.
// The default HostIntrinsicProceduresLibrary is built with <cmath> and
// <complex> functions that are guaranteed to exist from the C++ standard.

#include "intrinsics-library-templates.h"
#include <cmath>
#include <complex>

namespace Fortran::evaluate {

// Note: argument passing is ignored in equivalence
bool HostIntrinsicProceduresLibrary::HasEquivalentProcedure(
    const IntrinsicProcedureRuntimeDescription &sym) const {
  const auto rteProcRange{procedures.equal_range(sym.name)};
  const size_t nargs{sym.argumentsType.size()};
  for (auto iter{rteProcRange.first}; iter != rteProcRange.second; ++iter) {
    if (nargs == iter->second.argumentsType.size() &&
        sym.returnType == iter->second.returnType &&
        (sym.isElemental || iter->second.isElemental)) {
      bool match{true};
      int pos{0};
      for (const auto &type : sym.argumentsType) {
        if (type != iter->second.argumentsType[pos++]) {
          match = false;
          break;
        }
      }
      if (match) {
        return true;
      }
    }
  }
  return false;
}

// Map numerical intrinsic to  <cmath>/<complex> functions

// Define which host runtime functions will be used for folding

// C++ Bessel functions take a floating point as first arguments.
// Fortran Bessel functions take an integer.
template<typename HostT> static HostT Bessel_j0(HostT x) {
  return std::cyl_bessel_j(0., x);
}
template<typename HostT> static HostT Bessel_y0(HostT x) {
  return std::cyl_neumann(0., x);
}
template<typename HostT> static HostT Bessel_j1(HostT x) {
  return std::cyl_bessel_j(1., x);
}
template<typename HostT> static HostT Bessel_y1(HostT x) {
  return std::cyl_neumann(1., x);
}
template<typename HostT> static HostT Bessel_jn(int n, HostT x) {
  return std::cyl_bessel_j(static_cast<HostT>(n), x);
}
template<typename HostT> static HostT Bessel_yn(int n, HostT x) {
  return std::cyl_neumann(static_cast<HostT>(n), x);
}

template<typename HostT>
static void AddLibmRealHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  using F = FuncPointer<HostT, HostT>;
  using F2 = FuncPointer<HostT, HostT, HostT>;
  HostRuntimeIntrinsicProcedure libmSymbols[]{{"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true}, {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true}, {"atan", F{std::atan}, true},
      {"atan", F2{std::atan2}, true}, {"atanh", F{std::atanh}, true},
      {"bessel_j0", &Bessel_j0<HostT>, true},
      {"bessel_y0", &Bessel_y0<HostT>, true},
      {"bessel_j1", &Bessel_j1<HostT>, true},
      {"bessel_y1", &Bessel_y1<HostT>, true},
      {"bessel_jn", &Bessel_jn<HostT>, true},
      {"bessel_yn", &Bessel_yn<HostT>, true}, {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true}, {"erf", F{std::erf}, true},
      {"erfc", F{std::erfc}, true}, {"exp", F{std::exp}, true},
      {"gamma", F{std::tgamma}, true}, {"hypot", F2{std::hypot}, true},
      {"log", F{std::log}, true}, {"log10", F{std::log10}, true},
      {"log_gamma", F{std::lgamma}, true}, {"mod", F2{std::fmod}, true},
      {"sin", F{std::sin}, true}, {"sinh", F{std::sinh}, true},
      {"sqrt", F{std::sqrt}, true}, {"tan", F{std::tan}, true},
      {"tanh", F{std::tanh}, true}};
  // Note: cmath does not have modulo and erfc_scaled equivalent

  for (auto sym : libmSymbols) {
    if (!hostIntrinsicLibrary.HasEquivalentProcedure(sym)) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

template<typename HostT>
static void AddLibmComplexHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  using F = FuncPointer<std::complex<HostT>, const std::complex<HostT> &>;
  HostRuntimeIntrinsicProcedure libmSymbols[]{
      {"abs", FuncPointer<HostT, const std::complex<HostT> &>{std::abs}, true},
      {"acos", F{std::acos}, true}, {"acosh", F{std::acosh}, true},
      {"asin", F{std::asin}, true}, {"asinh", F{std::asinh}, true},
      {"atan", F{std::atan}, true}, {"atanh", F{std::atanh}, true},
      {"cos", F{std::cos}, true}, {"cosh", F{std::cosh}, true},
      {"exp", F{std::exp}, true}, {"log", F{std::log}, true},
      {"sin", F{std::sin}, true}, {"sinh", F{std::sinh}, true},
      {"sqrt", F{std::sqrt}, true}, {"tan", F{std::tan}, true},
      {"tanh", F{std::tanh}, true}};

  for (auto sym : libmSymbols) {
    if (!hostIntrinsicLibrary.HasEquivalentProcedure(sym)) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

void InitHostIntrinsicLibraryWithLibm(HostIntrinsicProceduresLibrary &lib) {
  AddLibmRealHostProcedures<float>(lib);
  AddLibmRealHostProcedures<double>(lib);
  AddLibmRealHostProcedures<long double>(lib);
  AddLibmComplexHostProcedures<float>(lib);
  AddLibmComplexHostProcedures<double>(lib);
  AddLibmComplexHostProcedures<long double>(lib);
}

void HostIntrinsicProceduresLibrary::DefaultInit() {
  InitHostIntrinsicLibraryWithLibm(*this);
}
}
