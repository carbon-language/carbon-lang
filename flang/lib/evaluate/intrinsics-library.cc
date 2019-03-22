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

// C++ Bessel functions take a floating point as first argument.
// Fortran Bessel functions take an integer.
template<typename HostT> static HostT Bessel_jn(std::int64_t n, HostT x) {
  return std::cyl_bessel_j(static_cast<HostT>(n), x);
}
template<typename HostT> static HostT Bessel_yn(std::int64_t n, HostT x) {
  return std::cyl_neumann(static_cast<HostT>(n), x);
}

template<typename HostT>
void AddLibmRealHostProcedure(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  using F = FuncPointer<HostT, HostT>;
  using F2 = FuncPointer<HostT, HostT, HostT>;
  HostRuntimeIntrinsicProcedure libmSymbols[]{{"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true}, {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true}, {"atan", F{std::atan}, true},
      {"atan", F2{std::atan2}, true}, {"atanh", F{std::atanh}, true},
      {"bessel_jn", &Bessel_jn<HostT>, true},
      {"bessel_yn", &Bessel_yn<HostT>, true}, {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true}, {"sin", F{std::sin}, true},
      {"sinh", F{std::sinh}, true}, {"sqrt", F{std::sqrt}, true},
      {"tan", F{std::tan}, true}, {"tanh", F{std::tanh}, true}};

  for (auto sym : libmSymbols) {
    if (!hostIntrinsicLibrary.HasEquivalentProcedure(sym)) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

template<typename HostT>
void AddLibmComplexHostProcedure(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  using F = FuncPointer<std::complex<HostT>, const std::complex<HostT> &>;
  HostRuntimeIntrinsicProcedure libmSymbols[]{{"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true}, {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true}, {"atan", F{std::atan}, true},
      {"atanh", F{std::atanh}, true}, {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true}, {"sin", F{std::sin}, true},
      {"sinh", F{std::sinh}, true}, {"sqrt", F{std::sqrt}, true},
      {"tan", F{std::tan}, true}, {"tanh", F{std::tanh}, true}};

  for (auto sym : libmSymbols) {
    if (!hostIntrinsicLibrary.HasEquivalentProcedure(sym)) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

// Define which host runtime functions will be used for folding

void HostIntrinsicProceduresLibrary::DefaultInit() {

  AddLibmRealHostProcedure<float>(*this);
  AddLibmRealHostProcedure<double>(*this);
  AddLibmRealHostProcedure<long double>(*this);

  AddLibmComplexHostProcedure<float>(*this);
  AddLibmComplexHostProcedure<double>(*this);
  AddLibmComplexHostProcedure<long double>(*this);
}
}
