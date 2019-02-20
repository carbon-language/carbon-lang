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

// This file defines the runtime libraries for the target as well as a default
// set of host rte functions that can be used for folding.
// The default HostRte is built with <cmath> and <complex> functions
// that are guaranteed to exist from the C++ standard.

#include "rte.h"
#include <cerrno>
#include <cfenv>

namespace Fortran::evaluate::rte {

using namespace Fortran::parser::literals;
// TODO mapping to <cmath> function to be tested.<cmath> func takes
// real arg for n
template<typename HostT> static HostT Bessel_jn(std::int64_t n, HostT x) {
  return std::cyl_bessel_j(static_cast<HostT>(n), x);
}
template<typename HostT> static HostT Bessel_yn(std::int64_t n, HostT x) {
  return std::cyl_neumann(static_cast<HostT>(n), x);
}

template<typename HostT> void AddLibmRealHostProcedure(HostRte &hostRte) {
  using F = FuncPointer<HostT, HostT>;
  using F2 = FuncPointer<HostT, HostT, HostT>;
  HostRteProcedureSymbol libmSymbols[]{{"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true}, {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true}, {"atan", F{std::atan}, true},
      {"atan", F2{std::atan2}, true}, {"atanh", F{std::atanh}, true},
      {"bessel_jn", &Bessel_jn<HostT>, true},
      {"bessel_yn", &Bessel_yn<HostT>, true}, {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true}, {"sin", F{std::sin}, true},
      {"sinh", F{std::sinh}, true}, {"sqrt", F{std::sqrt}, true},
      {"tan", F{std::tan}, true}, {"tanh", F{std::tanh}, true}};

  for (auto sym : libmSymbols) {
    hostRte.AddProcedure(std::move(sym));
  }
}

template<typename HostT> void AddLibmComplexHostProcedure(HostRte &hostRte) {
  using F = FuncPointer<std::complex<HostT>, const std::complex<HostT> &>;
  HostRteProcedureSymbol libmSymbols[]{{"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true}, {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true}, {"atan", F{std::atan}, true},
      {"atanh", F{std::atanh}, true}, {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true}, {"sin", F{std::sin}, true},
      {"sinh", F{std::sinh}, true}, {"sqrt", F{std::sqrt}, true},
      {"tan", F{std::tan}, true}, {"tanh", F{std::tanh}, true}};

  for (auto sym : libmSymbols) {
    hostRte.AddProcedure(std::move(sym));
  }
}

void HostRte::DefaultInit() {
  AddLibmRealHostProcedure<float>(*this);
  AddLibmRealHostProcedure<double>(*this);
  AddLibmRealHostProcedure<long double>(*this);

  AddLibmComplexHostProcedure<float>(*this);
  AddLibmComplexHostProcedure<double>(*this);
  AddLibmComplexHostProcedure<long double>(*this);
}

void HostFloatingPointEnvironment::SetUpHostFloatingPointEnvironment(
    FoldingContext &context) {
  errno = 0;
  if (feholdexcept(&originalFenv_) != 0) {
    context.messages().Say(
        "Folding with host runtime: feholdexcept() failed: %s"_en_US,
        std::strerror(errno));
    return;
  }
  if (fegetenv(&currentFenv_) != 0) {
    context.messages().Say(
        "Folding with host runtime: fegetenv() failed: %s"_en_US,
        std::strerror(errno));
    return;
  }
#if __x86_64__
  if (context.flushSubnormalsToZero()) {
    currentFenv_.__mxcsr |= 0x8000;  // result
    currentFenv_.__mxcsr |= 0x0040;  // operands
  } else {
    currentFenv_.__mxcsr &= ~0x8000;  // result
    currentFenv_.__mxcsr &= ~0x0040;  // operands
  }
#else
  // TODO other architecture
#endif
  errno = 0;
  if (fesetenv(&currentFenv_) != 0) {
    context.messages().Say("Folding with host rte: fesetenv() failed: %s"_en_US,
        std::strerror(errno));
    return;
  }
  switch (context.rounding().mode) {
  case RoundingMode::TiesToEven: fesetround(FE_TONEAREST); break;
  case RoundingMode::ToZero: fesetround(FE_TOWARDZERO); break;
  case RoundingMode::Up: fesetround(FE_UPWARD); break;
  case RoundingMode::Down: fesetround(FE_DOWNWARD); break;
  case RoundingMode::TiesAwayFromZero:
    context.messages().Say(
        "Folding with host runtime: TiesAwayFromZero rounding not available"_en_US);
    break;
  }
  errno = 0;
}
void HostFloatingPointEnvironment::CheckAndRestoreFloatingPointEnvironment(
    FoldingContext &context) {
  int errnoCapture{errno};
  int exceptions{fetestexcept(FE_ALL_EXCEPT)};
  RealFlags flags;
  if (exceptions & FE_INVALID) {
    flags.set(RealFlag::InvalidArgument);
  }
  if (exceptions & FE_DIVBYZERO) {
    flags.set(RealFlag::DivideByZero);
  }
  if (exceptions & FE_OVERFLOW) {
    flags.set(RealFlag::Overflow);
  }
  if (exceptions & FE_UNDERFLOW) {
    flags.set(RealFlag::Underflow);
  }
  if (exceptions & FE_INEXACT) {
    flags.set(RealFlag::Inexact);
  }

  if (!flags.empty()) {
    if (errnoCapture == EDOM) {
      flags.set(RealFlag::InvalidArgument);
    }
    if (errnoCapture == ERANGE) {
      // can't distinguish over/underflow from errno
      flags.set(RealFlag::Overflow);
    }
  }

  if (!flags.empty()) {
    RealFlagWarnings(context, flags, "folding function with host runtime");
  }
  errno = 0;
  if (fesetenv(&originalFenv_) != 0) {
    std::fprintf(stderr, "fesetenv() failed: %s\n", std::strerror(errno));
    context.messages().Say(
        "Folding with host rte: fesetenv() while restoring fenv failed: %s"_en_US,
        std::strerror(errno));
  }
  errno = 0;
}
}
