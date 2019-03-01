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
#include <sstream>
#if defined(__APPLE__) || defined(__unix__)
#define IS_POSIX_COMPLIANT
#include <dlfcn.h>
#endif

namespace Fortran::evaluate::rte {

using namespace Fortran::parser::literals;
// Note: argument passing is ignored in equivalence
bool HostRte::HasEquivalentProcedure(const RteProcedureSymbol &sym) const {
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

void HostRte::LoadTargetRteLibrary(const TargetRteLibrary &lib) {
  if (dynamicallyLoadedLibraries.find(lib.name) !=
      dynamicallyLoadedLibraries.end()) {
    return;  // already loaded
  }
#ifdef IS_POSIX_COMPLIANT
  void *handle = dlopen((lib.name + std::string{".so"}).c_str(), RTLD_LAZY);
  if (!handle) {
    return;
  }
  dynamicallyLoadedLibraries.insert(std::make_pair(lib.name, handle));
  for (const auto &sym : lib.procedures) {
    void *func{dlsym(handle, sym.second.symbol.c_str())};
    auto error{dlerror()};
    if (error) {
    } else {
      // Note: below is the only reinterpret_cast from an object pointer to
      // function pointer As per C++11 and later rules on reinterpret_cast, it is
      // implementation defined whether this is supported. POSIX mandates that
      // such cast from function pointers to void* are defined. Hence this
      // reinterpret_cast is and MUST REMAIN inside ifdef related to POSIX.
      AddProcedure(HostRteProcedureSymbol{
          sym.second, reinterpret_cast<FuncPointer<void *>>(func)});
    }
  }
#else
  // TODO: systems that do not support dlopen (e.g windows)
#endif
}

HostRte::~HostRte() {
  for (auto iter{dynamicallyLoadedLibraries.begin()};
       iter != dynamicallyLoadedLibraries.end(); ++iter) {
#ifdef IS_POSIX_COMPLIANT
    (void)dlclose(iter->second);
#endif
  }
}

// Map numerical intrinsic to  <cmath>/<complex> functions (for host folding
// only)

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
    if (!hostRte.HasEquivalentProcedure(sym)) {
      hostRte.AddProcedure(std::move(sym));
    }
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
    if (!hostRte.HasEquivalentProcedure(sym)) {
      hostRte.AddProcedure(std::move(sym));
    }
  }
}

// define mapping between numerical intrinsics and libpgmath symbols

enum class MathOption { Fast, Precise, Relaxed };

char constexpr inline EncodePgmMathOption(MathOption m) {
  switch (m) {
  case MathOption::Fast: return 'f';
  case MathOption::Precise: return 'p';
  case MathOption::Relaxed: return 'r';
  }
  return '\0';  // unreachable. Silencing bogus g++ warning
}

template<typename T> struct EncodePgmTypeHelper {};

template<> struct EncodePgmTypeHelper<Type<TypeCategory::Real, 4>> {
  static constexpr char value{'s'};
};
template<> struct EncodePgmTypeHelper<Type<TypeCategory::Real, 8>> {
  static constexpr char value{'d'};
};
template<> struct EncodePgmTypeHelper<Type<TypeCategory::Complex, 4>> {
  static constexpr char value{'c'};
};
template<> struct EncodePgmTypeHelper<Type<TypeCategory::Complex, 8>> {
  static constexpr char value{'z'};
};

template<typename T>
static constexpr char EncodePgmType{EncodePgmTypeHelper<T>::value};

template<typename T>
static std::string MakeLibpgmathName(const std::string &name, MathOption m) {
  std::ostringstream stream;
  stream << "__" << EncodePgmMathOption(m) << EncodePgmType<T> << "_" << name
         << "_1";
  // TODO Take mask and vector length into account
  return stream.str();
}

template<typename T>
static void AddLibpgmathTargetSymbols(TargetRteLibrary &lib, MathOption opt) {
  using F = Signature<T, ArgumentInfo<T, PassBy::Val>>;
  const std::string oneArgFuncs[]{"acos", "asin", "atan", "cos", "cosh", "exp",
      "log", "log10", "sin", "sinh", "tan", "tanh"};
  for (const std::string &name : oneArgFuncs) {
    lib.AddProcedure(TargetRteProcedureSymbol{
        F{name}, MakeLibpgmathName<T>(name, opt), true});
  }

  if constexpr (T::category == TypeCategory::Real) {
    using F2 = Signature<T, ArgumentInfo<T, PassBy::Val>,
        ArgumentInfo<T, PassBy::Val>>;
    lib.AddProcedure(TargetRteProcedureSymbol{
        F2{"atan2"}, MakeLibpgmathName<T>("acos", opt), true});
  } else {
    const std::string oneArgCmplxFuncs[]{
        "div", "sqrt"};  // for scalar, only complex available
    for (const std::string &name : oneArgCmplxFuncs) {
      lib.AddProcedure(TargetRteProcedureSymbol{
          F{name}, MakeLibpgmathName<T>(name, opt), true});
    }
  }
}

TargetRteLibrary BuildLibpgmTargetRteLibrary(MathOption opt) {
  TargetRteLibrary lib{"libpgmath"};
  AddLibpgmathTargetSymbols<Type<TypeCategory::Real, 4>>(lib, opt);
  AddLibpgmathTargetSymbols<Type<TypeCategory::Real, 8>>(lib, opt);
  AddLibpgmathTargetSymbols<Type<TypeCategory::Complex, 4>>(lib, opt);
  AddLibpgmathTargetSymbols<Type<TypeCategory::Complex, 8>>(lib, opt);
  return lib;
}

// Defines which host runtime functions will be used for folding

void HostRte::DefaultInit() {
  // TODO: when linkage information is available, this needs to be modified to
  // load runtime accordingly. For now, try loading libpgmath (libpgmath.so
  // needs to be in a directory from LD_LIBRARY_PATH) and then add libm symbols
  // when no equivalent symbols were already loaded
  TargetRteLibrary libpgmath{BuildLibpgmTargetRteLibrary(MathOption::Precise)};
  LoadTargetRteLibrary(libpgmath);

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
  // TODO other architectures
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

  if (flags.empty()) {
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
