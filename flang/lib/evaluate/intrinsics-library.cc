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
  const auto rteProcRange{procedures_.equal_range(sym.name)};
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

#if LINK_WITH_LIBPGMATH
namespace pgmath {
// Define mapping between numerical intrinsics and libpgmath symbols
// namespace is used to have shorter names on repeated patterns.
// A class would be better to hold all these defs, but GCC does not
// support specialization of template variables inside class even
// if it is C++14 standard compliant here because there are only full
// specializations.

// List of intrinsics that have libpgmath implementations that can be used for
// constant folding The tag names must match the name used inside libpgmath name
// so that the macro below work.
enum class I {
  acos,
  acosh,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  bessel_j0,
  bessel_j1,
  bessel_jn,
  bessel_y0,
  bessel_y1,
  bessel_yn,
  cos,
  cosh,
  erf,
  erfc,
  erfc_scaled,
  exp,
  gamma,
  hypot,
  log,
  log10,
  log_gamma,
  mod,
  sin,
  sinh,
  sqrt,
  tan,
  tanh
};

// Library versions: P for Precise, R for Relaxed, F for Fast
enum class L { F, R, P };

struct NoSuchRuntimeSymbol {};
template<L, I, typename> constexpr auto Sym{NoSuchRuntimeSymbol{}};

// Macros to declare fast/relaxed/precise libpgmath variants.
// Note: std::complex and _complex are layout compatible but only std::comlpex
// should be used here so that templatized functions work as expected (_Complex
// and std::complex are different from a type point of view).
#define DECLARE_PGMATH_FAST_REAL(func) \
  extern "C" float __fs_##func##_1(float); \
  extern "C" double __fd_##func##_1(double); \
  template<> constexpr auto Sym<L::F, I::func, float>{__fs_##func##_1}; \
  template<> constexpr auto Sym<L::F, I::func, double>{__fd_##func##_1};

#define DECLARE_PGMATH_FAST_COMPLEX(func) \
  extern "C" std::complex<float> __fc_##func##_1(std::complex<float>); \
  extern "C" std::complex<double> __fz_##func##_1(std::complex<double>); \
  template<> \
  constexpr auto Sym<L::F, I::func, std::complex<float>>{__fc_##func##_1}; \
  template<> \
  constexpr auto Sym<L::F, I::func, std::complex<double>>{__fz_##func##_1};

#define DECLARE_PGMATH_FAST_ALL_FP(func) \
  DECLARE_PGMATH_FAST_REAL(func) \
  DECLARE_PGMATH_FAST_COMPLEX(func)

#define DECLARE_PGMATH_PRECISE_REAL(func) \
  extern "C" float __ps_##func##_1(float); \
  extern "C" double __pd_##func##_1(double); \
  template<> constexpr auto Sym<L::P, I::func, float>{__ps_##func##_1}; \
  template<> constexpr auto Sym<L::P, I::func, double>{__pd_##func##_1};

#define DECLARE_PGMATH_PRECISE_COMPLEX(func) \
  extern "C" std::complex<float> __pc_##func##_1(std::complex<float>); \
  extern "C" std::complex<double> __pz_##func##_1(std::complex<double>); \
  template<> \
  constexpr auto Sym<L::P, I::func, std::complex<float>>{__pc_##func##_1}; \
  template<> \
  constexpr auto Sym<L::P, I::func, std::complex<double>>{__pz_##func##_1};

#define DECLARE_PGMATH_PRECISE_ALL_FP(func) \
  DECLARE_PGMATH_PRECISE_REAL(func) \
  DECLARE_PGMATH_PRECISE_COMPLEX(func)

#define DECLARE_PGMATH_RELAXED_REAL(func) \
  extern "C" float __rs_##func##_1(float); \
  extern "C" double __rd_##func##_1(double); \
  template<> constexpr auto Sym<L::R, I::func, float>{__rs_##func##_1}; \
  template<> constexpr auto Sym<L::R, I::func, double>{__rd_##func##_1};

#define DECLARE_PGMATH_RELAXED_COMPLEX(func) \
  extern "C" std::complex<float> __rc_##func##_1(std::complex<float>); \
  extern "C" std::complex<double> __rz_##func##_1(std::complex<double>); \
  template<> \
  constexpr auto Sym<L::R, I::func, std::complex<float>>{__rc_##func##_1}; \
  template<> \
  constexpr auto Sym<L::R, I::func, std::complex<double>>{__rz_##func##_1};

#define DECLARE_PGMATH_RELAXED_ALL_FP(func) \
  DECLARE_PGMATH_RELAXED_REAL(func) \
  DECLARE_PGMATH_RELAXED_COMPLEX(func)

#define DECLARE_PGMATH_REAL(func) \
  DECLARE_PGMATH_FAST_REAL(func) \
  DECLARE_PGMATH_PRECISE_REAL(func) \
  DECLARE_PGMATH_RELAXED_REAL(func)

#define DECLARE_PGMATH_COMPLEX(func) \
  DECLARE_PGMATH_FAST_COMPLEX(func) \
  DECLARE_PGMATH_PRECISE_COMPLEX(func) \
  DECLARE_PGMATH_RELAXED_COMPLEX(func)

#define DECLARE_PGMATH_ALL(func) \
  DECLARE_PGMATH_REAL(func) \
  DECLARE_PGMATH_COMPLEX(func)

// Marcos to declare __mth_i libpgmath variants
#define DECLARE_PGMATH_MTH_VERSION_REAL(func) \
  extern "C" float __mth_i_##func(float); \
  extern "C" double __mth_i_d##func(double); \
  template<> constexpr auto Sym<L::F, I::func, float>{__mth_i_##func}; \
  template<> constexpr auto Sym<L::F, I::func, double>{__mth_i_d##func}; \
  template<> constexpr auto Sym<L::P, I::func, float>{__mth_i_##func}; \
  template<> constexpr auto Sym<L::P, I::func, double>{__mth_i_d##func}; \
  template<> constexpr auto Sym<L::R, I::func, float>{__mth_i_##func}; \
  template<> constexpr auto Sym<L::R, I::func, double>{__mth_i_d##func};

// Actual libpgmath declarations
DECLARE_PGMATH_ALL(acos)
DECLARE_PGMATH_MTH_VERSION_REAL(acosh)
DECLARE_PGMATH_ALL(asin)
DECLARE_PGMATH_MTH_VERSION_REAL(asinh)
DECLARE_PGMATH_ALL(atan)
// atan2 has 2 args
extern "C" float __fs_atan2_1(float, float);
extern "C" double __fd_atan2_1(double, double);
extern "C" float __ps_atan2_1(float, float);
extern "C" double __pd_atan2_1(double, double);
extern "C" float __rs_atan2_1(float, float);
extern "C" double __rd_atan2_1(double, double);
template<> constexpr auto Sym<L::F, I::atan2, float>{__fs_atan2_1};
template<> constexpr auto Sym<L::F, I::atan2, double>{__fd_atan2_1};
template<> constexpr auto Sym<L::P, I::atan2, float>{__ps_atan2_1};
template<> constexpr auto Sym<L::P, I::atan2, double>{__pd_atan2_1};
template<> constexpr auto Sym<L::R, I::atan2, float>{__rs_atan2_1};
template<> constexpr auto Sym<L::R, I::atan2, double>{__rd_atan2_1};
DECLARE_PGMATH_MTH_VERSION_REAL(atanh)
DECLARE_PGMATH_MTH_VERSION_REAL(bessel_j0)
DECLARE_PGMATH_MTH_VERSION_REAL(bessel_j1)
DECLARE_PGMATH_MTH_VERSION_REAL(bessel_y0)
DECLARE_PGMATH_MTH_VERSION_REAL(bessel_y1)
// bessel_jn and bessel_yn takes an int as first arg
extern "C" float __mth_i_bessel_jn(int, float);
extern "C" double __mth_i_dbessel_jn(int, double);
template<> constexpr auto Sym<L::F, I::bessel_jn, float>{__mth_i_bessel_jn};
template<> constexpr auto Sym<L::F, I::bessel_jn, double>{__mth_i_dbessel_jn};
template<> constexpr auto Sym<L::P, I::bessel_jn, float>{__mth_i_bessel_jn};
template<> constexpr auto Sym<L::P, I::bessel_jn, double>{__mth_i_dbessel_jn};
template<> constexpr auto Sym<L::R, I::bessel_jn, float>{__mth_i_bessel_jn};
template<> constexpr auto Sym<L::R, I::bessel_jn, double>{__mth_i_dbessel_jn};
extern "C" float __mth_i_bessel_yn(int, float);
extern "C" double __mth_i_dbessel_yn(int, double);
template<> constexpr auto Sym<L::F, I::bessel_yn, float>{__mth_i_bessel_yn};
template<> constexpr auto Sym<L::F, I::bessel_yn, double>{__mth_i_dbessel_yn};
template<> constexpr auto Sym<L::P, I::bessel_yn, float>{__mth_i_bessel_yn};
template<> constexpr auto Sym<L::P, I::bessel_yn, double>{__mth_i_dbessel_yn};
template<> constexpr auto Sym<L::R, I::bessel_yn, float>{__mth_i_bessel_yn};
template<> constexpr auto Sym<L::R, I::bessel_yn, double>{__mth_i_dbessel_yn};
DECLARE_PGMATH_ALL(cos)
DECLARE_PGMATH_ALL(cosh)
DECLARE_PGMATH_MTH_VERSION_REAL(erf)
DECLARE_PGMATH_MTH_VERSION_REAL(erfc)
DECLARE_PGMATH_MTH_VERSION_REAL(erfc_scaled)
DECLARE_PGMATH_ALL(exp)
DECLARE_PGMATH_MTH_VERSION_REAL(gamma)
extern "C" float __mth_i_hypot(float, float);
extern "C" double __mth_i_dhypot(double, double);
template<> constexpr auto Sym<L::F, I::hypot, float>{__mth_i_hypot};
template<> constexpr auto Sym<L::F, I::hypot, double>{__mth_i_dhypot};
template<> constexpr auto Sym<L::P, I::hypot, float>{__mth_i_hypot};
template<> constexpr auto Sym<L::P, I::hypot, double>{__mth_i_dhypot};
template<> constexpr auto Sym<L::R, I::hypot, float>{__mth_i_hypot};
template<> constexpr auto Sym<L::R, I::hypot, double>{__mth_i_dhypot};
DECLARE_PGMATH_ALL(log)
DECLARE_PGMATH_REAL(log10)
DECLARE_PGMATH_MTH_VERSION_REAL(log_gamma)
// no function for modulo in libpgmath
extern "C" float __fs_mod_1(float, float);
extern "C" double __fd_mod_1(double, double);
template<> constexpr auto Sym<L::F, I::mod, float>{__fs_mod_1};
template<> constexpr auto Sym<L::F, I::mod, double>{__fd_mod_1};
template<> constexpr auto Sym<L::P, I::mod, float>{__fs_mod_1};
template<> constexpr auto Sym<L::P, I::mod, double>{__fd_mod_1};
template<> constexpr auto Sym<L::R, I::mod, float>{__fs_mod_1};
template<> constexpr auto Sym<L::R, I::mod, double>{__fd_mod_1};
DECLARE_PGMATH_ALL(sin)
DECLARE_PGMATH_ALL(sinh)
DECLARE_PGMATH_MTH_VERSION_REAL(sqrt)
DECLARE_PGMATH_COMPLEX(sqrt)  // real versions are __mth_i...
DECLARE_PGMATH_ALL(tan)
DECLARE_PGMATH_ALL(tanh)

// Fill the function map used for folding with libpgmath symbols
template<L Lib, typename HostT>
static void AddLibpgmathRealHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  static_assert(std::is_same_v<HostT, float> || std::is_same_v<HostT, double>);
  HostRuntimeIntrinsicProcedure pgmathSymbols[]{
      {"acos", Sym<Lib, I::acos, HostT>, true},
      {"acosh", Sym<Lib, I::acosh, HostT>, true},
      {"asin", Sym<Lib, I::asin, HostT>, true},
      {"asinh", Sym<Lib, I::asinh, HostT>, true},
      {"atan", Sym<Lib, I::atan, HostT>, true},
      {"atan", Sym<Lib, I::atan2, HostT>,
          true},  // atan is also the generic name for atan2
      {"atanh", Sym<Lib, I::atanh, HostT>, true},
      {"bessel_j0", Sym<Lib, I::bessel_j0, HostT>, true},
      {"bessel_j1", Sym<Lib, I::bessel_j1, HostT>, true},
      {"bessel_jn", Sym<Lib, I::bessel_jn, HostT>, true},
      {"bessel_y0", Sym<Lib, I::bessel_y0, HostT>, true},
      {"bessel_y1", Sym<Lib, I::bessel_y1, HostT>, true},
      {"bessel_yn", Sym<Lib, I::bessel_yn, HostT>, true},
      {"cos", Sym<Lib, I::cos, HostT>, true},
      {"cosh", Sym<Lib, I::cosh, HostT>, true},
      {"erf", Sym<Lib, I::erf, HostT>, true},
      {"erfc", Sym<Lib, I::erfc, HostT>, true},
      {"erfc_scaled", Sym<Lib, I::erfc_scaled, HostT>, true},
      {"exp", Sym<Lib, I::exp, HostT>, true},
      {"gamma", Sym<Lib, I::gamma, HostT>, true},
      {"hypot", Sym<Lib, I::hypot, HostT>, true},
      {"log", Sym<Lib, I::log, HostT>, true},
      {"log10", Sym<Lib, I::log10, HostT>, true},
      {"log_gamma", Sym<Lib, I::log_gamma, HostT>, true},
      {"mod", Sym<Lib, I::mod, HostT>, true},
      {"sin", Sym<Lib, I::sin, HostT>, true},
      {"sinh", Sym<Lib, I::sinh, HostT>, true},
      {"sqrt", Sym<Lib, I::sqrt, HostT>, true},
      {"tan", Sym<Lib, I::tan, HostT>, true},
      {"tanh", Sym<Lib, I::tanh, HostT>, true}};

  for (auto sym : pgmathSymbols) {
    hostIntrinsicLibrary.AddProcedure(std::move(sym));
  }
}

template<L Lib, typename HostT>
static void AddLibpgmathComplexHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  static_assert(std::is_same_v<HostT, float> || std::is_same_v<HostT, double>);
  using CHostT = std::complex<HostT>;
  // cmath is used to complement pgmath when symbols are not available
  using CmathF = FuncPointer<CHostT, const CHostT &>;
  HostRuntimeIntrinsicProcedure pgmathSymbols[]{
      {"abs", FuncPointer<HostT, const CHostT &>{std::abs}, true},
      {"acos", Sym<Lib, I::acos, CHostT>, true},
      {"acosh", CmathF{std::acosh}, true},
      {"asin", Sym<Lib, I::asin, CHostT>, true},
      {"asinh", CmathF{std::asinh}, true}, {"atan", CmathF{std::atan}, true},
      {"atanh", CmathF{std::atanh}, true},
      {"cos", Sym<Lib, I::cos, CHostT>, true},
      {"cosh", Sym<Lib, I::cosh, CHostT>, true},
      {"exp", Sym<Lib, I::exp, CHostT>, true},
      {"log", Sym<Lib, I::log, CHostT>, true},
      {"sin", Sym<Lib, I::sin, CHostT>, true},
      {"sinh", Sym<Lib, I::sinh, CHostT>, true},
      {"sqrt", Sym<Lib, I::sqrt, CHostT>, true},
      {"tan", Sym<Lib, I::tan, CHostT>, true},
      {"tanh", Sym<Lib, I::tanh, CHostT>, true}};

  for (auto sym : pgmathSymbols) {
    hostIntrinsicLibrary.AddProcedure(std::move(sym));
  }
}

template<L Lib>
static void InitHostIntrinsicLibraryWithLibpgmath(
    HostIntrinsicProceduresLibrary &lib) {
  AddLibpgmathRealHostProcedures<Lib, float>(lib);
  AddLibpgmathRealHostProcedures<Lib, double>(lib);
  AddLibpgmathComplexHostProcedures<Lib, float>(lib);
  AddLibpgmathComplexHostProcedures<Lib, double>(lib);
  // No long double functions in libpgmath
  AddLibmRealHostProcedures<long double>(lib);
  AddLibmComplexHostProcedures<long double>(lib);
}
}
#endif  // LINK_WITH_LIBPGMATH

// Define which host runtime functions will be used for folding
HostIntrinsicProceduresLibrary::HostIntrinsicProceduresLibrary() {
  // TODO: When command line options regarding targeted numerical library is
  // available, this needs to be revisited to take it into account. So far,
  // default to libpgmath if F18 is built with it.
#if LINK_WITH_LIBPGMATH
  pgmath::InitHostIntrinsicLibraryWithLibpgmath<pgmath::L::P>(*this);
#else
  InitHostIntrinsicLibraryWithLibm(*this);
#endif
}
}
