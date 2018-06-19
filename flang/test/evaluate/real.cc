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

#include "fp-testing.h"
#include "testing.h"
#include "../../lib/evaluate/type.h"
#include <cstdio>
#include <cstdlib>

using namespace Fortran::evaluate;

using Real2 = typename Type<Category::Real, 2>::Value;
using Real4 = typename Type<Category::Real, 4>::Value;
using Real8 = typename Type<Category::Real, 8>::Value;
using Real10 = typename Type<Category::Real, 10>::Value;
using Real16 = typename Type<Category::Real, 16>::Value;
using Integer4 = typename Type<Category::Integer, 4>::Value;
using Integer8 = typename Type<Category::Integer, 8>::Value;

template<typename R> void basicTests(int rm, Rounding rounding) {
  char desc[64];
  using Word = typename R::Word;
  std::snprintf(
      desc, sizeof desc, "bits=%d, le=%d", R::bits, Word::littleEndian);
  R zero;
  TEST(!zero.IsNegative())(desc);
  TEST(!zero.IsNotANumber())(desc);
  TEST(!zero.IsInfinite())(desc);
  TEST(zero.IsZero())(desc);
  MATCH(0, zero.Exponent())(desc);
  TEST(zero.RawBits().IsZero())(desc);
  MATCH(0, zero.RawBits().ToUInt64())(desc);
  TEST(zero.ABS().RawBits().IsZero())(desc);
  TEST(zero.Negate().RawBits().IEOR(Word::MASKL(1)).IsZero())(desc);
  TEST(zero.Compare(zero) == Relation::Equal)(desc);
  R minusZero{Word{std::uint64_t{1}}.SHIFTL(R::bits - 1)};
  TEST(minusZero.IsNegative())(desc);
  TEST(!minusZero.IsNotANumber())(desc);
  TEST(!minusZero.IsInfinite())(desc);
  TEST(minusZero.IsZero())(desc);
  TEST(minusZero.ABS().RawBits().IsZero())(desc);
  TEST(minusZero.Negate().RawBits().IsZero())(desc);
  MATCH(0, minusZero.Exponent())(desc);
  MATCH(0, minusZero.RawBits().LEADZ())(desc);
  MATCH(1, minusZero.RawBits().POPCNT())(desc);
  TEST(minusZero.Compare(minusZero) == Relation::Equal)(desc);
  TEST(zero.Compare(minusZero) == Relation::Equal)(desc);
  ValueWithRealFlags<R> vr;
  MATCH(0, vr.value.RawBits().ToUInt64())(desc);
  TEST(vr.flags.empty())(desc);
  R nan{Word{std::uint64_t{1}}
            .SHIFTL(R::bits)
            .SubtractSigned(Word{std::uint64_t{1}})
            .value};
  MATCH(R::bits, nan.RawBits().POPCNT())(desc);
  TEST(!nan.IsNegative())(desc);
  TEST(nan.IsNotANumber())(desc);
  TEST(!nan.IsInfinite())(desc);
  TEST(!nan.IsZero())(desc);
  TEST(zero.Compare(nan) == Relation::Unordered)(desc);
  TEST(minusZero.Compare(nan) == Relation::Unordered)(desc);
  TEST(nan.Compare(zero) == Relation::Unordered)(desc);
  TEST(nan.Compare(minusZero) == Relation::Unordered)(desc);
  TEST(nan.Compare(nan) == Relation::Unordered)(desc);
  int significandBits{R::precision - R::implicitMSB};
  int exponentBits{R::bits - significandBits - 1};
  std::uint64_t maxExponent{(std::uint64_t{1} << exponentBits) - 1};
  MATCH(nan.Exponent(), maxExponent)(desc);
  R inf{Word{maxExponent}.SHIFTL(significandBits)};
  TEST(!inf.IsNegative())(desc);
  TEST(!inf.IsNotANumber())(desc);
  TEST(inf.IsInfinite())(desc);
  TEST(!inf.IsZero())(desc);
  TEST(inf.RawBits().CompareUnsigned(inf.ABS().RawBits()) == Ordering::Equal)
  (desc);
  TEST(zero.Compare(inf) == Relation::Less)(desc);
  TEST(minusZero.Compare(inf) == Relation::Less)(desc);
  TEST(nan.Compare(inf) == Relation::Unordered)(desc);
  TEST(inf.Compare(inf) == Relation::Equal)(desc);
  R negInf{Word{maxExponent}.SHIFTL(significandBits).IOR(Word::MASKL(1))};
  TEST(negInf.IsNegative())(desc);
  TEST(!negInf.IsNotANumber())(desc);
  TEST(negInf.IsInfinite())(desc);
  TEST(!negInf.IsZero())(desc);
  TEST(inf.RawBits().CompareUnsigned(negInf.ABS().RawBits()) == Ordering::Equal)
  (desc);
  TEST(inf.RawBits().CompareUnsigned(negInf.Negate().RawBits()) ==
      Ordering::Equal)
  (desc);
  TEST(inf.Negate().RawBits().CompareUnsigned(negInf.RawBits()) ==
      Ordering::Equal)
  (desc);
  TEST(zero.Compare(negInf) == Relation::Greater)(desc);
  TEST(minusZero.Compare(negInf) == Relation::Greater)(desc);
  TEST(nan.Compare(negInf) == Relation::Unordered)(desc);
  TEST(inf.Compare(negInf) == Relation::Greater)(desc);
  TEST(negInf.Compare(negInf) == Relation::Equal)(desc);
  for (std::uint64_t j{0}; j < 63; ++j) {
    char ldesc[128];
    std::uint64_t x{1};
    x <<= j;
    std::snprintf(ldesc, sizeof ldesc, "%s j=%d x=0x%llx rm=%d", desc,
        static_cast<int>(j), static_cast<unsigned long long>(x), rm);
    Integer8 ix{x};
    TEST(!ix.IsNegative())(ldesc);
    MATCH(x, ix.ToUInt64())(ldesc);
    vr = R::FromInteger(ix, rounding);
    TEST(!vr.value.IsNegative())(ldesc);
    TEST(!vr.value.IsNotANumber())(ldesc);
    TEST(!vr.value.IsZero())(ldesc);
    auto ivf = vr.value.template ToInteger<Integer8>();
    if (j > (maxExponent / 2)) {
      TEST(vr.flags.test(RealFlag::Overflow))(ldesc);
      TEST(vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.test(RealFlag::Overflow))(ldesc);
      MATCH(0x7fffffffffffffff, ivf.value.ToUInt64())(ldesc);
    } else {
      TEST(vr.flags.empty())(ldesc);
      TEST(!vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.empty())(ldesc);
      MATCH(x, ivf.value.ToUInt64())(ldesc);
    }
    ix = ix.Negate().value;
    TEST(ix.IsNegative())(ldesc);
    x = -x;
    std::int64_t nx = x;
    MATCH(x, ix.ToUInt64())(ldesc);
    MATCH(nx, ix.ToInt64())(ldesc);
    vr = R::FromInteger(ix);
    TEST(vr.value.IsNegative())(ldesc);
    TEST(!vr.value.IsNotANumber())(ldesc);
    TEST(!vr.value.IsZero())(ldesc);
    ivf = vr.value.template ToInteger<Integer8>();
    if (j > (maxExponent / 2)) {
      TEST(vr.flags.test(RealFlag::Overflow))(ldesc);
      TEST(vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.test(RealFlag::Overflow))(ldesc);
      MATCH(0x8000000000000000, ivf.value.ToUInt64())(ldesc);
    } else {
      TEST(vr.flags.empty())(ldesc);
      TEST(!vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.empty())(ldesc);
      MATCH(x, ivf.value.ToUInt64())(ldesc);
      MATCH(nx, ivf.value.ToInt64())(ldesc);
    }
  }
}

// Takes an integer and distributes its bits across a floating
// point value.  The LSB is used to complement the result.
std::uint32_t MakeReal(std::uint32_t n) {
  int shifts[] = { -1, 31, 23, 30, 22, 0, 24, 29, 25, 28, 26, 1, 16, 21, 2, -1 };
  std::uint32_t x{0};
  for (int j{1}; shifts[j] >= 0; ++j) {
    x |= ((n >> j) & 1) << shifts[j];
  }
  x ^= -(n & 1);
  return x;
}

std::uint64_t MakeReal(std::uint64_t n) {
  int shifts[] = { -1, 63, 52, 62, 51, 0, 53, 61, 54, 60, 55, 59, 1, 16, 50, 2, -1 };
  std::uint64_t x{0};
  for (int j{1}; shifts[j] >= 0; ++j) {
    x |= ((n >> j) & 1) << shifts[j];
  }
  x ^= -(n & 1);
  return x;
}

std::uint32_t NormalizeNaN(std::uint32_t x) {
  if ((x & 0x7f800000) == 0x7f800000 && (x & 0x007fffff) != 0) {
    x = 0x7fe00000;
  }
  return x;
}

std::uint64_t NormalizeNaN(std::uint64_t x) {
  if ((x & 0x7ff0000000000000) == 0x7ff0000000000000 &&
      (x & 0x000fffffffffffff) != 0) {
    x = 0x7ffc000000000000;
  }
  return x;
}

std::uint32_t FlagsToBits(const RealFlags &flags) {
  std::uint32_t bits{0};
#ifndef __clang__
  // TODO: clang support for fenv.h is broken, so tests of flag settings
  // are disabled.
  if (flags.test(RealFlag::Overflow)) {
    bits |= 1;
  }
  if (flags.test(RealFlag::DivideByZero)) {
    bits |= 2;
  }
  if (flags.test(RealFlag::InvalidArgument)) {
    bits |= 4;
  }
  if (flags.test(RealFlag::Underflow)) {
    bits |= 8;
  }
  if (flags.test(RealFlag::Inexact)) {
    bits |= 0x10;
  }
#endif  // __clang__
  return bits;
}

template<typename UINT = std::uint32_t, typename FLT = float, typename REAL>
void inttest(std::int64_t x, int pass, Rounding rounding) {
  union {
    UINT ui;
    FLT f;
  } u;
  ScopedHostFloatingPointEnvironment fpenv;
  Integer8 ix{x};
  ValueWithRealFlags<REAL> real;
  real = real.value.FromInteger(ix, rounding);
#ifndef __clang__  // broken and also slow
  fpenv.ClearFlags();
#endif
  FLT fcheck = x;  // TODO unsigned too
  auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
  u.f = fcheck;
  UINT rcheck{NormalizeNaN(u.ui)};
  UINT check = real.value.RawBits().ToUInt64();
  MATCH(rcheck, check)("%d 0x%llx", pass, x);
  MATCH(actualFlags, FlagsToBits(real.flags))("%d 0x%llx", pass, x);
}

template<typename UINT = std::uint32_t, typename FLT = float, typename REAL = Real4>
void subsetTests(int pass, Rounding rounding, std::uint32_t opds) {
  for (int j{0}; j < 63; ++j) {
    std::int64_t x{1};
    x <<= j;
    inttest<UINT,FLT,REAL>(x, pass, rounding);
    inttest<UINT,FLT,REAL>(-x, pass, rounding);
  }
  inttest<UINT,FLT,REAL>(0, pass, rounding);
  inttest<UINT,FLT,REAL>(static_cast<std::int64_t>(0x8000000000000000), pass, rounding);

  union {
    UINT ui;
    FLT f;
  } u;
  ScopedHostFloatingPointEnvironment fpenv;

  for (UINT j{0}; j < opds; ++j) {
    UINT rj{MakeReal(j)};
    u.ui = rj;
    FLT fj{u.f};
    REAL x{typename REAL::Word{std::uint64_t{rj}}};
    for (UINT k{0}; k < opds; ++k) {
      UINT rk{MakeReal(k)};
      u.ui = rk;
      FLT fk{u.f};
      REAL y{typename REAL::Word{std::uint64_t{rk}}};
      {
        ValueWithRealFlags<REAL> sum{x.Add(y, rounding)};
#ifndef __clang__  // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj + fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = sum.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
          ("%d 0x%llx + 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
        MATCH(actualFlags, FlagsToBits(sum.flags))
          ("%d 0x%llx + 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
      }
      {
        ValueWithRealFlags<REAL> diff{x.Subtract(y, rounding)};
#ifndef __clang__  // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj - fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = diff.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
          ("%d 0x%llx - 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
        MATCH(actualFlags, FlagsToBits(diff.flags))
          ("%d 0x%llx - 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
      }
      {
        ValueWithRealFlags<REAL> prod{x.Multiply(y, rounding)};
#ifndef __clang__  // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj * fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = prod.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
          ("%d 0x%llx * 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
        MATCH(actualFlags, FlagsToBits(prod.flags))
          ("%d 0x%llx * 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
      }
      {
        ValueWithRealFlags<REAL> quot{x.Divide(y, rounding)};
#ifndef __clang__  // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj / fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = quot.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
          ("%d 0x%llx / 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
        MATCH(actualFlags, FlagsToBits(quot.flags))
          ("%d 0x%llx / 0x%llx", pass, static_cast<long long>(rj),
           static_cast<long long>(rk));
      }
    }
  }
}

void roundTest(int rm, Rounding rounding, std::uint32_t opds) {
  basicTests<Real2>(rm, rounding);
  basicTests<Real4>(rm, rounding);
  basicTests<Real8>(rm, rounding);
  basicTests<Real10>(rm, rounding);
  basicTests<Real16>(rm, rounding);
  ScopedHostFloatingPointEnvironment::SetRounding(rounding);
  subsetTests<std::uint32_t, float, Real4>(rm, rounding, opds);
  subsetTests<std::uint64_t, double, Real8>(rm, rounding, opds);
}

int main() {
  std::uint32_t opds{512};  // for quick testing by default
  if (const char *p{std::getenv("REAL_TEST_OPERANDS")}) {
    // Use 8192 or 16384 for more exhaustive testing.
    opds = std::atol(p);
  }
  roundTest(0, Rounding::TiesToEven, opds);
  roundTest(1, Rounding::ToZero, opds);
  roundTest(2, Rounding::Up, opds);
  roundTest(3, Rounding::Down, opds);
  // TODO: how to test Rounding::TiesAwayFromZero on x86?
  return testing::Complete();
}
