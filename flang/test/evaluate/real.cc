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

#include "../../lib/evaluate/integer.h"
#include "../../lib/evaluate/real.h"
#include "testing.h"
#include "fp-testing.h"
#include <cstdio>

using namespace Fortran::evaluate;

template<typename R> void tests() {
  char desc[64];
  using Word = typename R::Word;
  std::snprintf(desc, sizeof desc, "bits=%d, le=%d",
                R::bits, Word::littleEndian);
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
  R nan{Word{std::uint64_t{1}}.SHIFTL(R::bits).SubtractSigned(Word{std::uint64_t{1}}).value};
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
  TEST(inf.RawBits().CompareUnsigned(inf.ABS().RawBits()) == Ordering::Equal)(desc);
  TEST(zero.Compare(inf) == Relation::Less)(desc);
  TEST(minusZero.Compare(inf) == Relation::Less)(desc);
  TEST(nan.Compare(inf) == Relation::Unordered)(desc);
  TEST(inf.Compare(inf) == Relation::Equal)(desc);
  R negInf{Word{maxExponent}.SHIFTL(significandBits).IOR(Word::MASKL(1))};
  TEST(negInf.IsNegative())(desc);
  TEST(!negInf.IsNotANumber())(desc);
  TEST(negInf.IsInfinite())(desc);
  TEST(!negInf.IsZero())(desc);
  TEST(inf.RawBits().CompareUnsigned(negInf.ABS().RawBits()) == Ordering::Equal)(desc);
  TEST(inf.RawBits().CompareUnsigned(negInf.Negate().RawBits()) == Ordering::Equal)(desc);
  TEST(inf.Negate().RawBits().CompareUnsigned(negInf.RawBits()) == Ordering::Equal)(desc);
  TEST(zero.Compare(negInf) == Relation::Greater)(desc);
  TEST(minusZero.Compare(negInf) == Relation::Greater)(desc);
  TEST(nan.Compare(negInf) == Relation::Unordered)(desc);
  TEST(inf.Compare(negInf) == Relation::Greater)(desc);
  TEST(negInf.Compare(negInf) == Relation::Equal)(desc);
  for (std::uint64_t j{0}; j < 63; ++j) {
    std::uint64_t x{1};
    x <<= j;
    Integer<64> ix{x};
    TEST(!ix.IsNegative())("%s,%d,0x%llx",desc,j,x);
    MATCH(x, ix.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    vr = R::ConvertSigned(ix);
    TEST(!vr.value.IsNegative())("%s,%d,0x%llx",desc,j,x);
    TEST(!vr.value.IsNotANumber())("%s,%d,0x%llx",desc,j,x);
    TEST(!vr.value.IsZero())("%s,%d,0x%llx",desc,j,x);
    auto ivf = vr.value.template ToInteger<Integer<64>>();
    if (j > (maxExponent / 2)) {
      TEST(vr.flags.test(RealFlag::Overflow))(desc);
      TEST(vr.value.IsInfinite())("%s,%d,0x%llx",desc,j,x);
      TEST(ivf.flags.test(RealFlag::Overflow))("%s,%d,0x%llx",desc,j,x);
      MATCH(0x7fffffffffffffff, ivf.value.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    } else {
      TEST(vr.flags.empty())(desc);
      TEST(!vr.value.IsInfinite())("%s,%d,0x%llx",desc,j,x);
      TEST(ivf.flags.empty())("%s,%d,0x%llx",desc,j,x);
      MATCH(x, ivf.value.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    }
    ix = ix.Negate().value;
    TEST(ix.IsNegative())("%s,%d,0x%llx",desc,j,x);
    x = -x;
    std::int64_t nx = x;
    MATCH(x, ix.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    MATCH(nx, ix.ToInt64())("%s,%d,0x%llx",desc,j,x);
    vr = R::ConvertSigned(ix);
    TEST(vr.value.IsNegative())("%s,%d,0x%llx",desc,j,x);
    TEST(!vr.value.IsNotANumber())("%s,%d,0x%llx",desc,j,x);
    TEST(!vr.value.IsZero())("%s,%d,0x%llx",desc,j,x);
    ivf = vr.value.template ToInteger<Integer<64>>();
    if (j > (maxExponent / 2)) {
      TEST(vr.flags.test(RealFlag::Overflow))(desc);
      TEST(vr.value.IsInfinite())("%s,%d,0x%llx",desc,j,x);
      TEST(ivf.flags.test(RealFlag::Overflow))("%s,%d,0x%llx",desc,j,x);
      MATCH(0x8000000000000000, ivf.value.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    } else {
      TEST(vr.flags.empty())(desc);
      TEST(!vr.value.IsInfinite())("%s,%d,0x%llx",desc,j,x);
      TEST(ivf.flags.empty())("%s,%d,0x%llx",desc,j,x);
      MATCH(x, ivf.value.ToUInt64())("%s,%d,0x%llx",desc,j,x);
      MATCH(nx, ivf.value.ToInt64())("%s,%d,0x%llx",desc,j,x);
    }
  }
}

// Takes a 12-bit number and distributes its bits across a 32-bit single
// precision real.  All sign and exponent bit positions are tested, plus
// the upper two bits and lowest bit in the significand.
std::uint32_t MakeReal(std::uint32_t n) {
  return (n << 23) | (n >> 11) | ((n & 6) << 20);
}

std::uint32_t NormalizeNaN(std::uint32_t x) {
  if ((x & 0x7f800000) == 0x7f800000 &&
      (x & 0x007fffff) != 0) {
    x = 0x7fe00000;
  }
  return x;
}

void subset32bit() {
  union {
    std::uint32_t u32;
    float f;
  } u;
  for (std::uint32_t j{0}; j < 4096; ++j) {
    std::uint32_t rj{MakeReal(j)};
    u.u32 = rj;
    float fj{u.f};
    RealKind4 x{Integer<32>{std::uint64_t{rj}}};
    for (std::uint32_t k{0}; k < 4096; ++k) {
      std::uint32_t rk{MakeReal(k)};
      u.u32 = rk;
      float fk{u.f};
      RealKind4 y{Integer<32>{std::uint64_t{rk}}};
      { ValueWithRealFlags<RealKind4> sum{x.Add(y)};
        ScopedHostFloatingPointEnvironment fpenv;
        float fcheck{fj + fk};
        u.f = fcheck;
        std::uint32_t rcheck{NormalizeNaN(u.u32)};
        std::uint32_t check = sum.value.RawBits().ToUInt64();
        MATCH(rcheck, check)("0x%x + 0x%x", rj, rk);
      }
      { ValueWithRealFlags<RealKind4> diff{x.Subtract(y)};
        ScopedHostFloatingPointEnvironment fpenv;
        float fcheck{fj - fk};
        u.f = fcheck;
        std::uint32_t rcheck{NormalizeNaN(u.u32)};
        std::uint32_t check = diff.value.RawBits().ToUInt64();
        MATCH(rcheck, check)("0x%x - 0x%x", rj, rk);
      }
#if 0
      { ValueWithRealFlags<RealKind4> prod{x.Multiply(y)};
        ScopedHostFloatingPointEnvironment fpenv;
        float fcheck{fj * fk};
        u.f = fcheck;
        std::uint32_t rcheck{NormalizeNaN(u.u32)};
        std::uint32_t check = prod.value.RawBits().ToUInt64();
        MATCH(rcheck, check)("0x%x * 0x%x", rj, rk);
      }
      { ValueWithRealFlags<RealKind4> quot{x.Divide(y)};
        ScopedHostFloatingPointEnvironment fpenv;
        float fcheck{fj * fk};
        u.f = fcheck;
        std::uint32_t rcheck{NormalizeNaN(u.u32)};
        std::uint32_t check = quot.value.RawBits().ToUInt64();
        MATCH(rcheck, check)("0x%x / 0x%x", rj, rk);
      }
#endif
    }
  }
}

int main() {
  tests<RealKind2>();
  tests<RealKind4>();
  tests<RealKind8>();
  tests<RealKind10>();
  tests<RealKind16>();
  subset32bit();
  return testing::Complete();
}
