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
  MATCH(0, vr.flags)(desc);
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
    MATCH(x, ix.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    vr = R::ConvertSigned(ix);
    TEST(!vr.value.IsNegative())("%s,%d,0x%llx",desc,j,x);
    TEST(!vr.value.IsNotANumber())("%s,%d,0x%llx",desc,j,x);
    TEST(!vr.value.IsZero())("%s,%d,0x%llx",desc,j,x);
    auto ivf = vr.value.template ToInteger<Integer<64>>();
    if (j > (maxExponent / 2)) {
      MATCH(RealFlag::Overflow, vr.flags)(desc);
      TEST(vr.value.IsInfinite())("%s,%d,0x%llx",desc,j,x);
      MATCH(RealFlag::Overflow, ivf.flags)("%s,%d,0x%llx",desc,j,x);
      MATCH(0x7fffffffffffffff, ivf.value.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    } else {
      MATCH(RealFlag::Ok, vr.flags)(desc);
      TEST(!vr.value.IsInfinite())("%s,%d,0x%llx",desc,j,x);
      MATCH(RealFlag::Ok, ivf.flags)("%s,%d,0x%llx",desc,j,x);
      MATCH(x, ivf.value.ToUInt64())("%s,%d,0x%llx",desc,j,x);
    }
  }
}

int main() {
  tests<RealKind2>();
  tests<RealKind4>();
  tests<RealKind8>();
  tests<RealKind10>();
  tests<RealKind16>();
  return testing::Complete();
}
