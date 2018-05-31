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

#include "testing.h"
#include "../../lib/evaluate/fixed-point.h"
#include <cstdio>

using Fortran::evaluate::FixedPoint;
using Fortran::evaluate::Ordering;

template<int BITS, typename FP = FixedPoint<BITS>> void exhaustiveTesting() {
  COMPARE(BITS, ==, FP::bits);
  std::uint64_t maxUnsignedValue{(std::uint64_t{1} << BITS) - 1};
  std::int64_t maxPositiveSignedValue{(std::int64_t{1} << (BITS - 1)) - 1};
  std::int64_t mostNegativeSignedValue{-(std::int64_t{1} << (BITS - 1))};
  char desc[64];
  std::snprintf(desc, sizeof desc, "BITS=%d, PARTBITS=%d, sizeof(Part)=%d, LE=%d",
      BITS, FP::partBits, static_cast<int>(sizeof(typename FP::Part)),
      FP::littleEndian);
  FP zero;
  TEST(zero.IsZero())(desc);
  for (std::uint64_t x{0}; x <= maxUnsignedValue; ++x) {
    FP a{x};
    MATCH(x, a.ToUInt64())(desc);
    FP copy{a};
    MATCH(x, copy.ToUInt64())(desc);
    copy = a;
    MATCH(x, copy.ToUInt64())(desc);
    MATCH(x == 0, a.IsZero())("%s, x=0x%llx", desc, x);
    FP t{a.NOT()};
    MATCH(x ^ maxUnsignedValue, t.ToUInt64())("%s, x=0x%llx", desc, x);
    auto negated{a.Negate()};
    MATCH(x == std::uint64_t{1} << (BITS - 1), negated.overflow)
      ("%s, x=0x%llx", desc, x);
    MATCH(negated.value.ToUInt64(), -x & maxUnsignedValue)
      ("%s, x=0x%llx", desc, x);
    int lzbc{a.LEADZ()};
    COMPARE(lzbc, >=, 0)("%s, x=0x%llx", desc, x);
    COMPARE(lzbc, <=, BITS)("%s, x=0x%llx", desc, x);
    MATCH(x == 0, lzbc == BITS)("%s, x=0x%llx, lzbc=%d", desc, x, lzbc);
    std::uint64_t lzcheck{std::uint64_t{1} << (BITS - lzbc)};
    COMPARE(x, <, lzcheck)("%s, x=0x%llx, lzbc=%d", desc, x, lzbc);
    COMPARE(x + x + !x, >=, lzcheck)("%s, x=0x%llx, lzbc=%d", desc, x, lzbc);
    int popcheck{0};
    for (int j{0}; j < BITS; ++j) {
      popcheck += (x >> j) & 1;
    }
    MATCH(popcheck, a.POPCNT())("%s, x=0x%llx", desc, x);
    MATCH(popcheck & 1, a.POPPAR())("%s, x=0x%llx", desc, x);
    Ordering ord{Ordering::Equal};
    std::int64_t sx = x;
    if (x + x > maxUnsignedValue) {
      TEST(a.IsNegative())("%s, x=0x%llx", desc, x);
      sx = x | (~std::uint64_t{0} << BITS);
      TEST(sx < 0)("%s, x=0x%llx %lld", desc, x, sx);
      ord = Ordering::Less;
    } else {
      TEST(!a.IsNegative())("%s, x=0x%llx", desc, x);
      TEST(sx >= 0)("%s, x=0x%llx %lld", desc, x, sx);
      if (sx > 0) {
        ord = Ordering::Greater;
      } else {
        ord = Ordering::Equal;
      }
    }
    TEST(sx == a.ToInt64())("%s, x=0x%llx %lld", desc, x, sx);
    TEST(a.CompareToZeroSigned() == ord)("%s, x=0x%llx %lld", desc, x, sx);
    for (int count{0}; count <= BITS + 1; ++count) {
      t = a.SHIFTL(count);
      MATCH((x << count) & maxUnsignedValue, t.ToUInt64())
        ("%s, x=0x%llx, count=%d", desc, x, count);
      t = a.ISHFT(count);
      MATCH((x << count) & maxUnsignedValue, t.ToUInt64())
        ("%s, x=0x%llx, count=%d", desc, x, count);
      t = a.SHIFTR(count);
      MATCH(x >> count, t.ToUInt64())
        ("%s, x=0x%llx, count=%d", desc, x, count);
      t = a.ISHFT(-count);
      MATCH(x >> count, t.ToUInt64())("%s, x=0x%llx, count=%d", desc, x, count);
      t = a.SHIFTA(count);
      std::uint64_t fill{-(x >> (BITS-1))};
      std::uint64_t sra{count >= BITS ? fill : (x >> count) | (fill << (BITS-count))};
      MATCH(sra, t.ToInt64())
        ("%s, x=0x%llx, count=%d", desc, x, count);
    }
    for (std::uint64_t y{0}; y <= maxUnsignedValue; ++y) {
      std::int64_t sy = y;
      if (y + y > maxUnsignedValue) {
        sy = y | (~std::uint64_t{0} << BITS);
      }
      FP b{y};
      if (x < y) {
        ord = Ordering::Less;
      } else if (x > y) {
        ord = Ordering::Greater;
      } else {
        ord = Ordering::Equal;
      }
      TEST(a.CompareUnsigned(b) == ord)("%s, x=0x%llx, y=0x%llx", desc, x, y);
      if (sx < sy) {
        ord = Ordering::Less;
      } else if (sx > sy) {
        ord = Ordering::Greater;
      } else {
        ord = Ordering::Equal;
      }
      TEST(a.CompareSigned(b) == ord)
        ("%s, x=0x%llx %lld %d, y=0x%llx %lld %d", desc, x, sx,
         a.IsNegative(), y, sy, b.IsNegative());
      t = a.IAND(b);
      MATCH(x & y, t.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      t = a.IOR(b);
      MATCH(x | y, t.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      t = a.IEOR(b);
      MATCH(x ^ y, t.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      auto sum{a.AddUnsigned(b)};
      COMPARE(x + y, ==, sum.value.ToUInt64() + (std::uint64_t{sum.carry} << BITS))
        ("%s, x=0x%llx, y=0x%llx, carry=%d", desc, x, y, sum.carry);
      auto ssum{a.AddSigned(b)};
      MATCH((sx + sy) & maxUnsignedValue, ssum.value.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      MATCH(sx + sy < mostNegativeSignedValue ||
            sx + sy > maxPositiveSignedValue, ssum.overflow)
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      auto diff{a.SubtractSigned(b)};
      MATCH((sx - sy) & maxUnsignedValue, diff.value.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      MATCH(sx - sy < mostNegativeSignedValue ||
            sx - sy > maxPositiveSignedValue, diff.overflow)
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      auto product{a.MultiplyUnsigned(b)};
      MATCH(x * y, (product.upper.ToUInt64() << BITS) ^ product.lower.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx, lower=0x%llx, upper=0x%llx", desc, x, y,
          product.lower.ToUInt64(), product.upper.ToUInt64());
      product = a.MultiplySigned(b);
      MATCH((sx * sy) & maxUnsignedValue, product.lower.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      MATCH(((sx * sy) >> BITS) & maxUnsignedValue, product.upper.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      auto quot{a.DivideUnsigned(b)};
      MATCH(y == 0, quot.divisionByZero)("%s, x=0x%llx, y=0x%llx", desc, x, y);
      if (y == 0) {
        MATCH(maxUnsignedValue, quot.quotient.ToUInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        MATCH(0, quot.remainder.ToUInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      } else {
        MATCH(x / y, quot.quotient.ToUInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        MATCH(x % y, quot.remainder.ToUInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      }
      quot = a.DivideSigned(b);
      bool badCase{sx == mostNegativeSignedValue &&
          ((sy == -1 && sx != sy) || (BITS == 1 && sx == sy))};
      MATCH(y == 0, quot.divisionByZero)("%s, x=0x%llx, y=0x%llx", desc, x, y);
      MATCH(badCase, quot.overflow)("%s, x=0x%llx, y=0x%llx", desc, x, y);
      if (y == 0) {
        if (sx >= 0) {
          MATCH(maxPositiveSignedValue, quot.quotient.ToInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        } else {
          MATCH(mostNegativeSignedValue, quot.quotient.ToInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        }
        MATCH(0, quot.remainder.ToUInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      } else if (badCase) {
        MATCH(x, quot.quotient.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
        MATCH(0, quot.remainder.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      } else {
        MATCH(sx / sy, quot.quotient.ToInt64())
          ("%s, x=0x%llx %lld, y=0x%llx %lld; unsigned 0x%llx", desc, x, sx, y,
            sy, quot.quotient.ToUInt64());
        MATCH(sx - sy * (sx / sy), quot.remainder.ToInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      }
      // TODO test ABS, B[GL][ET], BTEST, DIM, HUGE, MODULO, ISHFTC, DSHIFTL/R
      // TODO test IBCLR, IBSET, IBITS, MAX, MIN, MERGE_BITS, RANGE, SIGN
      // TODO test TRAILZ
    }
  }
}

int main() {
  TEST(Reverse(Ordering::Less) == Ordering::Greater);
  TEST(Reverse(Ordering::Greater) == Ordering::Less);
  TEST(Reverse(Ordering::Equal) == Ordering::Equal);
  exhaustiveTesting<1>();
  exhaustiveTesting<2>();
  exhaustiveTesting<7>();
  exhaustiveTesting<8>();
  exhaustiveTesting<9>();
  exhaustiveTesting<9, FixedPoint<9, 1>>();
  exhaustiveTesting<9, FixedPoint<9, 1, std::uint8_t, std::uint16_t>>();
  exhaustiveTesting<9, FixedPoint<9, 2>>();
  exhaustiveTesting<9, FixedPoint<9, 2, std::uint8_t, std::uint16_t>>();
  exhaustiveTesting<9, FixedPoint<9, 8, std::uint8_t, std::uint16_t>>();
  exhaustiveTesting<9, FixedPoint<9, 8, std::uint8_t, std::uint16_t, false>>();
  return testing::Complete();
}
