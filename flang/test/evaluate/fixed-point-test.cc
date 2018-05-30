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
  std::snprintf(desc, sizeof desc, "BITS=%d, PARTBITS=%d, sizeof(Part)=%d",
      BITS, FP::partBits, static_cast<int>(sizeof(typename FP::Part)));
  FP zero;
  TEST(zero.IsZero())(desc);
  for (std::uint64_t x{0}; x <= maxUnsignedValue; ++x) {
    FP a{x};
    COMPARE(x, ==, a.ToUInt64())(desc);
    FP copy{a};
    COMPARE(x, ==, copy.ToUInt64())(desc);
    copy = a;
    COMPARE(x, ==, copy.ToUInt64())(desc);
    COMPARE(x == 0, ==, a.IsZero())("%s, x=0x%llx", desc, x);
    copy.OnesComplement();
    COMPARE(x ^ maxUnsignedValue, ==, copy.ToUInt64())("%s, x=0x%llx", desc, x);
    copy = a;
    bool over{copy.TwosComplement()};
    COMPARE(over, ==, x == std::uint64_t{1} << (BITS - 1))
    ("%s, x=0x%llx", desc, x);
    COMPARE(-x & maxUnsignedValue, ==, copy.ToUInt64())
    ("%s, x=0x%llx", desc, x);
    int lzbc{a.LeadingZeroBitCount()};
    COMPARE(lzbc, >=, 0)("%s, x=0x%llx", desc, x);
    COMPARE(lzbc, <=, BITS)("%s, x=0x%llx", desc, x);
    COMPARE(x == 0, ==, lzbc == BITS)("%s, x=0x%llx, lzbc=%d", desc, x, lzbc);
    std::uint64_t lzcheck{std::uint64_t{1} << (BITS - lzbc)};
    COMPARE(x, <, lzcheck)("%s, x=0x%llx, lzbc=%d", desc, x, lzbc);
    COMPARE(x + x + !x, >=, lzcheck)("%s, x=0x%llx, lzbc=%d", desc, x, lzbc);
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
      copy = a;
      copy.ShiftLeft(count);
      COMPARE((x << count) & maxUnsignedValue, ==, copy.ToUInt64())
      ("%s, x=0x%llx, count=%d", desc, x, count);
      copy = a;
      copy.ShiftRightLogical(count);
      COMPARE(x >> count, ==, copy.ToUInt64())
      ("%s, x=0x%llx, count=%d", desc, x, count);
      copy = a;
      copy.ShiftLeft(-count);
      COMPARE(x >> count, ==, copy.ToUInt64())
      ("%s, x=0x%llx, count=%d", desc, x, count);
      copy = a;
      copy.ShiftRightLogical(-count);
      COMPARE((x << count) & maxUnsignedValue, ==, copy.ToUInt64())
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
      ("%s, x=0x%llx %lld %d, y=0x%llx %lld %d", desc, x, sx, a.IsNegative(), y,
          sy, b.IsNegative());
      copy = a;
      copy.And(b);
      COMPARE(x & y, ==, copy.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      copy = a;
      copy.Or(b);
      COMPARE(x | y, ==, copy.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      copy = a;
      copy.Xor(b);
      COMPARE(x ^ y, ==, copy.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      copy = a;
      bool carry{copy.AddUnsigned(b)};
      COMPARE(x + y, ==, copy.ToUInt64() + (std::uint64_t{carry} << BITS))
      ("%s, x=0x%llx, y=0x%llx, carry=%d", desc, x, y, carry);
      copy = a;
      over = copy.AddSigned(b);
      COMPARE((sx + sy) & maxUnsignedValue, ==, copy.ToUInt64())
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      COMPARE(over, ==,
          sx + sy < mostNegativeSignedValue || sx + sy > maxPositiveSignedValue)
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      copy = a;
      over = copy.SubtractSigned(b);
      COMPARE((sx - sy) & maxUnsignedValue, ==, copy.ToUInt64())
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      COMPARE(over, ==,
          sx - sy < mostNegativeSignedValue || sx - sy > maxPositiveSignedValue)
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      copy = a;
      FP upper;
      copy.MultiplyUnsigned(b, upper);
      COMPARE(x * y, ==, (upper.ToUInt64() << BITS) ^ copy.ToUInt64())
      ("%s, x=0x%llx, y=0x%llx, lower=0x%llx, upper=0x%llx", desc, x, y,
          copy.ToUInt64(), upper.ToUInt64());
      copy = a;
      copy.MultiplySigned(b, upper);
      COMPARE((sx * sy) & maxUnsignedValue, ==, copy.ToUInt64())
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      COMPARE(((sx * sy) >> BITS) & maxUnsignedValue, ==, upper.ToUInt64())
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      copy = a;
      FP rem;
      COMPARE(y == 0, ==, copy.DivideUnsigned(b, rem))
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      if (y == 0) {
        COMPARE(maxUnsignedValue, ==, copy.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        COMPARE(0, ==, rem.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      } else {
        COMPARE(x / y, ==, copy.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        COMPARE(x % y, ==, rem.ToUInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      }
      copy = a;
      bool badCase{sx == mostNegativeSignedValue &&
          ((sy == -1 && sx != sy) || (BITS == 1 && sx == sy))};
      COMPARE(y == 0 || badCase, ==, copy.DivideSigned(b, rem))
      ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      if (y == 0) {
        if (sx >= 0) {
          COMPARE(maxPositiveSignedValue, ==, copy.ToInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        } else {
          COMPARE(mostNegativeSignedValue, ==, copy.ToInt64())
          ("%s, x=0x%llx, y=0x%llx", desc, x, y);
        }
        COMPARE(0, ==, rem.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      } else if (badCase) {
        COMPARE(x, ==, copy.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
        COMPARE(0, ==, rem.ToUInt64())("%s, x=0x%llx, y=0x%llx", desc, x, y);
      } else {
        COMPARE(sx / sy, ==, copy.ToInt64())
        ("%s, x=0x%llx %lld, y=0x%llx %lld; unsigned 0x%llx", desc, x, sx, y,
            sy, copy.ToUInt64());
        COMPARE(sx - sy * (sx / sy), ==, rem.ToInt64())
        ("%s, x=0x%llx, y=0x%llx", desc, x, y);
      }
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
  // exhaustiveTesting<15>();
  // exhaustiveTesting<16>();
  return testing::Complete();
}
