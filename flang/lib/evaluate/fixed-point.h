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

#ifndef FORTRAN_EVALUATE_FIXED_POINT_H_
#define FORTRAN_EVALUATE_FIXED_POINT_H_

// Emulates integers of a nearly arbitrary fixed size for use when the C++
// environment does not support it.  The size must be some multiple of
// 32 bits.  Signed and unsigned operations are distinct.

#include "leading-zero-bit-count.h"
#include <cinttypes>
#include <cstddef>

namespace Fortran::evaluate {

enum class Ordering { Less, Equal, Greater };
static constexpr Ordering Reverse Ordering ordering) {
  if (ordering == Ordering::Less) {
    return Ordering::Greater;
  }
  if (ordering == Ordering::Greater) {
    return Ordering::Less;
  }
  return Ordering::Equal;
}

typedef <int BITS>
class FixedPoint {
private:
  using Part = std::uint32_t;
  using BigPart = std::uint64_t;
  static constexpr int bits{BITS};
  static constexpr int partBits{CHAR_BIT * sizeof(Part)};
  static_assert(bits >= partBits);
  static_assert(sizeof(BigPart) == 2 * partBits);
  static constexpr int parts{bits / partBits};
  static_assert(bits * partBits == parts);  // no partial part

public:
  FixedPoint() = delete;
  constexpr FixedPoint(const FixedPoint &) = default;
  constexpr FixedPoint(std::uint64_t n) {
    for (int j{0}; j < parts; ++j) {
      part_[j] = n;
      if constexpr (partBits < 64) {
        n >>= partBits;
      } else {
        n = 0;
      }
    }
  }
  constexpr FixedPoint(std::int64_t n) {
    std::int64_t signExtension{-(n < 0) << partBits};
    for (int j{0}; j < parts; ++j) {
      part_[j] = n;
      if constexpr (partBits < 64) {
        n = (n >> partBits) | signExtension;
      } else {
        n = signExtension;
      }
    }
  }

  constexpr FixedPoint &operator=(const FixedPoint &) = default;

  constexpr Ordering CompareToZeroUnsigned() const {
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        return Ordering::Greater;
      }
    }
    return Ordering::Equal;
  }

  constexpr Ordering CompareToZeroSigned() const {
    if (IsNegative()) {
      return Ordering::Less;
    }
    return CompareToZeroUnsigned();
  }

  constexpr Ordering CompareUnsigned(const FixedPoint &y) const {
    for (int j{parts}; j-- > 0; ) {
      if (part_[j] > y.part_[j]) {
        return Ordering::Greater;
      }
      if (part_[j] < y.part_[j]) {
        return Ordering::Less;
      }
    }
    return Ordering::Equal;
  }

  constexpr Ordering CompareSigned(const FixedPoint &y) const {
    if (IsNegative()) {
      if (!y.IsNegative()) {
        return Ordering::Less;
      }
      return Reverse(CompareUnsigned(y));
    } else if (y.IsNegative()) {
      return Ordering::Greater;
    } else {
      return CompareUnsigned(y);
    }
  }

  constexpr int LeadingZeroBitCount() const {
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        return (j * partBits) + evaluate::LeadingZeroBitCount(part_[j]);
      }
    }
    return bits;
  }

  constexpr std::uint64_t ToUInt64() const {
    std::uint64_t n{0};
    int filled{0};
    static constexpr int toFill{bits < 64 ? bits : 64};
    for (int j{0}; filled < 64; ++j, filled += partBits) {
      n |= part_[j] << filled;
    }
    return n;
  }

  constexpr std::int64_t ToInt64() const {
    return static_cast<std::int64_t>(ToUInt64());
  }

  constexpr void OnesComplement() {
    for (int j{0}; j < parts; ++j) {
      part_[j] = ~part_[j];
    }
  }

  // Returns true on overflow (i.e., negating the most negative number)
  constexpr bool TwosComplement() {
    Part carry{1};
    for (int j{0}; j < parts; ++j) {
      Part newCarry{part_[j] == 0 && carry};
      part_[j] = ~part_[j] + carry;
      carry = newCarry;
    }
    return carry != IsNegative();
  }

  constexpr void And(const FixedPoint &y) {
    for (int j{0}; j < parts; ++j) {
      part_[j] &= y.part_[j];
    }
  }

  constexpr void Or(const FixedPoint &y) {
    for (int j{0}; j < parts; ++j) {
      part_[j] |= y.part_[j];
    }
  }

  constexpr void Xor(const FixedPoint &y) {
    for (int j{0}; j < parts; ++j) {
      part_[j] ^= y.part_[j];
    }
  }

  constexpr void ShiftLeft(int count) {
    if (count < 0) {
      ShiftRight(-count);
    } else {
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{parts-1};
      if (bitShift == 0) {
        for (; j >= shiftParts; --j) {
          part_[j] = part_[j - shiftParts];
        }
        for (; j >= 0; --j) {
          part_[j] = 0;
        }
      } else {
        for (; j > shiftParts; --j) {
          part_[j] = (part_[j - shiftParts] << bitShift) |
                     (part_[j - shiftParts - 1] >> (partBits - bitShift);
        }
        if (j == shiftParts) {
          part_[j--] = part_[0] << bitShift;
        }
        for (; j >= 0; --j) {
          part_[j] = 0;
        }
      }
    }
  }

  constexpr void ShiftRightLogical(int count) {  // i.e., unsigned
    if (count < 0) {
      ShiftLeft(-count);
    } else {
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{0};
      if (bitShift == 0) {
        for (; j + shiftParts < parts; ++j) {
          part_[j] = part_[j + shiftParts];
        }
        for (; j < parts; ++j) {
          part_[j] = 0;
        }
      } else {
        for (; j + shiftParts + 1 < parts; ++j) {
          part_[j] = (part_[j + shiftParts] >> bitShift) |
                     (part_[j + shiftParts + 1] << (partBits - bitShift);
        }
        if (j + shiftParts + 1 == parts) {
          part_[j++] = part_[parts - 1] >> bitShift;
        }
        for (; j < parts; ++j) {
          part_[j] = 0;
        }
      }
    }
  }

  // Returns carry out.
  constexpr bool AddUnsigned(const FixedPoint &y, bool carryIn{false}) {
    BigPart carry{carryIn};
    for (int j{0}; j < parts; ++j) {
      carry += part_[j];
      part_[j] = carry += y.part_[j];
      carry >>= 32;
    }
    return carry != 0;
  }

  // Returns true on overflow.
  constexpr bool AddSigned(const FixedPoint &y) {
    bool carry{AddUnsigned(y)};
    return carry != IsNegative();
  }

  // Returns true on overflow.
  constexpr bool SubtractSigned(const FixedPoint &y) {
    FixedPoint minusy{y};
    minusy.TwosComplement();
    return AddSigned(minusy);
  }

  // Overwrites *this with lower half of full product.
  constexpr void MultiplyUnsigned(const FixedPoint &y, FixedPoint &upper) {
    Part product[2 * parts]{};  // little-endian full product
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        for (int k{0}; k < parts; ++k) {
          if (y.part_[k] != 0) {
            BigPart x{part_[j]};
            x *= y.part_[k];
            for (int to{j+k}; xy != 0; ++to) {
              product[to] = xy += product[to];
              xy >>= partBits;
            }
          }
        }
      }
    }
    for (int j{0}; j < parts; ++j) {
      part_[j] = product[j];
      upper.part_[j] = product[j + parts];
    }
  }

  // Overwrites *this with lower half of full product.
  constexpr void MultiplySigned(const FixedPoint &y, FixedPoint &upper) {
    bool yIsNegative{y.IsNegative()};
    FixedPoint yprime{y};
    if (yIsNegative) {
      yprime.TwosComplement();
    }
    bool isNegative{IsNegative()};
    if (isNegative) {
      TwosComplement();
    }
    MultiplyUnsigned(yprime, upper);
    if (isNegative != yIsNegative) {
      OnesComplement();
      upper.OnesComplement();
      FixedPoint one{std::uint64_t{1}};
      if (AddUnsigned(one)) {
        upper.AddUnsigned(one);
      }
    }
  }

  // Overwrites *this with quotient.
  constexpr void DivideUnsigned(const FixedPoint &divisor, FixedPoint &remainder) {
    FixedPoint top{*this};
    *this = remainder = FixedPoint{0};
    int bitsDone{top.LeadingZeroBitCount()};
    top.ShiftLeft(bitsDone);
    for (; bitsDone < bits; ++bitsDone) {
      remainder.AddUnsigned(remainder, top.AddUnsigned(top));
      bool nextBit{remainder.CompareUnsigned(divisor) != Ordering::Less};
      quotient.AddUnsigned(quotient, nextBit);
      if (nextBit) {
        remainder.SubtractSigned(divisor);
      }
    }
  }

  // Overwrites *this with quotient.  Returns true on overflow (viz.,
  // the most negative value divided by -1) or division by zero.
  constexpr bool DivideSigned(FixedPoint divisor, FixedPoint &remainder) {
    bool negateQuotient{false}, negateRemainder{false};
    if (IsNegative()) {
      negateQuotient = negateRemainder = true;
      TwosComplement();
    }
    Ordering divisorOrdering{divisor.CompareToZeroSigned()};
    bool overflow{divisorOrdering == Ordering::Equal};
    if (divisorOrdering == Ordering::Less) {
      negateQuotient = !negateQuotient;
      divisor.TwosComplement();
    }
    DivideUnsigned(divisor, remainder);
    overflow |= IsNegative();
    if (negateQuotient) {
      TwosComplement();
    }
    if (negateRemainder) {
      remainder.TwosComplement();
    }
    return overflow;
  }

private:
  constexpr bool IsNegative() const {
    return (part_[parts-1] >> (partBits - 1)) & 1;
  }

  Part part_[parts];  // little-endian order: [parts-1] is most significant
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_FIXED_POINT_H_
