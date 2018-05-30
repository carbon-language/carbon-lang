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

// Emulates integers of a arbitrary static size for use when the C++
// environment does not support it.  Signed and unsigned operations are
// distinguished by member function interface; the data are typeless.

#include "leading-zero-bit-count.h"
#include <cinttypes>
#include <climits>
#include <cstddef>

namespace Fortran::evaluate {

enum class Ordering { Less, Equal, Greater };
static constexpr Ordering Reverse(Ordering ordering) {
  if (ordering == Ordering::Less) {
    return Ordering::Greater;
  } else if (ordering == Ordering::Greater) {
    return Ordering::Less;
  } else {
    return Ordering::Equal;
  }
}

// Implement an integer as an assembly of smaller (i.e., 32-bit) integers.
// These are stored in little-endian order.  To facilitate exhaustive
// testing of what would otherwise be more rare edge cases, this template class
// may be configured to use other part types &/or partial fields in the
// parts.
template<int BITS, int PARTBITS = 32, typename PART = std::uint32_t,
    typename BIGPART = std::uint64_t>
class FixedPoint {
public:
  static constexpr int bits{BITS};
  static constexpr int partBits{PARTBITS};
  using Part = PART;
  using BigPart = BIGPART;
  static_assert(sizeof(BigPart) >= 2 * sizeof(Part));

private:
  static constexpr int maxPartBits{CHAR_BIT * sizeof(Part)};
  static_assert(partBits > 0 && partBits <= maxPartBits);
  static constexpr int extraPartBits{maxPartBits - partBits};
  static constexpr int parts{(bits + partBits - 1) / partBits};
  static_assert(parts >= 1);
  static constexpr int extraTopPartBits{
      extraPartBits + (parts * partBits) - bits};
  static constexpr int topPartBits{maxPartBits - extraTopPartBits};
  static_assert(topPartBits > 0 && topPartBits <= partBits);
  static_assert((parts - 1) * partBits + topPartBits == bits);
  static constexpr Part partMask{static_cast<Part>(~0) >> extraPartBits};
  static constexpr Part topPartMask{static_cast<Part>(~0) >> extraTopPartBits};

public:
  constexpr FixedPoint() {}  // zero
  constexpr FixedPoint(const FixedPoint &) = default;
  constexpr FixedPoint(std::uint64_t n) {
    for (int j{0}; j < parts - 1; ++j) {
      part_[j] = n & partMask;
      if constexpr (partBits < 64) {
        n >>= partBits;
      } else {
        n = 0;
      }
    }
    part_[parts - 1] = n & topPartMask;
  }
  constexpr FixedPoint(std::int64_t n) {
    std::int64_t signExtension{-(n < 0)};
    signExtension <<= partBits;
    for (int j{0}; j < parts - 1; ++j) {
      part_[j] = n & partMask;
      if constexpr (partBits < 64) {
        n = (n >> partBits) | signExtension;
      } else {
        n = signExtension;
      }
    }
    part_[parts - 1] = n & topPartMask;
  }

  constexpr FixedPoint &operator=(const FixedPoint &) = default;

  constexpr bool IsZero() const {
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  constexpr bool IsNegative() const {
    return (part_[parts - 1] >> (topPartBits - 1)) & 1;
  }

  constexpr Ordering CompareToZeroSigned() const {
    if (IsNegative()) {
      return Ordering::Less;
    }
    return IsZero() ? Ordering::Equal : Ordering::Greater;
  }

  constexpr Ordering CompareUnsigned(const FixedPoint &y) const {
    for (int j{parts}; j-- > 0;) {
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
    bool isNegative{IsNegative()};
    if (isNegative != y.IsNegative()) {
      return isNegative ? Ordering::Less : Ordering::Greater;
    }
    return CompareUnsigned(y);
  }

  constexpr int LeadingZeroBitCount() const {
    if (part_[parts - 1] != 0) {
      int lzbc{evaluate::LeadingZeroBitCount(part_[parts - 1])};
      return lzbc - extraTopPartBits;
    }
    int upperZeroes{topPartBits};
    for (int j{1}; j < parts; ++j) {
      if (Part p{part_[parts - 1 - j]}) {
        int lzbc{evaluate::LeadingZeroBitCount(p)};
        return upperZeroes + lzbc - extraPartBits;
      }
      upperZeroes += partBits;
    }
    return bits;
  }

  constexpr std::uint64_t ToUInt64() const {
    std::uint64_t n{part_[0]};
    int filled{partBits};
    for (int j{1}; filled < 64 && j < parts; ++j, filled += partBits) {
      n |= part_[j] << filled;
    }
    return n;
  }

  constexpr std::int64_t ToInt64() const {
    std::int64_t signExtended = ToUInt64();
    if (bits < 64) {
      signExtended |= -(signExtended >> (bits - 1)) << bits;
    }
    return signExtended;
  }

  constexpr void OnesComplement() {
    for (int j{0}; j + 1 < parts; ++j) {
      part_[j] = ~part_[j] & partMask;
    }
    part_[parts - 1] = ~part_[parts - 1] & topPartMask;
  }

  // Returns true on overflow (i.e., negating the most negative signed number)
  constexpr bool TwosComplement() {
    Part carry{1};
    for (int j{0}; j + 1 < parts; ++j) {
      Part newCarry{part_[j] == 0 && carry};
      part_[j] = (~part_[j] + carry) & partMask;
      carry = newCarry;
    }
    Part before{part_[parts - 1]};
    part_[parts - 1] = (~before + carry) & topPartMask;
    return before != 0 && part_[parts - 1] == before;
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
      ShiftRightLogical(-count);
    } else if (count > 0) {
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{parts - 1};
      if (bitShift == 0) {
        for (; j >= shiftParts; --j) {
          part_[j] = part_[j - shiftParts] & PartMask(j);
        }
        for (; j >= 0; --j) {
          part_[j] = 0;
        }
      } else {
        for (; j > shiftParts; --j) {
          part_[j] = ((part_[j - shiftParts] << bitShift) |
                         (part_[j - shiftParts - 1] >> (partBits - bitShift))) &
              PartMask(j);
        }
        if (j == shiftParts) {
          part_[j] = (part_[0] << bitShift) & PartMask(j);
          --j;
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
    } else if (count > 0) {
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
          part_[j] = ((part_[j + shiftParts] >> bitShift) |
                         (part_[j + shiftParts + 1] << (partBits - bitShift))) &
              partMask;
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
  constexpr bool AddUnsigned(const FixedPoint &y, bool carryIn = false) {
    BigPart carry{carryIn};
    for (int j{0}; j + 1 < parts; ++j) {
      carry += part_[j];
      carry += y.part_[j];
      part_[j] = carry & partMask;
      carry >>= partBits;
    }
    carry += part_[parts - 1];
    carry += y.part_[parts - 1];
    part_[parts - 1] = carry & topPartMask;
    return carry > topPartMask;
  }

  // Returns true on overflow.
  constexpr bool AddSigned(const FixedPoint &y) {
    bool isNegative{IsNegative()};
    bool sameSign{isNegative == y.IsNegative()};
    AddUnsigned(y);
    return sameSign && IsNegative() != isNegative;
  }

  // Returns true on overflow.
  constexpr bool SubtractSigned(const FixedPoint &y) {
    bool isNegative{IsNegative()};
    bool sameSign{isNegative == y.IsNegative()};
    FixedPoint minusy{y};
    minusy.TwosComplement();
    AddUnsigned(minusy);
    return !sameSign && IsNegative() != isNegative;
  }

  // Overwrites *this with lower half of full product.
  constexpr void MultiplyUnsigned(const FixedPoint &y, FixedPoint &upper) {
    Part product[2 * parts]{};  // little-endian full product
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        for (int k{0}; k < parts; ++k) {
          if (y.part_[k] != 0) {
            BigPart xy{part_[j]};
            xy *= y.part_[k];
            for (int to{j + k}; xy != 0; ++to) {
              xy += product[to];
              product[to] = xy & partMask;
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
    if (topPartBits < partBits) {
      upper.ShiftLeft(partBits - topPartBits);
      upper.part_[0] |= part_[parts - 1] >> topPartBits;
      part_[parts - 1] &= topPartMask;
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

  // Overwrites *this with quotient.  Returns true on division by zero.
  constexpr bool DivideUnsigned(
      const FixedPoint &divisor, FixedPoint &remainder) {
    remainder.Clear();
    if (divisor.IsZero()) {
      RightMask(bits);
      return true;
    }
    FixedPoint top{*this};
    Clear();
    int bitsDone{top.LeadingZeroBitCount()};
    top.ShiftLeft(bitsDone);
    for (; bitsDone < bits; ++bitsDone) {
      remainder.AddUnsigned(remainder, top.AddUnsigned(top));
      bool nextBit{remainder.CompareUnsigned(divisor) != Ordering::Less};
      AddUnsigned(*this, nextBit);
      if (nextBit) {
        remainder.SubtractSigned(divisor);
      }
    }
    return false;
  }

  // Overwrites *this with quotient.  Returns true on overflow (viz.,
  // the most negative value divided by -1) or division by zero.
  // A nonzero remainder has the sign of the dividend, i.e., it is
  // the MOD intrinsic (X-INT(X/Y)*Y), not MODULO.
  constexpr bool DivideSigned(FixedPoint divisor, FixedPoint &remainder) {
    bool dividendIsNegative{IsNegative()};
    bool negateQuotient{dividendIsNegative};
    Ordering divisorOrdering{divisor.CompareToZeroSigned()};
    if (divisorOrdering == Ordering::Less) {
      negateQuotient = !negateQuotient;
      if (divisor.TwosComplement()) {
        // divisor was (and is) the most negative number
        if (CompareUnsigned(divisor) == Ordering::Equal) {
          RightMask(1);
          remainder.Clear();
          return bits <= 1;  // edge case: 1-bit signed numbers overflow on 1!
        } else {
          remainder = *this;
          Clear();
          return false;
        }
      }
    } else if (divisorOrdering == Ordering::Equal) {
      // division by zero
      remainder.Clear();
      if (dividendIsNegative) {
        LeftMask(1);  // most negative signed number
      } else {
        RightMask(bits - 1);  // most positive signed number
      }
      return true;
    }
    if (dividendIsNegative) {
      if (TwosComplement()) {
        // Dividend was (and remains) the most negative number.
        // See whether the original divisor was -1 (if so, it's 1 now).
        if (divisorOrdering == Ordering::Less &&
            divisor.CompareUnsigned(FixedPoint{std::uint64_t{1}}) ==
                Ordering::Equal) {
          // most negative number / -1 is the sole overflow case
          remainder.Clear();
          return true;
        }
      }
    }
    // Overflow is not possible, and both the dividend (*this) and divisor
    // are now positive.
    DivideUnsigned(divisor, remainder);
    if (negateQuotient) {
      TwosComplement();
    }
    if (dividendIsNegative) {
      remainder.TwosComplement();
    }
    return false;
  }

  // MASKR intrinsic
  constexpr void RightMask(int places) {
    int j{0};
    for (; j + 1 < parts && places >= partBits; ++j, places -= partBits) {
      part_[j] = partMask;
    }
    if (places > 0) {
      if (j + 1 < parts) {
        part_[j++] = partMask >> (partBits - places);
      } else if (j + 1 == parts) {
        if (places >= topPartBits) {
          part_[j++] = topPartMask;
        } else {
          part_[j++] = topPartMask >> (topPartBits - places);
        }
      }
    }
    for (; j < parts; ++j) {
      part_[j] = 0;
    }
  }

  // MASKL intrinsic
  constexpr void LeftMask(int places) {
    if (places < 0) {
      Clear();
    } else if (places >= bits) {
      RightMask(bits);
    } else {
      RightMask(bits - places);
      OnesComplement();
    }
  }

private:
  static constexpr Part PartMask(int part) {
    return part == parts - 1 ? topPartMask : partMask;
  }

  constexpr void Clear() {
    for (int j{0}; j < parts; ++j) {
      part_[j] = 0;
    }
  }

  Part part_[parts]{};  // little-endian order: [parts-1] is most significant
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_FIXED_POINT_H_
