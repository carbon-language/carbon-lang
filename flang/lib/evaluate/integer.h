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

#ifndef FORTRAN_EVALUATE_INTEGER_H_
#define FORTRAN_EVALUATE_INTEGER_H_

// Emulates binary integers of an arbitrary (but fixed) bit size for use
// when the host C++ environment does not support that size or when the
// full suite of Fortran's integer intrinsic scalar functions are needed.
// The data model is typeless, so signed* and unsigned operations
// are distinguished from each other with distinct member function interfaces.
// (*"Signed" here means two's-complement, just to be clear.  Ones'-complement
// and signed-magnitude encodings appear to be extinct in 2018.)

#include "bit-population-count.h"
#include "common.h"
#include "leading-zero-bit-count.h"
#include <cinttypes>
#include <climits>
#include <cstddef>
#include <type_traits>

namespace Fortran::evaluate {

// Implements an integer as an assembly of smaller host integer parts
// that constitute the digits of a large-radix fixed-point number.
// For best performance, the type of these parts should be half of the
// size of the largest efficient integer supported by the host processor.
// These parts are stored in either little- or big-endian order, which can
// match that of the host's endianness or not; but if the ordering matches
// that of the host, raw host data can be overlaid with a properly configured
// instance of this class and used in situ.
// To facilitate exhaustive testing of what would otherwise be more rare
// edge cases, this class template may be configured to use other part
// types &/or partial fields in the parts.  The radix (i.e., the number
// of possible values in a part), however, must be a power of two; this
// template class is not generalized to enable, say, decimal arithmetic.
// Member functions that correspond to Fortran intrinsic functions are
// named accordingly so that they can be referenced easily in the
// language standard.
template<int BITS,
    int PARTBITS =
        BITS<32 ? BITS : 32, typename PART = HostUnsignedInt<PARTBITS>,
            typename BIGPART = HostUnsignedInt<PARTBITS * 2>,
            bool LITTLE_ENDIAN = IsHostLittleEndian> class Integer {
public:
  static constexpr int bits{BITS};
  static constexpr int partBits{PARTBITS};
  using Part = PART;
  using BigPart = BIGPART;
  static_assert(CHAR_BIT * sizeof(BigPart) >= 2 * partBits);
  static constexpr bool littleEndian{LITTLE_ENDIAN};

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
  struct ValueWithOverflow {
    Integer value;
    bool overflow;
  };

  struct ValueWithCarry {
    Integer value;
    bool carry;
  };

  struct Product {
    Integer upper, lower;
  };

  struct QuotientWithRemainder {
    Integer quotient, remainder;
    bool divisionByZero, overflow;
  };

  // Constructors and value-generating static functions
  constexpr Integer() { Clear(); }  // default constructor: zero
  constexpr Integer(const Integer &) = default;
  constexpr Integer(std::uint64_t n) {
    for (int j{0}; j + 1 < parts; ++j) {
      SetLEPart(j, n);
      if constexpr (partBits < 64) {
        n >>= partBits;
      } else {
        n = 0;
      }
    }
    SetLEPart(parts - 1, n);
  }
  constexpr Integer(std::int64_t n) {
    std::int64_t signExtension{-(n < 0)};
    signExtension <<= partBits;
    for (int j{0}; j + 1 < parts; ++j) {
      SetLEPart(j, n);
      if constexpr (partBits < 64) {
        n = (n >> partBits) | signExtension;
      } else {
        n = signExtension;
      }
    }
    SetLEPart(parts - 1, n);
  }

  // Right-justified mask (e.g., MASKR(1) == 1, MASKR(2) == 3, &c.)
  static constexpr Integer MASKR(int places) {
    Integer result{nullptr};
    int j{0};
    for (; j + 1 < parts && places >= partBits; ++j, places -= partBits) {
      result.LEPart(j) = partMask;
    }
    if (places > 0) {
      if (j + 1 < parts) {
        result.LEPart(j++) = partMask >> (partBits - places);
      } else if (j + 1 == parts) {
        if (places >= topPartBits) {
          result.LEPart(j++) = topPartMask;
        } else {
          result.LEPart(j++) = topPartMask >> (topPartBits - places);
        }
      }
    }
    for (; j < parts; ++j) {
      result.LEPart(j) = 0;
    }
    return result;
  }

  // Left-justified mask (e.g., MASKL(1) has only its sign bit set)
  static constexpr Integer MASKL(int places) {
    if (places <= 0) {
      return {};
    } else if (places >= bits) {
      return MASKR(bits);
    } else {
      return MASKR(bits - places).NOT();
    }
  }

  static constexpr ValueWithOverflow ReadUnsigned(
      const char *&pp, std::uint64_t base = 10) {
    Integer result;
    bool overflow{false};
    const char *p{pp};
    while (*p == ' ' || *p == '\t') {
      ++p;
    }
    if (*p == '+') {
      ++p;
    }
    Integer radix{base};
    for (; std::uint64_t digit = *p; ++p) {
      if (digit >= '0' && digit < '0' + base) {
        digit -= '0';
      } else if (base > 10 && digit >= 'A' && digit < 'A' + base - 10) {
        digit -= 'A' - 10;
      } else if (base > 10 && digit >= 'a' && digit < 'a' + base - 10) {
        digit -= 'a' - 10;
      } else {
        break;
      }
      Product shifted{result.MultiplyUnsigned(radix)};
      overflow |= !shifted.upper.IsZero();
      ValueWithCarry next{shifted.lower.AddUnsigned(Integer{digit})};
      overflow |= next.carry;
      result = next.value;
    }
    pp = p;
    return {result, overflow};
  }

  static constexpr Integer HUGE() { return MASKR(bits - 1); }

  // Returns the number of full decimal digits that can be represented.
  static constexpr int RANGE() {
    if (bits < 4) {
      return 0;
    }
    Integer x{HUGE()}, ten{std::uint64_t{10}};
    int digits{0};
    while (x.CompareUnsigned(ten) != Ordering::Less) {
      ++digits;
      x = x.DivideUnsigned(ten).quotient;
    }
    return digits;
  }

  constexpr Integer &operator=(const Integer &) = default;

  constexpr bool IsZero() const {
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  constexpr bool IsNegative() const {
    return (LEPart(parts - 1) >> (topPartBits - 1)) & 1;
  }

  constexpr Ordering CompareToZeroSigned() const {
    if (IsNegative()) {
      return Ordering::Less;
    } else if (IsZero()) {
      return Ordering::Equal;
    } else {
      return Ordering::Greater;
    }
  }

  // Count the number of contiguous most-significant bit positions
  // that are clear.
  constexpr int LEADZ() const {
    if (LEPart(parts - 1) != 0) {
      int lzbc{LeadingZeroBitCount(LEPart(parts - 1))};
      return lzbc - extraTopPartBits;
    }
    int upperZeroes{topPartBits};
    for (int j{1}; j < parts; ++j) {
      if (Part p{LEPart(parts - 1 - j)}) {
        int lzbc{LeadingZeroBitCount(p)};
        return upperZeroes + lzbc - extraPartBits;
      }
      upperZeroes += partBits;
    }
    return bits;
  }

  // Count the number of bit positions that are set.
  constexpr int POPCNT() const {
    int count{0};
    for (int j{0}; j < parts; ++j) {
      count += BitPopulationCount(part_[j]);
    }
    return count;
  }

  // True when POPCNT is odd.
  constexpr bool POPPAR() const { return POPCNT() & 1; }

  constexpr int TRAILZ() const {
    auto minus1{AddUnsigned(MASKR(bits))};  // { x-1, carry = x > 0 }
    if (!minus1.carry) {
      return bits;  // was zero
    } else {
      // x ^ (x-1) has all bits set at and below original least-order set bit.
      return IEOR(minus1.value).POPCNT() - 1;
    }
  }

  constexpr bool BTEST(int pos) const {
    if (pos < 0 || pos >= bits) {
      return false;
    } else {
      return (LEPart(pos / partBits) >> (pos % partBits)) & 1;
    }
  }

  constexpr Ordering CompareUnsigned(const Integer &y) const {
    for (int j{parts}; j-- > 0;) {
      if (LEPart(j) > y.LEPart(j)) {
        return Ordering::Greater;
      }
      if (LEPart(j) < y.LEPart(j)) {
        return Ordering::Less;
      }
    }
    return Ordering::Equal;
  }

  constexpr bool BGE(const Integer &y) const {
    return CompareUnsigned(y) != Ordering::Less;
  }
  constexpr bool BGT(const Integer &y) const {
    return CompareUnsigned(y) == Ordering::Greater;
  }
  constexpr bool BLE(const Integer &y) const { return !BGT(y); }
  constexpr bool BLT(const Integer &y) const { return !BGE(y); }

  constexpr Ordering CompareSigned(const Integer &y) const {
    bool isNegative{IsNegative()};
    if (isNegative != y.IsNegative()) {
      return isNegative ? Ordering::Less : Ordering::Greater;
    }
    return CompareUnsigned(y);
  }

  constexpr std::uint64_t ToUInt64() const {
    std::uint64_t n{LEPart(0)};
    int filled{partBits};
    for (int j{1}; filled < 64 && j < parts; ++j, filled += partBits) {
      n |= LEPart(j) << filled;
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

  // Ones'-complement (i.e., C's ~)
  constexpr Integer NOT() const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.SetLEPart(j, ~LEPart(j));
    }
    return result;
  }

  // Two's-complement negation (-x = ~x + 1).
  // An overflow flag accompanies the result, and will be true when the
  // operand is the most negative signed number (MASKL(1)).
  constexpr ValueWithOverflow Negate() const {
    Integer result{nullptr};
    Part carry{1};
    for (int j{0}; j + 1 < parts; ++j) {
      Part newCarry{LEPart(j) == 0 && carry};
      result.SetLEPart(j, ~LEPart(j) + carry);
      carry = newCarry;
    }
    Part top{LEPart(parts - 1)};
    result.SetLEPart(parts - 1, ~top + carry);
    bool overflow{top != 0 && result.LEPart(parts - 1) == top};
    return {result, overflow};
  }

  constexpr ValueWithOverflow ABS() const {
    if (IsNegative()) {
      return Negate();
    } else {
      return {*this, false};
    }
  }

  // Shifts the operand left when the count is positive, right when negative.
  // Vacated bit positions are filled with zeroes.
  constexpr Integer ISHFT(int count) const {
    if (count < 0) {
      return SHIFTR(-count);
    } else {
      return SHIFTL(count);
    }
  }

  // Left shift with zero fill.
  constexpr Integer SHIFTL(int count) const {
    if (count <= 0) {
      return *this;
    } else {
      Integer result{nullptr};
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{parts - 1};
      if (bitShift == 0) {
        for (; j >= shiftParts; --j) {
          result.SetLEPart(j, LEPart(j - shiftParts));
        }
        for (; j >= 0; --j) {
          result.LEPart(j) = 0;
        }
      } else {
        for (; j > shiftParts; --j) {
          result.SetLEPart(j,
              ((LEPart(j - shiftParts) << bitShift) |
                  (LEPart(j - shiftParts - 1) >> (partBits - bitShift))));
        }
        if (j == shiftParts) {
          result.SetLEPart(j, LEPart(0) << bitShift);
          --j;
        }
        for (; j >= 0; --j) {
          result.LEPart(j) = 0;
        }
      }
      return result;
    }
  }

  // Circular shift of a field of least-significant bits.  The least-order
  // "size" bits are shifted circularly in place by "count" positions;
  // the shift is leftward if count is nonnegative, rightward otherwise.
  // Higher-order bits are unchanged.
  constexpr Integer ISHFTC(int count, int size = bits) const {
    if (count == 0 || size <= 0) {
      return *this;
    }
    if (size > bits) {
      size = bits;
    }
    if ((count %= size) == 0) {
      return *this;
    }
    int middleBits{size - count}, leastBits{count};
    if (count < 0) {
      middleBits = -count;
      leastBits = size + count;
    }
    if (size == bits) {
      return SHIFTL(leastBits).IOR(SHIFTR(middleBits));
    }
    Integer unchanged{IAND(MASKL(bits - size))};
    Integer middle{IAND(MASKR(middleBits)).SHIFTL(leastBits)};
    Integer least{SHIFTR(middleBits).IAND(MASKR(leastBits))};
    return unchanged.IOR(middle).IOR(least);
  }

  // Double shifts, aka shifts with specific fill
  constexpr Integer DSHIFTL(const Integer &fill, int count) const {
    if (count <= 0) {
      return *this;
    } else if (count >= 2 * bits) {
      return {};
    } else if (count > bits) {
      return fill.SHIFTL(count - bits);
    } else if (count == bits) {
      return fill;
    } else {
      return SHIFTL(count).IOR(fill.SHIFTR(bits - count));
    }
  }

  constexpr Integer DSHIFTR(const Integer &fill, int count) const {
    if (count <= 0) {
      return *this;
    } else if (count >= 2 * bits) {
      return {};
    } else if (count > bits) {
      return fill.SHIFTR(count - bits);
    } else if (count == bits) {
      return fill;
    } else {
      return SHIFTR(count).IOR(fill.SHIFTL(bits - count));
    }
  }

  // Vacated upper bits are filled with zeroes.
  constexpr Integer SHIFTR(int count) const {
    if (count <= 0) {
      return *this;
    } else {
      Integer result{nullptr};
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{0};
      if (bitShift == 0) {
        for (; j + shiftParts < parts; ++j) {
          result.LEPart(j) = LEPart(j + shiftParts);
        }
        for (; j < parts; ++j) {
          result.LEPart(j) = 0;
        }
      } else {
        for (; j + shiftParts + 1 < parts; ++j) {
          result.SetLEPart(j,
              (LEPart(j + shiftParts) >> bitShift) |
                  (LEPart(j + shiftParts + 1) << (partBits - bitShift)));
        }
        if (j + shiftParts + 1 == parts) {
          result.LEPart(j++) = LEPart(parts - 1) >> bitShift;
        }
        for (; j < parts; ++j) {
          result.LEPart(j) = 0;
        }
      }
      return result;
    }
  }

  // Be advised, an arithmetic (sign-filling) right shift is not
  // the same as a division by a power of two in all cases.
  constexpr Integer SHIFTA(int count) const {
    if (count <= 0) {
      return *this;
    } else if (IsNegative()) {
      return SHIFTR(count).IOR(MASKL(count));
    } else {
      return SHIFTR(count);
    }
  }

  // Clears a single bit.
  constexpr Integer IBCLR(int pos) const {
    if (pos < 0 || pos >= bits) {
      return *this;
    } else {
      Integer result{*this};
      result.LEPart(pos / partBits) &= ~(Part{1} << (pos % partBits));
      return result;
    }
  }

  // Sets a single bit.
  constexpr Integer IBSET(int pos) const {
    if (pos < 0 || pos >= bits) {
      return *this;
    } else {
      Integer result{*this};
      result.LEPart(pos / partBits) |= Part{1} << (pos % partBits);
      return result;
    }
  }

  // Extracts a field.
  constexpr Integer IBITS(int pos, int size) const {
    return SHIFTR(pos).IAND(MASKR(size));
  }

  constexpr Integer IAND(const Integer &y) const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.LEPart(j) = LEPart(j) & y.LEPart(j);
    }
    return result;
  }

  constexpr Integer IOR(const Integer &y) const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.LEPart(j) = LEPart(j) | y.LEPart(j);
    }
    return result;
  }

  constexpr Integer IEOR(const Integer &y) const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.LEPart(j) = LEPart(j) ^ y.LEPart(j);
    }
    return result;
  }

  constexpr Integer MERGE_BITS(const Integer &y, const Integer &mask) const {
    return IAND(mask).IOR(y.IAND(mask.NOT()));
  }

  constexpr Integer MAX(const Integer &y) const {
    if (CompareSigned(y) == Ordering::Less) {
      return y;
    } else {
      return *this;
    }
  }

  constexpr Integer MIN(const Integer &y) const {
    if (CompareSigned(y) == Ordering::Less) {
      return *this;
    } else {
      return y;
    }
  }

  // Unsigned addition with carry.
  constexpr ValueWithCarry AddUnsigned(
      const Integer &y, bool carryIn = false) const {
    Integer sum{nullptr};
    BigPart carry{carryIn};
    for (int j{0}; j + 1 < parts; ++j) {
      carry += LEPart(j);
      carry += y.LEPart(j);
      sum.SetLEPart(j, carry);
      carry >>= partBits;
    }
    carry += LEPart(parts - 1);
    carry += y.LEPart(parts - 1);
    sum.SetLEPart(parts - 1, carry);
    return {sum, carry > topPartMask};
  }

  constexpr ValueWithOverflow AddSigned(const Integer &y) const {
    bool isNegative{IsNegative()};
    bool sameSign{isNegative == y.IsNegative()};
    ValueWithCarry sum{AddUnsigned(y)};
    bool overflow{sameSign && sum.value.IsNegative() != isNegative};
    return {sum.value, overflow};
  }

  constexpr ValueWithOverflow SubtractSigned(const Integer &y) const {
    bool isNegative{IsNegative()};
    bool sameSign{isNegative == y.IsNegative()};
    ValueWithCarry diff{AddUnsigned(y.Negate().value)};
    bool overflow{!sameSign && diff.value.IsNegative() != isNegative};
    return {diff.value, overflow};
  }

  // MAX(X-Y, 0)
  constexpr Integer DIM(const Integer &y) const {
    if (CompareSigned(y) != Ordering::Greater) {
      return {};
    } else {
      return SubtractSigned(y).value;
    }
  }

  constexpr ValueWithOverflow SIGN(const Integer &sign) const {
    bool goNegative{sign.IsNegative()};
    if (goNegative == IsNegative()) {
      return {*this, false};
    } else if (goNegative) {
      return Negate();
    } else {
      return ABS();
    }
  }

  constexpr Product MultiplyUnsigned(const Integer &y) const {
    Part product[2 * parts]{};  // little-endian full product
    for (int j{0}; j < parts; ++j) {
      if (Part xpart{LEPart(j)}) {
        for (int k{0}; k < parts; ++k) {
          if (Part ypart{y.LEPart(k)}) {
            BigPart xy{xpart};
            xy *= ypart;
            for (int to{j + k}; xy != 0; ++to) {
              xy += product[to];
              product[to] = xy & partMask;
              xy >>= partBits;
            }
          }
        }
      }
    }
    Integer upper{nullptr}, lower{nullptr};
    for (int j{0}; j < parts; ++j) {
      lower.LEPart(j) = product[j];
      upper.LEPart(j) = product[j + parts];
    }
    if constexpr (topPartBits < partBits) {
      upper = upper.SHIFTL(partBits - topPartBits);
      upper.LEPart(0) |= lower.LEPart(parts - 1) >> topPartBits;
      lower.LEPart(parts - 1) &= topPartMask;
    }
    return {upper, lower};
  }

  constexpr Product MultiplySigned(const Integer &y) const {
    bool yIsNegative{y.IsNegative()};
    Integer absy{y};
    if (yIsNegative) {
      absy = y.Negate().value;
    }
    bool isNegative{IsNegative()};
    Integer absx{*this};
    if (isNegative) {
      absx = Negate().value;
    }
    Product product{absx.MultiplyUnsigned(absy)};
    if (isNegative != yIsNegative) {
      product.lower = product.lower.NOT();
      product.upper = product.upper.NOT();
      Integer one{std::uint64_t{1}};
      auto incremented{product.lower.AddUnsigned(one)};
      product.lower = incremented.value;
      if (incremented.carry) {
        product.upper = product.upper.AddUnsigned(one).value;
      }
    }
    return product;
  }

  constexpr QuotientWithRemainder DivideUnsigned(const Integer &divisor) const {
    if (divisor.IsZero()) {
      return {MASKR(bits), Integer{}, true, false};  // overflow to max value
    }
    int bitsDone{LEADZ()};
    Integer top{SHIFTL(bitsDone)};
    Integer quotient, remainder;
    for (; bitsDone < bits; ++bitsDone) {
      auto doubledTop{top.AddUnsigned(top)};
      top = doubledTop.value;
      remainder = remainder.AddUnsigned(remainder, doubledTop.carry).value;
      bool nextBit{remainder.CompareUnsigned(divisor) != Ordering::Less};
      quotient = quotient.AddUnsigned(quotient, nextBit).value;
      if (nextBit) {
        remainder = remainder.SubtractSigned(divisor).value;
      }
    }
    return {quotient, remainder, false, false};
  }

  // A nonzero remainder has the sign of the dividend, i.e., it computes
  // the MOD intrinsic (X-INT(X/Y)*Y), not MODULO (which is below).
  // 8/5 = 1r3;  -8/5 = -1r-3;  8/-5 = -1r3;  -8/-5 = 1r-3
  constexpr QuotientWithRemainder DivideSigned(Integer divisor) const {
    bool dividendIsNegative{IsNegative()};
    bool negateQuotient{dividendIsNegative};
    Ordering divisorOrdering{divisor.CompareToZeroSigned()};
    if (divisorOrdering == Ordering::Less) {
      negateQuotient = !negateQuotient;
      auto negated{divisor.Negate()};
      if (negated.overflow) {
        // divisor was (and is) the most negative number
        if (CompareUnsigned(divisor) == Ordering::Equal) {
          return {MASKR(1), Integer{}, false, bits <= 1};
        } else {
          return {Integer{}, *this, false, false};
        }
      }
      divisor = negated.value;
    } else if (divisorOrdering == Ordering::Equal) {
      // division by zero
      if (dividendIsNegative) {
        return {MASKL(1), Integer{}, true, false};
      } else {
        return {MASKR(bits - 1), Integer{}, true, false};
      }
    }
    Integer dividend{*this};
    if (dividendIsNegative) {
      auto negated{Negate()};
      if (negated.overflow) {
        // Dividend was (and remains) the most negative number.
        // See whether the original divisor was -1 (if so, it's 1 now).
        if (divisorOrdering == Ordering::Less &&
            divisor.CompareUnsigned(Integer{std::uint64_t{1}}) ==
                Ordering::Equal) {
          // most negative number / -1 is the sole overflow case
          return {*this, Integer{}, false, true};
        }
      } else {
        dividend = negated.value;
      }
    }
    // Overflow is not possible, and both the dividend and divisor
    // are now positive.
    QuotientWithRemainder result{dividend.DivideUnsigned(divisor)};
    if (negateQuotient) {
      result.quotient = result.quotient.Negate().value;
    }
    if (dividendIsNegative) {
      result.remainder = result.remainder.Negate().value;
    }
    return result;
  }

  // Result has the sign of the divisor argument.
  // 8 mod 5 = 3;  -8 mod 5 = 2;  8 mod -5 = -2;  -8 mod -5 = -3
  constexpr ValueWithOverflow MODULO(const Integer &divisor) const {
    bool negativeDivisor{divisor.IsNegative()};
    bool distinctSigns{IsNegative() != negativeDivisor};
    QuotientWithRemainder divided{DivideSigned(divisor)};
    if (distinctSigns && !divided.remainder.IsZero()) {
      return {divided.remainder.AddUnsigned(divisor).value, divided.overflow};
    } else {
      return {divided.remainder, divided.overflow};
    }
  }

private:
  constexpr Integer(std::nullptr_t) {}  // does not initialize parts

  // Accesses parts in little-endian order.
  constexpr const Part &LEPart(int part) const {
    if constexpr (littleEndian) {
      return part_[part];
    } else {
      return part_[parts - 1 - part];
    }
  }

  constexpr Part &LEPart(int part) {
    if constexpr (littleEndian) {
      return part_[part];
    } else {
      return part_[parts - 1 - part];
    }
  }

  constexpr void SetLEPart(int part, Part x) {
    LEPart(part) = x & PartMask(part);
  }

  static constexpr Part PartMask(int part) {
    return part == parts - 1 ? topPartMask : partMask;
  }

  constexpr void Clear() {
    for (int j{0}; j < parts; ++j) {
      part_[j] = 0;
    }
  }

  Part part_[parts];
};

extern template class Integer<8>;
extern template class Integer<16>;
extern template class Integer<32>;
extern template class Integer<64>;
extern template class Integer<128>;

template<int KIND> using IntrinsicInteger = Integer<KIND * CHAR_BIT>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_INTEGER_H_
