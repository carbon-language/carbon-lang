//===-- llvm/Support/APInt.h - For Arbitrary Precision Integer -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Sheng Zhou and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision integral
// constant values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APINT_H
#define LLVM_APINT_H

#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <string>

namespace llvm {

/// Forward declaration.
class APInt;
namespace APIntOps {
  APInt udiv(const APInt& LHS, const APInt& RHS);
  APInt urem(const APInt& LHS, const APInt& RHS);
}

//===----------------------------------------------------------------------===//
//                              APInt Class
//===----------------------------------------------------------------------===//

/// APInt - This class represents arbitrary precision constant integral values.
/// It is a functional replacement for common case unsigned integer type like 
/// "unsigned", "unsigned long" or "uint64_t", but also allows non-byte-width 
/// integer sizes and large integer value types such as 3-bits, 15-bits, or more
/// than 64-bits of precision. APInt provides a variety of arithmetic operators 
/// and methods to manipulate integer values of any bit-width. It supports both
/// the typical integer arithmetic and comparison operations as well as bitwise
/// manipulation.
///
/// The class has several invariants worth noting:
///   * All bit, byte, and word positions are zero-based.
///   * Once the bit width is set, it doesn't change except by the Truncate, 
///     SignExtend, or ZeroExtend operations.
///   * All binary operators must be on APInt instances of the same bit width.
///     Attempting to use these operators on instances with different bit 
///     widths will yield an assertion.
///   * The value is stored canonically as an unsigned value. For operations
///     where it makes a difference, there are both signed and unsigned variants
///     of the operation. For example, sdiv and udiv. However, because the bit
///     widths must be the same, operations such as Mul and Add produce the same
///     results regardless of whether the values are interpreted as signed or
///     not.
///   * In general, the class tries to follow the style of computation that LLVM
///     uses in its IR. This simplifies its use for LLVM.
///
/// @brief Class for arbitrary precision integers.
class APInt {
  unsigned BitWidth;      ///< The number of bits in this APInt.

  /// This union is used to store the integer value. When the
  /// integer bit-width <= 64, it uses VAL; 
  /// otherwise it uses the pVal.
  union {
    uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal;  ///< Used to store the >64 bits integer value.
  };

  /// This enum is just used to hold a constant we needed for APInt.
  enum {
    APINT_BITS_PER_WORD = sizeof(uint64_t) * 8
  };

  /// Here one word's bitwidth equals to that of uint64_t.
  /// @returns the number of words to hold the integer value of this APInt.
  /// @brief Get the number of words.
  inline unsigned getNumWords() const {
    return (BitWidth + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
  }

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this APInt just has one word to store value.
  inline bool isSingleWord() const { 
    return BitWidth <= APINT_BITS_PER_WORD; 
  }

  /// @returns the word position for the specified bit position.
  static inline unsigned whichWord(unsigned bitPosition) { 
    return bitPosition / APINT_BITS_PER_WORD; 
  }

  /// @returns the byte position for the specified bit position.
  static inline unsigned whichByte(unsigned bitPosition) { 
    return (bitPosition % APINT_BITS_PER_WORD) / 8; 
  }

  /// @returns the bit position in a word for the specified bit position 
  /// in APInt.
  static inline unsigned whichBit(unsigned bitPosition) { 
    return bitPosition % APINT_BITS_PER_WORD; 
  }

  /// @returns a uint64_t type integer with just bit position at
  /// "whichBit(bitPosition)" setting, others zero.
  static inline uint64_t maskBit(unsigned bitPosition) { 
    return (static_cast<uint64_t>(1)) << whichBit(bitPosition); 
  }

  /// This method is used internally to clear the to "N" bits that are not used
  /// by the APInt. This is needed after a word is assigned a value to ensure 
  /// that those bits are zero'd out.
  /// @brief Clear high order bits
  inline void clearUnusedBits() {
    if (isSingleWord())
      VAL &= ~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - BitWidth);
    else
      pVal[getNumWords() - 1] &= ~uint64_t(0ULL) >> 
        (APINT_BITS_PER_WORD - (whichBit(BitWidth - 1) + 1));
  }

  /// @returns the corresponding word for the specified bit position.
  inline uint64_t& getWord(unsigned bitPosition) { 
    return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; 
  }

  /// @returns the corresponding word for the specified bit position.
  /// This is a constant version.
  inline uint64_t getWord(unsigned bitPosition) const { 
    return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; 
  }

  /// @brief Converts a char array into an integer.
  void fromString(unsigned numBits, const char *StrStart, unsigned slen, 
                  uint8_t radix);

public:
  /// @brief Create a new APInt of numBits bit-width, and initialized as val.
  APInt(unsigned numBits, uint64_t val);

  /// @brief Create a new APInt of numBits bit-width, and initialized as 
  /// bigVal[].
  APInt(unsigned numBits, unsigned numWords, uint64_t bigVal[]);

  /// @brief Create a new APInt by translating the string represented 
  /// integer value.
  APInt(unsigned numBits, const std::string& Val, uint8_t radix);

  /// @brief Create a new APInt by translating the char array represented
  /// integer value.
  APInt(unsigned numBits, const char StrStart[], unsigned slen, uint8_t radix);

  /// @brief Copy Constructor.
  APInt(const APInt& API);

  /// @brief Destructor.
  ~APInt();

  /// @brief Copy assignment operator. 
  APInt& operator=(const APInt& RHS);

  /// Assigns an integer value to the APInt.
  /// @brief Assignment operator. 
  APInt& operator=(uint64_t RHS);

  /// Increments the APInt by one.
  /// @brief Postfix increment operator.
  inline const APInt operator++(int) {
    APInt API(*this);
    ++(*this);
    return API;
  }

  /// Increments the APInt by one.
  /// @brief Prefix increment operator.
  APInt& operator++();

  /// Decrements the APInt by one.
  /// @brief Postfix decrement operator. 
  inline const APInt operator--(int) {
    APInt API(*this);
    --(*this);
    return API;
  }

  /// Decrements the APInt by one.
  /// @brief Prefix decrement operator. 
  APInt& operator--();

  /// Performs bitwise AND operation on this APInt and the given APInt& RHS, 
  /// assigns the result to this APInt.
  /// @brief Bitwise AND assignment operator. 
  APInt& operator&=(const APInt& RHS);

  /// Performs bitwise OR operation on this APInt and the given APInt& RHS, 
  /// assigns the result to this APInt.
  /// @brief Bitwise OR assignment operator. 
  APInt& operator|=(const APInt& RHS);

  /// Performs bitwise XOR operation on this APInt and the given APInt& RHS, 
  /// assigns the result to this APInt.
  /// @brief Bitwise XOR assignment operator. 
  APInt& operator^=(const APInt& RHS);

  /// Performs a bitwise complement operation on this APInt.
  /// @brief Bitwise complement operator. 
  APInt operator~() const;

  /// Multiplies this APInt by the  given APInt& RHS and 
  /// assigns the result to this APInt.
  /// @brief Multiplication assignment operator. 
  APInt& operator*=(const APInt& RHS);

  /// Adds this APInt by the given APInt& RHS and 
  /// assigns the result to this APInt.
  /// @brief Addition assignment operator. 
  APInt& operator+=(const APInt& RHS);

  /// Subtracts this APInt by the given APInt &RHS and 
  /// assigns the result to this APInt.
  /// @brief Subtraction assignment operator. 
  APInt& operator-=(const APInt& RHS);

  /// Performs bitwise AND operation on this APInt and 
  /// the given APInt& RHS.
  /// @brief Bitwise AND operator. 
  APInt operator&(const APInt& RHS) const;

  /// Performs bitwise OR operation on this APInt and the given APInt& RHS.
  /// @brief Bitwise OR operator. 
  APInt operator|(const APInt& RHS) const;

  /// Performs bitwise XOR operation on this APInt and the given APInt& RHS.
  /// @brief Bitwise XOR operator. 
  APInt operator^(const APInt& RHS) const;

  /// Performs logical negation operation on this APInt.
  /// @brief Logical negation operator. 
  bool operator !() const;

  /// Multiplies this APInt by the given APInt& RHS.
  /// @brief Multiplication operator. 
  APInt operator*(const APInt& RHS) const;

  /// Adds this APInt by the given APInt& RHS.
  /// @brief Addition operator. 
  APInt operator+(const APInt& RHS) const;

  /// Subtracts this APInt by the given APInt& RHS
  /// @brief Subtraction operator. 
  APInt operator-(const APInt& RHS) const;

  /// @brief Unary negation operator
  inline APInt operator-() const {
    return APInt(BitWidth, 0) - (*this);
  }

  /// @brief Array-indexing support.
  bool operator[](unsigned bitPosition) const;

  /// Compare this APInt with the given APInt& RHS 
  /// for the validity of the equality relationship.
  /// @brief Equality operator. 
  bool operator==(const APInt& RHS) const;

  /// Compare this APInt with the given uint64_t value
  /// for the validity of the equality relationship.
  /// @brief Equality operator.
  bool operator==(uint64_t Val) const;

  /// Compare this APInt with the given APInt& RHS 
  /// for the validity of the inequality relationship.
  /// @brief Inequality operator. 
  inline bool operator!=(const APInt& RHS) const {
    return !((*this) == RHS);
  }

  /// Compare this APInt with the given uint64_t value 
  /// for the validity of the inequality relationship.
  /// @brief Inequality operator. 
  inline bool operator!=(uint64_t Val) const {
    return !((*this) == Val);
  }
  
  /// @brief Equality comparison
  bool eq(const APInt &RHS) const {
    return (*this) == RHS; 
  }

  /// @brief Inequality comparison
  bool ne(const APInt &RHS) const {
    return !((*this) == RHS);
  }

  /// @brief Unsigned less than comparison
  bool ult(const APInt& RHS) const;

  /// @brief Signed less than comparison
  bool slt(const APInt& RHS) const;

  /// @brief Unsigned less or equal comparison
  bool ule(const APInt& RHS) const {
    return ult(RHS) || eq(RHS);
  }

  /// @brief Signed less or equal comparison
  bool sle(const APInt& RHS) const {
    return slt(RHS) || eq(RHS);
  }

  /// @brief Unsigned greather than comparison
  bool ugt(const APInt& RHS) const {
    return !ult(RHS) && !eq(RHS);
  }

  /// @brief Signed greather than comparison
  bool sgt(const APInt& RHS) const {
    return !slt(RHS) && !eq(RHS);
  }

  /// @brief Unsigned greater or equal comparison
  bool uge(const APInt& RHS) const {
    return !ult(RHS);
  }

  /// @brief Signed greather or equal comparison
  bool sge(const APInt& RHS) const {
    return !slt(RHS);
  }

  /// Arithmetic right-shift this APInt by shiftAmt.
  /// @brief Arithmetic right-shift function.
  APInt ashr(unsigned shiftAmt) const;

  /// Logical right-shift this APInt by shiftAmt.
  /// @brief Logical right-shift function.
  APInt lshr(unsigned shiftAmt) const;

  /// Left-shift this APInt by shiftAmt.
  /// @brief Left-shift function.
  APInt shl(unsigned shiftAmt) const;

  /// Signed divide this APInt by APInt RHS.
  /// @brief Signed division function for APInt.
  inline APInt sdiv(const APInt& RHS) const {
    bool isNegativeLHS = (*this)[BitWidth - 1];
    bool isNegativeRHS = RHS[RHS.BitWidth - 1];
    APInt API = APIntOps::udiv(
        isNegativeLHS ? -(*this) : (*this), isNegativeRHS ? -RHS : RHS);
    return isNegativeLHS != isNegativeRHS ? -API : API;;
  }

  /// Unsigned divide this APInt by APInt RHS.
  /// @brief Unsigned division function for APInt.
  APInt udiv(const APInt& RHS) const;

  /// Signed remainder operation on APInt.
  /// @brief Function for signed remainder operation.
  inline APInt srem(const APInt& RHS) const {
    bool isNegativeLHS = (*this)[BitWidth - 1];
    bool isNegativeRHS = RHS[RHS.BitWidth - 1];
    APInt API = APIntOps::urem(
        isNegativeLHS ? -(*this) : (*this), isNegativeRHS ? -RHS : RHS);
    return isNegativeLHS ? -API : API;
  }

  /// Unsigned remainder operation on APInt.
  /// @brief Function for unsigned remainder operation.
  APInt urem(const APInt& RHS) const;

  /// Truncate the APInt to a specified width. It is an error to specify a width
  /// that is greater than or equal to the current width. 
  /// @brief Truncate to new width.
  void trunc(unsigned width);

  /// This operation sign extends the APInt to a new width. If the high order
  /// bit is set, the fill on the left will be done with 1 bits, otherwise zero.
  /// It is an error to specify a width that is less than or equal to the 
  /// current width.
  /// @brief Sign extend to a new width.
  void sext(unsigned width);

  /// This operation zero extends the APInt to a new width. Thie high order bits
  /// are filled with 0 bits.  It is an error to specify a width that is less 
  /// than or equal to the current width.
  /// @brief Zero extend to a new width.
  void zext(unsigned width);

  /// @brief Set every bit to 1.
  APInt& set();

  /// Set the given bit to 1 whose position is given as "bitPosition".
  /// @brief Set a given bit to 1.
  APInt& set(unsigned bitPosition);

  /// @brief Set every bit to 0.
  APInt& clear();

  /// Set the given bit to 0 whose position is given as "bitPosition".
  /// @brief Set a given bit to 0.
  APInt& clear(unsigned bitPosition);

  /// @brief Toggle every bit to its opposite value.
  APInt& flip();

  /// Toggle a given bit to its opposite value whose position is given 
  /// as "bitPosition".
  /// @brief Toggles a given bit to its opposite value.
  APInt& flip(unsigned bitPosition);

  /// This function returns the number of active bits which is defined as the
  /// bit width minus the number of leading zeros. This is used in several
  /// computations to see how "wide" the value is.
  /// @brief Compute the number of active bits in the value
  inline unsigned getActiveBits() const {
    return getNumWords() * APINT_BITS_PER_WORD - countLeadingZeros();
  }

  /// @returns a uint64_t value from this APInt. If this APInt contains a single
  /// word, just returns VAL, otherwise pVal[0].
  inline uint64_t getValue(bool isSigned = false) const {
    if (isSingleWord())
      return isSigned ? int64_t(VAL << (64 - BitWidth)) >> 
                                       (64 - BitWidth) : VAL;
    unsigned n = getActiveBits();
    if (n <= 64)
      return pVal[0];
    assert(0 && "This APInt's bitwidth > 64");
  }

  /// @returns the largest value for an APInt of the specified bit-width and 
  /// if isSign == true, it should be largest signed value, otherwise largest
  /// unsigned value.
  /// @brief Gets max value of the APInt with bitwidth <= 64.
  static APInt getMaxValue(unsigned numBits, bool isSign);

  /// @returns the smallest value for an APInt of the given bit-width and
  /// if isSign == true, it should be smallest signed value, otherwise zero.
  /// @brief Gets min value of the APInt with bitwidth <= 64.
  static APInt getMinValue(unsigned numBits, bool isSign);

  /// @returns the all-ones value for an APInt of the specified bit-width.
  /// @brief Get the all-ones value.
  static APInt getAllOnesValue(unsigned numBits);

  /// @returns the '0' value for an APInt of the specified bit-width.
  /// @brief Get the '0' value.
  static APInt getNullValue(unsigned numBits);

  /// This converts the APInt to a boolean valy as a test against zero.
  /// @brief Boolean conversion function. 
  inline bool getBoolValue() const {
    return countLeadingZeros() != BitWidth;
  }

  /// @returns a character interpretation of the APInt.
  std::string toString(uint8_t radix = 10) const;

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the low bits and right shift to the least significant bit.
  /// @returns the high "numBits" bits of this APInt.
  APInt getHiBits(unsigned numBits) const;

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the high bits.
  /// @returns the low "numBits" bits of this APInt.
  APInt getLoBits(unsigned numBits) const;

  /// @returns true if the argument APInt value is a power of two > 0.
  bool isPowerOf2() const; 

  /// @returns the number of zeros from the most significant bit to the first
  /// one bits.
  unsigned countLeadingZeros() const;

  /// @returns the number of zeros from the least significant bit to the first
  /// one bit.
  unsigned countTrailingZeros() const;

  /// @returns the number of set bits.
  unsigned countPopulation() const; 

  /// @returns the total number of bits.
  inline unsigned getBitWidth() const { 
    return BitWidth; 
  }

  /// @brief Check if this APInt has a N-bits integer value.
  inline bool isIntN(unsigned N) const {
    assert(N && "N == 0 ???");
    if (isSingleWord()) {
      return VAL == (VAL & (~0ULL >> (64 - N)));
    } else {
      APInt Tmp(N, getNumWords(), pVal);
      return Tmp == (*this);
    }
  }

  /// @returns a byte-swapped representation of this APInt Value.
  APInt byteSwap() const;

  /// @returns the floor log base 2 of this APInt.
  inline unsigned logBase2() const {
    return getNumWords() * APINT_BITS_PER_WORD - 1 - countLeadingZeros();
  }

  /// @brief Converts this APInt to a double value.
  double roundToDouble(bool isSigned = false) const;

};

namespace APIntOps {

/// @brief Check if the specified APInt has a N-bits integer value.
inline bool isIntN(unsigned N, const APInt& APIVal) {
  return APIVal.isIntN(N);
}

/// @returns true if the argument APInt value is a sequence of ones
/// starting at the least significant bit with the remainder zero.
inline const bool isMask(unsigned numBits, const APInt& APIVal) {
  return APIVal.getBoolValue() && ((APIVal + APInt(numBits,1)) & APIVal) == 0;
}

/// @returns true if the argument APInt value contains a sequence of ones
/// with the remainder zero.
inline const bool isShiftedMask(unsigned numBits, const APInt& APIVal) {
  return isMask(numBits, (APIVal - APInt(numBits,1)) | APIVal);
}

/// @returns a byte-swapped representation of the specified APInt Value.
inline APInt byteSwap(const APInt& APIVal) {
  return APIVal.byteSwap();
}

/// @returns the floor log base 2 of the specified APInt value.
inline unsigned logBase2(const APInt& APIVal) {
  return APIVal.logBase2(); 
}

/// @returns the greatest common divisor of the two values 
/// using Euclid's algorithm.
APInt GreatestCommonDivisor(const APInt& API1, const APInt& API2);

/// @brief Converts the given APInt to a double value.
inline double RoundAPIntToDouble(const APInt& APIVal, bool isSigned = false) {
  return APIVal.roundToDouble(isSigned);
}

/// @brief Converts the given APInt to a float vlalue.
inline float RoundAPIntToFloat(const APInt& APIVal) {
  return float(RoundAPIntToDouble(APIVal));
}

/// @brief Converts the given double value into a APInt.
APInt RoundDoubleToAPInt(double Double);

/// @brief Converts the given float value into a APInt.
inline APInt RoundFloatToAPInt(float Float) {
  return RoundDoubleToAPInt(double(Float));
}

/// Arithmetic right-shift the APInt by shiftAmt.
/// @brief Arithmetic right-shift function.
inline APInt ashr(const APInt& LHS, unsigned shiftAmt) {
  return LHS.ashr(shiftAmt);
}

/// Logical right-shift the APInt by shiftAmt.
/// @brief Logical right-shift function.
inline APInt lshr(const APInt& LHS, unsigned shiftAmt) {
  return LHS.lshr(shiftAmt);
}

/// Left-shift the APInt by shiftAmt.
/// @brief Left-shift function.
inline APInt shl(const APInt& LHS, unsigned shiftAmt) {
  return LHS.shl(shiftAmt);
}

/// Signed divide APInt LHS by APInt RHS.
/// @brief Signed division function for APInt.
inline APInt sdiv(const APInt& LHS, const APInt& RHS) {
  return LHS.sdiv(RHS);
}

/// Unsigned divide APInt LHS by APInt RHS.
/// @brief Unsigned division function for APInt.
inline APInt udiv(const APInt& LHS, const APInt& RHS) {
  return LHS.udiv(RHS);
}

/// Signed remainder operation on APInt.
/// @brief Function for signed remainder operation.
inline APInt srem(const APInt& LHS, const APInt& RHS) {
  return LHS.srem(RHS);
}

/// Unsigned remainder operation on APInt.
/// @brief Function for unsigned remainder operation.
inline APInt urem(const APInt& LHS, const APInt& RHS) {
  return LHS.urem(RHS);
}

/// Performs multiplication on APInt values.
/// @brief Function for multiplication operation.
inline APInt mul(const APInt& LHS, const APInt& RHS) {
  return LHS * RHS;
}

/// Performs addition on APInt values.
/// @brief Function for addition operation.
inline APInt add(const APInt& LHS, const APInt& RHS) {
  return LHS + RHS;
}

/// Performs subtraction on APInt values.
/// @brief Function for subtraction operation.
inline APInt sub(const APInt& LHS, const APInt& RHS) {
  return LHS - RHS;
}

/// Performs bitwise AND operation on APInt LHS and 
/// APInt RHS.
/// @brief Bitwise AND function for APInt.
inline APInt And(const APInt& LHS, const APInt& RHS) {
  return LHS & RHS;
}

/// Performs bitwise OR operation on APInt LHS and APInt RHS.
/// @brief Bitwise OR function for APInt. 
inline APInt Or(const APInt& LHS, const APInt& RHS) {
  return LHS | RHS;
}

/// Performs bitwise XOR operation on APInt.
/// @brief Bitwise XOR function for APInt.
inline APInt Xor(const APInt& LHS, const APInt& RHS) {
  return LHS ^ RHS;
} 

/// Performs a bitwise complement operation on APInt.
/// @brief Bitwise complement function. 
inline APInt Not(const APInt& APIVal) {
  return ~APIVal;
}

} // End of APIntOps namespace

} // End of llvm namespace

#endif
