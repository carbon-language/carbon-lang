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
/// integer type and large integer value types such as 3-bits, 15-bits, or more
/// than 64-bits of precision. APInt provides a variety of arithmetic operators 
/// and methods to manipulate integer values of any bit-width. It supports not 
/// only all the operations of uint64_t but also bitwise manipulation.
///
/// @brief Class for arbitrary precision integers.
///
/// Note: In this class, all bit/byte/word positions are zero-based.
///
class APInt {
  unsigned BitsNum;      ///< The number of bits.

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
    return (BitsNum + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
  }

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this APInt just has one word to store value.
  inline bool isSingleWord() const
  { return BitsNum <= APINT_BITS_PER_WORD; }

  /// @returns the word position for the specified bit position.
  static inline unsigned whichWord(unsigned bitPosition)
  { return bitPosition / APINT_BITS_PER_WORD; }

  /// @returns the byte position for the specified bit position.
  static inline unsigned whichByte(unsigned bitPosition)
  { return (bitPosition % APINT_BITS_PER_WORD) / 8; }

  /// @returns the bit position in a word for the specified bit position 
  /// in APInt.
  static inline unsigned whichBit(unsigned bitPosition)
  { return bitPosition % APINT_BITS_PER_WORD; }

  /// @returns a uint64_t type integer with just bit position at
  /// "whichBit(bitPosition)" setting, others zero.
  static inline uint64_t maskBit(unsigned bitPosition)
  { return (static_cast<uint64_t>(1)) << whichBit(bitPosition); }

  inline void TruncToBits() {
    if (isSingleWord())
      VAL &= ~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - BitsNum);
    else
      pVal[getNumWords() - 1] &= ~uint64_t(0ULL) >> 
        (APINT_BITS_PER_WORD - (whichBit(BitsNum - 1) + 1));
  }

  /// @returns the corresponding word for the specified bit position.
  inline uint64_t& getWord(unsigned bitPosition)
  { return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; }

  /// @returns the corresponding word for the specified bit position.
  /// This is a constant version.
  inline uint64_t getWord(unsigned bitPosition) const
  { return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; }

  /// @brief Converts a char array into an integer.
  void StrToAPInt(const char *StrStart, unsigned slen, uint8_t radix);

public:
  /// @brief Create a new APInt of numBits bit-width, and initialized as val.
  APInt(uint64_t val = 0, unsigned numBits = APINT_BITS_PER_WORD);

  /// @brief Create a new APInt of numBits bit-width, and initialized as 
  /// bigVal[].
  APInt(unsigned numBits, uint64_t bigVal[]);

  /// @brief Create a new APInt by translating the string represented 
  /// integer value.
  APInt(const std::string& Val, uint8_t radix = 10);

  /// @brief Create a new APInt by translating the char array represented
  /// integer value.
  APInt(const char StrStart[], unsigned slen, uint8_t radix);

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
    return ++API;
  }

  /// Increments the APInt by one.
  /// @brief Prefix increment operator.
  APInt& operator++();

  /// Decrements the APInt by one.
  /// @brief Postfix decrement operator. 
  inline const APInt operator--(int) {
    APInt API(*this);
    return --API;
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

  /// Performs logical AND operation on this APInt and the given APInt& RHS.
  /// @brief Logical AND operator. 
  bool operator&&(const APInt& RHS) const;

  /// Performs logical OR operation on this APInt and the given APInt& RHS.
  /// @brief Logical OR operator. 
  bool operator||(const APInt& RHS) const;

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

  ///
  inline APInt operator-() const {
    return APInt(0, BitsNum) - (*this);
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
  
  /// Compare this APInt with the given APInt& RHS for 
  /// the validity of the less-than relationship.
  /// @brief Less-than operator. 
  bool operator <(const APInt& RHS) const;

  /// Compare this APInt with the given APInt& RHS for the validity 
  /// of the less-than-or-equal relationship.
  /// @brief Less-than-or-equal operator. 
  bool operator<=(const APInt& RHS) const;

  /// Compare this APInt with the given APInt& RHS for the validity 
  /// of the greater-than relationship.
  /// @brief Greater-than operator. 
  bool operator> (const APInt& RHS) const;

  /// @brief Greater-than-or-equal operator. 
  /// Compare this APInt with the given APInt& RHS for the validity 
  /// of the greater-than-or-equal relationship.
  bool operator>=(const APInt& RHS) const;

  /// @returns a uint64_t value from this APInt. If this APInt contains a single
  /// word, just returns VAL, otherwise pVal[0].
  inline uint64_t getValue() {
    if (isSingleWord())
      return VAL;
    unsigned n = getNumWords() * 64 - CountLeadingZeros();
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

  /// @brief Set every bit to 1.
  APInt& set();

  /// Set the given bit to 1 whose position is given as "bitPosition".
  /// @brief Set a given bit to 1.
  APInt& set(unsigned bitPosition);

  /// @returns the '0' value for an APInt of the specified bit-width.
  /// @brief Get the '0' value.
  static APInt getNullValue(unsigned numBits);

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

  /// @returns a character interpretation of the APInt.
  std::string to_string(uint8_t radix = 10) const;

  /// Get an APInt with the same BitsNum as this APInt, just zero mask
  /// the low bits and right shift to the least significant bit.
  /// @returns the high "numBits" bits of this APInt.
  APInt HiBits(unsigned numBits) const;

  /// Get an APInt with the same BitsNum as this APInt, just zero mask
  /// the high bits.
  /// @returns the low "numBits" bits of this APInt.
  APInt LoBits(unsigned numBits) const;

  /// @returns true if the argument APInt value is a power of two > 0.
  inline const bool isPowerOf2() const {
    return (!!*this) && !(*this & (*this - 1));
  }

  /// @returns the number of zeros from the most significant bit to the first
  /// one bits.
  unsigned CountLeadingZeros() const;

  /// @returns the number of zeros from the least significant bit to the first
  /// one bit.
  unsigned CountTrailingZeros() const;

  /// @returns the number of set bits.
  unsigned CountPopulation() const; 

  /// @returns the total number of bits.
  inline unsigned getNumBits() const
  { return BitsNum; }

  /// @brief Check if this APInt has a N-bits integer value.
  inline bool isIntN(unsigned N) const {
    if (isSingleWord()) {
      return VAL == VAL & (~uint64_t(0ULL) >> (64 - N));
    } else {
      APInt Tmp(N, pVal);
      return Tmp == (*this);
    }
  }

  /// @returns a byte-swapped representation of this APInt Value.
  APInt ByteSwap() const;

  /// @returns the floor log base 2 of this APInt.
  inline unsigned LogBase2() const {
    return getNumWords() * APINT_BITS_PER_WORD - 
           CountLeadingZeros();
  }

  /// @brief Converts this APInt to a double value.
  double APIntRoundToDouble(bool isSigned = false) const;

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
    bool isSignedLHS = (*this)[BitsNum - 1], isSignedRHS = RHS[RHS.BitsNum - 1];
    APInt API = APIntOps::udiv(isSignedLHS ? -(*this) : (*this), isSignedRHS ? -RHS : RHS);
    return isSignedLHS != isSignedRHS ? -API : API;;
  }

  /// Unsigned divide this APInt by APInt RHS.
  /// @brief Unsigned division function for APInt.
  APInt udiv(const APInt& RHS) const;

  /// Signed remainder operation on APInt.
  /// @brief Function for signed remainder operation.
  inline APInt srem(const APInt& RHS) const {
    bool isSignedLHS = (*this)[BitsNum - 1], isSignedRHS = RHS[RHS.BitsNum - 1];
    APInt API = APIntOps::urem(isSignedLHS ? -(*this) : (*this), isSignedRHS ? -RHS : RHS);
    return isSignedLHS ? -API : API;
  }

  /// Unsigned remainder operation on APInt.
  /// @brief Function for unsigned remainder operation.
  APInt urem(const APInt& RHS) const;

};

namespace APIntOps {

/// @brief Check if the specified APInt has a N-bits integer value.
inline bool isIntN(unsigned N, const APInt& APIVal) {
  return APIVal.isIntN(N);
}

/// @returns true if the argument APInt value is a sequence of ones
/// starting at the least significant bit with the remainder zero.
inline const bool isMask(unsigned numBits, const APInt& APIVal) {
  return APIVal && ((APIVal + 1) & APIVal) == 0;
}

/// @returns true if the argument APInt value contains a sequence of ones
/// with the remainder zero.
inline const bool isShiftedMask(unsigned numBits, const APInt& APIVal) {
  return isMask(numBits, (APIVal - 1) | APIVal);
}

/// @returns a byte-swapped representation of the specified APInt Value.
inline APInt ByteSwap(const APInt& APIVal) {
  return APIVal.ByteSwap();
}

/// @returns the floor log base 2 of the specified APInt value.
inline unsigned LogBase2(const APInt& APIVal) {
  return APIVal.LogBase2(); 
}

/// @returns the greatest common divisor of the two values 
/// using Euclid's algorithm.
APInt GreatestCommonDivisor(const APInt& API1, const APInt& API2);

/// @brief Converts the given APInt to a double value.
inline double APIntRoundToDouble(const APInt& APIVal, bool isSigned = false) {
  return APIVal.APIntRoundToDouble(isSigned);
}

/// @brief Converts the given APInt to a float vlalue.
inline float APIntRoundToFloat(const APInt& APIVal) {
  return float(APIntRoundToDouble(APIVal));
}

/// @brief Converts the given double value into a APInt.
APInt DoubleRoundToAPInt(double Double);

/// @brief Converts the given float value into a APInt.
inline APInt FloatRoundToAPInt(float Float) {
  return DoubleRoundToAPInt(double(Float));
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
