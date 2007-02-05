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
#include <string>

namespace llvm {

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
class APInt {
  /// Friend Functions of APInt Declared here. For detailed comments,
  /// see bottom of this file.
  friend bool isIntN(unsigned N, const APInt& APIVal);
  friend APInt ByteSwap(const APInt& APIVal);
  friend APInt LogBase2(const APInt& APIVal);
  friend double APIntToDouble(const APInt& APIVal);
  friend float APIntToFloat(const APInt& APIVal);

  unsigned bitsnum;      ///< The number of bits.
  bool isSigned;         ///< The sign flag for this APInt.

  /// This union is used to store the integer value. When the
  /// integer bit-width <= 64, it is used as an uint64_t; 
  /// otherwise it uses an uint64_t array.
  union {
    uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal;  ///< Used to store the >64 bits integer value.
  };

  /// This enum is just used to hold constant we needed for APInt.
  enum {
    APINT_BITS_PER_WORD = sizeof(uint64_t) * 8
  };

  /// @returns the number of words to hold the integer value of this APInt.
  /// Here one word's bitwidth equals to that of uint64_t.
  /// @brief Get the number of the words.
  inline unsigned numWords() const {
    return bitsnum < 1 ? 0 : (bitsnum + APINT_BITS_PER_WORD - 1) /
                             APINT_BITS_PER_WORD;
  }

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this APInt just has one word to store value.
  inline bool isSingleWord() const
  { return bitsnum <= APINT_BITS_PER_WORD; }

  /// @returns the word position for the specified bit position.
  /// Note: the bitPosition and the return value are zero-based.
  static inline unsigned whichWord(unsigned bitPosition)
  { return bitPosition / APINT_BITS_PER_WORD; }

  /// @returns the byte position for the specified bit position.
  /// Note: the bitPosition and the return value are zero-based.
  static inline unsigned whichByte(unsigned bitPosition);

  /// @returns the bit position in a word for the specified bit position 
  /// in APInt.
  /// Note: the bitPosition and the return value are zero-based.
  static inline unsigned whichBit(unsigned bitPosition)
  { return bitPosition % APINT_BITS_PER_WORD; }

  /// @returns a uint64_t type integer with just bit position at
  /// "whichBit(bitPosition)" setting, others zero.
  /// Note: the bitPosition and the return value are zero-based.
  static inline uint64_t maskBit(unsigned bitPosition)
  { return (static_cast<uint64_t>(1)) << whichBit(bitPosition); }

  inline void TruncToBits() {
    if (isSingleWord())
      VAL &= ~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - bitsnum);
    else
      pVal[numWords() - 1] &= ~uint64_t(0ULL) >> 
        (APINT_BITS_PER_WORD - (whichBit(bitsnum - 1) + 1));
  }

  /// @returns the corresponding word for the specified bit position.
  /// Note: the bitPosition is zero-based.
  inline uint64_t& getWord(unsigned bitPosition);

  /// @returns the corresponding word for the specified bit position.
  /// This is a constant version.
  /// Note: the bitPosition is zero-based.
  inline uint64_t getWord(unsigned bitPosition) const;

  /// mul_1 - This function performs the multiplication operation on a
  /// large integer (represented as a integer array) and a uint64_t integer.
  /// @returns the carry of the multiplication.
  static uint64_t mul_1(uint64_t dest[], uint64_t x[], 
                        unsigned len, uint64_t y);

  /// mul - This function performs the multiplication operation on two large
  /// integers (represented as integer arrays).
  static void mul(uint64_t dest[], uint64_t x[], unsigned xlen,
                  uint64_t y[], unsigned ylen);

  /// add_1 - This function performs the addition operation on a large integer
  /// and a uint64_t integer.
  /// @returns the carry of the addtion.
  static uint64_t add_1(uint64_t dest[], uint64_t x[], 
                        unsigned len, uint64_t y);

  /// add - This function performs the addtion operation on two large integers.
  static uint64_t add(uint64_t dest[], uint64_t x[], 
                      uint64_t y[], unsigned len);

  /// sub_1 - This function performs the subtraction operation on a large 
  /// integer and a uint64_t integer.
  static uint64_t sub_1(uint64_t x[], unsigned len, uint64_t y);

  /// sub - This function performs the subtraction operation on two large 
  /// integers.
  static uint64_t sub(uint64_t dest[], uint64_t x[], 
                      uint64_t y[], unsigned len);

  /// unitDiv - This function divides uint64_t N by unsigned D.
  /// @returns (remainder << 32) + quotient.
  /// @assumes (N >> 32) < D.
  static uint64_t unitDiv(uint64_t N, unsigned D);

  /// subMul - This function subtract x[len-1 : 0] * y from 
  /// dest[offset+len-1 : offset].
  /// @returns the most significant word of the product, minus borrow-out from
  /// the subtraction.
  static unsigned subMul(unsigned dest[], unsigned offset, 
                         unsigned x[], unsigned len, unsigned y);

  /// div - This function divides the large integer zds[] by y[].
  /// The remainder ends up in zds[ny-1 : 0].
  /// The quotient ends up in zds[ny : nx].
  /// @assumes nx > ny and (int)y[ny-1] < 0.
  static void div(unsigned zds[], unsigned nx, unsigned y[], unsigned ny);

  /// lshift - This function shifts x[len-1 : 0] by shiftAmt, and store the 
  /// "len" least significant words of the result in 
  /// dest[d_offset+len-1 : d_offset].
  /// @returns the bits shifted out from the most significant digit.
  static uint64_t lshift(uint64_t dest[], unsigned d_offset, 
                         uint64_t x[], unsigned len, unsigned shiftAmt);

public:
  /// Create a new APInt of numBits bit-width, and initalized as val.
  APInt(uint64_t val = 0, unsigned numBits = APINT_BITS_PER_WORD, 
        bool sign = false);

  /// Create a new APInt of numBits bit-width, and initalized as bigVal[].
  APInt(unsigned numBits, uint64_t bigVal[], bool sign = false);

  /// Create a new APInt by translating the string represented integer value.
  APInt(std::string& Val, uint8_t radix = 10, bool sign = false);

  /// Copy Constructor.
  APInt(const APInt& API);

  /// Destructor.
  ~APInt();

  /// @brief Copy assignment operator. Create a new object from the given
  /// APInt one by initialization.
  APInt& operator=(const APInt& RHS);

  /// @brief Assignment operator. Assigns a common case integer value to 
  /// the APInt.
  APInt& operator=(uint64_t RHS);

  /// @brief Postfix increment operator. Increments the APInt by one.
  const APInt operator++(int);

  /// @brief Prefix increment operator. Increments the APInt by one.
  APInt& operator++();

  /// @brief Postfix decrement operator. Decrements the APInt by one.
  const APInt operator--(int);

  /// @brief Prefix decrement operator. Decrements the APInt by one.
  APInt& operator--();

  /// @brief Bitwise AND assignment operator. Performs bitwise AND operation on
  /// this APInt and the given APInt& RHS, assigns the result to this APInt.
  APInt& operator&=(const APInt& RHS);

  /// @brief Bitwise OR assignment operator. Performs bitwise OR operation on 
  /// this APInt and the given APInt& RHS, assigns the result to this APInt.
  APInt& operator|=(const APInt& RHS);

  /// @brief Bitwise XOR assignment operator. Performs bitwise XOR operation on
  /// this APInt and the given APInt& RHS, assigns the result to this APInt.
  APInt& operator^=(const APInt& RHS);

  /// @brief Left-shift assignment operator. Left-shift the APInt by shiftAmt
  /// and assigns the result to this APInt.
  APInt& operator<<=(unsigned shiftAmt);

  /// @brief Right-shift assignment operator. Right-shift the APInt by shiftAmt
  /// and assigns the result to this APInt.
  APInt& operator>>=(unsigned shiftAmt);

  /// @brief Bitwise complement operator. Performs a bitwise complement 
  /// operation on this APInt.
  APInt operator~() const;

  /// @brief Multiplication assignment operator. Multiplies this APInt by the 
  /// given APInt& RHS and assigns the result to this APInt.
  APInt& operator*=(const APInt& RHS);

  /// @brief Division assignment operator. Divides this APInt by the given APInt
  /// &RHS and assigns the result to this APInt.
  APInt& operator/=(const APInt& RHS);

  /// @brief Addition assignment operator. Adds this APInt by the given APInt&
  /// RHS and assigns the result to this APInt.
  APInt& operator+=(const APInt& RHS);

  /// @brief Subtraction assignment operator. Subtracts this APInt by the given
  /// APInt &RHS and assigns the result to this APInt.
  APInt& operator-=(const APInt& RHS);

  /// @brief Remainder assignment operator. Yields the remainder from the 
  /// division of this APInt by the given APInt& RHS and assigns the remainder 
  /// to this APInt.
  APInt& operator%=(const APInt& RHS);

  /// @brief Bitwise AND operator. Performs bitwise AND operation on this APInt
  /// and the given APInt& RHS.
  APInt operator&(const APInt& RHS) const;

  /// @brief Bitwise OR operator. Performs bitwise OR operation on this APInt 
  /// and the given APInt& RHS.
  APInt operator|(const APInt& RHS) const;

  /// @brief Bitwise XOR operator. Performs bitwise XOR operation on this APInt
  /// and the given APInt& RHS.
  APInt operator^(const APInt& RHS) const;

  /// @brief Logical AND operator. Performs logical AND operation on this APInt
  /// and the given APInt& RHS.
  bool operator&&(const APInt& RHS) const;

  /// @brief Logical OR operator. Performs logical OR operation on this APInt 
  /// and the given APInt& RHS.
  bool operator||(const APInt& RHS) const;

  /// @brief Logical negation operator. Performs logical negation operation on
  /// this APInt.
  bool operator !() const;

  /// @brief Multiplication operator. Multiplies this APInt by the given APInt& 
  /// RHS.
  APInt operator*(const APInt& RHS) const;

  /// @brief Division operator. Divides this APInt by the given APInt& RHS.
  APInt operator/(const APInt& RHS) const;

  /// @brief Remainder operator. Yields the remainder from the division of this
  /// APInt and the given APInt& RHS.
  APInt operator%(const APInt& RHS) const;

  /// @brief Addition operator. Adds this APInt by the given APInt& RHS.
  APInt operator+(const APInt& RHS) const;

  /// @brief Subtraction operator. Subtracts this APInt by the given APInt& RHS
  APInt operator-(const APInt& RHS) const;

  /// @brief Left-shift operator. Left-shift the APInt by shiftAmt.
  APInt operator<<(unsigned shiftAmt) const;

  /// @brief Right-shift operator. Right-shift the APInt by shiftAmt.
  APInt operator>>(unsigned shiftAmt) const;

  /// @brief Array-indexing support.
  bool operator[](unsigned bitPosition) const;

  /// @brief Equality operator. Compare this APInt with the given APInt& RHS 
  /// for the validity of the equality relationship.
  bool operator==(const APInt& RHS) const;

  /// @brief Inequality operator. Compare this APInt with the given APInt& RHS
  /// for the validity of the inequality relationship.
  bool operator!=(const APInt& RHS) const;

  /// @brief Less-than operator. Compare this APInt with the given APInt& RHS
  /// for the validity of the less-than relationship.
  bool operator <(const APInt& RHS) const;

  /// @brief Less-than-or-equal operator. Compare this APInt with the given 
  /// APInt& RHS for the validity of the less-than-or-equal relationship.
  bool operator<=(const APInt& RHS) const;

  /// @brief Greater-than operator. Compare this APInt with the given APInt& RHS
  /// for the validity of the greater-than relationship.
  bool operator> (const APInt& RHS) const;

  /// @brief Greater-than-or-equal operator. Compare this APInt with the given 
  /// APInt& RHS for the validity of the greater-than-or-equal relationship.
  bool operator>=(const APInt& RHS) const;

  /// @returns a uint64_t value from this APInt. If this APInt contains a single
  /// word, just returns VAL, otherwise pVal[0].
  inline uint64_t getValue() {
    if (isSingleWord())
      return isSigned ? ((int64_t(VAL) << (APINT_BITS_PER_WORD - bitsnum)) >> 
                         (APINT_BITS_PER_WORD - bitsnum)) :
                        VAL;
    else
      return pVal[0];
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

  /// Set the given bit to 1 whose poition is given as "bitPosition".
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

  /// @returns the high "numBits" bits of this APInt.
  APInt HiBits(unsigned numBits) const;

  /// @returns the low "numBits" bits of this APInt.
  APInt LoBits(unsigned numBits) const;

  /// @returns true if the argument APInt value is a power of two > 0.
  inline const bool isPowerOf2() const {
    return *this && !(*this & (*this - 1));
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
  { return bitsnum; }

};

/// @brief Check if the specified APInt has a N-bits integer value.
inline bool isIntN(unsigned N, const APInt& APIVal) {
  if (APIVal.isSingleWord()) {
    APInt Tmp(N, APIVal.VAL);
    return Tmp == APIVal;
  } else {
    APInt Tmp(N, APIVal.pVal);
    return Tmp == APIVal;
  }
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
APInt ByteSwap(const APInt& APIVal);

/// @returns the floor log base 2 of the specified APInt value.
inline APInt LogBase2(const APInt& APIVal) {
  return APIVal.numWords() * APInt::APINT_BITS_PER_WORD - 
         APIVal.CountLeadingZeros();
}

/// @returns the greatest common divisor of the two values 
/// using Euclid's algorithm.
APInt GreatestCommonDivisor(const APInt& API1, const APInt& API2);

/// @returns the bit equivalent double.
/// If the APInt numBits > 64, truncated first and then convert to double.
inline double APIntToDouble(const APInt& APIVal) {
  uint64_t value = APIVal.isSingleWord() ? APIVal.VAL : APIVal.pVal[0];
  union {
    uint64_t I;
    double D;
  } T;
  T.I = value;
  return T.D;
}

/// @returns the bit equivalent float.
/// If the APInt numBits > 32, truncated first and then convert to double.
inline float APIntToFloat(const APInt& APIVal) {
  unsigned value = APIVal.isSingleWord() ? APIVal.VAL : APIVal.pVal[0];
  union {
    unsigned I;
    float F;
  } T;
  T.I = value;
  return T.F;
}

/// @returns the bit equivalent APInt.
inline APInt DoubleToAPInt(double Double) {
  union {
    uint64_t L;
    double D;
  } T;
  T.D = Double;
  return APInt(T.L);
}

/// @returns the bit equivalent APInt.
inline APInt FloatToAPInt(float Float) {
  union {
    uint32_t I;
    float F;
  } T;
  T.F = Float;
  return APInt(uint64_t(T.I));
}

} // End of llvm namespace

#endif
