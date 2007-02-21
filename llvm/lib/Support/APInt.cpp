//===-- APInt.cpp - Implement APInt class ---------------------------------===//
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

#include "llvm/ADT/APInt.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/MathExtras.h"
#include <cstring>
#include <cstdlib>
#ifndef NDEBUG
#include <iostream>
#include <iomanip>
#endif

using namespace llvm;

// A utility function for allocating memory, checking for allocation failures,
// and ensuring the contents is zeroed.
inline static uint64_t* getClearedMemory(uint32_t numWords) {
  uint64_t * result = new uint64_t[numWords];
  assert(result && "APInt memory allocation fails!");
  memset(result, 0, numWords * sizeof(uint64_t));
  return result;
}

// A utility function for allocating memory and checking for allocation failure.
inline static uint64_t* getMemory(uint32_t numWords) {
  uint64_t * result = new uint64_t[numWords];
  assert(result && "APInt memory allocation fails!");
  return result;
}

APInt::APInt(uint32_t numBits, uint64_t val)
  : BitWidth(numBits), VAL(0) {
  assert(BitWidth >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(BitWidth <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  if (isSingleWord()) 
    VAL = val & (~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - BitWidth));
  else {
    pVal = getClearedMemory(getNumWords());
    pVal[0] = val;
  }
}

APInt::APInt(uint32_t numBits, uint32_t numWords, uint64_t bigVal[])
  : BitWidth(numBits), VAL(0)  {
  assert(BitWidth >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(BitWidth <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  assert(bigVal && "Null pointer detected!");
  if (isSingleWord())
    VAL = bigVal[0] & (~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - BitWidth));
  else {
    pVal = getMemory(getNumWords());
    // Calculate the actual length of bigVal[].
    uint32_t maxN = std::max<uint32_t>(numWords, getNumWords());
    uint32_t minN = std::min<uint32_t>(numWords, getNumWords());
    memcpy(pVal, bigVal, (minN - 1) * APINT_WORD_SIZE);
    pVal[minN-1] = bigVal[minN-1] & 
                    (~uint64_t(0ULL) >> 
                     (APINT_BITS_PER_WORD - BitWidth % APINT_BITS_PER_WORD));
    if (maxN == getNumWords())
      memset(pVal+numWords, 0, (getNumWords() - numWords) * APINT_WORD_SIZE);
  }
}

/// @brief Create a new APInt by translating the char array represented
/// integer value.
APInt::APInt(uint32_t numbits, const char StrStart[], uint32_t slen, 
             uint8_t radix) 
  : BitWidth(numbits), VAL(0) {
  fromString(numbits, StrStart, slen, radix);
}

/// @brief Create a new APInt by translating the string represented
/// integer value.
APInt::APInt(uint32_t numbits, const std::string& Val, uint8_t radix)
  : BitWidth(numbits), VAL(0) {
  assert(!Val.empty() && "String empty?");
  fromString(numbits, Val.c_str(), Val.size(), radix);
}

/// @brief Copy constructor
APInt::APInt(const APInt& that)
  : BitWidth(that.BitWidth), VAL(0) {
  if (isSingleWord()) 
    VAL = that.VAL;
  else {
    pVal = getMemory(getNumWords());
    memcpy(pVal, that.pVal, getNumWords() * APINT_WORD_SIZE);
  }
}

APInt::~APInt() {
  if (!isSingleWord() && pVal) 
    delete[] pVal;
}

/// @brief Copy assignment operator. Create a new object from the given
/// APInt one by initialization.
APInt& APInt::operator=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) 
    VAL = RHS.VAL;
  else
    memcpy(pVal, RHS.pVal, getNumWords() * APINT_WORD_SIZE);
  return *this;
}

/// @brief Assignment operator. Assigns a common case integer value to 
/// the APInt.
APInt& APInt::operator=(uint64_t RHS) {
  if (isSingleWord()) 
    VAL = RHS;
  else {
    pVal[0] = RHS;
    memset(pVal+1, 0, (getNumWords() - 1) * APINT_WORD_SIZE);
  }
  return *this;
}

/// add_1 - This function adds a single "digit" integer, y, to the multiple 
/// "digit" integer array,  x[]. x[] is modified to reflect the addition and
/// 1 is returned if there is a carry out, otherwise 0 is returned.
/// @returns the carry of the addition.
static uint64_t add_1(uint64_t dest[], 
                             uint64_t x[], uint32_t len, 
                             uint64_t y) {
  for (uint32_t i = 0; i < len; ++i) {
    dest[i] = y + x[i];
    if (dest[i] < y)
      y = 1;
    else {
      y = 0;
      break;
    }
  }
  return y;
}

/// @brief Prefix increment operator. Increments the APInt by one.
APInt& APInt::operator++() {
  if (isSingleWord()) 
    ++VAL;
  else
    add_1(pVal, pVal, getNumWords(), 1);
  clearUnusedBits();
  return *this;
}

/// sub_1 - This function subtracts a single "digit" (64-bit word), y, from 
/// the multi-digit integer array, x[], propagating the borrowed 1 value until 
/// no further borrowing is neeeded or it runs out of "digits" in x.  The result
/// is 1 if "borrowing" exhausted the digits in x, or 0 if x was not exhausted.
/// In other words, if y > x then this function returns 1, otherwise 0.
static uint64_t sub_1(uint64_t x[], uint32_t len, 
                             uint64_t y) {
  for (uint32_t i = 0; i < len; ++i) {
    uint64_t X = x[i];
    x[i] -= y;
    if (y > X) 
      y = 1;  // We have to "borrow 1" from next "digit"
    else {
      y = 0;  // No need to borrow
      break;  // Remaining digits are unchanged so exit early
    }
  }
  return y;
}

/// @brief Prefix decrement operator. Decrements the APInt by one.
APInt& APInt::operator--() {
  if (isSingleWord()) 
    --VAL;
  else
    sub_1(pVal, getNumWords(), 1);
  clearUnusedBits();
  return *this;
}

/// add - This function adds the integer array x[] by integer array
/// y[] and returns the carry.
static uint64_t add(uint64_t dest[], uint64_t x[], uint64_t y[], uint32_t len) {
  uint64_t carry = 0;
  for (uint32_t i = 0; i< len; ++i) {
    dest[i] = x[i] + y[i] + carry;
    uint64_t limit = std::min(x[i],y[i]);
    carry = dest[i] < limit || (carry && dest[i] == limit);
  }
  return carry;
}

/// @brief Addition assignment operator. Adds this APInt by the given APInt&
/// RHS and assigns the result to this APInt.
APInt& APInt::operator+=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) 
    VAL += RHS.VAL;
  else {
    add(pVal, pVal, RHS.pVal, getNumWords());
  }
  clearUnusedBits();
  return *this;
}

/// sub - This function subtracts the integer array x[] by
/// integer array y[], and returns the borrow-out carry.
static uint64_t sub(uint64_t *dest, const uint64_t *x, const uint64_t *y, 
                    uint32_t len) {
  bool borrow = false;
  for (uint32_t i = 0; i < len; ++i) {
    uint64_t x_tmp = borrow ? x[i] - 1 : x[i];
    borrow = y[i] > x_tmp || (borrow && x[i] == 0);
    dest[i] = x_tmp - y[i];
  }
  return borrow;
}

/// @brief Subtraction assignment operator. Subtracts this APInt by the given
/// APInt &RHS and assigns the result to this APInt.
APInt& APInt::operator-=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) 
    VAL -= RHS.VAL;
  else
    sub(pVal, pVal, RHS.pVal, getNumWords());
  clearUnusedBits();
  return *this;
}

/// mul_1 - This function performs the multiplication operation on a
/// large integer (represented as an integer array) and a uint64_t integer.
/// @returns the carry of the multiplication.
static uint64_t mul_1(uint64_t dest[], 
                             uint64_t x[], uint32_t len, 
                             uint64_t y) {
  // Split y into high 32-bit part and low 32-bit part.
  uint64_t ly = y & 0xffffffffULL, hy = y >> 32;
  uint64_t carry = 0, lx, hx;
  for (uint32_t i = 0; i < len; ++i) {
    lx = x[i] & 0xffffffffULL;
    hx = x[i] >> 32;
    // hasCarry - A flag to indicate if has carry.
    // hasCarry == 0, no carry
    // hasCarry == 1, has carry
    // hasCarry == 2, no carry and the calculation result == 0.
    uint8_t hasCarry = 0;
    dest[i] = carry + lx * ly;
    // Determine if the add above introduces carry.
    hasCarry = (dest[i] < carry) ? 1 : 0;
    carry = hx * ly + (dest[i] >> 32) + (hasCarry ? (1ULL << 32) : 0);
    // The upper limit of carry can be (2^32 - 1)(2^32 - 1) + 
    // (2^32 - 1) + 2^32 = 2^64.
    hasCarry = (!carry && hasCarry) ? 1 : (!carry ? 2 : 0);

    carry += (lx * hy) & 0xffffffffULL;
    dest[i] = (carry << 32) | (dest[i] & 0xffffffffULL);
    carry = (((!carry && hasCarry != 2) || hasCarry == 1) ? (1ULL << 32) : 0) + 
            (carry >> 32) + ((lx * hy) >> 32) + hx * hy;
  }

  return carry;
}

/// mul - This function multiplies integer array x[] by integer array y[] and
/// stores the result into integer array dest[].
/// Note the array dest[]'s size should no less than xlen + ylen.
static void mul(uint64_t dest[], uint64_t x[], uint32_t xlen,
                uint64_t y[], uint32_t ylen) {
  dest[xlen] = mul_1(dest, x, xlen, y[0]);

  for (uint32_t i = 1; i < ylen; ++i) {
    uint64_t ly = y[i] & 0xffffffffULL, hy = y[i] >> 32;
    uint64_t carry = 0, lx, hx;
    for (uint32_t j = 0; j < xlen; ++j) {
      lx = x[j] & 0xffffffffULL;
      hx = x[j] >> 32;
      // hasCarry - A flag to indicate if has carry.
      // hasCarry == 0, no carry
      // hasCarry == 1, has carry
      // hasCarry == 2, no carry and the calculation result == 0.
      uint8_t hasCarry = 0;
      uint64_t resul = carry + lx * ly;
      hasCarry = (resul < carry) ? 1 : 0;
      carry = (hasCarry ? (1ULL << 32) : 0) + hx * ly + (resul >> 32);
      hasCarry = (!carry && hasCarry) ? 1 : (!carry ? 2 : 0);

      carry += (lx * hy) & 0xffffffffULL;
      resul = (carry << 32) | (resul & 0xffffffffULL);
      dest[i+j] += resul;
      carry = (((!carry && hasCarry != 2) || hasCarry == 1) ? (1ULL << 32) : 0)+
              (carry >> 32) + (dest[i+j] < resul ? 1 : 0) + 
              ((lx * hy) >> 32) + hx * hy;
    }
    dest[i+xlen] = carry;
  }
}

/// @brief Multiplication assignment operator. Multiplies this APInt by the 
/// given APInt& RHS and assigns the result to this APInt.
APInt& APInt::operator*=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    VAL *= RHS.VAL;
  else {
    // one-based first non-zero bit position.
    uint32_t first = getActiveBits();
    uint32_t xlen = !first ? 0 : whichWord(first - 1) + 1;
    if (!xlen) 
      return *this;
    else {
      first = RHS.getActiveBits();
      uint32_t ylen = !first ? 0 : whichWord(first - 1) + 1;
      if (!ylen) {
        memset(pVal, 0, getNumWords() * APINT_WORD_SIZE);
        return *this;
      }
      uint64_t *dest = getMemory(xlen+ylen);
      mul(dest, pVal, xlen, RHS.pVal, ylen);
      memcpy(pVal, dest, ((xlen + ylen >= getNumWords()) ? 
                         getNumWords() : xlen + ylen) * APINT_WORD_SIZE);
      delete[] dest;
    }
  }
  clearUnusedBits();
  return *this;
}

/// @brief Bitwise AND assignment operator. Performs bitwise AND operation on
/// this APInt and the given APInt& RHS, assigns the result to this APInt.
APInt& APInt::operator&=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL &= RHS.VAL;
    return *this;
  }
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] &= RHS.pVal[i];
  return *this;
}

/// @brief Bitwise OR assignment operator. Performs bitwise OR operation on 
/// this APInt and the given APInt& RHS, assigns the result to this APInt.
APInt& APInt::operator|=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL |= RHS.VAL;
    return *this;
  }
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] |= RHS.pVal[i];
  return *this;
}

/// @brief Bitwise XOR assignment operator. Performs bitwise XOR operation on
/// this APInt and the given APInt& RHS, assigns the result to this APInt.
APInt& APInt::operator^=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL ^= RHS.VAL;
    this->clearUnusedBits();
    return *this;
  } 
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] ^= RHS.pVal[i];
  this->clearUnusedBits();
  return *this;
}

/// @brief Bitwise AND operator. Performs bitwise AND operation on this APInt
/// and the given APInt& RHS.
APInt APInt::operator&(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(getBitWidth(), VAL & RHS.VAL);

  APInt Result(*this);
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    Result.pVal[i] &= RHS.pVal[i];
  return Result;
}

/// @brief Bitwise OR operator. Performs bitwise OR operation on this APInt 
/// and the given APInt& RHS.
APInt APInt::operator|(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(getBitWidth(), VAL | RHS.VAL);

  APInt Result(*this);
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    Result.pVal[i] |= RHS.pVal[i];
  return Result;
}

/// @brief Bitwise XOR operator. Performs bitwise XOR operation on this APInt
/// and the given APInt& RHS.
APInt APInt::operator^(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    APInt Result(BitWidth, VAL ^ RHS.VAL);
    Result.clearUnusedBits();
    return Result;
  }
  APInt Result(*this);
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    Result.pVal[i] ^= RHS.pVal[i];
  return Result;
}

/// @brief Logical negation operator. Performs logical negation operation on
/// this APInt.
bool APInt::operator !() const {
  if (isSingleWord())
    return !VAL;

  for (uint32_t i = 0; i < getNumWords(); ++i)
    if (pVal[i]) 
      return false;
  return true;
}

/// @brief Multiplication operator. Multiplies this APInt by the given APInt& 
/// RHS.
APInt APInt::operator*(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    APInt Result(BitWidth, VAL * RHS.VAL);
    Result.clearUnusedBits();
    return Result;
  }
  APInt Result(*this);
  Result *= RHS;
  Result.clearUnusedBits();
  return Result;
}

/// @brief Addition operator. Adds this APInt by the given APInt& RHS.
APInt APInt::operator+(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    APInt Result(BitWidth, VAL + RHS.VAL);
    Result.clearUnusedBits();
    return Result;
  }
  APInt Result(BitWidth, 0);
  add(Result.pVal, this->pVal, RHS.pVal, getNumWords());
  Result.clearUnusedBits();
  return Result;
}

/// @brief Subtraction operator. Subtracts this APInt by the given APInt& RHS
APInt APInt::operator-(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    APInt Result(BitWidth, VAL - RHS.VAL);
    Result.clearUnusedBits();
    return Result;
  }
  APInt Result(BitWidth, 0);
  sub(Result.pVal, this->pVal, RHS.pVal, getNumWords());
  Result.clearUnusedBits();
  return Result;
}

/// @brief Array-indexing support.
bool APInt::operator[](uint32_t bitPosition) const {
  return (maskBit(bitPosition) & (isSingleWord() ? 
          VAL : pVal[whichWord(bitPosition)])) != 0;
}

/// @brief Equality operator. Compare this APInt with the given APInt& RHS 
/// for the validity of the equality relationship.
bool APInt::operator==(const APInt& RHS) const {
  if (isSingleWord())
    return VAL == RHS.VAL;

  uint32_t n1 = getActiveBits();
  uint32_t n2 = RHS.getActiveBits();
  if (n1 != n2) 
    return false;

  if (n1 <= APINT_BITS_PER_WORD)
    return pVal[0] == RHS.pVal[0];

  for (int i = whichWord(n1 - 1); i >= 0; --i)
    if (pVal[i] != RHS.pVal[i]) 
      return false;
  return true;
}

/// @brief Equality operator. Compare this APInt with the given uint64_t value 
/// for the validity of the equality relationship.
bool APInt::operator==(uint64_t Val) const {
  if (isSingleWord())
    return VAL == Val;

  uint32_t n = getActiveBits(); 
  if (n <= APINT_BITS_PER_WORD)
    return pVal[0] == Val;
  else
    return false;
}

/// @brief Unsigned less than comparison
bool APInt::ult(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be same for comparison");
  if (isSingleWord())
    return VAL < RHS.VAL;
  else {
    uint32_t n1 = getActiveBits();
    uint32_t n2 = RHS.getActiveBits();
    if (n1 < n2)
      return true;
    else if (n2 < n1)
      return false;
    else if (n1 <= APINT_BITS_PER_WORD && n2 <= APINT_BITS_PER_WORD)
      return pVal[0] < RHS.pVal[0];
    for (int i = whichWord(n1 - 1); i >= 0; --i) {
      if (pVal[i] > RHS.pVal[i]) return false;
      else if (pVal[i] < RHS.pVal[i]) return true;
    }
  }
  return false;
}

/// @brief Signed less than comparison
bool APInt::slt(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be same for comparison");
  if (isSingleWord()) {
    int64_t lhsSext = (int64_t(VAL) << (64-BitWidth)) >> (64-BitWidth);
    int64_t rhsSext = (int64_t(RHS.VAL) << (64-BitWidth)) >> (64-BitWidth);
    return lhsSext < rhsSext;
  }

  APInt lhs(*this);
  APInt rhs(*this);
  bool lhsNegative = false;
  bool rhsNegative = false;
  if (lhs[BitWidth-1]) {
    lhsNegative = true;
    lhs.flip();
    lhs++;
  }
  if (rhs[BitWidth-1]) {
    rhsNegative = true;
    rhs.flip();
    rhs++;
  }
  if (lhsNegative)
    if (rhsNegative)
      return !lhs.ult(rhs);
    else
      return true;
  else if (rhsNegative)
    return false;
  else 
    return lhs.ult(rhs);
}

/// Set the given bit to 1 whose poition is given as "bitPosition".
/// @brief Set a given bit to 1.
APInt& APInt::set(uint32_t bitPosition) {
  if (isSingleWord()) VAL |= maskBit(bitPosition);
  else pVal[whichWord(bitPosition)] |= maskBit(bitPosition);
  return *this;
}

/// @brief Set every bit to 1.
APInt& APInt::set() {
  if (isSingleWord()) 
    VAL = ~0ULL >> (APINT_BITS_PER_WORD - BitWidth);
  else {
    for (uint32_t i = 0; i < getNumWords() - 1; ++i)
      pVal[i] = -1ULL;
    pVal[getNumWords() - 1] = ~0ULL >> 
      (APINT_BITS_PER_WORD - BitWidth % APINT_BITS_PER_WORD);
  }
  return *this;
}

/// Set the given bit to 0 whose position is given as "bitPosition".
/// @brief Set a given bit to 0.
APInt& APInt::clear(uint32_t bitPosition) {
  if (isSingleWord()) 
    VAL &= ~maskBit(bitPosition);
  else 
    pVal[whichWord(bitPosition)] &= ~maskBit(bitPosition);
  return *this;
}

/// @brief Set every bit to 0.
APInt& APInt::clear() {
  if (isSingleWord()) 
    VAL = 0;
  else 
    memset(pVal, 0, getNumWords() * APINT_WORD_SIZE);
  return *this;
}

/// @brief Bitwise NOT operator. Performs a bitwise logical NOT operation on
/// this APInt.
APInt APInt::operator~() const {
  APInt API(*this);
  API.flip();
  return API;
}

/// @brief Toggle every bit to its opposite value.
APInt& APInt::flip() {
  if (isSingleWord()) VAL = (~(VAL << 
        (APINT_BITS_PER_WORD - BitWidth))) >> (APINT_BITS_PER_WORD - BitWidth);
  else {
    uint32_t i = 0;
    for (; i < getNumWords() - 1; ++i)
      pVal[i] = ~pVal[i];
    uint32_t offset = 
      APINT_BITS_PER_WORD - (BitWidth - APINT_BITS_PER_WORD * (i - 1));
    pVal[i] = (~(pVal[i] << offset)) >> offset;
  }
  return *this;
}

/// Toggle a given bit to its opposite value whose position is given 
/// as "bitPosition".
/// @brief Toggles a given bit to its opposite value.
APInt& APInt::flip(uint32_t bitPosition) {
  assert(bitPosition < BitWidth && "Out of the bit-width range!");
  if ((*this)[bitPosition]) clear(bitPosition);
  else set(bitPosition);
  return *this;
}

/// getMaxValue - This function returns the largest value
/// for an APInt of the specified bit-width and if isSign == true,
/// it should be largest signed value, otherwise unsigned value.
APInt APInt::getMaxValue(uint32_t numBits, bool isSign) {
  APInt Result(numBits, 0);
  Result.set();
  if (isSign) 
    Result.clear(numBits - 1);
  return Result;
}

/// getMinValue - This function returns the smallest value for
/// an APInt of the given bit-width and if isSign == true,
/// it should be smallest signed value, otherwise zero.
APInt APInt::getMinValue(uint32_t numBits, bool isSign) {
  APInt Result(numBits, 0);
  if (isSign) 
    Result.set(numBits - 1);
  return Result;
}

/// getAllOnesValue - This function returns an all-ones value for
/// an APInt of the specified bit-width.
APInt APInt::getAllOnesValue(uint32_t numBits) {
  return getMaxValue(numBits, false);
}

/// getNullValue - This function creates an '0' value for an
/// APInt of the specified bit-width.
APInt APInt::getNullValue(uint32_t numBits) {
  return getMinValue(numBits, false);
}

/// HiBits - This function returns the high "numBits" bits of this APInt.
APInt APInt::getHiBits(uint32_t numBits) const {
  return APIntOps::lshr(*this, BitWidth - numBits);
}

/// LoBits - This function returns the low "numBits" bits of this APInt.
APInt APInt::getLoBits(uint32_t numBits) const {
  return APIntOps::lshr(APIntOps::shl(*this, BitWidth - numBits), 
                        BitWidth - numBits);
}

bool APInt::isPowerOf2() const {
  return (!!*this) && !(*this & (*this - APInt(BitWidth,1)));
}

/// countLeadingZeros - This function is a APInt version corresponding to 
/// llvm/include/llvm/Support/MathExtras.h's function 
/// countLeadingZeros_{32, 64}. It performs platform optimal form of counting 
/// the number of zeros from the most significant bit to the first one bit.
/// @returns numWord() * 64 if the value is zero.
uint32_t APInt::countLeadingZeros() const {
  uint32_t Count = 0;
  if (isSingleWord())
    Count = CountLeadingZeros_64(VAL);
  else {
    for (uint32_t i = getNumWords(); i > 0u; --i) {
      if (pVal[i-1] == 0)
        Count += APINT_BITS_PER_WORD;
      else {
        Count += CountLeadingZeros_64(pVal[i-1]);
        break;
      }
    }
  }
  return Count - (APINT_BITS_PER_WORD - (BitWidth % APINT_BITS_PER_WORD));
}

/// countTrailingZeros - This function is a APInt version corresponding to
/// llvm/include/llvm/Support/MathExtras.h's function 
/// countTrailingZeros_{32, 64}. It performs platform optimal form of counting 
/// the number of zeros from the least significant bit to the first one bit.
/// @returns numWord() * 64 if the value is zero.
uint32_t APInt::countTrailingZeros() const {
  if (isSingleWord())
    return CountTrailingZeros_64(VAL);
  APInt Tmp( ~(*this) & ((*this) - APInt(BitWidth,1)) );
  return getNumWords() * APINT_BITS_PER_WORD - Tmp.countLeadingZeros();
}

/// countPopulation - This function is a APInt version corresponding to
/// llvm/include/llvm/Support/MathExtras.h's function
/// countPopulation_{32, 64}. It counts the number of set bits in a value.
/// @returns 0 if the value is zero.
uint32_t APInt::countPopulation() const {
  if (isSingleWord())
    return CountPopulation_64(VAL);
  uint32_t Count = 0;
  for (uint32_t i = 0; i < getNumWords(); ++i)
    Count += CountPopulation_64(pVal[i]);
  return Count;
}


/// byteSwap - This function returns a byte-swapped representation of the
/// this APInt.
APInt APInt::byteSwap() const {
  assert(BitWidth >= 16 && BitWidth % 16 == 0 && "Cannot byteswap!");
  if (BitWidth == 16)
    return APInt(BitWidth, ByteSwap_16(VAL));
  else if (BitWidth == 32)
    return APInt(BitWidth, ByteSwap_32(VAL));
  else if (BitWidth == 48) {
    uint64_t Tmp1 = ((VAL >> 32) << 16) | (VAL & 0xFFFF);
    Tmp1 = ByteSwap_32(Tmp1);
    uint64_t Tmp2 = (VAL >> 16) & 0xFFFF;
    Tmp2 = ByteSwap_16(Tmp2);
    return 
      APInt(BitWidth, 
            (Tmp1 & 0xff) | ((Tmp1<<16) & 0xffff00000000ULL) | (Tmp2 << 16));
  } else if (BitWidth == 64)
    return APInt(BitWidth, ByteSwap_64(VAL));
  else {
    APInt Result(BitWidth, 0);
    char *pByte = (char*)Result.pVal;
    for (uint32_t i = 0; i < BitWidth / APINT_WORD_SIZE / 2; ++i) {
      char Tmp = pByte[i];
      pByte[i] = pByte[BitWidth / APINT_WORD_SIZE - 1 - i];
      pByte[BitWidth / APINT_WORD_SIZE - i - 1] = Tmp;
    }
    return Result;
  }
}

/// GreatestCommonDivisor - This function returns the greatest common
/// divisor of the two APInt values using Enclid's algorithm.
APInt llvm::APIntOps::GreatestCommonDivisor(const APInt& API1, 
                                            const APInt& API2) {
  APInt A = API1, B = API2;
  while (!!B) {
    APInt T = B;
    B = APIntOps::urem(A, B);
    A = T;
  }
  return A;
}

/// DoubleRoundToAPInt - This function convert a double value to
/// a APInt value.
APInt llvm::APIntOps::RoundDoubleToAPInt(double Double) {
  union {
    double D;
    uint64_t I;
  } T;
  T.D = Double;
  bool isNeg = T.I >> 63;
  int64_t exp = ((T.I >> 52) & 0x7ff) - 1023;
  if (exp < 0)
    return APInt(64ull, 0u);
  uint64_t mantissa = ((T.I << 12) >> 12) | (1ULL << 52);
  if (exp < 52)
    return isNeg ? -APInt(64u, mantissa >> (52 - exp)) : 
                    APInt(64u, mantissa >> (52 - exp));
  APInt Tmp(exp + 1, mantissa);
  Tmp = Tmp.shl(exp - 52);
  return isNeg ? -Tmp : Tmp;
}

/// RoundToDouble - This function convert this APInt to a double.
/// The layout for double is as following (IEEE Standard 754):
///  --------------------------------------
/// |  Sign    Exponent    Fraction    Bias |
/// |-------------------------------------- |
/// |  1[63]   11[62-52]   52[51-00]   1023 |
///  -------------------------------------- 
double APInt::roundToDouble(bool isSigned) const {

  // Handle the simple case where the value is contained in one uint64_t.
  if (isSingleWord() || getActiveBits() <= APINT_BITS_PER_WORD) {
    if (isSigned) {
      int64_t sext = (int64_t(VAL) << (64-BitWidth)) >> (64-BitWidth);
      return double(sext);
    } else
      return double(VAL);
  }

  // Determine if the value is negative.
  bool isNeg = isSigned ? (*this)[BitWidth-1] : false;

  // Construct the absolute value if we're negative.
  APInt Tmp(isNeg ? -(*this) : (*this));

  // Figure out how many bits we're using.
  uint32_t n = Tmp.getActiveBits();

  // The exponent (without bias normalization) is just the number of bits
  // we are using. Note that the sign bit is gone since we constructed the
  // absolute value.
  uint64_t exp = n;

  // Return infinity for exponent overflow
  if (exp > 1023) {
    if (!isSigned || !isNeg)
      return double(1.0E300 * 1.0E300); // positive infinity
    else 
      return double(-1.0E300 * 1.0E300); // negative infinity
  }
  exp += 1023; // Increment for 1023 bias

  // Number of bits in mantissa is 52. To obtain the mantissa value, we must
  // extract the high 52 bits from the correct words in pVal.
  uint64_t mantissa;
  unsigned hiWord = whichWord(n-1);
  if (hiWord == 0) {
    mantissa = Tmp.pVal[0];
    if (n > 52)
      mantissa >>= n - 52; // shift down, we want the top 52 bits.
  } else {
    assert(hiWord > 0 && "huh?");
    uint64_t hibits = Tmp.pVal[hiWord] << (52 - n % APINT_BITS_PER_WORD);
    uint64_t lobits = Tmp.pVal[hiWord-1] >> (11 + n % APINT_BITS_PER_WORD);
    mantissa = hibits | lobits;
  }

  // The leading bit of mantissa is implicit, so get rid of it.
  uint64_t sign = isNeg ? (1ULL << (APINT_BITS_PER_WORD - 1)) : 0;
  union {
    double D;
    uint64_t I;
  } T;
  T.I = sign | (exp << 52) | mantissa;
  return T.D;
}

// Truncate to new width.
void APInt::trunc(uint32_t width) {
  assert(width < BitWidth && "Invalid APInt Truncate request");
}

// Sign extend to a new width.
void APInt::sext(uint32_t width) {
  assert(width > BitWidth && "Invalid APInt SignExtend request");
}

//  Zero extend to a new width.
void APInt::zext(uint32_t width) {
  assert(width > BitWidth && "Invalid APInt ZeroExtend request");
}

/// Arithmetic right-shift this APInt by shiftAmt.
/// @brief Arithmetic right-shift function.
APInt APInt::ashr(uint32_t shiftAmt) const {
  APInt API(*this);
  if (API.isSingleWord())
    API.VAL = 
      (((int64_t(API.VAL) << (APINT_BITS_PER_WORD - API.BitWidth)) >> 
          (APINT_BITS_PER_WORD - API.BitWidth)) >> shiftAmt) & 
      (~uint64_t(0UL) >> (APINT_BITS_PER_WORD - API.BitWidth));
  else {
    if (shiftAmt >= API.BitWidth) {
      memset(API.pVal, API[API.BitWidth-1] ? 1 : 0, 
             (API.getNumWords()-1) * APINT_WORD_SIZE);
      API.pVal[API.getNumWords() - 1] = 
        ~uint64_t(0UL) >> 
          (APINT_BITS_PER_WORD - API.BitWidth % APINT_BITS_PER_WORD);
    } else {
      uint32_t i = 0;
      for (; i < API.BitWidth - shiftAmt; ++i)
        if (API[i+shiftAmt]) 
          API.set(i);
        else
          API.clear(i);
      for (; i < API.BitWidth; ++i)
        if (API[API.BitWidth-1]) 
          API.set(i);
        else API.clear(i);
    }
  }
  return API;
}

/// Logical right-shift this APInt by shiftAmt.
/// @brief Logical right-shift function.
APInt APInt::lshr(uint32_t shiftAmt) const {
  APInt API(*this);
  if (API.isSingleWord())
    API.VAL >>= shiftAmt;
  else {
    if (shiftAmt >= API.BitWidth)
      memset(API.pVal, 0, API.getNumWords() * APINT_WORD_SIZE);
    uint32_t i = 0;
    for (i = 0; i < API.BitWidth - shiftAmt; ++i)
      if (API[i+shiftAmt]) API.set(i);
      else API.clear(i);
    for (; i < API.BitWidth; ++i)
      API.clear(i);
  }
  return API;
}

/// Left-shift this APInt by shiftAmt.
/// @brief Left-shift function.
APInt APInt::shl(uint32_t shiftAmt) const {
  APInt API(*this);
  if (API.isSingleWord())
    API.VAL <<= shiftAmt;
  else if (shiftAmt >= API.BitWidth)
    memset(API.pVal, 0, API.getNumWords() * APINT_WORD_SIZE);
  else {
    if (uint32_t offset = shiftAmt / APINT_BITS_PER_WORD) {
      for (uint32_t i = API.getNumWords() - 1; i > offset - 1; --i)
        API.pVal[i] = API.pVal[i-offset];
      memset(API.pVal, 0, offset * APINT_WORD_SIZE);
    }
    shiftAmt %= APINT_BITS_PER_WORD;
    uint32_t i;
    for (i = API.getNumWords() - 1; i > 0; --i)
      API.pVal[i] = (API.pVal[i] << shiftAmt) | 
                    (API.pVal[i-1] >> (APINT_BITS_PER_WORD - shiftAmt));
    API.pVal[i] <<= shiftAmt;
  }
  API.clearUnusedBits();
  return API;
}

#if 0
/// subMul - This function substracts x[len-1:0] * y from 
/// dest[offset+len-1:offset], and returns the most significant 
/// word of the product, minus the borrow-out from the subtraction.
static uint32_t subMul(uint32_t dest[], uint32_t offset, 
                        uint32_t x[], uint32_t len, uint32_t y) {
  uint64_t yl = (uint64_t) y & 0xffffffffL;
  uint32_t carry = 0;
  uint32_t j = 0;
  do {
    uint64_t prod = ((uint64_t) x[j] & 0xffffffffUL) * yl;
    uint32_t prod_low = (uint32_t) prod;
    uint32_t prod_high = (uint32_t) (prod >> 32);
    prod_low += carry;
    carry = (prod_low < carry ? 1 : 0) + prod_high;
    uint32_t x_j = dest[offset+j];
    prod_low = x_j - prod_low;
    if (prod_low > x_j) ++carry;
    dest[offset+j] = prod_low;
  } while (++j < len);
  return carry;
}

/// unitDiv - This function divides N by D, 
/// and returns (remainder << 32) | quotient.
/// Assumes (N >> 32) < D.
static uint64_t unitDiv(uint64_t N, uint32_t D) {
  uint64_t q, r;                   // q: quotient, r: remainder.
  uint64_t a1 = N >> 32;           // a1: high 32-bit part of N.
  uint64_t a0 = N & 0xffffffffL;   // a0: low 32-bit part of N
  if (a1 < ((D - a1 - (a0 >> 31)) & 0xffffffffL)) {
      q = N / D;
      r = N % D;
  }
  else {
    // Compute c1*2^32 + c0 = a1*2^32 + a0 - 2^31*d
    uint64_t c = N - ((uint64_t) D << 31);
    // Divide (c1*2^32 + c0) by d
    q = c / D;
    r = c % D;
    // Add 2^31 to quotient 
    q += 1 << 31;
  }

  return (r << 32) | (q & 0xFFFFFFFFl);
}

#endif

/// div - This is basically Knuth's formulation of the classical algorithm.
/// Correspondance with Knuth's notation:
/// Knuth's u[0:m+n] == zds[nx:0].
/// Knuth's v[1:n] == y[ny-1:0]
/// Knuth's n == ny.
/// Knuth's m == nx-ny.
/// Our nx == Knuth's m+n.
/// Could be re-implemented using gmp's mpn_divrem:
/// zds[nx] = mpn_divrem (&zds[ny], 0, zds, nx, y, ny).

/// Implementation of Knuth's Algorithm D (Division of nonnegative integers)
/// from "Art of Computer Programming, Volume 2", section 4.3.1, p. 272. The
/// variables here have the same names as in the algorithm. Comments explain
/// the algorithm and any deviation from it.
static void KnuthDiv(uint32_t *u, uint32_t *v, uint32_t *q, uint32_t* r, 
                     uint32_t m, uint32_t n) {
  assert(u && "Must provide dividend");
  assert(v && "Must provide divisor");
  assert(q && "Must provide quotient");
  assert(n>1 && "n must be > 1");

  // Knuth uses the value b as the base of the number system. In our case b
  // is 2^31 so we just set it to -1u.
  uint64_t b = uint64_t(1) << 32;

  // D1. [Normalize.] Set d = b / (v[n-1] + 1) and multiply all the digits of 
  // u and v by d. Note that we have taken Knuth's advice here to use a power 
  // of 2 value for d such that d * v[n-1] >= b/2 (b is the base). A power of 
  // 2 allows us to shift instead of multiply and it is easy to determine the 
  // shift amount from the leading zeros.  We are basically normalizing the u
  // and v so that its high bits are shifted to the top of v's range without
  // overflow. Note that this can require an extra word in u so that u must
  // be of length m+n+1.
  uint32_t shift = CountLeadingZeros_32(v[n-1]);
  uint32_t v_carry = 0;
  uint32_t u_carry = 0;
  if (shift) {
    for (uint32_t i = 0; i < m+n; ++i) {
      uint32_t u_tmp = u[i] >> (32 - shift);
      u[i] = (u[i] << shift) | u_carry;
      u_carry = u_tmp;
    }
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t v_tmp = v[i] >> (32 - shift);
      v[i] = (v[i] << shift) | v_carry;
      v_carry = v_tmp;
    }
  }
  u[m+n] = u_carry;

  // D2. [Initialize j.]  Set j to m. This is the loop counter over the places.
  int j = m;
  do {
    // D3. [Calculate q'.]. 
    //     Set qp = (u[j+n]*b + u[j+n-1]) / v[n-1]. (qp=qprime=q')
    //     Set rp = (u[j+n]*b + u[j+n-1]) % v[n-1]. (rp=rprime=r')
    // Now test if qp == b or qp*v[n-2] > b*rp + u[j+n-2]; if so, decrease
    // qp by 1, inrease rp by v[n-1], and repeat this test if rp < b. The test
    // on v[n-2] determines at high speed most of the cases in which the trial
    // value qp is one too large, and it eliminates all cases where qp is two 
    // too large. 
    uint64_t qp = ((uint64_t(u[j+n]) << 32) | uint64_t(u[j+n-1])) / v[n-1];
    uint64_t rp = ((uint64_t(u[j+n]) << 32) | uint64_t(u[j+n-1])) % v[n-1];
    if (qp == b || qp*v[n-2] > b*rp + u[j+n-2]) {
      qp--;
      rp += v[n-1];
    }
    if (rp < b) 
      if (qp == b || qp*v[n-2] > b*rp + u[j+n-2]) {
        qp--;
        rp += v[n-1];
      }

    // D4. [Multiply and subtract.] Replace u with u - q*v (for each word).
    uint32_t borrow = 0;
    for (uint32_t i = 0; i < n; i++) {
      uint32_t save = u[j+i];
      u[j+i] = uint64_t(u[j+i]) - (qp * v[i]) - borrow;
      if (u[j+i] > save) {
        borrow = 1;
        u[j+i+1] += b;
      } else {
        borrow = 0;
      }
    }
    if (borrow)
      u[j+n] += 1;

    // D5. [Test remainder.] Set q[j] = qp. If the result of step D4 was 
    // negative, go to step D6; otherwise go on to step D7.
    q[j] = qp;
    if (borrow) {
      // D6. [Add back]. The probability that this step is necessary is very 
      // small, on the order of only 2/b. Make sure that test data accounts for
      // this possibility. Decreate qj by 1 and add v[...] to u[...]. A carry 
      // will occur to the left of u[j+n], and it should be ignored since it 
      // cancels with the borrow that occurred in D4.
      uint32_t carry = 0;
      for (uint32_t i = 0; i < n; i++) {
        uint32_t save = u[j+i];
        u[j+i] += v[i] + carry;
        carry = u[j+i] < save;
      }
    }

    // D7. [Loop on j.]  Decreate j by one. Now if j >= 0, go back to D3.
    j--;
  } while (j >= 0);

  // D8. [Unnormalize]. Now q[...] is the desired quotient, and the desired
  // remainder may be obtained by dividing u[...] by d. If r is non-null we
  // compute the remainder (urem uses this).
  if (r) {
    // The value d is expressed by the "shift" value above since we avoided
    // multiplication by d by using a shift left. So, all we have to do is
    // shift right here. In order to mak
    uint32_t mask = ~0u >> (32 - shift);
    uint32_t carry = 0;
    for (int i = n-1; i >= 0; i--) {
      uint32_t save = u[i] & mask;
      r[i] = (u[i] >> shift) | carry;
      carry = save;
    }
  }
}

// This function makes calling KnuthDiv a little more convenient. It uses
// APInt parameters instead of uint32_t* parameters. It can also divide APInt
// values of different widths.
void APInt::divide(const APInt LHS, uint32_t lhsWords, 
                   const APInt &RHS, uint32_t rhsWords,
                   APInt *Quotient, APInt *Remainder)
{
  assert(lhsWords >= rhsWords && "Fractional result");

  // First, compose the values into an array of 32-bit words instead of 
  // 64-bit words. This is a necessity of both the "short division" algorithm
  // and the the Knuth "classical algorithm" which requires there to be native 
  // operations for +, -, and * on an m bit value with an m*2 bit result. We 
  // can't use 64-bit operands here because we don't have native results of 
  // 128-bits. Furthremore, casting the 64-bit values to 32-bit values won't 
  // work on large-endian machines.
  uint64_t mask = ~0ull >> (sizeof(uint32_t)*8);
  uint32_t n = rhsWords * 2;
  uint32_t m = (lhsWords * 2) - n;
  // FIXME: allocate space on stack if m and n are sufficiently small.
  uint32_t *U = new uint32_t[m + n + 1];
  memset(U, 0, (m+n+1)*sizeof(uint32_t));
  for (unsigned i = 0; i < lhsWords; ++i) {
    uint64_t tmp = (lhsWords == 1 ? LHS.VAL : LHS.pVal[i]);
    U[i * 2] = tmp & mask;
    U[i * 2 + 1] = tmp >> (sizeof(uint32_t)*8);
  }
  U[m+n] = 0; // this extra word is for "spill" in the Knuth algorithm.

  uint32_t *V = new uint32_t[n];
  memset(V, 0, (n)*sizeof(uint32_t));
  for (unsigned i = 0; i < rhsWords; ++i) {
    uint64_t tmp = (rhsWords == 1 ? RHS.VAL : RHS.pVal[i]);
    V[i * 2] = tmp & mask;
    V[i * 2 + 1] = tmp >> (sizeof(uint32_t)*8);
  }

  // Set up the quotient and remainder
  uint32_t *Q = new uint32_t[m+n];
  memset(Q, 0, (m+n) * sizeof(uint32_t));
  uint32_t *R = 0;
  if (Remainder) {
    R = new uint32_t[n];
    memset(R, 0, n * sizeof(uint32_t));
  }

  // Now, adjust m and n for the Knuth division. n is the number of words in 
  // the divisor. m is the number of words by which the dividend exceeds the
  // divisor (i.e. m+n is the length of the dividend). These sizes must not 
  // contain any zero words or the Knuth algorithm fails.
  for (unsigned i = n; i > 0 && V[i-1] == 0; i--) {
    n--;
    m++;
  }
  for (unsigned i = m+n; i > 0 && U[i-1] == 0; i--)
    m--;

  // If we're left with only a single word for the divisor, Knuth doesn't work
  // so we implement the short division algorithm here. This is much simpler
  // and faster because we are certain that we can divide a 64-bit quantity
  // by a 32-bit quantity at hardware speed and short division is simply a
  // series of such operations. This is just like doing short division but we
  // are using base 2^32 instead of base 10.
  assert(n != 0 && "Divide by zero?");
  if (n == 1) {
    uint32_t divisor = V[0];
    uint32_t remainder = 0;
    for (int i = m+n-1; i >= 0; i--) {
      uint64_t partial_dividend = uint64_t(remainder) << 32 | U[i];
      if (partial_dividend == 0) {
        Q[i] = 0;
        remainder = 0;
      } else if (partial_dividend < divisor) {
        Q[i] = 0;
        remainder = partial_dividend;
      } else if (partial_dividend == divisor) {
        Q[i] = 1;
        remainder = 0;
      } else {
        Q[i] = partial_dividend / divisor;
        remainder = partial_dividend - (Q[i] * divisor);
      }
    }
    if (R)
      R[0] = remainder;
  } else {
    // Now we're ready to invoke the Knuth classical divide algorithm. In this
    // case n > 1.
    KnuthDiv(U, V, Q, R, m, n);
  }

  // If the caller wants the quotient
  if (Quotient) {
    // Set up the Quotient value's memory.
    if (Quotient->BitWidth != LHS.BitWidth) {
      if (Quotient->isSingleWord())
        Quotient->VAL = 0;
      else
        delete Quotient->pVal;
      Quotient->BitWidth = LHS.BitWidth;
      if (!Quotient->isSingleWord())
        Quotient->pVal = getClearedMemory(lhsWords);
    } else
      Quotient->clear();

    // The quotient is in Q. Reconstitute the quotient into Quotient's low 
    // order words.
    if (lhsWords == 1) {
      uint64_t tmp = 
        uint64_t(Q[0]) | (uint64_t(Q[1]) << (APINT_BITS_PER_WORD / 2));
      if (Quotient->isSingleWord())
        Quotient->VAL = tmp;
      else
        Quotient->pVal[0] = tmp;
    } else {
      assert(!Quotient->isSingleWord() && "Quotient APInt not large enough");
      for (unsigned i = 0; i < lhsWords; ++i)
        Quotient->pVal[i] = 
          uint64_t(Q[i*2]) | (uint64_t(Q[i*2+1]) << (APINT_BITS_PER_WORD / 2));
    }
  }

  // If the caller wants the remainder
  if (Remainder) {
    // Set up the Remainder value's memory.
    if (Remainder->BitWidth != RHS.BitWidth) {
      if (Remainder->isSingleWord())
        Remainder->VAL = 0;
      else
        delete Remainder->pVal;
      Remainder->BitWidth = RHS.BitWidth;
      if (!Remainder->isSingleWord())
        Remainder->pVal = getClearedMemory(rhsWords);
    } else
      Remainder->clear();

    // The remainder is in R. Reconstitute the remainder into Remainder's low
    // order words.
    if (rhsWords == 1) {
      uint64_t tmp = 
        uint64_t(R[0]) | (uint64_t(R[1]) << (APINT_BITS_PER_WORD / 2));
      if (Remainder->isSingleWord())
        Remainder->VAL = tmp;
      else
        Remainder->pVal[0] = tmp;
    } else {
      assert(!Remainder->isSingleWord() && "Remainder APInt not large enough");
      for (unsigned i = 0; i < rhsWords; ++i)
        Remainder->pVal[i] = 
          uint64_t(R[i*2]) | (uint64_t(R[i*2+1]) << (APINT_BITS_PER_WORD / 2));
    }
  }

  // Clean up the memory we allocated.
  delete [] U;
  delete [] V;
  delete [] Q;
  delete [] R;
}

/// Unsigned divide this APInt by APInt RHS.
/// @brief Unsigned division function for APInt.
APInt APInt::udiv(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");

  // First, deal with the easy case
  if (isSingleWord()) {
    assert(RHS.VAL != 0 && "Divide by zero?");
    return APInt(BitWidth, VAL / RHS.VAL);
  }

  // Get some facts about the LHS and RHS number of bits and words
  uint32_t rhsBits = RHS.getActiveBits();
  uint32_t rhsWords = !rhsBits ? 0 : (APInt::whichWord(rhsBits - 1) + 1);
  assert(rhsWords && "Divided by zero???");
  uint32_t lhsBits = this->getActiveBits();
  uint32_t lhsWords = !lhsBits ? 0 : (APInt::whichWord(lhsBits - 1) + 1);

  // Make a temporary to hold the result
  APInt Result(*this);

  // Deal with some degenerate cases
  if (!lhsWords) 
    return Result; // 0 / X == 0
  else if (lhsWords < rhsWords || Result.ult(RHS)) {
    // X / Y with X < Y == 0
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    return Result;
  } else if (Result == RHS) {
    // X / X == 1
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    Result.pVal[0] = 1;
    return Result;
  } else if (lhsWords == 1 && rhsWords == 1) {
    // All high words are zero, just use native divide
    Result.pVal[0] /= RHS.pVal[0];
    return Result;
  }

  // We have to compute it the hard way. Invoke the Knuth divide algorithm.
  APInt Quotient(1,0); // to hold result.
  divide(*this, lhsWords, RHS, rhsWords, &Quotient, 0);
  return Quotient;
}

/// Unsigned remainder operation on APInt.
/// @brief Function for unsigned remainder operation.
APInt APInt::urem(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    assert(RHS.VAL != 0 && "Remainder by zero?");
    return APInt(BitWidth, VAL % RHS.VAL);
  }

  // Make a temporary to hold the result
  APInt Result(*this);

  // Get some facts about the RHS
  uint32_t rhsBits = RHS.getActiveBits();
  uint32_t rhsWords = !rhsBits ? 0 : (APInt::whichWord(rhsBits - 1) + 1);
  assert(rhsWords && "Performing remainder operation by zero ???");

  // Get some facts about the LHS
  uint32_t lhsBits = Result.getActiveBits();
  uint32_t lhsWords = !lhsBits ? 0 : (Result.whichWord(lhsBits - 1) + 1);

  // Check the degenerate cases
  if (lhsWords == 0) {
    // 0 % Y == 0
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    return Result;
  } else if (lhsWords < rhsWords || Result.ult(RHS)) {
    // X % Y == X iff X < Y
    return Result;
  } else if (Result == RHS) {
    // X % X == 0;
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    return Result;
  } else if (lhsWords == 1) {
    // All high words are zero, just use native remainder
    Result.pVal[0] %=  RHS.pVal[0];
    return Result;
  }

  // We have to compute it the hard way. Invoke the Knute divide algorithm.
  APInt Remainder(1,0);
  divide(*this, lhsWords, RHS, rhsWords, 0, &Remainder);
  return Remainder;
}

/// @brief Converts a char array into an integer.
void APInt::fromString(uint32_t numbits, const char *str, uint32_t slen, 
                       uint8_t radix) {
  // Check our assumptions here
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  assert(str && "String is null?");
  assert(slen <= numbits || radix != 2 && "Insufficient bit width");
  assert(slen*3 <= numbits || radix != 8 && "Insufficient bit width");
  assert(slen*4 <= numbits || radix != 16 && "Insufficient bit width");
  assert((slen*64)/20 <= numbits || radix != 10 && "Insufficient bit width");

  // Allocate memory
  if (!isSingleWord())
    pVal = getClearedMemory(getNumWords());

  // Figure out if we can shift instead of multiply
  uint32_t shift = (radix == 16 ? 4 : radix == 8 ? 3 : radix == 2 ? 1 : 0);

  // Set up an APInt for the digit to add outside the loop so we don't
  // constantly construct/destruct it.
  APInt apdigit(getBitWidth(), 0);
  APInt apradix(getBitWidth(), radix);

  // Enter digit traversal loop
  for (unsigned i = 0; i < slen; i++) {
    // Get a digit
    uint32_t digit = 0;
    char cdigit = str[i];
    if (isdigit(cdigit))
      digit = cdigit - '0';
    else if (isxdigit(cdigit))
      if (cdigit >= 'a')
        digit = cdigit - 'a' + 10;
      else if (cdigit >= 'A')
        digit = cdigit - 'A' + 10;
      else
        assert(0 && "huh?");
    else
      assert(0 && "Invalid character in digit string");

    // Shift or multiple the value by the radix
    if (shift)
      this->shl(shift);
    else
      *this *= apradix;

    // Add in the digit we just interpreted
    apdigit.pVal[0] = digit;
    *this += apdigit;
  }
}

/// to_string - This function translates the APInt into a string.
std::string APInt::toString(uint8_t radix, bool wantSigned) const {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  static const char *digits[] = { 
    "0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F" 
  };
  std::string result;
  uint32_t bits_used = getActiveBits();
  if (isSingleWord()) {
    char buf[65];
    const char *format = (radix == 10 ? (wantSigned ? "%lld" : "%llu") :
       (radix == 16 ? "%llX" : (radix == 8 ? "%llo" : 0)));
    if (format) {
      if (wantSigned) {
        int64_t sextVal = (int64_t(VAL) << (APINT_BITS_PER_WORD-BitWidth)) >> 
                           (APINT_BITS_PER_WORD-BitWidth);
        sprintf(buf, format, sextVal);
      } else 
        sprintf(buf, format, VAL);
    } else {
      memset(buf, 0, 65);
      uint64_t v = VAL;
      while (bits_used) {
        uint32_t bit = v & 1;
        bits_used--;
        buf[bits_used] = digits[bit][0];
        v >>=1;
      }
    }
    result = buf;
    return result;
  }

  if (radix != 10) {
    uint64_t mask = radix - 1;
    uint32_t shift = (radix == 16 ? 4 : radix  == 8 ? 3 : 1);
    uint32_t nibbles = APINT_BITS_PER_WORD / shift;
    for (uint32_t i = 0; i < getNumWords(); ++i) {
      uint64_t value = pVal[i];
      for (uint32_t j = 0; j < nibbles; ++j) {
        result.insert(0, digits[ value & mask ]);
        value >>= shift;
      }
    }
    return result;
  }

  APInt tmp(*this);
  APInt divisor(4, radix);
  APInt zero(tmp.getBitWidth(), 0);
  size_t insert_at = 0;
  if (wantSigned && tmp[BitWidth-1]) {
    // They want to print the signed version and it is a negative value
    // Flip the bits and add one to turn it into the equivalent positive
    // value and put a '-' in the result.
    tmp.flip();
    tmp++;
    result = "-";
    insert_at = 1;
  }
  if (tmp == APInt(tmp.getBitWidth(), 0))
    result = "0";
  else while (tmp.ne(zero)) {
    APInt APdigit(1,0);
    APInt tmp2(tmp.getBitWidth(), 0);
    divide(tmp, tmp.getNumWords(), divisor, divisor.getNumWords(), &tmp2, 
           &APdigit);
    uint32_t digit = APdigit.getValue();
    assert(digit < radix && "divide failed");
    result.insert(insert_at,digits[digit]);
    tmp = tmp2;
  }

  return result;
}

#ifndef NDEBUG
void APInt::dump() const
{
  std::cerr << "APInt(" << BitWidth << ")=" << std::setbase(16);
  if (isSingleWord())
    std::cerr << VAL;
  else for (unsigned i = getNumWords(); i > 0; i--) {
    std::cerr << pVal[i-1] << " ";
  }
  std::cerr << " (" << this->toString(10, false) << ")\n" << std::setbase(10);
}
#endif
