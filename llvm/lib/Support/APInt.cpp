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
  : BitWidth(numBits) {
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
  : BitWidth(numBits) {
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
             uint8_t radix) {
  fromString(numbits, StrStart, slen, radix);
}

/// @brief Create a new APInt by translating the string represented
/// integer value.
APInt::APInt(uint32_t numbits, const std::string& Val, uint8_t radix) {
  assert(!Val.empty() && "String empty?");
  fromString(numbits, Val.c_str(), Val.size(), radix);
}

/// @brief Copy constructor
APInt::APInt(const APInt& APIVal)
  : BitWidth(APIVal.BitWidth) {
  if (isSingleWord()) 
    VAL = APIVal.VAL;
  else {
    pVal = getMemory(getNumWords());
    memcpy(pVal, APIVal.pVal, getNumWords() * APINT_WORD_SIZE);
  }
}

APInt::~APInt() {
  if (!isSingleWord() && pVal) delete[] pVal;
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
static uint64_t add(uint64_t dest[], uint64_t x[], 
                           uint64_t y[], uint32_t len) {
  uint32_t carry = 0;
  for (uint32_t i = 0; i< len; ++i) {
    carry += x[i];
    dest[i] = carry + y[i];
    carry = carry < x[i] ? 1 : (dest[i] < carry ? 1 : 0);
  }
  return carry;
}

/// @brief Addition assignment operator. Adds this APInt by the given APInt&
/// RHS and assigns the result to this APInt.
APInt& APInt::operator+=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) VAL += RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    if (RHS.isSingleWord()) add_1(pVal, pVal, getNumWords(), RHS.VAL);
    else {
      if (getNumWords() <= RHS.getNumWords()) 
        add(pVal, pVal, RHS.pVal, getNumWords());
      else {
        uint64_t carry = add(pVal, pVal, RHS.pVal, RHS.getNumWords());
        add_1(pVal + RHS.getNumWords(), pVal + RHS.getNumWords(), 
              getNumWords() - RHS.getNumWords(), carry);
      }
    }
  }
  clearUnusedBits();
  return *this;
}

/// sub - This function subtracts the integer array x[] by
/// integer array y[], and returns the borrow-out carry.
static uint64_t sub(uint64_t dest[], uint64_t x[], 
                           uint64_t y[], uint32_t len) {
  // Carry indicator.
  uint64_t cy = 0;
  
  for (uint32_t i = 0; i < len; ++i) {
    uint64_t Y = y[i], X = x[i];
    Y += cy;

    cy = Y < cy ? 1 : 0;
    Y = X - Y;
    cy += Y > X ? 1 : 0;
    dest[i] = Y;
  }
  return cy;
}

/// @brief Subtraction assignment operator. Subtracts this APInt by the given
/// APInt &RHS and assigns the result to this APInt.
APInt& APInt::operator-=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) 
    VAL -= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    if (RHS.isSingleWord())
      sub_1(pVal, getNumWords(), RHS.VAL);
    else {
      if (RHS.getNumWords() < getNumWords()) { 
        uint64_t carry = sub(pVal, pVal, RHS.pVal, RHS.getNumWords());
        sub_1(pVal + RHS.getNumWords(), getNumWords() - RHS.getNumWords(), 
              carry); 
      }
      else
        sub(pVal, pVal, RHS.pVal, getNumWords());
    }
  }
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
  if (isSingleWord()) VAL *= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    // one-based first non-zero bit position.
    uint32_t first = getActiveBits();
    uint32_t xlen = !first ? 0 : whichWord(first - 1) + 1;
    if (!xlen) 
      return *this;
    else if (RHS.isSingleWord()) 
      mul_1(pVal, pVal, xlen, RHS.VAL);
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
    return *this;
  } 
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] ^= RHS.pVal[i];
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
  if (isSingleWord())
    return APInt(getBitWidth(), VAL ^ RHS.VAL);
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
  APInt API(RHS);
  API *= *this;
  API.clearUnusedBits();
  return API;
}

/// @brief Addition operator. Adds this APInt by the given APInt& RHS.
APInt APInt::operator+(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  APInt API(*this);
  API += RHS;
  API.clearUnusedBits();
  return API;
}

/// @brief Subtraction operator. Subtracts this APInt by the given APInt& RHS
APInt APInt::operator-(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  APInt API(*this);
  API -= RHS;
  return API;
}

/// @brief Array-indexing support.
bool APInt::operator[](uint32_t bitPosition) const {
  return (maskBit(bitPosition) & (isSingleWord() ? 
          VAL : pVal[whichWord(bitPosition)])) != 0;
}

/// @brief Equality operator. Compare this APInt with the given APInt& RHS 
/// for the validity of the equality relationship.
bool APInt::operator==(const APInt& RHS) const {
  uint32_t n1 = getActiveBits();
  uint32_t n2 = RHS.getActiveBits();
  if (n1 != n2) return false;
  else if (isSingleWord()) 
    return VAL == (RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0]);
  else {
    if (n1 <= APINT_BITS_PER_WORD)
      return pVal[0] == (RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0]);
    for (int i = whichWord(n1 - 1); i >= 0; --i)
      if (pVal[i] != RHS.pVal[i]) return false;
  }
  return true;
}

/// @brief Equality operator. Compare this APInt with the given uint64_t value 
/// for the validity of the equality relationship.
bool APInt::operator==(uint64_t Val) const {
  if (isSingleWord())
    return VAL == Val;
  else {
    uint32_t n = getActiveBits(); 
    if (n <= APINT_BITS_PER_WORD)
      return pVal[0] == Val;
    else
      return false;
  }
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

  APInt tmp(*this);
  APInt divisor(tmp.getBitWidth(), radix);
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
  if (tmp == 0)
    result = "0";
  else while (tmp.ne(zero)) {
    APInt APdigit = APIntOps::urem(tmp,divisor);
    uint32_t digit = APdigit.getValue();
    assert(digit < radix && "urem failed");
    result.insert(insert_at,digits[digit]);
    tmp = APIntOps::udiv(tmp, divisor);
  }

  return result;
}

/// getMaxValue - This function returns the largest value
/// for an APInt of the specified bit-width and if isSign == true,
/// it should be largest signed value, otherwise unsigned value.
APInt APInt::getMaxValue(uint32_t numBits, bool isSign) {
  APInt APIVal(numBits, 0);
  APIVal.set();
  if (isSign) APIVal.clear(numBits - 1);
  return APIVal;
}

/// getMinValue - This function returns the smallest value for
/// an APInt of the given bit-width and if isSign == true,
/// it should be smallest signed value, otherwise zero.
APInt APInt::getMinValue(uint32_t numBits, bool isSign) {
  APInt APIVal(numBits, 0);
  if (isSign) APIVal.set(numBits - 1);
  return APIVal;
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
  if (isSingleWord())
    return CountLeadingZeros_64(VAL) - (APINT_BITS_PER_WORD - BitWidth);
  uint32_t Count = 0;
  for (uint32_t i = getNumWords(); i > 0u; --i) {
    uint32_t tmp = CountLeadingZeros_64(pVal[i-1]);
    Count += tmp;
    if (tmp != APINT_BITS_PER_WORD)
      if (i == getNumWords())
        Count -= (APINT_BITS_PER_WORD - whichBit(BitWidth));
      break;
  }
  return Count;
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
  if (isSingleWord() || getActiveBits() <= APINT_BITS_PER_WORD) {
    if (isSigned) {
      int64_t sext = (int64_t(VAL) << (64-BitWidth)) >> (64-BitWidth);
      return double(sext);
    } else
      return double(VAL);
  }

  bool isNeg = isSigned ? (*this)[BitWidth-1] : false;
  APInt Tmp(isNeg ? -(*this) : (*this));
  uint32_t n = Tmp.getActiveBits();
  // Exponent when normalized to have decimal point directly after
  // leading one. This is stored excess 1023 in the exponent bit field.
  uint64_t exp = n - 1;

  // Gross overflow.
  assert(exp <= 1023 && "Infinity value!");

  // Number of bits in mantissa including the leading one
  // equals to 53.
  uint64_t mantissa;
  if (n % APINT_BITS_PER_WORD >= 53)
    mantissa = Tmp.pVal[whichWord(n - 1)] >> (n % APINT_BITS_PER_WORD - 53);
  else
    mantissa = (Tmp.pVal[whichWord(n - 1)] << (53 - n % APINT_BITS_PER_WORD)) | 
               (Tmp.pVal[whichWord(n - 1) - 1] >> 
                (11 + n % APINT_BITS_PER_WORD));
  // The leading bit of mantissa is implicit, so get rid of it.
  mantissa &= ~(1ULL << 52);
  uint64_t sign = isNeg ? (1ULL << (APINT_BITS_PER_WORD - 1)) : 0;
  exp += 1023;
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

/// div - This is basically Knuth's formulation of the classical algorithm.
/// Correspondance with Knuth's notation:
/// Knuth's u[0:m+n] == zds[nx:0].
/// Knuth's v[1:n] == y[ny-1:0]
/// Knuth's n == ny.
/// Knuth's m == nx-ny.
/// Our nx == Knuth's m+n.
/// Could be re-implemented using gmp's mpn_divrem:
/// zds[nx] = mpn_divrem (&zds[ny], 0, zds, nx, y, ny).
static void div(uint32_t zds[], uint32_t nx, uint32_t y[], uint32_t ny) {
  uint32_t j = nx;
  do {                          // loop over digits of quotient
    // Knuth's j == our nx-j.
    // Knuth's u[j:j+n] == our zds[j:j-ny].
    uint32_t qhat;  // treated as unsigned
    if (zds[j] == y[ny-1]) 
      qhat = -1U;  // 0xffffffff
    else {
      uint64_t w = (((uint64_t)(zds[j])) << 32) + 
                   ((uint64_t)zds[j-1] & 0xffffffffL);
      qhat = (uint32_t) unitDiv(w, y[ny-1]);
    }
    if (qhat) {
      uint32_t borrow = subMul(zds, j - ny, y, ny, qhat);
      uint32_t save = zds[j];
      uint64_t num = ((uint64_t)save&0xffffffffL) - 
                     ((uint64_t)borrow&0xffffffffL);
      while (num) {
        qhat--;
        uint64_t carry = 0;
        for (uint32_t i = 0;  i < ny; i++) {
          carry += ((uint64_t) zds[j-ny+i] & 0xffffffffL)
            + ((uint64_t) y[i] & 0xffffffffL);
          zds[j-ny+i] = (uint32_t) carry;
          carry >>= 32;
        }
        zds[j] += carry;
        num = carry - 1;
      }
    }
    zds[j] = qhat;
  } while (--j >= ny);
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

  // Make a temporary to hold the result
  APInt Result(*this);

  // Get some facts about the LHS and RHS number of bits and words
  uint32_t rhsBits = RHS.getActiveBits();
  uint32_t rhsWords = !rhsBits ? 0 : (APInt::whichWord(rhsBits - 1) + 1);
  assert(rhsWords && "Divided by zero???");
  uint32_t lhsBits = Result.getActiveBits();
  uint32_t lhsWords = !lhsBits ? 0 : (APInt::whichWord(lhsBits - 1) + 1);

  // Deal with some degenerate cases
  if (!lhsWords) 
    return Result; // 0 / X == 0
  else if (lhsWords < rhsWords || Result.ult(RHS))
    // X / Y with X < Y == 0
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
  else if (Result == RHS) {
    // X / X == 1
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    Result.pVal[0] = 1;
  } else if (lhsWords == 1)
    // All high words are zero, just use native divide
    Result.pVal[0] /= RHS.pVal[0];
  else {
    // Compute it the hard way ..
    APInt X(BitWidth, 0);
    APInt Y(BitWidth, 0);
    uint32_t nshift = 
      (APINT_BITS_PER_WORD - 1) - ((rhsBits - 1) % APINT_BITS_PER_WORD );
    if (nshift) {
      Y = APIntOps::shl(RHS, nshift);
      X = APIntOps::shl(Result, nshift);
      ++lhsWords;
    }
    div((uint32_t*)X.pVal, lhsWords * 2 - 1, 
        (uint32_t*)(Y.isSingleWord()? &Y.VAL : Y.pVal), rhsWords*2);
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    memcpy(Result.pVal, X.pVal + rhsWords, 
           (lhsWords - rhsWords) * APINT_WORD_SIZE);
  }
  return Result;
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
  if (lhsWords == 0)
    // 0 % Y == 0
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
  else if (lhsWords < rhsWords || Result.ult(RHS))
    // X % Y == X iff X < Y
    return Result;
  else if (Result == RHS)
    // X % X == 0;
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
  else if (lhsWords == 1) 
    // All high words are zero, just use native remainder
    Result.pVal[0] %=  RHS.pVal[0];
  else {
    // Do it the hard way
    APInt X((lhsWords+1)*APINT_BITS_PER_WORD, 0);
    APInt Y(rhsWords*APINT_BITS_PER_WORD, 0);
    uint32_t nshift = 
      (APINT_BITS_PER_WORD - 1) - (rhsBits - 1) % APINT_BITS_PER_WORD;
    if (nshift) {
      APIntOps::shl(Y, nshift);
      APIntOps::shl(X, nshift);
    }
    div((uint32_t*)X.pVal, rhsWords*2-1, 
        (uint32_t*)(Y.isSingleWord()? &Y.VAL : Y.pVal), rhsWords*2);
    memset(Result.pVal, 0, Result.getNumWords() * APINT_WORD_SIZE);
    for (uint32_t i = 0; i < rhsWords-1; ++i)
      Result.pVal[i] = (X.pVal[i] >> nshift) | 
                        (X.pVal[i+1] << (APINT_BITS_PER_WORD - nshift));
    Result.pVal[rhsWords-1] = X.pVal[rhsWords-1] >> nshift;
  }
  return Result;
}

/// @brief Converts a char array into an integer.
void APInt::fromString(uint32_t numbits, const char *StrStart, uint32_t slen, 
                       uint8_t radix) {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  assert(StrStart && "String is null?");
  uint32_t size = 0;
  // If the radix is a power of 2, read the input
  // from most significant to least significant.
  if ((radix & (radix - 1)) == 0) {
    uint32_t nextBitPos = 0; 
    uint32_t bits_per_digit = radix / 8 + 2;
    uint64_t resDigit = 0;
    BitWidth = slen * bits_per_digit;
    if (getNumWords() > 1)
      pVal = getMemory(getNumWords());
    for (int i = slen - 1; i >= 0; --i) {
      uint64_t digit = StrStart[i] - '0';
      resDigit |= digit << nextBitPos;
      nextBitPos += bits_per_digit;
      if (nextBitPos >= APINT_BITS_PER_WORD) {
        if (isSingleWord()) {
          VAL = resDigit;
           break;
        }
        pVal[size++] = resDigit;
        nextBitPos -= APINT_BITS_PER_WORD;
        resDigit = digit >> (bits_per_digit - nextBitPos);
      }
    }
    if (!isSingleWord() && size <= getNumWords()) 
      pVal[size] = resDigit;
  } else {   // General case.  The radix is not a power of 2.
    // For 10-radix, the max value of 64-bit integer is 18446744073709551615,
    // and its digits number is 20.
    const uint32_t chars_per_word = 20;
    if (slen < chars_per_word || 
        (slen == chars_per_word &&             // In case the value <= 2^64 - 1
         strcmp(StrStart, "18446744073709551615") <= 0)) {
      BitWidth = APINT_BITS_PER_WORD;
      VAL = strtoull(StrStart, 0, 10);
    } else { // In case the value > 2^64 - 1
      BitWidth = (slen / chars_per_word + 1) * APINT_BITS_PER_WORD;
      pVal = getClearedMemory(getNumWords());
      uint32_t str_pos = 0;
      while (str_pos < slen) {
        uint32_t chunk = slen - str_pos;
        if (chunk > chars_per_word - 1)
          chunk = chars_per_word - 1;
        uint64_t resDigit = StrStart[str_pos++] - '0';
        uint64_t big_base = radix;
        while (--chunk > 0) {
          resDigit = resDigit * radix + StrStart[str_pos++] - '0';
          big_base *= radix;
        }
       
        uint64_t carry;
        if (!size)
          carry = resDigit;
        else {
          carry = mul_1(pVal, pVal, size, big_base);
          carry += add_1(pVal, pVal, size, resDigit);
        }
        
        if (carry) pVal[size++] = carry;
      }
    }
  }
}
