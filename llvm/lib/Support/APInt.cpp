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

#if 0
#include "llvm/DerivedTypes.h"
#include "llvm/Support/MathExtras.h"
#include <cstring>
#include <cstdlib>
using namespace llvm;

/// mul_1 - This function performs the multiplication operation on a
/// large integer (represented as an integer array) and a uint64_t integer.
/// @returns the carry of the multiplication.
static uint64_t mul_1(uint64_t dest[], uint64_t x[],
                     unsigned len, uint64_t y) {
  // Split y into high 32-bit part and low 32-bit part.
  uint64_t ly = y & 0xffffffffULL, hy = y >> 32;
  uint64_t carry = 0, lx, hx;
  for (unsigned i = 0; i < len; ++i) {
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
static void mul(uint64_t dest[], uint64_t x[], unsigned xlen,
               uint64_t y[], unsigned ylen) {
  dest[xlen] = mul_1(dest, x, xlen, y[0]);

  for (unsigned i = 1; i < ylen; ++i) {
    uint64_t ly = y[i] & 0xffffffffULL, hy = y[i] >> 32;
    uint64_t carry = 0, lx, hx;
    for (unsigned j = 0; j < xlen; ++j) {
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

/// add_1 - This function adds the integer array x[] by integer y and
/// returns the carry.
/// @returns the carry of the addition.
static uint64_t add_1(uint64_t dest[], uint64_t x[],
                      unsigned len, uint64_t y) {
  uint64_t carry = y;

  for (unsigned i = 0; i < len; ++i) {
    dest[i] = carry + x[i];
    carry = (dest[i] < carry) ? 1 : 0;
  }
  return carry;
}

/// add - This function adds the integer array x[] by integer array
/// y[] and returns the carry.
static uint64_t add(uint64_t dest[], uint64_t x[],
                    uint64_t y[], unsigned len) {
  unsigned carry = 0;
  
  for (unsigned i = 0; i< len; ++i) {
    carry += x[i];
    dest[i] = carry + y[i];
    carry = carry < x[i] ? 1 : (dest[i] < carry ? 1 : 0);
  }
  return carry;
}

/// sub_1 - This function subtracts the integer array x[] by
/// integer y and returns the borrow-out carry.
static uint64_t sub_1(uint64_t x[], unsigned len, uint64_t y) {
  uint64_t cy = y;

  for (unsigned i = 0; i < len; ++i) {
    uint64_t X = x[i];
    x[i] -= cy;
    if (cy > X) 
      cy = 1;
    else {
      cy = 0;
      break;
    }
  }

  return cy;
}

/// sub - This function subtracts the integer array x[] by
/// integer array y[], and returns the borrow-out carry.
static uint64_t sub(uint64_t dest[], uint64_t x[],
                    uint64_t y[], unsigned len) {
  // Carry indicator.
  uint64_t cy = 0;
  
  for (unsigned i = 0; i < len; ++i) {
    uint64_t Y = y[i], X = x[i];
    Y += cy;

    cy = Y < cy ? 1 : 0;
    Y = X - Y;
    cy += Y > X ? 1 : 0;
    dest[i] = Y;
  }
  return cy;
}

/// UnitDiv - This function divides N by D, 
/// and returns (remainder << 32) | quotient.
/// Assumes (N >> 32) < D.
static uint64_t unitDiv(uint64_t N, unsigned D) {
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

/// subMul - This function substracts x[len-1:0] * y from 
/// dest[offset+len-1:offset], and returns the most significant 
/// word of the product, minus the borrow-out from the subtraction.
static unsigned subMul(unsigned dest[], unsigned offset, 
                        unsigned x[], unsigned len, unsigned y) {
  uint64_t yl = (uint64_t) y & 0xffffffffL;
  unsigned carry = 0;
  unsigned j = 0;
  do {
    uint64_t prod = ((uint64_t) x[j] & 0xffffffffL) * yl;
    unsigned prod_low = (unsigned) prod;
    unsigned prod_high = (unsigned) (prod >> 32);
    prod_low += carry;
    carry = (prod_low < carry ? 1 : 0) + prod_high;
    unsigned x_j = dest[offset+j];
    prod_low = x_j - prod_low;
    if (prod_low > x_j) ++carry;
    dest[offset+j] = prod_low;
  } while (++j < len);
  return carry;
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
static void div(unsigned zds[], unsigned nx, unsigned y[], unsigned ny) {
  unsigned j = nx;
  do {                          // loop over digits of quotient
    // Knuth's j == our nx-j.
    // Knuth's u[j:j+n] == our zds[j:j-ny].
    unsigned qhat;  // treated as unsigned
    if (zds[j] == y[ny-1]) qhat = -1U;  // 0xffffffff
    else {
      uint64_t w = (((uint64_t)(zds[j])) << 32) + 
                   ((uint64_t)zds[j-1] & 0xffffffffL);
      qhat = (unsigned) unitDiv(w, y[ny-1]);
    }
    if (qhat) {
      unsigned borrow = subMul(zds, j - ny, y, ny, qhat);
      unsigned save = zds[j];
      uint64_t num = ((uint64_t)save&0xffffffffL) - 
                     ((uint64_t)borrow&0xffffffffL);
      while (num) {
        qhat--;
        uint64_t carry = 0;
        for (unsigned i = 0;  i < ny; i++) {
          carry += ((uint64_t) zds[j-ny+i] & 0xffffffffL)
            + ((uint64_t) y[i] & 0xffffffffL);
          zds[j-ny+i] = (unsigned) carry;
          carry >>= 32;
        }
        zds[j] += carry;
        num = carry - 1;
      }
    }
    zds[j] = qhat;
  } while (--j >= ny);
}

/// lshift - This function shift x[0:len-1] left by shiftAmt bits, and 
/// store the len least significant words of the result in 
/// dest[d_offset:d_offset+len-1]. It returns the bits shifted out from 
/// the most significant digit.
static uint64_t lshift(uint64_t dest[], unsigned d_offset,
                       uint64_t x[], unsigned len, unsigned shiftAmt) {
  unsigned count = 64 - shiftAmt;
  int i = len - 1;
  uint64_t high_word = x[i], retVal = high_word >> count;
  ++d_offset;
  while (--i >= 0) {
    uint64_t low_word = x[i];
    dest[d_offset+i] = (high_word << shiftAmt) | (low_word >> count);
    high_word = low_word;
  }
  dest[d_offset+i] = high_word << shiftAmt;
  return retVal;
}

APInt::APInt(uint64_t val, unsigned numBits)
  : BitsNum(numBits) {
  assert(BitsNum >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(BitsNum <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  if (isSingleWord()) 
    VAL = val & (~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - BitsNum));
  else {
    // Memory allocation and check if successful.
    assert((pVal = new uint64_t[getNumWords()]) && 
            "APInt memory allocation fails!");
    memset(pVal, 0, getNumWords() * 8);
    pVal[0] = val;
  }
}

APInt::APInt(unsigned numBits, uint64_t bigVal[])
  : BitsNum(numBits) {
  assert(BitsNum >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(BitsNum <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  assert(bigVal && "Null pointer detected!");
  if (isSingleWord())
    VAL = bigVal[0] & (~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - BitsNum));
  else {
    // Memory allocation and check if successful.
    assert((pVal = new uint64_t[getNumWords()]) && 
           "APInt memory allocation fails!");
    // Calculate the actual length of bigVal[].
    unsigned n = sizeof(*bigVal) / sizeof(bigVal[0]);
    unsigned maxN = std::max<unsigned>(n, getNumWords());
    unsigned minN = std::min<unsigned>(n, getNumWords());
    memcpy(pVal, bigVal, (minN - 1) * 8);
    pVal[minN-1] = bigVal[minN-1] & (~uint64_t(0ULL) >> (64 - BitsNum % 64));
    if (maxN == getNumWords())
      memset(pVal+n, 0, (getNumWords() - n) * 8);
  }
}

/// @brief Create a new APInt by translating the char array represented
/// integer value.
APInt::APInt(const char StrStart[], unsigned slen, uint8_t radix) {
  StrToAPInt(StrStart, slen, radix);
}

/// @brief Create a new APInt by translating the string represented
/// integer value.
APInt::APInt(const std::string& Val, uint8_t radix) {
  assert(!Val.empty() && "String empty?");
  StrToAPInt(Val.c_str(), Val.size(), radix);
}

/// @brief Converts a char array into an integer.
void APInt::StrToAPInt(const char *StrStart, unsigned slen, uint8_t radix) {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  assert(StrStart && "String empty?");
  unsigned size = 0;
  // If the radix is a power of 2, read the input
  // from most significant to least significant.
  if ((radix & (radix - 1)) == 0) {
    unsigned nextBitPos = 0, bits_per_digit = radix / 8 + 2;
    uint64_t resDigit = 0;
    BitsNum = slen * bits_per_digit;
    if (getNumWords() > 1)
      assert((pVal = new uint64_t[getNumWords()]) && 
             "APInt memory allocation fails!");
    for (int i = slen - 1; i >= 0; --i) {
      uint64_t digit = StrStart[i] - 48;             // '0' == 48.
      resDigit |= digit << nextBitPos;
      nextBitPos += bits_per_digit;
      if (nextBitPos >= 64) {
        if (isSingleWord()) {
          VAL = resDigit;
           break;
        }
        pVal[size++] = resDigit;
        nextBitPos -= 64;
        resDigit = digit >> (bits_per_digit - nextBitPos);
      }
    }
    if (!isSingleWord() && size <= getNumWords()) 
      pVal[size] = resDigit;
  } else {   // General case.  The radix is not a power of 2.
    // For 10-radix, the max value of 64-bit integer is 18446744073709551615,
    // and its digits number is 14.
    const unsigned chars_per_word = 20;
    if (slen < chars_per_word || 
        (slen == chars_per_word &&             // In case the value <= 2^64 - 1
         strcmp(StrStart, "18446744073709551615") <= 0)) {
      BitsNum = 64;
      VAL = strtoull(StrStart, 0, 10);
    } else { // In case the value > 2^64 - 1
      BitsNum = (slen / chars_per_word + 1) * 64;
      assert((pVal = new uint64_t[getNumWords()]) && 
             "APInt memory allocation fails!");
      memset(pVal, 0, getNumWords() * 8);
      unsigned str_pos = 0;
      while (str_pos < slen) {
        unsigned chunk = slen - str_pos;
        if (chunk > chars_per_word - 1)
          chunk = chars_per_word - 1;
        uint64_t resDigit = StrStart[str_pos++] - 48;  // 48 == '0'.
        uint64_t big_base = radix;
        while (--chunk > 0) {
          resDigit = resDigit * radix + StrStart[str_pos++] - 48;
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

APInt::APInt(const APInt& APIVal)
  : BitsNum(APIVal.BitsNum) {
  if (isSingleWord()) VAL = APIVal.VAL;
  else {
    // Memory allocation and check if successful.
    assert((pVal = new uint64_t[getNumWords()]) && 
           "APInt memory allocation fails!");
    memcpy(pVal, APIVal.pVal, getNumWords() * 8);
  }
}

APInt::~APInt() {
  if (!isSingleWord() && pVal) delete[] pVal;
}

/// @brief Copy assignment operator. Create a new object from the given
/// APInt one by initialization.
APInt& APInt::operator=(const APInt& RHS) {
  if (isSingleWord()) VAL = RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    unsigned minN = std::min(getNumWords(), RHS.getNumWords());
    memcpy(pVal, RHS.isSingleWord() ? &RHS.VAL : RHS.pVal, minN * 8);
    if (getNumWords() != minN)
      memset(pVal + minN, 0, (getNumWords() - minN) * 8);
  }
  return *this;
}

/// @brief Assignment operator. Assigns a common case integer value to 
/// the APInt.
APInt& APInt::operator=(uint64_t RHS) {
  if (isSingleWord()) VAL = RHS;
  else {
    pVal[0] = RHS;
    memset(pVal, 0, (getNumWords() - 1) * 8);
  }
  TruncToBits();
  return *this;
}

/// @brief Prefix increment operator. Increments the APInt by one.
APInt& APInt::operator++() {
  if (isSingleWord()) ++VAL;
  else
    add_1(pVal, pVal, getNumWords(), 1);
  TruncToBits();
  return *this;
}

/// @brief Prefix decrement operator. Decrements the APInt by one.
APInt& APInt::operator--() {
  if (isSingleWord()) --VAL;
  else
    sub_1(pVal, getNumWords(), 1);
  TruncToBits();
  return *this;
}

/// @brief Addition assignment operator. Adds this APInt by the given APInt&
/// RHS and assigns the result to this APInt.
APInt& APInt::operator+=(const APInt& RHS) {
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
  TruncToBits();
  return *this;
}

/// @brief Subtraction assignment operator. Subtracts this APInt by the given
/// APInt &RHS and assigns the result to this APInt.
APInt& APInt::operator-=(const APInt& RHS) {
  if (isSingleWord()) 
    VAL -= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    if (RHS.isSingleWord())
      sub_1(pVal, getNumWords(), RHS.VAL);
    else {
      if (RHS.getNumWords() < getNumWords()) { 
        uint64_t carry = sub(pVal, pVal, RHS.pVal, RHS.getNumWords());
        sub_1(pVal + RHS.getNumWords(), getNumWords() - RHS.getNumWords(), carry); 
      }
      else
        sub(pVal, pVal, RHS.pVal, getNumWords());
    }
  }
  TruncToBits();
  return *this;
}

/// @brief Multiplication assignment operator. Multiplies this APInt by the 
/// given APInt& RHS and assigns the result to this APInt.
APInt& APInt::operator*=(const APInt& RHS) {
  if (isSingleWord()) VAL *= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    // one-based first non-zero bit position.
    unsigned first = getNumWords() * APINT_BITS_PER_WORD - CountLeadingZeros();
    unsigned xlen = !first ? 0 : whichWord(first - 1) + 1;
    if (!xlen) 
      return *this;
    else if (RHS.isSingleWord()) 
      mul_1(pVal, pVal, xlen, RHS.VAL);
    else {
      first = RHS.getNumWords() * APINT_BITS_PER_WORD - RHS.CountLeadingZeros();
      unsigned ylen = !first ? 0 : whichWord(first - 1) + 1;
      if (!ylen) {
        memset(pVal, 0, getNumWords() * 8);
        return *this;
      }
      uint64_t *dest = new uint64_t[xlen+ylen];
      assert(dest && "Memory Allocation Failed!");
      mul(dest, pVal, xlen, RHS.pVal, ylen);
      memcpy(pVal, dest, ((xlen + ylen >= getNumWords()) ? 
                         getNumWords() : xlen + ylen) * 8);
      delete[] dest;
    }
  }
  TruncToBits();
  return *this;
}

/// @brief Bitwise AND assignment operator. Performs bitwise AND operation on
/// this APInt and the given APInt& RHS, assigns the result to this APInt.
APInt& APInt::operator&=(const APInt& RHS) {
  if (isSingleWord()) {
    if (RHS.isSingleWord()) VAL &= RHS.VAL;
    else VAL &= RHS.pVal[0];
  } else {
    if (RHS.isSingleWord()) {
      memset(pVal, 0, (getNumWords() - 1) * 8);
      pVal[0] &= RHS.VAL;
    } else {
      unsigned minwords = getNumWords() < RHS.getNumWords() ? 
                          getNumWords() : RHS.getNumWords();
      for (unsigned i = 0; i < minwords; ++i)
        pVal[i] &= RHS.pVal[i];
      if (getNumWords() > minwords) 
        memset(pVal+minwords, 0, (getNumWords() - minwords) * 8);
    }
  }
  return *this;
}

/// @brief Bitwise OR assignment operator. Performs bitwise OR operation on 
/// this APInt and the given APInt& RHS, assigns the result to this APInt.
APInt& APInt::operator|=(const APInt& RHS) {
  if (isSingleWord()) {
    if (RHS.isSingleWord()) VAL |= RHS.VAL;
    else VAL |= RHS.pVal[0];
  } else {
    if (RHS.isSingleWord()) {
      pVal[0] |= RHS.VAL;
    } else {
      unsigned minwords = getNumWords() < RHS.getNumWords() ? 
                          getNumWords() : RHS.getNumWords();
      for (unsigned i = 0; i < minwords; ++i)
        pVal[i] |= RHS.pVal[i];
    }
  }
  TruncToBits();
  return *this;
}

/// @brief Bitwise XOR assignment operator. Performs bitwise XOR operation on
/// this APInt and the given APInt& RHS, assigns the result to this APInt.
APInt& APInt::operator^=(const APInt& RHS) {
  if (isSingleWord()) {
    if (RHS.isSingleWord()) VAL ^= RHS.VAL;
    else VAL ^= RHS.pVal[0];
  } else {
    if (RHS.isSingleWord()) {
      for (unsigned i = 0; i < getNumWords(); ++i)
        pVal[i] ^= RHS.VAL;
    } else {
      unsigned minwords = getNumWords() < RHS.getNumWords() ? 
                          getNumWords() : RHS.getNumWords();
      for (unsigned i = 0; i < minwords; ++i)
        pVal[i] ^= RHS.pVal[i];
      if (getNumWords() > minwords)
        for (unsigned i = minwords; i < getNumWords(); ++i)
          pVal[i] ^= 0;
    }
  }
  TruncToBits();
  return *this;
}

/// @brief Bitwise AND operator. Performs bitwise AND operation on this APInt
/// and the given APInt& RHS.
APInt APInt::operator&(const APInt& RHS) const {
  APInt API(RHS);
  return API &= *this;
}

/// @brief Bitwise OR operator. Performs bitwise OR operation on this APInt 
/// and the given APInt& RHS.
APInt APInt::operator|(const APInt& RHS) const {
  APInt API(RHS);
  API |= *this;
  API.TruncToBits();
  return API;
}

/// @brief Bitwise XOR operator. Performs bitwise XOR operation on this APInt
/// and the given APInt& RHS.
APInt APInt::operator^(const APInt& RHS) const {
  APInt API(RHS);
  API ^= *this;
  API.TruncToBits();
  return API;
}

/// @brief Logical AND operator. Performs logical AND operation on this APInt
/// and the given APInt& RHS.
bool APInt::operator&&(const APInt& RHS) const {
  if (isSingleWord()) 
    return RHS.isSingleWord() ? VAL && RHS.VAL : VAL && RHS.pVal[0];
  else if (RHS.isSingleWord())
    return RHS.VAL && pVal[0];
  else {
    unsigned minN = std::min(getNumWords(), RHS.getNumWords());
    for (unsigned i = 0; i < minN; ++i)
      if (pVal[i] && RHS.pVal[i])
        return true;
  }
  return false;
}

/// @brief Logical OR operator. Performs logical OR operation on this APInt 
/// and the given APInt& RHS.
bool APInt::operator||(const APInt& RHS) const {
  if (isSingleWord()) 
    return RHS.isSingleWord() ? VAL || RHS.VAL : VAL || RHS.pVal[0];
  else if (RHS.isSingleWord())
    return RHS.VAL || pVal[0];
  else {
    unsigned minN = std::min(getNumWords(), RHS.getNumWords());
    for (unsigned i = 0; i < minN; ++i)
      if (pVal[i] || RHS.pVal[i])
        return true;
  }
  return false;
}

/// @brief Logical negation operator. Performs logical negation operation on
/// this APInt.
bool APInt::operator !() const {
  if (isSingleWord())
    return !VAL;
  else
    for (unsigned i = 0; i < getNumWords(); ++i)
       if (pVal[i]) 
         return false;
  return true;
}

/// @brief Multiplication operator. Multiplies this APInt by the given APInt& 
/// RHS.
APInt APInt::operator*(const APInt& RHS) const {
  APInt API(RHS);
  API *= *this;
  API.TruncToBits();
  return API;
}

/// @brief Addition operator. Adds this APInt by the given APInt& RHS.
APInt APInt::operator+(const APInt& RHS) const {
  APInt API(*this);
  API += RHS;
  API.TruncToBits();
  return API;
}

/// @brief Subtraction operator. Subtracts this APInt by the given APInt& RHS
APInt APInt::operator-(const APInt& RHS) const {
  APInt API(*this);
  API -= RHS;
  return API;
}

/// @brief Array-indexing support.
bool APInt::operator[](unsigned bitPosition) const {
  return maskBit(bitPosition) & (isSingleWord() ? 
         VAL : pVal[whichWord(bitPosition)]) != 0;
}

/// @brief Equality operator. Compare this APInt with the given APInt& RHS 
/// for the validity of the equality relationship.
bool APInt::operator==(const APInt& RHS) const {
  unsigned n1 = getNumWords() * APINT_BITS_PER_WORD - CountLeadingZeros(), 
    n2 = RHS.getNumWords() * APINT_BITS_PER_WORD - RHS.CountLeadingZeros();
  if (n1 != n2) return false;
  else if (isSingleWord()) 
    return VAL == (RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0]);
  else {
    if (n1 <= 64)
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
    unsigned n = getNumWords() * APINT_BITS_PER_WORD - CountLeadingZeros();
    if (n <= 64)
      return pVal[0] == Val;
    else
      return false;
  }
}

/// @brief Less-than operator. Compare this APInt with the given APInt& RHS
/// for the validity of the less-than relationship.
bool APInt::operator <(const APInt& RHS) const {
  unsigned n1 = getNumWords() * 64 - CountLeadingZeros(), 
           n2 = RHS.getNumWords() * 64 - RHS.CountLeadingZeros();
  if (n1 < n2) return true;
  else if (n1 > n2) return false;
  else if (isSingleWord())
    return VAL < (RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0]);
  else {
    if (n1 <= 64)
      return pVal[0] < (RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0]);
    for (int i = whichWord(n1 - 1); i >= 0; --i) {
      if (pVal[i] > RHS.pVal[i]) return false;
      else if (pVal[i] < RHS.pVal[i]) return true;
    }
  }
  return false;
}

/// @brief Less-than-or-equal operator. Compare this APInt with the given 
/// APInt& RHS for the validity of the less-than-or-equal relationship.
bool APInt::operator<=(const APInt& RHS) const {
  return (*this) == RHS || (*this) < RHS;
}

/// @brief Greater-than operator. Compare this APInt with the given APInt& RHS
/// for the validity of the greater-than relationship.
bool APInt::operator >(const APInt& RHS) const {
  return !((*this) <= RHS);
}

/// @brief Greater-than-or-equal operator. Compare this APInt with the given 
/// APInt& RHS for the validity of the greater-than-or-equal relationship.
bool APInt::operator>=(const APInt& RHS) const {
  return !((*this) < RHS);
}  

/// Set the given bit to 1 whose poition is given as "bitPosition".
/// @brief Set a given bit to 1.
APInt& APInt::set(unsigned bitPosition) {
  if (isSingleWord()) VAL |= maskBit(bitPosition);
  else pVal[whichWord(bitPosition)] |= maskBit(bitPosition);
  return *this;
}

/// @brief Set every bit to 1.
APInt& APInt::set() {
  if (isSingleWord()) VAL = -1ULL;
  else
    for (unsigned i = 0; i < getNumWords(); ++i)
      pVal[i] = -1ULL;
  return *this;
}

/// Set the given bit to 0 whose position is given as "bitPosition".
/// @brief Set a given bit to 0.
APInt& APInt::clear(unsigned bitPosition) {
  if (isSingleWord()) VAL &= ~maskBit(bitPosition);
  else pVal[whichWord(bitPosition)] &= ~maskBit(bitPosition);
  return *this;
}

/// @brief Set every bit to 0.
APInt& APInt::clear() {
  if (isSingleWord()) VAL = 0;
  else 
    memset(pVal, 0, getNumWords() * 8);
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
  if (isSingleWord()) VAL = (~(VAL << (64 - BitsNum))) >> (64 - BitsNum);
  else {
    unsigned i = 0;
    for (; i < getNumWords() - 1; ++i)
      pVal[i] = ~pVal[i];
    unsigned offset = 64 - (BitsNum - 64 * (i - 1));
    pVal[i] = (~(pVal[i] << offset)) >> offset;
  }
  return *this;
}

/// Toggle a given bit to its opposite value whose position is given 
/// as "bitPosition".
/// @brief Toggles a given bit to its opposite value.
APInt& APInt::flip(unsigned bitPosition) {
  assert(bitPosition < BitsNum && "Out of the bit-width range!");
  if ((*this)[bitPosition]) clear(bitPosition);
  else set(bitPosition);
  return *this;
}

/// to_string - This function translates the APInt into a string.
std::string APInt::to_string(uint8_t radix) const {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  char *buf = 0;
  unsigned n = getNumWords() * 64 - CountLeadingZeros();
  std::string format = radix == 8 ? 
                       "%0*llo" : (radix == 10 ? "%0*llu" : "%0*llx");
  // If the radix is a power of 2, set the format of ostringstream,
  // and output the value into buf.
  if ((radix & (radix - 1)) == 0) {
    assert((buf = new char[n / Log2_32(radix) + 2]) && 
           "Memory allocation failed");
    if (isSingleWord())
      sprintf(buf, format.c_str(), 0, VAL);
    else {
      unsigned offset = sprintf(buf, format.c_str(), 0, pVal[whichWord(n-1)]);
      for (int i = whichWord(n-1) - 1; i >= 0; --i)
        offset += sprintf(buf + offset, format.c_str(), 
          64 / Log2_32(radix) + (64 % Log2_32(radix) ? 1 : 0), pVal[i]);
    }
  }
  else {  // If the radix = 10, need to translate the value into a
          // string.
    assert((buf = new char[(n / 64 + 1) * 20]) && "Memory allocation failed");
    if (isSingleWord())
      sprintf(buf, format.c_str(), 0, VAL);
    else {
      // FIXME: To be supported.
    }
  }
  std::string retStr(buf);
  delete[] buf;
  return retStr;
}

/// getMaxValue - This function returns the largest value
/// for an APInt of the specified bit-width and if isSign == true,
/// it should be largest signed value, otherwise unsigned value.
APInt APInt::getMaxValue(unsigned numBits, bool isSign) {
  APInt APIVal(numBits, 1);
  APIVal.set();
  return isSign ? APIVal.clear(numBits) : APIVal;
}

/// getMinValue - This function returns the smallest value for
/// an APInt of the given bit-width and if isSign == true,
/// it should be smallest signed value, otherwise zero.
APInt APInt::getMinValue(unsigned numBits, bool isSign) {
  APInt APIVal(0, numBits);
  return isSign ? APIVal : APIVal.set(numBits);
}

/// getAllOnesValue - This function returns an all-ones value for
/// an APInt of the specified bit-width.
APInt APInt::getAllOnesValue(unsigned numBits) {
  return getMaxValue(numBits, false);
}

/// getNullValue - This function creates an '0' value for an
/// APInt of the specified bit-width.
APInt APInt::getNullValue(unsigned numBits) {
  return getMinValue(numBits, true);
}

/// HiBits - This function returns the high "numBits" bits of this APInt.
APInt APInt::HiBits(unsigned numBits) const {
  return APIntOps::lshr(*this, BitsNum - numBits);
}

/// LoBits - This function returns the low "numBits" bits of this APInt.
APInt APInt::LoBits(unsigned numBits) const {
  return APIntOps::lshr(APIntOps::shl(*this, BitsNum - numBits), 
                        BitsNum - numBits);
}

/// CountLeadingZeros - This function is a APInt version corresponding to 
/// llvm/include/llvm/Support/MathExtras.h's function 
/// CountLeadingZeros_{32, 64}. It performs platform optimal form of counting 
/// the number of zeros from the most significant bit to the first one bit.
/// @returns numWord() * 64 if the value is zero.
unsigned APInt::CountLeadingZeros() const {
  if (isSingleWord())
    return CountLeadingZeros_64(VAL);
  unsigned Count = 0;
  for (int i = getNumWords() - 1; i >= 0; --i) {
    unsigned tmp = CountLeadingZeros_64(pVal[i]);
    Count += tmp;
    if (tmp != 64)
      break;
  }
  return Count;
}

/// CountTrailingZero - This function is a APInt version corresponding to
/// llvm/include/llvm/Support/MathExtras.h's function 
/// CountTrailingZeros_{32, 64}. It performs platform optimal form of counting 
/// the number of zeros from the least significant bit to the first one bit.
/// @returns numWord() * 64 if the value is zero.
unsigned APInt::CountTrailingZeros() const {
  if (isSingleWord())
    return CountTrailingZeros_64(~VAL & (VAL - 1));
  APInt Tmp = ~(*this) & ((*this) - 1);
  return getNumWords() * 64 - Tmp.CountLeadingZeros();
}

/// CountPopulation - This function is a APInt version corresponding to
/// llvm/include/llvm/Support/MathExtras.h's function
/// CountPopulation_{32, 64}. It counts the number of set bits in a value.
/// @returns 0 if the value is zero.
unsigned APInt::CountPopulation() const {
  if (isSingleWord())
    return CountPopulation_64(VAL);
  unsigned Count = 0;
  for (unsigned i = 0; i < getNumWords(); ++i)
    Count += CountPopulation_64(pVal[i]);
  return Count;
}


/// ByteSwap - This function returns a byte-swapped representation of the
/// APInt argument, APIVal.
APInt llvm::APIntOps::ByteSwap(const APInt& APIVal) {
  if (APIVal.BitsNum <= 32)
    return APInt(APIVal.BitsNum, ByteSwap_32(unsigned(APIVal.VAL)));
  else if (APIVal.BitsNum <= 64)
    return APInt(APIVal.BitsNum, ByteSwap_64(APIVal.VAL));
  else
    return APIVal;
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

/// Arithmetic right-shift the APInt by shiftAmt.
/// @brief Arithmetic right-shift function.
APInt llvm::APIntOps::ashr(const APInt& LHS, unsigned shiftAmt) {
  APInt API(LHS);
  if (API.isSingleWord())
    API.VAL = (((int64_t(API.VAL) << (64 - API.BitsNum)) >> (64 - API.BitsNum))
               >> shiftAmt) & (~uint64_t(0UL) >> (64 - API.BitsNum));
  else {
    if (shiftAmt >= API.BitsNum) {
      memset(API.pVal, API[API.BitsNum-1] ? 1 : 0, (API.getNumWords()-1) * 8);
      API.pVal[API.getNumWords() - 1] = ~uint64_t(0UL) >> 
                                        (64 - API.BitsNum % 64);
    } else {
      unsigned i = 0;
      for (; i < API.BitsNum - shiftAmt; ++i)
        if (API[i+shiftAmt]) 
          API.set(i);
        else
          API.clear(i);
      for (; i < API.BitsNum; ++i)
        API[API.BitsNum-1] ? API.set(i) : API.clear(i);
    }
  }
  return API;
}

/// Logical right-shift the APInt by shiftAmt.
/// @brief Logical right-shift function.
APInt llvm::APIntOps::lshr(const APInt& RHS, unsigned shiftAmt) {
  APInt API(RHS);
  if (API.isSingleWord())
    API.VAL >>= shiftAmt;
  else {
    if (shiftAmt >= API.BitsNum)
      memset(API.pVal, 0, API.getNumWords() * 8);
    unsigned i = 0;
    for (i = 0; i < API.BitsNum - shiftAmt; ++i)
      if (API[i+shiftAmt]) API.set(i);
      else API.clear(i);
    for (; i < API.BitsNum; ++i)
      API.clear(i);
  }
  return API;
}

/// Left-shift the APInt by shiftAmt.
/// @brief Left-shift function.
APInt llvm::APIntOps::shl(const APInt& RHS, unsigned shiftAmt) {
  APInt API(RHS);
  if (shiftAmt >= API.BitsNum) {
    if (API.isSingleWord()) 
      API.VAL = 0;
    else 
      memset(API.pVal, 0, API.getNumWords() * 8);
  } else {
    for (unsigned i = 0; i < shiftAmt; ++i) API.clear(i);
    for (unsigned i = shiftAmt; i < API.BitsNum; ++i) {
      if (API[i-shiftAmt]) API.set(i);
      else API.clear(i);
    }
  }
  return API;
}

/// Unsigned divide APInt LHS by APInt RHS.
/// @brief Unsigned division function for APInt.
APInt llvm::APIntOps::udiv(const APInt& LHS, const APInt& RHS) {
  APInt API(LHS);
  unsigned first = RHS.getNumWords() * APInt::APINT_BITS_PER_WORD - 
                   RHS.CountLeadingZeros();
  unsigned ylen = !first ? 0 : APInt::whichWord(first - 1) + 1;
  assert(ylen && "Divided by zero???");
  if (API.isSingleWord()) {
    API.VAL = RHS.isSingleWord() ? (API.VAL / RHS.VAL) : 
              (ylen > 1 ? 0 : API.VAL / RHS.pVal[0]);
  } else {
    unsigned first2 = API.getNumWords() * APInt::APINT_BITS_PER_WORD - 
                      API.CountLeadingZeros();
    unsigned xlen = !first2 ? 0 : APInt::whichWord(first2 - 1) + 1;
    if (!xlen)
      return API;
    else if (API < RHS)
      memset(API.pVal, 0, API.getNumWords() * 8);
    else if (API == RHS) {
      memset(API.pVal, 0, API.getNumWords() * 8);
      API.pVal[0] = 1;
    } else if (xlen == 1)
      API.pVal[0] /= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
    else {
      uint64_t *xwords = new uint64_t[xlen+1], *ywords = new uint64_t[ylen];
      assert(xwords && ywords && "Memory Allocation Failed!");
      memcpy(xwords, API.pVal, xlen * 8);
      xwords[xlen] = 0;
      memcpy(ywords, RHS.isSingleWord() ? &RHS.VAL : RHS.pVal, ylen * 8);
      if (unsigned nshift = 63 - (first - 1) % 64) {
        lshift(ywords, 0, ywords, ylen, nshift);
        unsigned xlentmp = xlen;
        xwords[xlen++] = lshift(xwords, 0, xwords, xlentmp, nshift);
      }
      div((unsigned*)xwords, xlen*2-1, (unsigned*)ywords, ylen*2);
      memset(API.pVal, 0, API.getNumWords() * 8);
      memcpy(API.pVal, xwords + ylen, (xlen - ylen) * 8);
      delete[] xwords;
      delete[] ywords;
    }
  }
  return API;
}

/// Unsigned remainder operation on APInt.
/// @brief Function for unsigned remainder operation.
APInt llvm::APIntOps::urem(const APInt& LHS, const APInt& RHS) {
  APInt API(LHS);
  unsigned first = RHS.getNumWords() * APInt::APINT_BITS_PER_WORD -
                   RHS.CountLeadingZeros();
  unsigned ylen = !first ? 0 : APInt::whichWord(first - 1) + 1;
  assert(ylen && "Performing remainder operation by zero ???");
  if (API.isSingleWord()) {
    API.VAL = RHS.isSingleWord() ? (API.VAL % RHS.VAL) : 
              (ylen > 1 ? API.VAL : API.VAL % RHS.pVal[0]);
  } else {
    unsigned first2 = API.getNumWords() * APInt::APINT_BITS_PER_WORD - 
                      API.CountLeadingZeros();
    unsigned xlen = !first2 ? 0 : API.whichWord(first2 - 1) + 1;
    if (!xlen || API < RHS)
      return API;
    else if (API == RHS) 
      memset(API.pVal, 0, API.getNumWords() * 8);
    else if (xlen == 1) 
      API.pVal[0] %= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
    else {
      uint64_t *xwords = new uint64_t[xlen+1], *ywords = new uint64_t[ylen];
      assert(xwords && ywords && "Memory Allocation Failed!");
      memcpy(xwords, API.pVal, xlen * 8);
      xwords[xlen] = 0;
      memcpy(ywords, RHS.isSingleWord() ? &RHS.VAL : RHS.pVal, ylen * 8);
      unsigned nshift = 63 - (first - 1) % 64;
      if (nshift) {
        lshift(ywords, 0, ywords, ylen, nshift);
        unsigned xlentmp = xlen;
        xwords[xlen++] = lshift(xwords, 0, xwords, xlentmp, nshift);
      }
      div((unsigned*)xwords, xlen*2-1, (unsigned*)ywords, ylen*2);
      memset(API.pVal, 0, API.getNumWords() * 8);
      for (unsigned i = 0; i < ylen-1; ++i)
        API.pVal[i] = (xwords[i] >> nshift) | (xwords[i+1] << (64 - nshift));
      API.pVal[ylen-1] = xwords[ylen-1] >> nshift;
      delete[] xwords;
      delete[] ywords;
    }
  }
  return API;
}

#endif

