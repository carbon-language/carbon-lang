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
#include <strings.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
using namespace llvm;

APInt::APInt(uint64_t val, unsigned numBits, bool sign)
  : bitsnum(numBits), isSigned(sign) {
  assert(bitsnum >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(bitsnum <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  if (isSingleWord()) 
    VAL = val & (~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - bitsnum));
  else {
    // Memory allocation and check if successful.
    assert((pVal = new uint64_t[numWords()]) && 
            "APInt memory allocation fails!");
    bzero(pVal, numWords() * 8);
    pVal[0] = val;
  }
}

APInt::APInt(unsigned numBits, uint64_t bigVal[], bool sign)
  : bitsnum(numBits), isSigned(sign) {
  assert(bitsnum >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(bitsnum <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  assert(bigVal && "Null pointer detected!");
  if (isSingleWord())
    VAL = bigVal[0] & (~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - bitsnum));
  else {
    // Memory allocation and check if successful.
    assert((pVal = new uint64_t[numWords()]) && 
           "APInt memory allocation fails!");
    // Calculate the actual length of bigVal[].
    unsigned n = sizeof(*bigVal) / sizeof(bigVal[0]);
    unsigned maxN = std::max<unsigned>(n, numWords());
    unsigned minN = std::min<unsigned>(n, numWords());
    memcpy(pVal, bigVal, (minN - 1) * 8);
    pVal[minN-1] = bigVal[minN-1] & (~uint64_t(0ULL) >> (64 - bitsnum % 64));
    if (maxN == numWords())
      bzero(pVal+n, (numWords() - n) * 8);
  }
}

APInt::APInt(std::string& Val, uint8_t radix, bool sign)
  : isSigned(sign) {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  assert(!Val.empty() && "String empty?");
  unsigned slen = Val.size();
  unsigned size = 0;
  // If the radix is a power of 2, read the input
  // from most significant to least significant.
  if ((radix & (radix - 1)) == 0) {
    unsigned nextBitPos = 0, bits_per_digit = radix / 8 + 2;
    uint64_t resDigit = 0;
    bitsnum = slen * bits_per_digit;
    if (numWords() > 1)
      assert((pVal = new uint64_t[numWords()]) && 
             "APInt memory allocation fails!");
    for (int i = slen - 1; i >= 0; --i) {
      uint64_t digit = Val[i] - 48;             // '0' == 48.
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
    if (!isSingleWord() && size <= numWords()) 
      pVal[size] = resDigit;
  } else {   // General case.  The radix is not a power of 2.
    // For 10-radix, the max value of 64-bit integer is 18446744073709551615,
    // and its digits number is 14.
    const unsigned chars_per_word = 20;
    if (slen < chars_per_word || 
        (Val <= "18446744073709551615" && 
         slen == chars_per_word)) { // In case Val <= 2^64 - 1
      bitsnum = 64;
      VAL = strtoull(Val.c_str(), 0, 10);
    } else { // In case Val > 2^64 - 1
      bitsnum = (slen / chars_per_word + 1) * 64;
      assert((pVal = new uint64_t[numWords()]) && 
             "APInt memory allocation fails!");
      bzero(pVal, numWords() * 8);
      unsigned str_pos = 0;
      while (str_pos < slen) {
        unsigned chunk = slen - str_pos;
        if (chunk > chars_per_word - 1)
          chunk = chars_per_word - 1;
        uint64_t resDigit = Val[str_pos++] - 48;  // 48 == '0'.
        uint64_t big_base = radix;
        while (--chunk > 0) {
          resDigit = resDigit * radix + Val[str_pos++] - 48;
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
  : bitsnum(APIVal.bitsnum), isSigned(APIVal.isSigned) {
  if (isSingleWord()) VAL = APIVal.VAL;
  else {
    // Memory allocation and check if successful.
    assert((pVal = new uint64_t[numWords()]) && 
           "APInt memory allocation fails!");
    memcpy(pVal, APIVal.pVal, numWords() * 8);
  }
}

APInt::~APInt() {
  if (!isSingleWord() && pVal) delete[] pVal;
}

/// whichByte - This function returns the word position 
/// for the specified bit position.
inline unsigned APInt::whichByte(unsigned bitPosition)
{ return (bitPosition % APINT_BITS_PER_WORD) / 8; }

/// getWord - returns the corresponding word for the specified bit position.
inline uint64_t& APInt::getWord(unsigned bitPosition)
{ return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; }

/// getWord - returns the corresponding word for the specified bit position.
/// This is a constant version.
inline uint64_t APInt::getWord(unsigned bitPosition) const
{ return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; }

/// mul_1 - This function multiplies the integer array x[] by a integer y and 
/// returns the carry.
uint64_t APInt::mul_1(uint64_t dest[], uint64_t x[],
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
void APInt::mul(uint64_t dest[], uint64_t x[], unsigned xlen,
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
uint64_t APInt::add_1(uint64_t dest[], uint64_t x[],
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
uint64_t APInt::add(uint64_t dest[], uint64_t x[],
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
uint64_t APInt::sub_1(uint64_t x[], unsigned len, uint64_t y) {
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
uint64_t APInt::sub(uint64_t dest[], uint64_t x[],
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
uint64_t APInt::unitDiv(uint64_t N, unsigned D) {
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
unsigned APInt::subMul(unsigned dest[], unsigned offset, 
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
void APInt::div(unsigned zds[], unsigned nx, unsigned y[], unsigned ny) {
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
uint64_t APInt::lshift(uint64_t dest[], unsigned d_offset,
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

/// @brief Copy assignment operator. Create a new object from the given
/// APInt one by initialization.
APInt& APInt::operator=(const APInt& RHS) {
  if (isSingleWord()) VAL = RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    unsigned minN = std::min(numWords(), RHS.numWords());
    memcpy(pVal, RHS.isSingleWord() ? &RHS.VAL : RHS.pVal, minN * 8);
    if (numWords() != minN)
      bzero(pVal + minN, (numWords() - minN) * 8);
  }
  return *this;
}

/// @brief Assignment operator. Assigns a common case integer value to 
/// the APInt.
APInt& APInt::operator=(uint64_t RHS) {
  if (isSingleWord()) VAL = RHS;
  else {
    pVal[0] = RHS;
    bzero(pVal, (numWords() - 1) * 8);
  }
  return *this;
}

/// @brief Postfix increment operator. Increments the APInt by one.
const APInt APInt::operator++(int) {
  APInt API(*this);
  if (isSingleWord()) ++VAL;
  else
    add_1(pVal, pVal, numWords(), 1);
  API.TruncToBits();
  return API;
}

/// @brief Prefix increment operator. Increments the APInt by one.
APInt& APInt::operator++() {
  if (isSingleWord()) ++VAL;
  else
    add_1(pVal, pVal, numWords(), 1);
  TruncToBits();
  return *this;
}

/// @brief Postfix decrement operator. Decrements the APInt by one.
const APInt APInt::operator--(int) {
  APInt API(*this);
  if (isSingleWord()) --VAL;
  else
    sub_1(API.pVal, API.numWords(), 1);
  API.TruncToBits();
  return API;
}

/// @brief Prefix decrement operator. Decrements the APInt by one.
APInt& APInt::operator--() {
  if (isSingleWord()) --VAL;
  else
    sub_1(pVal, numWords(), 1);
  TruncToBits();
  return *this;
}

/// @brief Addition assignment operator. Adds this APInt by the given APInt&
/// RHS and assigns the result to this APInt.
APInt& APInt::operator+=(const APInt& RHS) {
  if (isSingleWord()) VAL += RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
  else {
    if (RHS.isSingleWord()) add_1(pVal, pVal, numWords(), RHS.VAL);
    else {
      if (numWords() <= RHS.numWords()) 
        add(pVal, pVal, RHS.pVal, numWords());
      else {
        uint64_t carry = add(pVal, pVal, RHS.pVal, RHS.numWords());
        add_1(pVal + RHS.numWords(), pVal + RHS.numWords(), 
              numWords() - RHS.numWords(), carry);
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
      sub_1(pVal, numWords(), RHS.VAL);
    else {
      if (RHS.numWords() < numWords()) { 
        uint64_t carry = sub(pVal, pVal, RHS.pVal, RHS.numWords());
        sub_1(pVal + RHS.numWords(), numWords() - RHS.numWords(), carry); 
      }
      else
        sub(pVal, pVal, RHS.pVal, numWords());
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
    unsigned first = numWords() * APINT_BITS_PER_WORD - CountLeadingZeros();
    unsigned xlen = !first ? 0 : whichWord(first - 1) + 1;
    if (!xlen) 
      return *this;
    else if (RHS.isSingleWord()) 
      mul_1(pVal, pVal, xlen, RHS.VAL);
    else {
      first = RHS.numWords() * APINT_BITS_PER_WORD - RHS.CountLeadingZeros();
      unsigned ylen = !first ? 0 : whichWord(first - 1) + 1;
      if (!ylen) {
        bzero(pVal, numWords() * 8);
        return *this;
      }
      uint64_t *dest = new uint64_t[xlen+ylen];
      assert(dest && "Memory Allocation Failed!");
      mul(dest, pVal, xlen, RHS.pVal, ylen);
      memcpy(pVal, dest, ((xlen + ylen >= numWords()) ? numWords() : xlen + ylen) * 8);
      delete[] dest;
    }
  }
  TruncToBits();
  return *this;
}

/// @brief Division assignment operator. Divides this APInt by the given APInt
/// &RHS and assigns the result to this APInt.
APInt& APInt::operator/=(const APInt& RHS) {
  unsigned first = RHS.numWords() * APINT_BITS_PER_WORD - 
                   RHS.CountLeadingZeros();
  unsigned ylen = !first ? 0 : whichWord(first - 1) + 1;
  assert(ylen && "Divided by zero???");
  if (isSingleWord()) {
    if (isSigned && RHS.isSigned)
      VAL = RHS.isSingleWord() ? (int64_t(VAL) / int64_t(RHS.VAL)) :
            (ylen > 1 ? 0 : int64_t(VAL) / int64_t(RHS.pVal[0]));
    else
      VAL = RHS.isSingleWord() ? (VAL / RHS.VAL) : 
          (ylen > 1 ? 0 : VAL / RHS.pVal[0]);
  } else {
    unsigned first2 = numWords() * APINT_BITS_PER_WORD - CountLeadingZeros();
    unsigned xlen = !first2 ? 0 : whichWord(first2 - 1) + 1;
    if (!xlen)
      return *this;
    else if ((*this) < RHS)
      bzero(pVal, numWords() * 8);
    else if ((*this) == RHS) {
      bzero(pVal, numWords() * 8);
      pVal[0] = 1;
    } else if (xlen == 1)
      pVal[0] /= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
    else {
      uint64_t *xwords = new uint64_t[xlen+1], *ywords = new uint64_t[ylen];
      assert(xwords && ywords && "Memory Allocation Failed!");
      memcpy(xwords, pVal, xlen * 8);
      xwords[xlen] = 0;
      memcpy(ywords, RHS.isSingleWord() ? &RHS.VAL : RHS.pVal, ylen * 8);
      if (unsigned nshift = 63 - (first - 1) % 64) {
        lshift(ywords, 0, ywords, ylen, nshift);
        unsigned xlentmp = xlen;
        xwords[xlen++] = lshift(xwords, 0, xwords, xlentmp, nshift);
      }
      div((unsigned*)xwords, xlen*2-1, (unsigned*)ywords, ylen*2);
      bzero(pVal, numWords() * 8);
      memcpy(pVal, xwords + ylen, (xlen - ylen) * 8);
      delete[] xwords;
      delete[] ywords;
    }
  }
  return *this;
}

/// @brief Remainder assignment operator. Yields the remainder from the 
/// division of this APInt by the given APInt& RHS and assigns the remainder 
/// to this APInt.
APInt& APInt::operator%=(const APInt& RHS) {
  unsigned first = RHS.numWords() * APINT_BITS_PER_WORD -
                   RHS.CountLeadingZeros();
  unsigned ylen = !first ? 0 : whichWord(first - 1) + 1;
  assert(ylen && "Performing remainder operation by zero ???");
  if (isSingleWord()) {
    if (isSigned && RHS.isSigned)
      VAL = RHS.isSingleWord() ? (int64_t(VAL) % int64_t(RHS.VAL)) :
            (ylen > 1 ? VAL : int64_t(VAL) % int64_t(RHS.pVal[0]));
    else
      VAL = RHS.isSingleWord() ? (VAL % RHS.VAL) : 
          (ylen > 1 ? VAL : VAL % RHS.pVal[0]);
  } else {
    unsigned first2 = numWords() * APINT_BITS_PER_WORD - CountLeadingZeros();
    unsigned xlen = !first2 ? 0 : whichWord(first2 - 1) + 1;
    if (!xlen || (*this) < RHS)
      return *this;
    else if ((*this) == RHS) 
      bzero(pVal, numWords() * 8);
    else if (xlen == 1) 
      pVal[0] %= RHS.isSingleWord() ? RHS.VAL : RHS.pVal[0];
    else {
      uint64_t *xwords = new uint64_t[xlen+1], *ywords = new uint64_t[ylen];
      assert(xwords && ywords && "Memory Allocation Failed!");
      memcpy(xwords, pVal, xlen * 8);
      xwords[xlen] = 0;
      memcpy(ywords, RHS.isSingleWord() ? &RHS.VAL : RHS.pVal, ylen * 8);
      unsigned nshift = 63 - (first - 1) % 64;
      if (nshift) {
        lshift(ywords, 0, ywords, ylen, nshift);
        unsigned xlentmp = xlen;
        xwords[xlen++] = lshift(xwords, 0, xwords, xlentmp, nshift);
      }
      div((unsigned*)xwords, xlen*2-1, (unsigned*)ywords, ylen*2);
      bzero(pVal, numWords() * 8);
      for (unsigned i = 0; i < ylen-1; ++i)
        pVal[i] = (xwords[i] >> nshift) | (xwords[i+1] << (64 - nshift));
      pVal[ylen-1] = xwords[ylen-1] >> nshift;
      delete[] xwords;
      delete[] ywords;
    }
  }
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
      bzero(pVal, (numWords() - 1) * 8);
      pVal[0] &= RHS.VAL;
    } else {
      unsigned minwords = numWords() < RHS.numWords() ? numWords() : RHS.numWords();
      for (unsigned i = 0; i < minwords; ++i)
        pVal[i] &= RHS.pVal[i];
      if (numWords() > minwords) bzero(pVal+minwords, (numWords() - minwords) * 8);
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
      unsigned minwords = numWords() < RHS.numWords() ? numWords() : RHS.numWords();
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
      for (unsigned i = 0; i < numWords(); ++i)
        pVal[i] ^= RHS.VAL;
    } else {
      unsigned minwords = numWords() < RHS.numWords() ? numWords() : RHS.numWords();
      for (unsigned i = 0; i < minwords; ++i)
        pVal[i] ^= RHS.pVal[i];
      if (numWords() > minwords)
        for (unsigned i = minwords; i < numWords(); ++i)
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
    unsigned minN = std::min(numWords(), RHS.numWords());
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
    unsigned minN = std::min(numWords(), RHS.numWords());
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
    for (unsigned i = 0; i < numWords(); ++i)
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

/// @brief Division operator. Divides this APInt by the given APInt& RHS.
APInt APInt::operator/(const APInt& RHS) const {
  APInt API(*this);
  return API /= RHS;
}

/// @brief Remainder operator. Yields the remainder from the division of this
/// APInt and the given APInt& RHS.
APInt APInt::operator%(const APInt& RHS) const {
  APInt API(*this);
  return API %= RHS;
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
  API.TruncToBits();
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
  unsigned n1 = numWords() * APINT_BITS_PER_WORD - CountLeadingZeros(), 
           n2 = RHS.numWords() * APINT_BITS_PER_WORD - RHS.CountLeadingZeros();
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

/// @brief Inequality operator. Compare this APInt with the given APInt& RHS
/// for the validity of the inequality relationship.
bool APInt::operator!=(const APInt& RHS) const {
  return !((*this) == RHS);
}

/// @brief Less-than operator. Compare this APInt with the given APInt& RHS
/// for the validity of the less-than relationship.
bool APInt::operator <(const APInt& RHS) const {
  if (isSigned && RHS.isSigned) {
    if ((*this)[bitsnum-1] > RHS[RHS.bitsnum-1])
      return false;
    else if ((*this)[bitsnum-1] < RHS[RHS.bitsnum-1])
      return true;
  }
  unsigned n1 = numWords() * 64 - CountLeadingZeros(), 
           n2 = RHS.numWords() * 64 - RHS.CountLeadingZeros();
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
    for (unsigned i = 0; i < numWords(); ++i)
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
  else bzero(pVal, numWords() * 8);
  return *this;
}

/// @brief Left-shift assignment operator. Left-shift the APInt by shiftAmt
/// and assigns the result to this APInt.
APInt& APInt::operator<<=(unsigned shiftAmt) {
  if (shiftAmt >= bitsnum) {
    if (isSingleWord()) VAL = 0;
    else bzero(pVal, numWords() * 8);
  } else {
    for (unsigned i = 0; i < shiftAmt; ++i) clear(i);
    for (unsigned i = shiftAmt; i < bitsnum; ++i) {
      if ((*this)[i-shiftAmt]) set(i);
      else clear(i);
    }
  }
  return *this;
}

/// @brief Left-shift operator. Left-shift the APInt by shiftAmt.
APInt APInt::operator<<(unsigned shiftAmt) const {
  APInt API(*this);
  API <<= shiftAmt;
  return API;
}

/// @brief Right-shift assignment operator. Right-shift the APInt by shiftAmt
/// and assigns the result to this APInt.
APInt& APInt::operator>>=(unsigned shiftAmt) {
  bool isAShr = isSigned && (*this)[bitsnum-1];
  if (isSingleWord())
    VAL = isAShr ? (int64_t(VAL) >> shiftAmt) : (VAL >> shiftAmt);
  else {
    unsigned i = 0;
    for (i = 0; i < bitsnum - shiftAmt; ++i)
      if ((*this)[i+shiftAmt]) set(i);
      else clear(i);
    for (; i < bitsnum; ++i)
      isAShr ? set(i) : clear(i);
  }
  return *this;
}

/// @brief Right-shift operator. Right-shift the APInt by shiftAmt.
APInt APInt::operator>>(unsigned shiftAmt) const {
  APInt API(*this);
  API >>= shiftAmt;
  return API;
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
  if (isSingleWord()) VAL = (~(VAL << (64 - bitsnum))) >> (64 - bitsnum);
  else {
    unsigned i = 0;
    for (; i < numWords() - 1; ++i)
      pVal[i] = ~pVal[i];
    unsigned offset = 64 - (bitsnum - 64 * (i - 1));
    pVal[i] = (~(pVal[i] << offset)) >> offset;
  }
  return *this;
}

/// Toggle a given bit to its opposite value whose position is given 
/// as "bitPosition".
/// @brief Toggles a given bit to its opposite value.
APInt& APInt::flip(unsigned bitPosition) {
  assert(bitPosition < bitsnum && "Out of the bit-width range!");
  if ((*this)[bitPosition]) clear(bitPosition);
  else set(bitPosition);
  return *this;
}

/// to_string - This function translates the APInt into a string.
std::string APInt::to_string(uint8_t radix) const {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  std::ostringstream buf;
  buf << std::setbase(radix);
  // If the radix is a power of 2, set the format of ostringstream,
  // and output the value into buf.
  if ((radix & (radix - 1)) == 0) {
    if (isSingleWord()) buf << VAL;
    else {
      buf << pVal[numWords()-1];
      buf << std::setw(64 / (radix / 8 + 2)) << std::setfill('0');
      for (int i = numWords() - 2; i >= 0; --i)
        buf << pVal[i];
    }
  }
  else {  // If the radix = 10, need to translate the value into a
          // string.
    if (isSingleWord()) buf << VAL;
    else {
      // FIXME: To be supported.
    }
  }
  return buf.str();
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
  return (*this) >> (bitsnum - numBits); 
}

/// LoBits - This function returns the low "numBits" bits of this APInt.
APInt APInt::LoBits(unsigned numBits) const {
  return ((*this) << (bitsnum - numBits)) >> (bitsnum - numBits);
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
  for (int i = numWords() - 1; i >= 0; --i) {
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
  return numWords() * 64 - Tmp.CountLeadingZeros();
}

/// CountPopulation - This function is a APInt version corresponding to
/// llvm/include/llvm/Support/MathExtras.h's function
/// CountPopulation_{32, 64}. It counts the number of set bits in a value.
/// @returns 0 if the value is zero.
unsigned APInt::CountPopulation() const {
  if (isSingleWord())
    return CountPopulation_64(VAL);
  unsigned Count = 0;
  for (unsigned i = 0; i < numWords(); ++i)
    Count += CountPopulation_64(pVal[i]);
  return Count;
}


/// ByteSwap - This function returns a byte-swapped representation of the
/// APInt argument, APIVal.
APInt llvm::ByteSwap(const APInt& APIVal) {
  if (APIVal.bitsnum <= 32)
    return APInt(APIVal.bitsnum, ByteSwap_32(unsigned(APIVal.VAL)));
  else if (APIVal.bitsnum <= 64)
    return APInt(APIVal.bitsnum, ByteSwap_64(APIVal.VAL));
  else
    return APIVal;
}

/// GreatestCommonDivisor - This function returns the greatest common
/// divisor of the two APInt values using Enclid's algorithm.
APInt llvm::GreatestCommonDivisor(const APInt& API1, const APInt& API2) {
  APInt A = API1, B = API2;
  while (!!B) {
    APInt T = B;
    B = A % B;
    A = T;
  }
  return A;
}
