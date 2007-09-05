//===-- APFloat.cpp - Implement APFloat class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Neil Booth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision floating
// point values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define convolve(lhs, rhs) ((lhs) * 4 + (rhs))

/* Assumed in hexadecimal significand parsing.  */
COMPILE_TIME_ASSERT(integerPartWidth % 4 == 0);

namespace llvm {

  /* Represents floating point arithmetic semantics.  */
  struct fltSemantics {
    /* The largest E such that 2^E is representable; this matches the
       definition of IEEE 754.  */
    exponent_t maxExponent;

    /* The smallest E such that 2^E is a normalized number; this
       matches the definition of IEEE 754.  */
    exponent_t minExponent;

    /* Number of bits in the significand.  This includes the integer
       bit.  */
    unsigned char precision;

    /* If the target format has an implicit integer bit.  */
    bool implicitIntegerBit;
  };

  const fltSemantics APFloat::IEEEsingle = { 127, -126, 24, true };
  const fltSemantics APFloat::IEEEdouble = { 1023, -1022, 53, true };
  const fltSemantics APFloat::IEEEquad = { 16383, -16382, 113, true };
  const fltSemantics APFloat::x87DoubleExtended = { 16383, -16382, 64, false };
  const fltSemantics APFloat::Bogus = { 0, 0, 0, false };
}

/* Put a bunch of private, handy routines in an anonymous namespace.  */
namespace {

  inline unsigned int
  partCountForBits(unsigned int bits)
  {
    return ((bits) + integerPartWidth - 1) / integerPartWidth;
  }

  unsigned int
  digitValue(unsigned int c)
  {
    unsigned int r;

    r = c - '0';
    if(r <= 9)
      return r;

    return -1U;
  }

  unsigned int
  hexDigitValue (unsigned int c)
  {
    unsigned int r;

    r = c - '0';
    if(r <= 9)
      return r;

    r = c - 'A';
    if(r <= 5)
      return r + 10;

    r = c - 'a';
    if(r <= 5)
      return r + 10;

    return -1U;
  }

  /* This is ugly and needs cleaning up, but I don't immediately see
     how whilst remaining safe.  */
  static int
  totalExponent(const char *p, int exponentAdjustment)
  {
    integerPart unsignedExponent;
    bool negative, overflow;
    long exponent;

    /* Move past the exponent letter and sign to the digits.  */
    p++;
    negative = *p == '-';
    if(*p == '-' || *p == '+')
      p++;

    unsignedExponent = 0;
    overflow = false;
    for(;;) {
      unsigned int value;

      value = digitValue(*p);
      if(value == -1U)
	break;

      p++;
      unsignedExponent = unsignedExponent * 10 + value;
      if(unsignedExponent > 65535)
	overflow = true;
    }

    if(exponentAdjustment > 65535 || exponentAdjustment < -65536)
      overflow = true;

    if(!overflow) {
      exponent = unsignedExponent;
      if(negative)
	exponent = -exponent;
      exponent += exponentAdjustment;
      if(exponent > 65535 || exponent < -65536)
	overflow = true;
    }

    if(overflow)
      exponent = negative ? -65536: 65535;

    return exponent;
  }

  const char *
  skipLeadingZeroesAndAnyDot(const char *p, const char **dot)
  {
    *dot = 0;
    while(*p == '0')
      p++;

    if(*p == '.') {
      *dot = p++;
      while(*p == '0')
	p++;
    }

    return p;
  }

  /* Return the trailing fraction of a hexadecimal number.
     DIGITVALUE is the first hex digit of the fraction, P points to
     the next digit.  */
  lostFraction
  trailingHexadecimalFraction(const char *p, unsigned int digitValue)
  {
    unsigned int hexDigit;

    /* If the first trailing digit isn't 0 or 8 we can work out the
       fraction immediately.  */
    if(digitValue > 8)
      return lfMoreThanHalf;
    else if(digitValue < 8 && digitValue > 0)
      return lfLessThanHalf;

    /* Otherwise we need to find the first non-zero digit.  */
    while(*p == '0')
      p++;

    hexDigit = hexDigitValue(*p);

    /* If we ran off the end it is exactly zero or one-half, otherwise
       a little more.  */
    if(hexDigit == -1U)
      return digitValue == 0 ? lfExactlyZero: lfExactlyHalf;
    else
      return digitValue == 0 ? lfLessThanHalf: lfMoreThanHalf;
  }

  /* Return the fraction lost were a bignum truncated.  */
  lostFraction
  lostFractionThroughTruncation(integerPart *parts,
				unsigned int partCount,
				unsigned int bits)
  {
    unsigned int lsb;

    lsb = APInt::tcLSB(parts, partCount);

    /* Note this is guaranteed true if bits == 0, or LSB == -1U.  */
    if(bits <= lsb)
      return lfExactlyZero;
    if(bits == lsb + 1)
      return lfExactlyHalf;
    if(bits <= partCount * integerPartWidth
       && APInt::tcExtractBit(parts, bits - 1))
      return lfMoreThanHalf;

    return lfLessThanHalf;
  }

  /* Shift DST right BITS bits noting lost fraction.  */
  lostFraction
  shiftRight(integerPart *dst, unsigned int parts, unsigned int bits)
  {
    lostFraction lost_fraction;

    lost_fraction = lostFractionThroughTruncation(dst, parts, bits);

    APInt::tcShiftRight(dst, parts, bits);

    return lost_fraction;
  }
}

/* Constructors.  */
void
APFloat::initialize(const fltSemantics *ourSemantics)
{
  unsigned int count;

  semantics = ourSemantics;
  count = partCount();
  if(count > 1)
    significand.parts = new integerPart[count];
}

void
APFloat::freeSignificand()
{
  if(partCount() > 1)
    delete [] significand.parts;
}

void
APFloat::assign(const APFloat &rhs)
{
  assert(semantics == rhs.semantics);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;
  if(category == fcNormal || category == fcNaN)
    copySignificand(rhs);
}

void
APFloat::copySignificand(const APFloat &rhs)
{
  assert(category == fcNormal || category == fcNaN);
  assert(rhs.partCount() >= partCount());

  APInt::tcAssign(significandParts(), rhs.significandParts(),
		  partCount());
}

APFloat &
APFloat::operator=(const APFloat &rhs)
{
  if(this != &rhs) {
    if(semantics != rhs.semantics) {
      freeSignificand();
      initialize(rhs.semantics);
    }
    assign(rhs);
  }

  return *this;
}

bool
APFloat::bitwiseIsEqual(const APFloat &rhs) const {
  if (this == &rhs)
    return true;
  if (semantics != rhs.semantics ||
      category != rhs.category ||
      sign != rhs.sign)
    return false;
  if (category==fcZero || category==fcInfinity)
    return true;
  else if (category==fcNormal && exponent!=rhs.exponent)
    return false;
  else {
    int i= partCount();
    const integerPart* p=significandParts();
    const integerPart* q=rhs.significandParts();
    for (; i>0; i--, p++, q++) {
      if (*p != *q)
        return false;
    }
    return true;
  }
}

APFloat::APFloat(const fltSemantics &ourSemantics, integerPart value)
{
  initialize(&ourSemantics);
  sign = 0;
  zeroSignificand();
  exponent = ourSemantics.precision - 1;
  significandParts()[0] = value;
  normalize(rmNearestTiesToEven, lfExactlyZero);
}

APFloat::APFloat(const fltSemantics &ourSemantics,
		 fltCategory ourCategory, bool negative)
{
  initialize(&ourSemantics);
  category = ourCategory;
  sign = negative;
  if(category == fcNormal)
    category = fcZero;
}

APFloat::APFloat(const fltSemantics &ourSemantics, const char *text)
{
  initialize(&ourSemantics);
  convertFromString(text, rmNearestTiesToEven);
}

APFloat::APFloat(const APFloat &rhs)
{
  initialize(rhs.semantics);
  assign(rhs);
}

APFloat::~APFloat()
{
  freeSignificand();
}

unsigned int
APFloat::partCount() const
{
  return partCountForBits(semantics->precision + 1);
}

unsigned int
APFloat::semanticsPrecision(const fltSemantics &semantics)
{
  return semantics.precision;
}

const integerPart *
APFloat::significandParts() const
{
  return const_cast<APFloat *>(this)->significandParts();
}

integerPart *
APFloat::significandParts()
{
  assert(category == fcNormal || category == fcNaN);

  if(partCount() > 1)
    return significand.parts;
  else
    return &significand.part;
}

/* Combine the effect of two lost fractions.  */
lostFraction
APFloat::combineLostFractions(lostFraction moreSignificant,
			      lostFraction lessSignificant)
{
  if(lessSignificant != lfExactlyZero) {
    if(moreSignificant == lfExactlyZero)
      moreSignificant = lfLessThanHalf;
    else if(moreSignificant == lfExactlyHalf)
      moreSignificant = lfMoreThanHalf;
  }

  return moreSignificant;
}

void
APFloat::zeroSignificand()
{
  category = fcNormal;
  APInt::tcSet(significandParts(), 0, partCount());
}

/* Increment an fcNormal floating point number's significand.  */
void
APFloat::incrementSignificand()
{
  integerPart carry;

  carry = APInt::tcIncrement(significandParts(), partCount());

  /* Our callers should never cause us to overflow.  */
  assert(carry == 0);
}

/* Add the significand of the RHS.  Returns the carry flag.  */
integerPart
APFloat::addSignificand(const APFloat &rhs)
{
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcAdd(parts, rhs.significandParts(), 0, partCount());
}

/* Subtract the significand of the RHS with a borrow flag.  Returns
   the borrow flag.  */
integerPart
APFloat::subtractSignificand(const APFloat &rhs, integerPart borrow)
{
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcSubtract(parts, rhs.significandParts(), borrow,
			   partCount());
}

/* Multiply the significand of the RHS.  If ADDEND is non-NULL, add it
   on to the full-precision result of the multiplication.  Returns the
   lost fraction.  */
lostFraction
APFloat::multiplySignificand(const APFloat &rhs, const APFloat *addend)
{
  unsigned int omsb;	// One, not zero, based MSB.
  unsigned int partsCount, newPartsCount, precision;
  integerPart *lhsSignificand;
  integerPart scratch[4];
  integerPart *fullSignificand;
  lostFraction lost_fraction;

  assert(semantics == rhs.semantics);

  precision = semantics->precision;
  newPartsCount = partCountForBits(precision * 2);

  if(newPartsCount > 4)
    fullSignificand = new integerPart[newPartsCount];
  else
    fullSignificand = scratch;

  lhsSignificand = significandParts();
  partsCount = partCount();

  APInt::tcFullMultiply(fullSignificand, lhsSignificand,
			rhs.significandParts(), partsCount);

  lost_fraction = lfExactlyZero;
  omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  exponent += rhs.exponent;

  if(addend) {
    Significand savedSignificand = significand;
    const fltSemantics *savedSemantics = semantics;
    fltSemantics extendedSemantics;
    opStatus status;
    unsigned int extendedPrecision;

    /* Normalize our MSB.  */
    extendedPrecision = precision + precision - 1;
    if(omsb != extendedPrecision)
      {
	APInt::tcShiftLeft(fullSignificand, newPartsCount,
			   extendedPrecision - omsb);
	exponent -= extendedPrecision - omsb;
      }

    /* Create new semantics.  */
    extendedSemantics = *semantics;
    extendedSemantics.precision = extendedPrecision;

    if(newPartsCount == 1)
      significand.part = fullSignificand[0];
    else
      significand.parts = fullSignificand;
    semantics = &extendedSemantics;

    APFloat extendedAddend(*addend);
    status = extendedAddend.convert(extendedSemantics, rmTowardZero);
    assert(status == opOK);
    lost_fraction = addOrSubtractSignificand(extendedAddend, false);

    /* Restore our state.  */
    if(newPartsCount == 1)
      fullSignificand[0] = significand.part;
    significand = savedSignificand;
    semantics = savedSemantics;

    omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  }

  exponent -= (precision - 1);

  if(omsb > precision) {
    unsigned int bits, significantParts;
    lostFraction lf;

    bits = omsb - precision;
    significantParts = partCountForBits(omsb);
    lf = shiftRight(fullSignificand, significantParts, bits);
    lost_fraction = combineLostFractions(lf, lost_fraction);
    exponent += bits;
  }

  APInt::tcAssign(lhsSignificand, fullSignificand, partsCount);

  if(newPartsCount > 4)
    delete [] fullSignificand;

  return lost_fraction;
}

/* Multiply the significands of LHS and RHS to DST.  */
lostFraction
APFloat::divideSignificand(const APFloat &rhs)
{
  unsigned int bit, i, partsCount;
  const integerPart *rhsSignificand;
  integerPart *lhsSignificand, *dividend, *divisor;
  integerPart scratch[4];
  lostFraction lost_fraction;

  assert(semantics == rhs.semantics);

  lhsSignificand = significandParts();
  rhsSignificand = rhs.significandParts();
  partsCount = partCount();

  if(partsCount > 2)
    dividend = new integerPart[partsCount * 2];
  else
    dividend = scratch;

  divisor = dividend + partsCount;

  /* Copy the dividend and divisor as they will be modified in-place.  */
  for(i = 0; i < partsCount; i++) {
    dividend[i] = lhsSignificand[i];
    divisor[i] = rhsSignificand[i];
    lhsSignificand[i] = 0;
  }

  exponent -= rhs.exponent;

  unsigned int precision = semantics->precision;

  /* Normalize the divisor.  */
  bit = precision - APInt::tcMSB(divisor, partsCount) - 1;
  if(bit) {
    exponent += bit;
    APInt::tcShiftLeft(divisor, partsCount, bit);
  }

  /* Normalize the dividend.  */
  bit = precision - APInt::tcMSB(dividend, partsCount) - 1;
  if(bit) {
    exponent -= bit;
    APInt::tcShiftLeft(dividend, partsCount, bit);
  }

  if(APInt::tcCompare(dividend, divisor, partsCount) < 0) {
    exponent--;
    APInt::tcShiftLeft(dividend, partsCount, 1);
    assert(APInt::tcCompare(dividend, divisor, partsCount) >= 0);
  }

  /* Long division.  */
  for(bit = precision; bit; bit -= 1) {
    if(APInt::tcCompare(dividend, divisor, partsCount) >= 0) {
      APInt::tcSubtract(dividend, divisor, 0, partsCount);
      APInt::tcSetBit(lhsSignificand, bit - 1);
    }

    APInt::tcShiftLeft(dividend, partsCount, 1);
  }

  /* Figure out the lost fraction.  */
  int cmp = APInt::tcCompare(dividend, divisor, partsCount);

  if(cmp > 0)
    lost_fraction = lfMoreThanHalf;
  else if(cmp == 0)
    lost_fraction = lfExactlyHalf;
  else if(APInt::tcIsZero(dividend, partsCount))
    lost_fraction = lfExactlyZero;
  else
    lost_fraction = lfLessThanHalf;

  if(partsCount > 2)
    delete [] dividend;

  return lost_fraction;
}

unsigned int
APFloat::significandMSB() const
{
  return APInt::tcMSB(significandParts(), partCount());
}

unsigned int
APFloat::significandLSB() const
{
  return APInt::tcLSB(significandParts(), partCount());
}

/* Note that a zero result is NOT normalized to fcZero.  */
lostFraction
APFloat::shiftSignificandRight(unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert((exponent_t) (exponent + bits) >= exponent);

  exponent += bits;

  return shiftRight(significandParts(), partCount(), bits);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
APFloat::shiftSignificandLeft(unsigned int bits)
{
  assert(bits < semantics->precision);

  if(bits) {
    unsigned int partsCount = partCount();

    APInt::tcShiftLeft(significandParts(), partsCount, bits);
    exponent -= bits;

    assert(!APInt::tcIsZero(significandParts(), partsCount));
  }
}

APFloat::cmpResult
APFloat::compareAbsoluteValue(const APFloat &rhs) const
{
  int compare;

  assert(semantics == rhs.semantics);
  assert(category == fcNormal);
  assert(rhs.category == fcNormal);

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if(compare == 0)
    compare = APInt::tcCompare(significandParts(), rhs.significandParts(),
			       partCount());

  if(compare > 0)
    return cmpGreaterThan;
  else if(compare < 0)
    return cmpLessThan;
  else
    return cmpEqual;
}

/* Handle overflow.  Sign is preserved.  We either become infinity or
   the largest finite number.  */
APFloat::opStatus
APFloat::handleOverflow(roundingMode rounding_mode)
{
  /* Infinity?  */
  if(rounding_mode == rmNearestTiesToEven
     || rounding_mode == rmNearestTiesToAway
     || (rounding_mode == rmTowardPositive && !sign)
     || (rounding_mode == rmTowardNegative && sign))
    {
      category = fcInfinity;
      return (opStatus) (opOverflow | opInexact);
    }

  /* Otherwise we become the largest finite number.  */
  category = fcNormal;
  exponent = semantics->maxExponent;
  APInt::tcSetLeastSignificantBits(significandParts(), partCount(),
				   semantics->precision);

  return opInexact;
}

/* This routine must work for fcZero of both signs, and fcNormal
   numbers.  */
bool
APFloat::roundAwayFromZero(roundingMode rounding_mode,
			   lostFraction lost_fraction)
{
  /* NaNs and infinities should not have lost fractions.  */
  assert(category == fcNormal || category == fcZero);

  /* Our caller has already handled this case.  */
  assert(lost_fraction != lfExactlyZero);

  switch(rounding_mode) {
  default:
    assert(0);

  case rmNearestTiesToAway:
    return lost_fraction == lfExactlyHalf || lost_fraction == lfMoreThanHalf;

  case rmNearestTiesToEven:
    if(lost_fraction == lfMoreThanHalf)
      return true;

    /* Our zeroes don't have a significand to test.  */
    if(lost_fraction == lfExactlyHalf && category != fcZero)
      return significandParts()[0] & 1;

    return false;

  case rmTowardZero:
    return false;

  case rmTowardPositive:
    return sign == false;

  case rmTowardNegative:
    return sign == true;
  }
}

APFloat::opStatus
APFloat::normalize(roundingMode rounding_mode,
		   lostFraction lost_fraction)
{
  unsigned int omsb;		/* One, not zero, based MSB.  */
  int exponentChange;

  if(category != fcNormal)
    return opOK;

  /* Before rounding normalize the exponent of fcNormal numbers.  */
  omsb = significandMSB() + 1;

  if(omsb) {
    /* OMSB is numbered from 1.  We want to place it in the integer
       bit numbered PRECISON if possible, with a compensating change in
       the exponent.  */
    exponentChange = omsb - semantics->precision;

    /* If the resulting exponent is too high, overflow according to
       the rounding mode.  */
    if(exponent + exponentChange > semantics->maxExponent)
      return handleOverflow(rounding_mode);

    /* Subnormal numbers have exponent minExponent, and their MSB
       is forced based on that.  */
    if(exponent + exponentChange < semantics->minExponent)
      exponentChange = semantics->minExponent - exponent;

    /* Shifting left is easy as we don't lose precision.  */
    if(exponentChange < 0) {
      assert(lost_fraction == lfExactlyZero);

      shiftSignificandLeft(-exponentChange);

      return opOK;
    }

    if(exponentChange > 0) {
      lostFraction lf;

      /* Shift right and capture any new lost fraction.  */
      lf = shiftSignificandRight(exponentChange);

      lost_fraction = combineLostFractions(lf, lost_fraction);

      /* Keep OMSB up-to-date.  */
      if(omsb > (unsigned) exponentChange)
	omsb -= (unsigned) exponentChange;
      else
	omsb = 0;
    }
  }

  /* Now round the number according to rounding_mode given the lost
     fraction.  */

  /* As specified in IEEE 754, since we do not trap we do not report
     underflow for exact results.  */
  if(lost_fraction == lfExactlyZero) {
    /* Canonicalize zeroes.  */
    if(omsb == 0)
      category = fcZero;

    return opOK;
  }

  /* Increment the significand if we're rounding away from zero.  */
  if(roundAwayFromZero(rounding_mode, lost_fraction)) {
    if(omsb == 0)
      exponent = semantics->minExponent;

    incrementSignificand();
    omsb = significandMSB() + 1;

    /* Did the significand increment overflow?  */
    if(omsb == (unsigned) semantics->precision + 1) {
      /* Renormalize by incrementing the exponent and shifting our
	 significand right one.  However if we already have the
	 maximum exponent we overflow to infinity.  */
      if(exponent == semantics->maxExponent) {
	category = fcInfinity;

	return (opStatus) (opOverflow | opInexact);
      }

      shiftSignificandRight(1);

      return opInexact;
    }
  }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if(omsb == semantics->precision)
    return opInexact;

  /* We have a non-zero denormal.  */
  assert(omsb < semantics->precision);
  assert(exponent == semantics->minExponent);

  /* Canonicalize zeroes.  */
  if(omsb == 0)
    category = fcZero;

  /* The fcZero case is a denormal that underflowed to zero.  */
  return (opStatus) (opUnderflow | opInexact);
}

APFloat::opStatus
APFloat::addOrSubtractSpecials(const APFloat &rhs, bool subtract)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcNormal, fcZero):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcZero):
    return opOK;

  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcZero, fcInfinity):
    category = fcInfinity;
    sign = rhs.sign ^ subtract;
    return opOK;

  case convolve(fcZero, fcNormal):
    assign(rhs);
    sign = rhs.sign ^ subtract;
    return opOK;

  case convolve(fcZero, fcZero):
    /* Sign depends on rounding mode; handled by caller.  */
    return opOK;

  case convolve(fcInfinity, fcInfinity):
    /* Differently signed infinities can only be validly
       subtracted.  */
    if(sign ^ rhs.sign != subtract) {
      category = fcNaN;
      // Arbitrary but deterministic value for significand
      APInt::tcSet(significandParts(), ~0U, partCount());
      return opInvalidOp;
    }

    return opOK;

  case convolve(fcNormal, fcNormal):
    return opDivByZero;
  }
}

/* Add or subtract two normal numbers.  */
lostFraction
APFloat::addOrSubtractSignificand(const APFloat &rhs, bool subtract)
{
  integerPart carry;
  lostFraction lost_fraction;
  int bits;

  /* Determine if the operation on the absolute values is effectively
     an addition or subtraction.  */
  subtract ^= (sign ^ rhs.sign);

  /* Are we bigger exponent-wise than the RHS?  */
  bits = exponent - rhs.exponent;

  /* Subtraction is more subtle than one might naively expect.  */
  if(subtract) {
    APFloat temp_rhs(rhs);
    bool reverse;

    if (bits == 0) {
      reverse = compareAbsoluteValue(temp_rhs) == cmpLessThan;
      lost_fraction = lfExactlyZero;
    } else if (bits > 0) {
      lost_fraction = temp_rhs.shiftSignificandRight(bits - 1);
      shiftSignificandLeft(1);
      reverse = false;
    } else {
      lost_fraction = shiftSignificandRight(-bits - 1);
      temp_rhs.shiftSignificandLeft(1);
      reverse = true;
    }

    if (reverse) {
      carry = temp_rhs.subtractSignificand
	(*this, lost_fraction != lfExactlyZero);
      copySignificand(temp_rhs);
      sign = !sign;
    } else {
      carry = subtractSignificand
	(temp_rhs, lost_fraction != lfExactlyZero);
    }

    /* Invert the lost fraction - it was on the RHS and
       subtracted.  */
    if(lost_fraction == lfLessThanHalf)
      lost_fraction = lfMoreThanHalf;
    else if(lost_fraction == lfMoreThanHalf)
      lost_fraction = lfLessThanHalf;

    /* The code above is intended to ensure that no borrow is
       necessary.  */
    assert(!carry);
  } else {
    if(bits > 0) {
      APFloat temp_rhs(rhs);

      lost_fraction = temp_rhs.shiftSignificandRight(bits);
      carry = addSignificand(temp_rhs);
    } else {
      lost_fraction = shiftSignificandRight(-bits);
      carry = addSignificand(rhs);
    }

    /* We have a guard bit; generating a carry cannot happen.  */
    assert(!carry);
  }

  return lost_fraction;
}

APFloat::opStatus
APFloat::multiplySpecials(const APFloat &rhs)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
    return opOK;

  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcInfinity):
    category = fcInfinity;
    return opOK;

  case convolve(fcZero, fcNormal):
  case convolve(fcNormal, fcZero):
  case convolve(fcZero, fcZero):
    category = fcZero;
    return opOK;

  case convolve(fcZero, fcInfinity):
  case convolve(fcInfinity, fcZero):
    category = fcNaN;
    // Arbitrary but deterministic value for significand
    APInt::tcSet(significandParts(), ~0U, partCount());
    return opInvalidOp;

  case convolve(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus
APFloat::divideSpecials(const APFloat &rhs)
{
  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcInfinity, fcZero):
  case convolve(fcInfinity, fcNormal):
  case convolve(fcZero, fcInfinity):
  case convolve(fcZero, fcNormal):
    return opOK;

  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case convolve(fcNormal, fcInfinity):
    category = fcZero;
    return opOK;

  case convolve(fcNormal, fcZero):
    category = fcInfinity;
    return opDivByZero;

  case convolve(fcInfinity, fcInfinity):
  case convolve(fcZero, fcZero):
    category = fcNaN;
    // Arbitrary but deterministic value for significand
    APInt::tcSet(significandParts(), ~0U, partCount());
    return opInvalidOp;

  case convolve(fcNormal, fcNormal):
    return opOK;
  }
}

/* Change sign.  */
void
APFloat::changeSign()
{
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

void
APFloat::clearSign()
{
  /* So is this one. */
  sign = 0;
}

void
APFloat::copySign(const APFloat &rhs)
{
  /* And this one. */
  sign = rhs.sign;
}

/* Normalized addition or subtraction.  */
APFloat::opStatus
APFloat::addOrSubtract(const APFloat &rhs, roundingMode rounding_mode,
		       bool subtract)
{
  opStatus fs;

  fs = addOrSubtractSpecials(rhs, subtract);

  /* This return code means it was not a simple case.  */
  if(fs == opDivByZero) {
    lostFraction lost_fraction;

    lost_fraction = addOrSubtractSignificand(rhs, subtract);
    fs = normalize(rounding_mode, lost_fraction);

    /* Can only be zero if we lost no fraction.  */
    assert(category != fcZero || lost_fraction == lfExactlyZero);
  }

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if(category == fcZero) {
    if(rhs.category != fcZero || (sign == rhs.sign) == subtract)
      sign = (rounding_mode == rmTowardNegative);
  }

  return fs;
}

/* Normalized addition.  */
APFloat::opStatus
APFloat::add(const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract(rhs, rounding_mode, false);
}

/* Normalized subtraction.  */
APFloat::opStatus
APFloat::subtract(const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract(rhs, rounding_mode, true);
}

/* Normalized multiply.  */
APFloat::opStatus
APFloat::multiply(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = multiplySpecials(rhs);

  if(category == fcNormal) {
    lostFraction lost_fraction = multiplySignificand(rhs, 0);
    fs = normalize(rounding_mode, lost_fraction);
    if(lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized divide.  */
APFloat::opStatus
APFloat::divide(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = divideSpecials(rhs);

  if(category == fcNormal) {
    lostFraction lost_fraction = divideSignificand(rhs);
    fs = normalize(rounding_mode, lost_fraction);
    if(lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized remainder. */
APFloat::opStatus
APFloat::mod(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;
  APFloat V = *this;
  unsigned int origSign = sign;
  fs = V.divide(rhs, rmNearestTiesToEven);
  if (fs == opDivByZero)
    return fs;

  int parts = partCount();
  integerPart *x = new integerPart[parts];
  fs = V.convertToInteger(x, parts * integerPartWidth, true, 
                          rmNearestTiesToEven);
  if (fs==opInvalidOp)
    return fs;

  fs = V.convertFromInteger(x, parts, true, rmNearestTiesToEven);
  assert(fs==opOK);   // should always work

  fs = V.multiply(rhs, rounding_mode);
  assert(fs==opOK || fs==opInexact);   // should not overflow or underflow

  fs = subtract(V, rounding_mode);
  assert(fs==opOK || fs==opInexact);   // likewise

  if (isZero())
    sign = origSign;    // IEEE754 requires this
  delete[] x;
  return fs;
}

/* Normalized fused-multiply-add.  */
APFloat::opStatus
APFloat::fusedMultiplyAdd(const APFloat &multiplicand,
			  const APFloat &addend,
			  roundingMode rounding_mode)
{
  opStatus fs;

  /* Post-multiplication sign, before addition.  */
  sign ^= multiplicand.sign;

  /* If and only if all arguments are normal do we need to do an
     extended-precision calculation.  */
  if(category == fcNormal
     && multiplicand.category == fcNormal
     && addend.category == fcNormal) {
    lostFraction lost_fraction;

    lost_fraction = multiplySignificand(multiplicand, &addend);
    fs = normalize(rounding_mode, lost_fraction);
    if(lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);

    /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
       positive zero unless rounding to minus infinity, except that
       adding two like-signed zeroes gives that zero.  */
    if(category == fcZero && sign != addend.sign)
      sign = (rounding_mode == rmTowardNegative);
  } else {
    fs = multiplySpecials(multiplicand);

    /* FS can only be opOK or opInvalidOp.  There is no more work
       to do in the latter case.  The IEEE-754R standard says it is
       implementation-defined in this case whether, if ADDEND is a
       quiet NaN, we raise invalid op; this implementation does so.

       If we need to do the addition we can do so with normal
       precision.  */
    if(fs == opOK)
      fs = addOrSubtract(addend, rounding_mode, false);
  }

  return fs;
}

/* Comparison requires normalized numbers.  */
APFloat::cmpResult
APFloat::compare(const APFloat &rhs) const
{
  cmpResult result;

  assert(semantics == rhs.semantics);

  switch(convolve(category, rhs.category)) {
  default:
    assert(0);

  case convolve(fcNaN, fcZero):
  case convolve(fcNaN, fcNormal):
  case convolve(fcNaN, fcInfinity):
  case convolve(fcNaN, fcNaN):
  case convolve(fcZero, fcNaN):
  case convolve(fcNormal, fcNaN):
  case convolve(fcInfinity, fcNaN):
    return cmpUnordered;

  case convolve(fcInfinity, fcNormal):
  case convolve(fcInfinity, fcZero):
  case convolve(fcNormal, fcZero):
    if(sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case convolve(fcNormal, fcInfinity):
  case convolve(fcZero, fcInfinity):
  case convolve(fcZero, fcNormal):
    if(rhs.sign)
      return cmpGreaterThan;
    else
      return cmpLessThan;

  case convolve(fcInfinity, fcInfinity):
    if(sign == rhs.sign)
      return cmpEqual;
    else if(sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case convolve(fcZero, fcZero):
    return cmpEqual;

  case convolve(fcNormal, fcNormal):
    break;
  }

  /* Two normal numbers.  Do they have the same sign?  */
  if(sign != rhs.sign) {
    if(sign)
      result = cmpLessThan;
    else
      result = cmpGreaterThan;
  } else {
    /* Compare absolute values; invert result if negative.  */
    result = compareAbsoluteValue(rhs);

    if(sign) {
      if(result == cmpLessThan)
	result = cmpGreaterThan;
      else if(result == cmpGreaterThan)
	result = cmpLessThan;
    }
  }

  return result;
}

APFloat::opStatus
APFloat::convert(const fltSemantics &toSemantics,
		 roundingMode rounding_mode)
{
  unsigned int newPartCount;
  opStatus fs;

  newPartCount = partCountForBits(toSemantics.precision + 1);

  /* If our new form is wider, re-allocate our bit pattern into wider
     storage.  */
  if(newPartCount > partCount()) {
    integerPart *newParts;

    newParts = new integerPart[newPartCount];
    APInt::tcSet(newParts, 0, newPartCount);
    APInt::tcAssign(newParts, significandParts(), partCount());
    freeSignificand();
    significand.parts = newParts;
  }

  if(category == fcNormal) {
    /* Re-interpret our bit-pattern.  */
    exponent += toSemantics.precision - semantics->precision;
    semantics = &toSemantics;
    fs = normalize(rounding_mode, lfExactlyZero);
  } else {
    semantics = &toSemantics;
    fs = opOK;
  }

  return fs;
}

/* Convert a floating point number to an integer according to the
   rounding mode.  If the rounded integer value is out of range this
   returns an invalid operation exception.  If the rounded value is in
   range but the floating point number is not the exact integer, the C
   standard doesn't require an inexact exception to be raised.  IEEE
   854 does require it so we do that.

   Note that for conversions to integer type the C standard requires
   round-to-zero to always be used.  */
APFloat::opStatus
APFloat::convertToInteger(integerPart *parts, unsigned int width,
			  bool isSigned,
			  roundingMode rounding_mode) const
{
  lostFraction lost_fraction;
  unsigned int msb, partsCount;
  int bits;

  /* Handle the three special cases first.  */
  if(category == fcInfinity || category == fcNaN)
    return opInvalidOp;

  partsCount = partCountForBits(width);

  if(category == fcZero) {
    APInt::tcSet(parts, 0, partsCount);
    return opOK;
  }

  /* Shift the bit pattern so the fraction is lost.  */
  APFloat tmp(*this);

  bits = (int) semantics->precision - 1 - exponent;

  if(bits > 0) {
    lost_fraction = tmp.shiftSignificandRight(bits);
  } else {
    tmp.shiftSignificandLeft(-bits);
    lost_fraction = lfExactlyZero;
  }

  if(lost_fraction != lfExactlyZero
     && tmp.roundAwayFromZero(rounding_mode, lost_fraction))
    tmp.incrementSignificand();

  msb = tmp.significandMSB();

  /* Negative numbers cannot be represented as unsigned.  */
  if(!isSigned && tmp.sign && msb != -1U)
    return opInvalidOp;

  /* It takes exponent + 1 bits to represent the truncated floating
     point number without its sign.  We lose a bit for the sign, but
     the maximally negative integer is a special case.  */
  if(msb + 1 > width)		/* !! Not same as msb >= width !! */
    return opInvalidOp;

  if(isSigned && msb + 1 == width
     && (!tmp.sign || tmp.significandLSB() != msb))
    return opInvalidOp;

  APInt::tcAssign(parts, tmp.significandParts(), partsCount);

  if(tmp.sign)
    APInt::tcNegate(parts, partsCount);

  if(lost_fraction == lfExactlyZero)
    return opOK;
  else
    return opInexact;
}

APFloat::opStatus
APFloat::convertFromUnsignedInteger(integerPart *parts,
				    unsigned int partCount,
				    roundingMode rounding_mode)
{
  unsigned int msb, precision;
  lostFraction lost_fraction;

  msb = APInt::tcMSB(parts, partCount) + 1;
  precision = semantics->precision;

  category = fcNormal;
  exponent = precision - 1;

  if(msb > precision) {
    exponent += (msb - precision);
    lost_fraction = shiftRight(parts, partCount, msb - precision);
    msb = precision;
  } else
    lost_fraction = lfExactlyZero;

  /* Copy the bit image.  */
  zeroSignificand();
  APInt::tcAssign(significandParts(), parts, partCountForBits(msb));

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::convertFromInteger(const integerPart *parts,
			    unsigned int partCount, bool isSigned,
			    roundingMode rounding_mode)
{
  unsigned int width;
  opStatus status;
  integerPart *copy;

  copy = new integerPart[partCount];
  APInt::tcAssign(copy, parts, partCount);

  width = partCount * integerPartWidth;

  sign = false;
  if(isSigned && APInt::tcExtractBit(parts, width - 1)) {
    sign = true;
    APInt::tcNegate(copy, partCount);
  }

  status = convertFromUnsignedInteger(copy, partCount, rounding_mode);
  delete [] copy;

  return status;
}

APFloat::opStatus
APFloat::convertFromHexadecimalString(const char *p,
				      roundingMode rounding_mode)
{
  lostFraction lost_fraction;
  integerPart *significand;
  unsigned int bitPos, partsCount;
  const char *dot, *firstSignificantDigit;

  zeroSignificand();
  exponent = 0;
  category = fcNormal;

  significand = significandParts();
  partsCount = partCount();
  bitPos = partsCount * integerPartWidth;

  /* Skip leading zeroes and any(hexa)decimal point.  */
  p = skipLeadingZeroesAndAnyDot(p, &dot);
  firstSignificantDigit = p;

  for(;;) {
    integerPart hex_value;

    if(*p == '.') {
      assert(dot == 0);
      dot = p++;
    }

    hex_value = hexDigitValue(*p);
    if(hex_value == -1U) {
      lost_fraction = lfExactlyZero;
      break;
    }

    p++;

    /* Store the number whilst 4-bit nibbles remain.  */
    if(bitPos) {
      bitPos -= 4;
      hex_value <<= bitPos % integerPartWidth;
      significand[bitPos / integerPartWidth] |= hex_value;
    } else {
      lost_fraction = trailingHexadecimalFraction(p, hex_value);
      while(hexDigitValue(*p) != -1U)
	p++;
      break;
    }
  }

  /* Hex floats require an exponent but not a hexadecimal point.  */
  assert(*p == 'p' || *p == 'P');

  /* Ignore the exponent if we are zero.  */
  if(p != firstSignificantDigit) {
    int expAdjustment;

    /* Implicit hexadecimal point?  */
    if(!dot)
      dot = p;

    /* Calculate the exponent adjustment implicit in the number of
       significant digits.  */
    expAdjustment = dot - firstSignificantDigit;
    if(expAdjustment < 0)
      expAdjustment++;
    expAdjustment = expAdjustment * 4 - 1;

    /* Adjust for writing the significand starting at the most
       significant nibble.  */
    expAdjustment += semantics->precision;
    expAdjustment -= partsCount * integerPartWidth;

    /* Adjust for the given exponent.  */
    exponent = totalExponent(p, expAdjustment);
  }

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::convertFromString(const char *p, roundingMode rounding_mode) {
  /* Handle a leading minus sign.  */
  if(*p == '-')
    sign = 1, p++;
  else
    sign = 0;

  if(p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
    return convertFromHexadecimalString(p + 2, rounding_mode);

  assert(0 && "Decimal to binary conversions not yet implemented");
  abort();
}

// For good performance it is desirable for different APFloats
// to produce different integers.
uint32_t
APFloat::getHashValue() const { 
  if (category==fcZero) return sign<<8 | semantics->precision ;
  else if (category==fcInfinity) return sign<<9 | semantics->precision;
  else if (category==fcNaN) return 1<<10 | semantics->precision;
  else {
    uint32_t hash = sign<<11 | semantics->precision | exponent<<12;
    const integerPart* p = significandParts();
    for (int i=partCount(); i>0; i--, p++)
      hash ^= ((uint32_t)*p) ^ (*p)>>32;
    return hash;
  }
}

// Conversion from APFloat to/from host float/double.  It may eventually be
// possible to eliminate these and have everybody deal with APFloats, but that
// will take a while.  This approach will not easily extend to long double.
// Current implementation requires partCount()==1, which is correct at the
// moment but could be made more general.

// Denormals have exponent minExponent in APFloat, but minExponent-1 in
// the actual IEEE respresentation.  We compensate for that here.

double
APFloat::convertToDouble() const {
  assert(semantics == (const llvm::fltSemantics* const)&IEEEdouble);
  assert (partCount()==1);

  uint64_t myexponent, mysignificand;

  if (category==fcNormal) {
    myexponent = exponent+1023; //bias
    mysignificand = *significandParts();
    if (myexponent==1 && !(mysignificand & 0x10000000000000LL))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0x7ff;
    mysignificand = 0;
  } else if (category==fcNaN) {
    myexponent = 0x7ff;
    mysignificand = *significandParts();
  } else
    assert(0);

  return BitsToDouble((((uint64_t)sign & 1) << 63) | 
        ((myexponent & 0x7ff) <<  52) | 
        (mysignificand & 0xfffffffffffffLL));
}

float
APFloat::convertToFloat() const {
  assert(semantics == (const llvm::fltSemantics* const)&IEEEsingle);
  assert (partCount()==1);

  uint32_t myexponent, mysignificand;

  if (category==fcNormal) {
    myexponent = exponent+127; //bias
    mysignificand = *significandParts();
    if (myexponent == 1 && !(mysignificand & 0x400000))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0xff;
    mysignificand = 0;
  } else if (category==fcNaN) {
    myexponent = 0xff;
    mysignificand = *significandParts();
  } else
    assert(0);

  return BitsToFloat(((sign&1) << 31) | ((myexponent&0xff) << 23) | 
        (mysignificand & 0x7fffff));
}

APFloat::APFloat(double d) {
  uint64_t i = DoubleToBits(d);
  uint64_t myexponent = (i >> 52) & 0x7ff;
  uint64_t mysignificand = i & 0xfffffffffffffLL;

  initialize(&APFloat::IEEEdouble);
  assert(partCount()==1);

  sign = i>>63;
  if (myexponent==0 && mysignificand==0) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0x7ff && mysignificand==0) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0x7ff && mysignificand!=0) {
    // exponent meaningless
    category = fcNaN;
    *significandParts() = mysignificand;
  } else {
    category = fcNormal;
    exponent = myexponent - 1023;
    *significandParts() = mysignificand;
    if (myexponent==0)          // denormal
      exponent = -1022;
    else
      *significandParts() |= 0x10000000000000LL;  // integer bit
 }
}

APFloat::APFloat(float f) {
  uint32_t i = FloatToBits(f);
  uint32_t myexponent = (i >> 23) & 0xff;
  uint32_t mysignificand = i & 0x7fffff;

  initialize(&APFloat::IEEEsingle);
  assert(partCount()==1);

  sign = i >> 31;
  if (myexponent==0 && mysignificand==0) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0xff && mysignificand==0) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0xff && (mysignificand & 0x400000)) {
    // sign, exponent, significand meaningless
    category = fcNaN;
    *significandParts() = mysignificand;
  } else {
    category = fcNormal;
    exponent = myexponent - 127;  //bias
    *significandParts() = mysignificand;
    if (myexponent==0)    // denormal
      exponent = -126;
    else
      *significandParts() |= 0x800000; // integer bit
  }
}
