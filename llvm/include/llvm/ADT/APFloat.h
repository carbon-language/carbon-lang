//== llvm/Support/APFloat.h - Arbitrary Precision Floating Point -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Neil Booth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a class to represent arbitrary precision floating
// point values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

/*  A self-contained host- and target-independent arbitrary-precision
    floating-point software implementation using bignum integer
    arithmetic, as provided by static functions in the APInt class.
    The library will work with bignum integers whose parts are any
    unsigned type at least 16 bits wide.  64 bits is recommended.

    Written for clarity rather than speed, in particular with a view
    to use in the front-end of a cross compiler so that target
    arithmetic can be correctly performed on the host.  Performance
    should nonetheless be reasonable, particularly for its intended
    use.  It may be useful as a base implementation for a run-time
    library during development of a faster target-specific one.

    All 5 rounding modes in the IEEE-754R draft are handled correctly
    for all implemented operations.  Currently implemented operations
    are add, subtract, multiply, divide, fused-multiply-add,
    conversion-to-float, conversion-to-integer and
    conversion-from-integer.  New rounding modes (e.g. away from zero)
    can be added with three or four lines of code.  The library reads
    and correctly rounds hexadecimal floating point numbers as per
    C99; syntax is required to have been validated by the caller.
    Conversion from decimal is not currently implemented.

    Four formats are built-in: IEEE single precision, double
    precision, quadruple precision, and x87 80-bit extended double
    (when operating with full extended precision).  Adding a new
    format that obeys IEEE semantics only requires adding two lines of
    code: a declaration and definition of the format.

    All operations return the status of that operation as an exception
    bit-mask, so multiple operations can be done consecutively with
    their results or-ed together.  The returned status can be useful
    for compiler diagnostics; e.g., inexact, underflow and overflow
    can be easily diagnosed on constant folding, and compiler
    optimizers can determine what exceptions would be raised by
    folding operations and optimize, or perhaps not optimize,
    accordingly.

    At present, underflow tininess is detected after rounding; it
    should be straight forward to add support for the before-rounding
    case too.

    Non-zero finite numbers are represented internally as a sign bit,
    a 16-bit signed exponent, and the significand as an array of
    integer parts.  After normalization of a number of precision P the
    exponent is within the range of the format, and if the number is
    not denormal the P-th bit of the significand is set as an explicit
    integer bit.  For denormals the most significant bit is shifted
    right so that the exponent is maintained at the format's minimum,
    so that the smallest denormal has just the least significant bit
    of the significand set.  The sign of zeroes and infinities is
    significant; the exponent and significand of such numbers is
    indeterminate and meaningless.  For QNaNs the sign bit, as well as
    the exponent and significand are indeterminate and meaningless.

    TODO
    ====

    Some features that may or may not be worth adding:

    Conversions to and from decimal strings (hard).

    Conversions to hexadecimal string.

    Read and write IEEE-format in-memory representations.

    Optional ability to detect underflow tininess before rounding.

    New formats: x87 in single and double precision mode (IEEE apart
    from extended exponent range) and IBM two-double extended
    precision (hard).

    New operations: sqrt, copysign, nextafter, nexttoward.
*/

#ifndef LLVM_FLOAT_H
#define LLVM_FLOAT_H

// APInt contains static functions implementing bignum arithmetic.
#include "llvm/ADT/APInt.h"

namespace llvm {

  /* Exponents are stored as signed numbers.  */
  typedef signed short exponent_t;

  struct fltSemantics;

  /* When bits of a floating point number are truncated, this enum is
     used to indicate what fraction of the LSB those bits represented.
     It essentially combines the roles of guard and sticky bits.  */
  enum lostFraction {		// Example of truncated bits:
    lfExactlyZero,		// 000000
    lfLessThanHalf,		// 0xxxxx  x's not all zero
    lfExactlyHalf,		// 100000
    lfMoreThanHalf		// 1xxxxx  x's not all zero
  };

  class APFloat {
  public:

    /* We support the following floating point semantics.  */
    static const fltSemantics IEEEsingle;
    static const fltSemantics IEEEdouble;
    static const fltSemantics IEEEquad;
    static const fltSemantics x87DoubleExtended;

    static unsigned int semanticsPrecision(const fltSemantics &);

    /* Floating point numbers have a four-state comparison relation.  */
    enum cmpResult {
      cmpLessThan,
      cmpEqual,
      cmpGreaterThan,
      cmpUnordered
    };

    /* IEEE-754R gives five rounding modes.  */
    enum roundingMode {
      rmNearestTiesToEven,
      rmTowardPositive,
      rmTowardNegative,
      rmTowardZero,
      rmNearestTiesToAway
    };

    /* Operation status.  opUnderflow or opOverflow are always returned
       or-ed with opInexact.  */
    enum opStatus {
      opOK          = 0x00,
      opInvalidOp   = 0x01,
      opDivByZero   = 0x02,
      opOverflow    = 0x04,
      opUnderflow   = 0x08,
      opInexact     = 0x10
    };

    /* Category of internally-represented number.  */
    enum fltCategory {
      fcInfinity,
      fcQNaN,
      fcNormal,
      fcZero
    };

    /* Constructors.  */
    APFloat(const fltSemantics &, const char *);
    APFloat(const fltSemantics &, integerPart);
    APFloat(const fltSemantics &, fltCategory, bool negative);
    APFloat(const APFloat &);
    ~APFloat();

    /* Arithmetic.  */
    opStatus add(const APFloat &, roundingMode);
    opStatus subtract(const APFloat &, roundingMode);
    opStatus multiply(const APFloat &, roundingMode);
    opStatus divide(const APFloat &, roundingMode);
    opStatus fusedMultiplyAdd(const APFloat &, const APFloat &, roundingMode);
    void changeSign();

    /* Conversions.  */
    opStatus convert(const fltSemantics &, roundingMode);
    opStatus convertToInteger(integerPart *, unsigned int, bool,
			      roundingMode) const;
    opStatus convertFromInteger(const integerPart *, unsigned int, bool,
				roundingMode);
    opStatus convertFromString(const char *, roundingMode);

    /* Comparison with another floating point number.  */
    cmpResult compare(const APFloat &) const;

    /* Simple queries.  */
    fltCategory getCategory() const { return category; }
    const fltSemantics &getSemantics() const { return *semantics; }
    bool isZero() const { return category == fcZero; }
    bool isNonZero() const { return category != fcZero; }
    bool isNegative() const { return sign; }

    APFloat& operator=(const APFloat &);

  private:

    /* Trivial queries.  */
    integerPart *significandParts();
    const integerPart *significandParts() const;
    unsigned int partCount() const;

    /* Significand operations.  */
    integerPart addSignificand(const APFloat &);
    integerPart subtractSignificand(const APFloat &, integerPart);
    lostFraction addOrSubtractSignificand(const APFloat &, bool subtract);
    lostFraction multiplySignificand(const APFloat &, const APFloat *);
    lostFraction divideSignificand(const APFloat &);
    void incrementSignificand();
    void initialize(const fltSemantics *);
    void shiftSignificandLeft(unsigned int);
    lostFraction shiftSignificandRight(unsigned int);
    unsigned int significandLSB() const;
    unsigned int significandMSB() const;
    void zeroSignificand();

    /* Arithmetic on special values.  */
    opStatus addOrSubtractSpecials(const APFloat &, bool subtract);
    opStatus divideSpecials(const APFloat &);
    opStatus multiplySpecials(const APFloat &);

    /* Miscellany.  */
    opStatus normalize(roundingMode, lostFraction);
    opStatus addOrSubtract(const APFloat &, roundingMode, bool subtract);
    cmpResult compareAbsoluteValue(const APFloat &) const;
    opStatus handleOverflow(roundingMode);
    bool roundAwayFromZero(roundingMode, lostFraction);
    opStatus convertFromUnsignedInteger(integerPart *, unsigned int,
					roundingMode);
    lostFraction combineLostFractions(lostFraction, lostFraction);
    opStatus convertFromHexadecimalString(const char *, roundingMode);

    void assign(const APFloat &);
    void copySignificand(const APFloat &);
    void freeSignificand();

    /* What kind of semantics does this value obey?  */
    const fltSemantics *semantics;

    /* Significand - the fraction with an explicit integer bit.  Must be
       at least one bit wider than the target precision.  */
    union Significand
    {
      integerPart part;
      integerPart *parts;
    } significand;

    /* The exponent - a signed number.  */
    exponent_t exponent;

    /* What kind of floating point number this is.  */
    fltCategory category: 2;

    /* The sign bit of this number.  */
    unsigned int sign: 1;
  };
} /* namespace llvm */

#endif /* LLVM_FLOAT_H */
