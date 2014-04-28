//==- BlockFrequencyInfoImpl.h - Block Frequency Implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared implementation of BlockFrequency for IR and Machine Instructions.
// See the documentation below for BlockFrequencyInfoImpl for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>
#include <list>

#define DEBUG_TYPE "block-freq"

//===----------------------------------------------------------------------===//
//
// UnsignedFloat definition.
//
// TODO: Make this private to BlockFrequencyInfoImpl or delete.
//
//===----------------------------------------------------------------------===//
namespace llvm {

class UnsignedFloatBase {
public:
  static const int32_t MaxExponent = 16383;
  static const int32_t MinExponent = -16382;
  static const int DefaultPrecision = 10;

  static void dump(uint64_t D, int16_t E, int Width);
  static raw_ostream &print(raw_ostream &OS, uint64_t D, int16_t E, int Width,
                            unsigned Precision);
  static std::string toString(uint64_t D, int16_t E, int Width,
                              unsigned Precision);
  static int countLeadingZeros32(uint32_t N) { return countLeadingZeros(N); }
  static int countLeadingZeros64(uint64_t N) { return countLeadingZeros(N); }
  static uint64_t getHalf(uint64_t N) { return (N >> 1) + (N & 1); }

  static std::pair<uint64_t, bool> splitSigned(int64_t N) {
    if (N >= 0)
      return std::make_pair(N, false);
    uint64_t Unsigned = N == INT64_MIN ? UINT64_C(1) << 63 : uint64_t(-N);
    return std::make_pair(Unsigned, true);
  }
  static int64_t joinSigned(uint64_t U, bool IsNeg) {
    if (U > uint64_t(INT64_MAX))
      return IsNeg ? INT64_MIN : INT64_MAX;
    return IsNeg ? -int64_t(U) : int64_t(U);
  }

  static int32_t extractLg(const std::pair<int32_t, int> &Lg) {
    return Lg.first;
  }
  static int32_t extractLgFloor(const std::pair<int32_t, int> &Lg) {
    return Lg.first - (Lg.second > 0);
  }
  static int32_t extractLgCeiling(const std::pair<int32_t, int> &Lg) {
    return Lg.first + (Lg.second < 0);
  }

  static std::pair<uint64_t, int16_t> divide64(uint64_t L, uint64_t R);
  static std::pair<uint64_t, int16_t> multiply64(uint64_t L, uint64_t R);

  static int compare(uint64_t L, uint64_t R, int Shift) {
    assert(Shift >= 0);
    assert(Shift < 64);

    uint64_t L_adjusted = L >> Shift;
    if (L_adjusted < R)
      return -1;
    if (L_adjusted > R)
      return 1;

    return L > L_adjusted << Shift ? 1 : 0;
  }
};

/// \brief Simple representation of an unsigned floating point.
///
/// UnsignedFloat is a unsigned floating point number.  It uses simple
/// saturation arithmetic, and every operation is well-defined for every value.
///
/// The number is split into a signed exponent and unsigned digits.  The number
/// represented is \c getDigits()*2^getExponent().  In this way, the digits are
/// much like the mantissa in the x87 long double, but there is no canonical
/// form, so the same number can be represented by many bit representations
/// (it's always in "denormal" mode).
///
/// UnsignedFloat is templated on the underlying integer type for digits, which
/// is expected to be one of uint64_t, uint32_t, uint16_t or uint8_t.
///
/// Unlike builtin floating point types, UnsignedFloat is portable.
///
/// Unlike APFloat, UnsignedFloat does not model architecture floating point
/// behaviour (this should make it a little faster), and implements most
/// operators (this makes it usable).
///
/// UnsignedFloat is totally ordered.  However, there is no canonical form, so
/// there are multiple representations of most scalars.  E.g.:
///
///     UnsignedFloat(8u, 0) == UnsignedFloat(4u, 1)
///     UnsignedFloat(4u, 1) == UnsignedFloat(2u, 2)
///     UnsignedFloat(2u, 2) == UnsignedFloat(1u, 3)
///
/// UnsignedFloat implements most arithmetic operations.  Precision is kept
/// where possible.  Uses simple saturation arithmetic, so that operations
/// saturate to 0.0 or getLargest() rather than under or overflowing.  It has
/// some extra arithmetic for unit inversion.  0.0/0.0 is defined to be 0.0.
/// Any other division by 0.0 is defined to be getLargest().
///
/// As a convenience for modifying the exponent, left and right shifting are
/// both implemented, and both interpret negative shifts as positive shifts in
/// the opposite direction.
///
/// Exponents are limited to the range accepted by x87 long double.  This makes
/// it trivial to add functionality to convert to APFloat (this is already
/// relied on for the implementation of printing).
///
/// The current plan is to gut this and make the necessary parts of it (even
/// more) private to BlockFrequencyInfo.
template <class DigitsT> class UnsignedFloat : UnsignedFloatBase {
public:
  static_assert(!std::numeric_limits<DigitsT>::is_signed,
                "only unsigned floats supported");

  typedef DigitsT DigitsType;

private:
  typedef std::numeric_limits<DigitsType> DigitsLimits;

  static const int Width = sizeof(DigitsType) * 8;
  static_assert(Width <= 64, "invalid integer width for digits");

private:
  DigitsType Digits;
  int16_t Exponent;

public:
  UnsignedFloat() : Digits(0), Exponent(0) {}

  UnsignedFloat(DigitsType Digits, int16_t Exponent)
      : Digits(Digits), Exponent(Exponent) {}

private:
  UnsignedFloat(const std::pair<uint64_t, int16_t> &X)
      : Digits(X.first), Exponent(X.second) {}

public:
  static UnsignedFloat getZero() { return UnsignedFloat(0, 0); }
  static UnsignedFloat getOne() { return UnsignedFloat(1, 0); }
  static UnsignedFloat getLargest() {
    return UnsignedFloat(DigitsLimits::max(), MaxExponent);
  }
  static UnsignedFloat getFloat(uint64_t N) { return adjustToWidth(N, 0); }
  static UnsignedFloat getInverseFloat(uint64_t N) {
    return getFloat(N).invert();
  }
  static UnsignedFloat getFraction(DigitsType N, DigitsType D) {
    return getQuotient(N, D);
  }

  int16_t getExponent() const { return Exponent; }
  DigitsType getDigits() const { return Digits; }

  /// \brief Convert to the given integer type.
  ///
  /// Convert to \c IntT using simple saturating arithmetic, truncating if
  /// necessary.
  template <class IntT> IntT toInt() const;

  bool isZero() const { return !Digits; }
  bool isLargest() const { return *this == getLargest(); }
  bool isOne() const {
    if (Exponent > 0 || Exponent <= -Width)
      return false;
    return Digits == DigitsType(1) << -Exponent;
  }

  /// \brief The log base 2, rounded.
  ///
  /// Get the lg of the scalar.  lg 0 is defined to be INT32_MIN.
  int32_t lg() const { return extractLg(lgImpl()); }

  /// \brief The log base 2, rounded towards INT32_MIN.
  ///
  /// Get the lg floor.  lg 0 is defined to be INT32_MIN.
  int32_t lgFloor() const { return extractLgFloor(lgImpl()); }

  /// \brief The log base 2, rounded towards INT32_MAX.
  ///
  /// Get the lg ceiling.  lg 0 is defined to be INT32_MIN.
  int32_t lgCeiling() const { return extractLgCeiling(lgImpl()); }

  bool operator==(const UnsignedFloat &X) const { return compare(X) == 0; }
  bool operator<(const UnsignedFloat &X) const { return compare(X) < 0; }
  bool operator!=(const UnsignedFloat &X) const { return compare(X) != 0; }
  bool operator>(const UnsignedFloat &X) const { return compare(X) > 0; }
  bool operator<=(const UnsignedFloat &X) const { return compare(X) <= 0; }
  bool operator>=(const UnsignedFloat &X) const { return compare(X) >= 0; }

  bool operator!() const { return isZero(); }

  /// \brief Convert to a decimal representation in a string.
  ///
  /// Convert to a string.  Uses scientific notation for very large/small
  /// numbers.  Scientific notation is used roughly for numbers outside of the
  /// range 2^-64 through 2^64.
  ///
  /// \c Precision indicates the number of decimal digits of precision to use;
  /// 0 requests the maximum available.
  ///
  /// As a special case to make debugging easier, if the number is small enough
  /// to convert without scientific notation and has more than \c Precision
  /// digits before the decimal place, it's printed accurately to the first
  /// digit past zero.  E.g., assuming 10 digits of precision:
  ///
  ///     98765432198.7654... => 98765432198.8
  ///      8765432198.7654... =>  8765432198.8
  ///       765432198.7654... =>   765432198.8
  ///        65432198.7654... =>    65432198.77
  ///         5432198.7654... =>     5432198.765
  std::string toString(unsigned Precision = DefaultPrecision) {
    return UnsignedFloatBase::toString(Digits, Exponent, Width, Precision);
  }

  /// \brief Print a decimal representation.
  ///
  /// Print a string.  See toString for documentation.
  raw_ostream &print(raw_ostream &OS,
                     unsigned Precision = DefaultPrecision) const {
    return UnsignedFloatBase::print(OS, Digits, Exponent, Width, Precision);
  }
  void dump() const { return UnsignedFloatBase::dump(Digits, Exponent, Width); }

  UnsignedFloat &operator+=(const UnsignedFloat &X);
  UnsignedFloat &operator-=(const UnsignedFloat &X);
  UnsignedFloat &operator*=(const UnsignedFloat &X);
  UnsignedFloat &operator/=(const UnsignedFloat &X);
  UnsignedFloat &operator<<=(int16_t Shift) { shiftLeft(Shift); return *this; }
  UnsignedFloat &operator>>=(int16_t Shift) { shiftRight(Shift); return *this; }

private:
  void shiftLeft(int32_t Shift);
  void shiftRight(int32_t Shift);

  /// \brief Adjust two floats to have matching exponents.
  ///
  /// Adjust \c this and \c X to have matching exponents.  Returns the new \c X
  /// by value.  Does nothing if \a isZero() for either.
  ///
  /// The value that compares smaller will lose precision, and possibly become
  /// \a isZero().
  UnsignedFloat matchExponents(UnsignedFloat X);

  /// \brief Increase exponent to match another float.
  ///
  /// Increases \c this to have an exponent matching \c X.  May decrease the
  /// exponent of \c X in the process, and \c this may possibly become \a
  /// isZero().
  void increaseExponentToMatch(UnsignedFloat &X, int32_t ExponentDiff);

public:
  /// \brief Scale a large number accurately.
  ///
  /// Scale N (multiply it by this).  Uses full precision multiplication, even
  /// if Width is smaller than 64, so information is not lost.
  uint64_t scale(uint64_t N) const;
  uint64_t scaleByInverse(uint64_t N) const {
    // TODO: implement directly, rather than relying on inverse.  Inverse is
    // expensive.
    return inverse().scale(N);
  }
  int64_t scale(int64_t N) const {
    std::pair<uint64_t, bool> Unsigned = splitSigned(N);
    return joinSigned(scale(Unsigned.first), Unsigned.second);
  }
  int64_t scaleByInverse(int64_t N) const {
    std::pair<uint64_t, bool> Unsigned = splitSigned(N);
    return joinSigned(scaleByInverse(Unsigned.first), Unsigned.second);
  }

  int compare(const UnsignedFloat &X) const;
  int compareTo(uint64_t N) const {
    UnsignedFloat Float = getFloat(N);
    int Compare = compare(Float);
    if (Width == 64 || Compare != 0)
      return Compare;

    // Check for precision loss.  We know *this == RoundTrip.
    uint64_t RoundTrip = Float.template toInt<uint64_t>();
    return N == RoundTrip ? 0 : RoundTrip < N ? -1 : 1;
  }
  int compareTo(int64_t N) const { return N < 0 ? 1 : compareTo(uint64_t(N)); }

  UnsignedFloat &invert() { return *this = UnsignedFloat::getFloat(1) / *this; }
  UnsignedFloat inverse() const { return UnsignedFloat(*this).invert(); }

private:
  static UnsignedFloat getProduct(DigitsType L, DigitsType R);
  static UnsignedFloat getQuotient(DigitsType Dividend, DigitsType Divisor);

  std::pair<int32_t, int> lgImpl() const;
  static int countLeadingZerosWidth(DigitsType Digits) {
    if (Width == 64)
      return countLeadingZeros64(Digits);
    if (Width == 32)
      return countLeadingZeros32(Digits);
    return countLeadingZeros32(Digits) + Width - 32;
  }

  static UnsignedFloat adjustToWidth(uint64_t N, int32_t S) {
    assert(S >= MinExponent);
    assert(S <= MaxExponent);
    if (Width == 64 || N <= DigitsLimits::max())
      return UnsignedFloat(N, S);

    // Shift right.
    int Shift = 64 - Width - countLeadingZeros64(N);
    DigitsType Shifted = N >> Shift;

    // Round.
    assert(S + Shift <= MaxExponent);
    return getRounded(UnsignedFloat(Shifted, S + Shift),
                      N & UINT64_C(1) << (Shift - 1));
  }

  static UnsignedFloat getRounded(UnsignedFloat P, bool Round) {
    if (!Round)
      return P;
    if (P.Digits == DigitsLimits::max())
      // Careful of overflow in the exponent.
      return UnsignedFloat(1, P.Exponent) <<= Width;
    return UnsignedFloat(P.Digits + 1, P.Exponent);
  }
};

#define UNSIGNED_FLOAT_BOP(op, base)                                           \
  template <class DigitsT>                                                     \
  UnsignedFloat<DigitsT> operator op(const UnsignedFloat<DigitsT> &L,          \
                                     const UnsignedFloat<DigitsT> &R) {        \
    return UnsignedFloat<DigitsT>(L) base R;                                   \
  }
UNSIGNED_FLOAT_BOP(+, += )
UNSIGNED_FLOAT_BOP(-, -= )
UNSIGNED_FLOAT_BOP(*, *= )
UNSIGNED_FLOAT_BOP(/, /= )
UNSIGNED_FLOAT_BOP(<<, <<= )
UNSIGNED_FLOAT_BOP(>>, >>= )
#undef UNSIGNED_FLOAT_BOP

template <class DigitsT>
raw_ostream &operator<<(raw_ostream &OS, const UnsignedFloat<DigitsT> &X) {
  return X.print(OS, 10);
}

#define UNSIGNED_FLOAT_COMPARE_TO_TYPE(op, T1, T2)                             \
  template <class DigitsT>                                                     \
  bool operator op(const UnsignedFloat<DigitsT> &L, T1 R) {                    \
    return L.compareTo(T2(R)) op 0;                                            \
  }                                                                            \
  template <class DigitsT>                                                     \
  bool operator op(T1 L, const UnsignedFloat<DigitsT> &R) {                    \
    return 0 op R.compareTo(T2(L));                                            \
  }
#define UNSIGNED_FLOAT_COMPARE_TO(op)                                          \
  UNSIGNED_FLOAT_COMPARE_TO_TYPE(op, uint64_t, uint64_t)                       \
  UNSIGNED_FLOAT_COMPARE_TO_TYPE(op, uint32_t, uint64_t)                       \
  UNSIGNED_FLOAT_COMPARE_TO_TYPE(op, int64_t, int64_t)                         \
  UNSIGNED_FLOAT_COMPARE_TO_TYPE(op, int32_t, int64_t)
UNSIGNED_FLOAT_COMPARE_TO(< )
UNSIGNED_FLOAT_COMPARE_TO(> )
UNSIGNED_FLOAT_COMPARE_TO(== )
UNSIGNED_FLOAT_COMPARE_TO(!= )
UNSIGNED_FLOAT_COMPARE_TO(<= )
UNSIGNED_FLOAT_COMPARE_TO(>= )
#undef UNSIGNED_FLOAT_COMPARE_TO
#undef UNSIGNED_FLOAT_COMPARE_TO_TYPE

template <class DigitsT>
uint64_t UnsignedFloat<DigitsT>::scale(uint64_t N) const {
  if (Width == 64 || N <= DigitsLimits::max())
    return (getFloat(N) * *this).template toInt<uint64_t>();

  // Defer to the 64-bit version.
  return UnsignedFloat<uint64_t>(Digits, Exponent).scale(N);
}

template <class DigitsT>
UnsignedFloat<DigitsT> UnsignedFloat<DigitsT>::getProduct(DigitsType L,
                                                          DigitsType R) {
  // Check for zero.
  if (!L || !R)
    return getZero();

  // Check for numbers that we can compute with 64-bit math.
  if (Width <= 32 || (L <= UINT32_MAX && R <= UINT32_MAX))
    return adjustToWidth(uint64_t(L) * uint64_t(R), 0);

  // Do the full thing.
  return UnsignedFloat(multiply64(L, R));
}
template <class DigitsT>
UnsignedFloat<DigitsT> UnsignedFloat<DigitsT>::getQuotient(DigitsType Dividend,
                                                           DigitsType Divisor) {
  // Check for zero.
  if (!Dividend)
    return getZero();
  if (!Divisor)
    return getLargest();

  if (Width == 64)
    return UnsignedFloat(divide64(Dividend, Divisor));

  // We can compute this with 64-bit math.
  int Shift = countLeadingZeros64(Dividend);
  uint64_t Shifted = uint64_t(Dividend) << Shift;
  uint64_t Quotient = Shifted / Divisor;

  // If Quotient needs to be shifted, then adjustToWidth will round.
  if (Quotient > DigitsLimits::max())
    return adjustToWidth(Quotient, -Shift);

  // Round based on the value of the next bit.
  return getRounded(UnsignedFloat(Quotient, -Shift),
                    Shifted % Divisor >= getHalf(Divisor));
}

template <class DigitsT>
template <class IntT>
IntT UnsignedFloat<DigitsT>::toInt() const {
  typedef std::numeric_limits<IntT> Limits;
  if (*this < 1)
    return 0;
  if (*this >= Limits::max())
    return Limits::max();

  IntT N = Digits;
  if (Exponent > 0) {
    assert(size_t(Exponent) < sizeof(IntT) * 8);
    return N << Exponent;
  }
  if (Exponent < 0) {
    assert(size_t(-Exponent) < sizeof(IntT) * 8);
    return N >> -Exponent;
  }
  return N;
}

template <class DigitsT>
std::pair<int32_t, int> UnsignedFloat<DigitsT>::lgImpl() const {
  if (isZero())
    return std::make_pair(INT32_MIN, 0);

  // Get the floor of the lg of Digits.
  int32_t LocalFloor = Width - countLeadingZerosWidth(Digits) - 1;

  // Get the floor of the lg of this.
  int32_t Floor = Exponent + LocalFloor;
  if (Digits == UINT64_C(1) << LocalFloor)
    return std::make_pair(Floor, 0);

  // Round based on the next digit.
  assert(LocalFloor >= 1);
  bool Round = Digits & UINT64_C(1) << (LocalFloor - 1);
  return std::make_pair(Floor + Round, Round ? 1 : -1);
}

template <class DigitsT>
UnsignedFloat<DigitsT> UnsignedFloat<DigitsT>::matchExponents(UnsignedFloat X) {
  if (isZero() || X.isZero() || Exponent == X.Exponent)
    return X;

  int32_t Diff = int32_t(X.Exponent) - int32_t(Exponent);
  if (Diff > 0)
    increaseExponentToMatch(X, Diff);
  else
    X.increaseExponentToMatch(*this, -Diff);
  return X;
}
template <class DigitsT>
void UnsignedFloat<DigitsT>::increaseExponentToMatch(UnsignedFloat &X,
                                                     int32_t ExponentDiff) {
  assert(ExponentDiff > 0);
  if (ExponentDiff >= 2 * Width) {
    *this = getZero();
    return;
  }

  // Use up any leading zeros on X, and then shift this.
  int32_t ShiftX = std::min(countLeadingZerosWidth(X.Digits), ExponentDiff);
  assert(ShiftX < Width);

  int32_t ShiftThis = ExponentDiff - ShiftX;
  if (ShiftThis >= Width) {
    *this = getZero();
    return;
  }

  X.Digits <<= ShiftX;
  X.Exponent -= ShiftX;
  Digits >>= ShiftThis;
  Exponent += ShiftThis;
  return;
}

template <class DigitsT>
UnsignedFloat<DigitsT> &UnsignedFloat<DigitsT>::
operator+=(const UnsignedFloat &X) {
  if (isLargest() || X.isZero())
    return *this;
  if (isZero() || X.isLargest())
    return *this = X;

  // Normalize exponents.
  UnsignedFloat Scaled = matchExponents(X);

  // Check for zero again.
  if (isZero())
    return *this = Scaled;
  if (Scaled.isZero())
    return *this;

  // Compute sum.
  DigitsType Sum = Digits + Scaled.Digits;
  bool DidOverflow = Sum < Digits;
  Digits = Sum;
  if (!DidOverflow)
    return *this;

  if (Exponent == MaxExponent)
    return *this = getLargest();

  ++Exponent;
  Digits = UINT64_C(1) << (Width - 1) | Digits >> 1;

  return *this;
}
template <class DigitsT>
UnsignedFloat<DigitsT> &UnsignedFloat<DigitsT>::
operator-=(const UnsignedFloat &X) {
  if (X.isZero())
    return *this;
  if (*this <= X)
    return *this = getZero();

  // Normalize exponents.
  UnsignedFloat Scaled = matchExponents(X);
  assert(Digits >= Scaled.Digits);

  // Compute difference.
  if (!Scaled.isZero()) {
    Digits -= Scaled.Digits;
    return *this;
  }

  // Check if X just barely lost its last bit.  E.g., for 32-bit:
  //
  //   1*2^32 - 1*2^0 == 0xffffffff != 1*2^32
  if (*this == UnsignedFloat(1, X.lgFloor() + Width)) {
    Digits = DigitsType(0) - 1;
    --Exponent;
  }
  return *this;
}
template <class DigitsT>
UnsignedFloat<DigitsT> &UnsignedFloat<DigitsT>::
operator*=(const UnsignedFloat &X) {
  if (isZero())
    return *this;
  if (X.isZero())
    return *this = X;

  // Save the exponents.
  int32_t Exponents = int32_t(Exponent) + int32_t(X.Exponent);

  // Get the raw product.
  *this = getProduct(Digits, X.Digits);

  // Combine with exponents.
  return *this <<= Exponents;
}
template <class DigitsT>
UnsignedFloat<DigitsT> &UnsignedFloat<DigitsT>::
operator/=(const UnsignedFloat &X) {
  if (isZero())
    return *this;
  if (X.isZero())
    return *this = getLargest();

  // Save the exponents.
  int32_t Exponents = int32_t(Exponent) - int32_t(X.Exponent);

  // Get the raw quotient.
  *this = getQuotient(Digits, X.Digits);

  // Combine with exponents.
  return *this <<= Exponents;
}
template <class DigitsT>
void UnsignedFloat<DigitsT>::shiftLeft(int32_t Shift) {
  if (!Shift || isZero())
    return;
  assert(Shift != INT32_MIN);
  if (Shift < 0) {
    shiftRight(-Shift);
    return;
  }

  // Shift as much as we can in the exponent.
  int32_t ExponentShift = std::min(Shift, MaxExponent - Exponent);
  Exponent += ExponentShift;
  if (ExponentShift == Shift)
    return;

  // Check this late, since it's rare.
  if (isLargest())
    return;

  // Shift the digits themselves.
  Shift -= ExponentShift;
  if (Shift > countLeadingZerosWidth(Digits)) {
    // Saturate.
    *this = getLargest();
    return;
  }

  Digits <<= Shift;
  return;
}

template <class DigitsT>
void UnsignedFloat<DigitsT>::shiftRight(int32_t Shift) {
  if (!Shift || isZero())
    return;
  assert(Shift != INT32_MIN);
  if (Shift < 0) {
    shiftLeft(-Shift);
    return;
  }

  // Shift as much as we can in the exponent.
  int32_t ExponentShift = std::min(Shift, Exponent - MinExponent);
  Exponent -= ExponentShift;
  if (ExponentShift == Shift)
    return;

  // Shift the digits themselves.
  Shift -= ExponentShift;
  if (Shift >= Width) {
    // Saturate.
    *this = getZero();
    return;
  }

  Digits >>= Shift;
  return;
}

template <class DigitsT>
int UnsignedFloat<DigitsT>::compare(const UnsignedFloat &X) const {
  // Check for zero.
  if (isZero())
    return X.isZero() ? 0 : -1;
  if (X.isZero())
    return 1;

  // Check for the scale.  Use lgFloor to be sure that the exponent difference
  // is always lower than 64.
  int32_t lgL = lgFloor(), lgR = X.lgFloor();
  if (lgL != lgR)
    return lgL < lgR ? -1 : 1;

  // Compare digits.
  if (Exponent < X.Exponent)
    return UnsignedFloatBase::compare(Digits, X.Digits, X.Exponent - Exponent);

  return -UnsignedFloatBase::compare(X.Digits, Digits, Exponent - X.Exponent);
}

template <class T> struct isPodLike<UnsignedFloat<T>> {
  static const bool value = true;
};
}

//===----------------------------------------------------------------------===//
//
// BlockMass definition.
//
// TODO: Make this private to BlockFrequencyInfoImpl or delete.
//
//===----------------------------------------------------------------------===//
namespace llvm {

/// \brief Mass of a block.
///
/// This class implements a sort of fixed-point fraction always between 0.0 and
/// 1.0.  getMass() == UINT64_MAX indicates a value of 1.0.
///
/// Masses can be added and subtracted.  Simple saturation arithmetic is used,
/// so arithmetic operations never overflow or underflow.
///
/// Masses can be multiplied.  Multiplication treats full mass as 1.0 and uses
/// an inexpensive floating-point algorithm that's off-by-one (almost, but not
/// quite, maximum precision).
///
/// Masses can be scaled by \a BranchProbability at maximum precision.
class BlockMass {
  uint64_t Mass;

public:
  BlockMass() : Mass(0) {}
  explicit BlockMass(uint64_t Mass) : Mass(Mass) {}

  static BlockMass getEmpty() { return BlockMass(); }
  static BlockMass getFull() { return BlockMass(UINT64_MAX); }

  uint64_t getMass() const { return Mass; }

  bool isFull() const { return Mass == UINT64_MAX; }
  bool isEmpty() const { return !Mass; }

  bool operator!() const { return isEmpty(); }

  /// \brief Add another mass.
  ///
  /// Adds another mass, saturating at \a isFull() rather than overflowing.
  BlockMass &operator+=(const BlockMass &X) {
    uint64_t Sum = Mass + X.Mass;
    Mass = Sum < Mass ? UINT64_MAX : Sum;
    return *this;
  }

  /// \brief Subtract another mass.
  ///
  /// Subtracts another mass, saturating at \a isEmpty() rather than
  /// undeflowing.
  BlockMass &operator-=(const BlockMass &X) {
    uint64_t Diff = Mass - X.Mass;
    Mass = Diff > Mass ? 0 : Diff;
    return *this;
  }

  /// \brief Scale by another mass.
  ///
  /// The current implementation is a little imprecise, but it's relatively
  /// fast, never overflows, and maintains the property that 1.0*1.0==1.0
  /// (where isFull represents the number 1.0).  It's an approximation of
  /// 128-bit multiply that gets right-shifted by 64-bits.
  ///
  /// For a given digit size, multiplying two-digit numbers looks like:
  ///
  ///                  U1 .    L1
  ///                * U2 .    L2
  ///                ============
  ///           0 .       . L1*L2
  ///     +     0 . U1*L2 .     0 // (shift left once by a digit-size)
  ///     +     0 . U2*L1 .     0 // (shift left once by a digit-size)
  ///     + U1*L2 .     0 .     0 // (shift left twice by a digit-size)
  ///
  /// BlockMass has 64-bit numbers.  Split each into two 32-bit digits, stored
  /// 64-bit.  Add 1 to the lower digits, to model isFull as 1.0; this won't
  /// overflow, since we have 64-bit storage for each digit.
  ///
  /// To do this accurately, (a) multiply into two 64-bit digits, incrementing
  /// the upper digit on overflows of the lower digit (carry), (b) subtract 1
  /// from the lower digit, decrementing the upper digit on underflow (carry),
  /// and (c) truncate the lower digit.  For the 1.0*1.0 case, the upper digit
  /// will be 0 at the end of step (a), and then will underflow back to isFull
  /// (1.0) in step (b).
  ///
  /// Instead, the implementation does something a little faster with a small
  /// loss of accuracy: ignore the lower 64-bit digit entirely.  The loss of
  /// accuracy is small, since the sum of the unmodelled carries is 0 or 1
  /// (i.e., step (a) will overflow at most once, and step (b) will underflow
  /// only if step (a) overflows).
  ///
  /// This is the formula we're calculating:
  ///
  ///     U1.L1 * U2.L2 == U1 * U2 + (U1 * (L2+1))>>32 + (U2 * (L1+1))>>32
  ///
  /// As a demonstration of 1.0*1.0, consider two 4-bit numbers that are both
  /// full (1111).
  ///
  ///     U1.L1 * U2.L2 == U1 * U2 + (U1 * (L2+1))>>2 + (U2 * (L1+1))>>2
  ///     11.11 * 11.11 == 11 * 11 + (11 * (11+1))/4 + (11 * (11+1))/4
  ///                   == 1001 + (11 * 100)/4 + (11 * 100)/4
  ///                   == 1001 + 1100/4 + 1100/4
  ///                   == 1001 + 0011 + 0011
  ///                   == 1111
  BlockMass &operator*=(const BlockMass &X) {
    uint64_t U1 = Mass >> 32, L1 = Mass & UINT32_MAX, U2 = X.Mass >> 32,
             L2 = X.Mass & UINT32_MAX;
    Mass = U1 * U2 + (U1 * (L2 + 1) >> 32) + ((L1 + 1) * U2 >> 32);
    return *this;
  }

  /// \brief Multiply by a branch probability.
  ///
  /// Multiply by P.  Guarantees full precision.
  ///
  /// This could be naively implemented by multiplying by the numerator and
  /// dividing by the denominator, but in what order?  Multiplying first can
  /// overflow, while dividing first will lose precision (potentially, changing
  /// a non-zero mass to zero).
  ///
  /// The implementation mixes the two methods.  Since \a BranchProbability
  /// uses 32-bits and \a BlockMass 64-bits, shift the mass as far to the left
  /// as there is room, then divide by the denominator to get a quotient.
  /// Multiplying by the numerator and right shifting gives a first
  /// approximation.
  ///
  /// Calculate the error in this first approximation by calculating the
  /// opposite mass (multiply by the opposite numerator and shift) and
  /// subtracting both from teh original mass.
  ///
  /// Add to the first approximation the correct fraction of this error value.
  /// This time, multiply first and then divide, since there is no danger of
  /// overflow.
  ///
  /// \pre P represents a fraction between 0.0 and 1.0.
  BlockMass &operator*=(const BranchProbability &P);

  bool operator==(const BlockMass &X) const { return Mass == X.Mass; }
  bool operator!=(const BlockMass &X) const { return Mass != X.Mass; }
  bool operator<=(const BlockMass &X) const { return Mass <= X.Mass; }
  bool operator>=(const BlockMass &X) const { return Mass >= X.Mass; }
  bool operator<(const BlockMass &X) const { return Mass < X.Mass; }
  bool operator>(const BlockMass &X) const { return Mass > X.Mass; }

  /// \brief Convert to floating point.
  ///
  /// Convert to a float.  \a isFull() gives 1.0, while \a isEmpty() gives
  /// slightly above 0.0.
  UnsignedFloat<uint64_t> toFloat() const;

  void dump() const;
  raw_ostream &print(raw_ostream &OS) const;
};

inline BlockMass operator+(const BlockMass &L, const BlockMass &R) {
  return BlockMass(L) += R;
}
inline BlockMass operator-(const BlockMass &L, const BlockMass &R) {
  return BlockMass(L) -= R;
}
inline BlockMass operator*(const BlockMass &L, const BlockMass &R) {
  return BlockMass(L) *= R;
}
inline BlockMass operator*(const BlockMass &L, const BranchProbability &R) {
  return BlockMass(L) *= R;
}
inline BlockMass operator*(const BranchProbability &L, const BlockMass &R) {
  return BlockMass(R) *= L;
}

inline raw_ostream &operator<<(raw_ostream &OS, const BlockMass &X) {
  return X.print(OS);
}

template <> struct isPodLike<BlockMass> {
  static const bool value = true;
};
}

//===----------------------------------------------------------------------===//
//
// BlockFrequencyInfoImpl definition.
//
//===----------------------------------------------------------------------===//
namespace llvm {

class BasicBlock;
class BranchProbabilityInfo;
class Function;
class Loop;
class LoopInfo;
class MachineBasicBlock;
class MachineBranchProbabilityInfo;
class MachineFunction;
class MachineLoop;
class MachineLoopInfo;

namespace bfi_detail {
struct IrreducibleGraph;

// This is part of a workaround for a GCC 4.7 crash on lambdas.
template <class BT> struct BlockEdgesAdder;
}

/// \brief Base class for BlockFrequencyInfoImpl
///
/// BlockFrequencyInfoImplBase has supporting data structures and some
/// algorithms for BlockFrequencyInfoImplBase.  Only algorithms that depend on
/// the block type (or that call such algorithms) are skipped here.
///
/// Nevertheless, the majority of the overall algorithm documention lives with
/// BlockFrequencyInfoImpl.  See there for details.
class BlockFrequencyInfoImplBase {
public:
  typedef UnsignedFloat<uint64_t> Float;

  /// \brief Representative of a block.
  ///
  /// This is a simple wrapper around an index into the reverse-post-order
  /// traversal of the blocks.
  ///
  /// Unlike a block pointer, its order has meaning (location in the
  /// topological sort) and it's class is the same regardless of block type.
  struct BlockNode {
    typedef uint32_t IndexType;
    IndexType Index;

    bool operator==(const BlockNode &X) const { return Index == X.Index; }
    bool operator!=(const BlockNode &X) const { return Index != X.Index; }
    bool operator<=(const BlockNode &X) const { return Index <= X.Index; }
    bool operator>=(const BlockNode &X) const { return Index >= X.Index; }
    bool operator<(const BlockNode &X) const { return Index < X.Index; }
    bool operator>(const BlockNode &X) const { return Index > X.Index; }

    BlockNode() : Index(UINT32_MAX) {}
    BlockNode(IndexType Index) : Index(Index) {}

    bool isValid() const { return Index <= getMaxIndex(); }
    static size_t getMaxIndex() { return UINT32_MAX - 1; }
  };

  /// \brief Stats about a block itself.
  struct FrequencyData {
    Float Floating;
    uint64_t Integer;
  };

  /// \brief Data about a loop.
  ///
  /// Contains the data necessary to represent represent a loop as a
  /// pseudo-node once it's packaged.
  struct LoopData {
    typedef SmallVector<std::pair<BlockNode, BlockMass>, 4> ExitMap;
    typedef SmallVector<BlockNode, 4> NodeList;
    LoopData *Parent;       ///< The parent loop.
    bool IsPackaged;        ///< Whether this has been packaged.
    uint32_t NumHeaders;    ///< Number of headers.
    ExitMap Exits;          ///< Successor edges (and weights).
    NodeList Nodes;         ///< Header and the members of the loop.
    BlockMass BackedgeMass; ///< Mass returned to loop header.
    BlockMass Mass;
    Float Scale;

    LoopData(LoopData *Parent, const BlockNode &Header)
        : Parent(Parent), IsPackaged(false), NumHeaders(1), Nodes(1, Header) {}
    template <class It1, class It2>
    LoopData(LoopData *Parent, It1 FirstHeader, It1 LastHeader, It2 FirstOther,
             It2 LastOther)
        : Parent(Parent), IsPackaged(false), Nodes(FirstHeader, LastHeader) {
      NumHeaders = Nodes.size();
      Nodes.insert(Nodes.end(), FirstOther, LastOther);
    }
    bool isHeader(const BlockNode &Node) const {
      if (isIrreducible())
        return std::binary_search(Nodes.begin(), Nodes.begin() + NumHeaders,
                                  Node);
      return Node == Nodes[0];
    }
    BlockNode getHeader() const { return Nodes[0]; }
    bool isIrreducible() const { return NumHeaders > 1; }

    NodeList::const_iterator members_begin() const {
      return Nodes.begin() + NumHeaders;
    }
    NodeList::const_iterator members_end() const { return Nodes.end(); }
    iterator_range<NodeList::const_iterator> members() const {
      return make_range(members_begin(), members_end());
    }
  };

  /// \brief Index of loop information.
  struct WorkingData {
    BlockNode Node; ///< This node.
    LoopData *Loop; ///< The loop this block is inside.
    BlockMass Mass; ///< Mass distribution from the entry block.

    WorkingData(const BlockNode &Node) : Node(Node), Loop(nullptr) {}

    bool isLoopHeader() const { return Loop && Loop->isHeader(Node); }
    bool isDoubleLoopHeader() const {
      return isLoopHeader() && Loop->Parent && Loop->Parent->isIrreducible() &&
             Loop->Parent->isHeader(Node);
    }

    LoopData *getContainingLoop() const {
      if (!isLoopHeader())
        return Loop;
      if (!isDoubleLoopHeader())
        return Loop->Parent;
      return Loop->Parent->Parent;
    }

    /// \brief Resolve a node to its representative.
    ///
    /// Get the node currently representing Node, which could be a containing
    /// loop.
    ///
    /// This function should only be called when distributing mass.  As long as
    /// there are no irreducilbe edges to Node, then it will have complexity
    /// O(1) in this context.
    ///
    /// In general, the complexity is O(L), where L is the number of loop
    /// headers Node has been packaged into.  Since this method is called in
    /// the context of distributing mass, L will be the number of loop headers
    /// an early exit edge jumps out of.
    BlockNode getResolvedNode() const {
      auto L = getPackagedLoop();
      return L ? L->getHeader() : Node;
    }
    LoopData *getPackagedLoop() const {
      if (!Loop || !Loop->IsPackaged)
        return nullptr;
      auto L = Loop;
      while (L->Parent && L->Parent->IsPackaged)
        L = L->Parent;
      return L;
    }

    /// \brief Get the appropriate mass for a node.
    ///
    /// Get appropriate mass for Node.  If Node is a loop-header (whose loop
    /// has been packaged), returns the mass of its pseudo-node.  If it's a
    /// node inside a packaged loop, it returns the loop's mass.
    BlockMass &getMass() {
      if (!isAPackage())
        return Mass;
      if (!isADoublePackage())
        return Loop->Mass;
      return Loop->Parent->Mass;
    }

    /// \brief Has ContainingLoop been packaged up?
    bool isPackaged() const { return getResolvedNode() != Node; }
    /// \brief Has Loop been packaged up?
    bool isAPackage() const { return isLoopHeader() && Loop->IsPackaged; }
    /// \brief Has Loop been packaged up twice?
    bool isADoublePackage() const {
      return isDoubleLoopHeader() && Loop->Parent->IsPackaged;
    }
  };

  /// \brief Unscaled probability weight.
  ///
  /// Probability weight for an edge in the graph (including the
  /// successor/target node).
  ///
  /// All edges in the original function are 32-bit.  However, exit edges from
  /// loop packages are taken from 64-bit exit masses, so we need 64-bits of
  /// space in general.
  ///
  /// In addition to the raw weight amount, Weight stores the type of the edge
  /// in the current context (i.e., the context of the loop being processed).
  /// Is this a local edge within the loop, an exit from the loop, or a
  /// backedge to the loop header?
  struct Weight {
    enum DistType { Local, Exit, Backedge };
    DistType Type;
    BlockNode TargetNode;
    uint64_t Amount;
    Weight() : Type(Local), Amount(0) {}
  };

  /// \brief Distribution of unscaled probability weight.
  ///
  /// Distribution of unscaled probability weight to a set of successors.
  ///
  /// This class collates the successor edge weights for later processing.
  ///
  /// \a DidOverflow indicates whether \a Total did overflow while adding to
  /// the distribution.  It should never overflow twice.
  struct Distribution {
    typedef SmallVector<Weight, 4> WeightList;
    WeightList Weights;    ///< Individual successor weights.
    uint64_t Total;        ///< Sum of all weights.
    bool DidOverflow;      ///< Whether \a Total did overflow.

    Distribution() : Total(0), DidOverflow(false) {}
    void addLocal(const BlockNode &Node, uint64_t Amount) {
      add(Node, Amount, Weight::Local);
    }
    void addExit(const BlockNode &Node, uint64_t Amount) {
      add(Node, Amount, Weight::Exit);
    }
    void addBackedge(const BlockNode &Node, uint64_t Amount) {
      add(Node, Amount, Weight::Backedge);
    }

    /// \brief Normalize the distribution.
    ///
    /// Combines multiple edges to the same \a Weight::TargetNode and scales
    /// down so that \a Total fits into 32-bits.
    ///
    /// This is linear in the size of \a Weights.  For the vast majority of
    /// cases, adjacent edge weights are combined by sorting WeightList and
    /// combining adjacent weights.  However, for very large edge lists an
    /// auxiliary hash table is used.
    void normalize();

  private:
    void add(const BlockNode &Node, uint64_t Amount, Weight::DistType Type);
  };

  /// \brief Data about each block.  This is used downstream.
  std::vector<FrequencyData> Freqs;

  /// \brief Loop data: see initializeLoops().
  std::vector<WorkingData> Working;

  /// \brief Indexed information about loops.
  std::list<LoopData> Loops;

  /// \brief Add all edges out of a packaged loop to the distribution.
  ///
  /// Adds all edges from LocalLoopHead to Dist.  Calls addToDist() to add each
  /// successor edge.
  ///
  /// \return \c true unless there's an irreducible backedge.
  bool addLoopSuccessorsToDist(const LoopData *OuterLoop, LoopData &Loop,
                               Distribution &Dist);

  /// \brief Add an edge to the distribution.
  ///
  /// Adds an edge to Succ to Dist.  If \c LoopHead.isValid(), then whether the
  /// edge is local/exit/backedge is in the context of LoopHead.  Otherwise,
  /// every edge should be a local edge (since all the loops are packaged up).
  ///
  /// \return \c true unless aborted due to an irreducible backedge.
  bool addToDist(Distribution &Dist, const LoopData *OuterLoop,
                 const BlockNode &Pred, const BlockNode &Succ, uint64_t Weight);

  LoopData &getLoopPackage(const BlockNode &Head) {
    assert(Head.Index < Working.size());
    assert(Working[Head.Index].isLoopHeader());
    return *Working[Head.Index].Loop;
  }

  /// \brief Analyze irreducible SCCs.
  ///
  /// Separate irreducible SCCs from \c G, which is an explict graph of \c
  /// OuterLoop (or the top-level function, if \c OuterLoop is \c nullptr).
  /// Insert them into \a Loops before \c Insert.
  ///
  /// \return the \c LoopData nodes representing the irreducible SCCs.
  iterator_range<std::list<LoopData>::iterator>
  analyzeIrreducible(const bfi_detail::IrreducibleGraph &G, LoopData *OuterLoop,
                     std::list<LoopData>::iterator Insert);

  /// \brief Update a loop after packaging irreducible SCCs inside of it.
  ///
  /// Update \c OuterLoop.  Before finding irreducible control flow, it was
  /// partway through \a computeMassInLoop(), so \a LoopData::Exits and \a
  /// LoopData::BackedgeMass need to be reset.  Also, nodes that were packaged
  /// up need to be removed from \a OuterLoop::Nodes.
  void updateLoopWithIrreducible(LoopData &OuterLoop);

  /// \brief Distribute mass according to a distribution.
  ///
  /// Distributes the mass in Source according to Dist.  If LoopHead.isValid(),
  /// backedges and exits are stored in its entry in Loops.
  ///
  /// Mass is distributed in parallel from two copies of the source mass.
  void distributeMass(const BlockNode &Source, LoopData *OuterLoop,
                      Distribution &Dist);

  /// \brief Compute the loop scale for a loop.
  void computeLoopScale(LoopData &Loop);

  /// \brief Package up a loop.
  void packageLoop(LoopData &Loop);

  /// \brief Unwrap loops.
  void unwrapLoops();

  /// \brief Finalize frequency metrics.
  ///
  /// Calculates final frequencies and cleans up no-longer-needed data
  /// structures.
  void finalizeMetrics();

  /// \brief Clear all memory.
  void clear();

  virtual std::string getBlockName(const BlockNode &Node) const;
  std::string getLoopName(const LoopData &Loop) const;

  virtual raw_ostream &print(raw_ostream &OS) const { return OS; }
  void dump() const { print(dbgs()); }

  Float getFloatingBlockFreq(const BlockNode &Node) const;

  BlockFrequency getBlockFreq(const BlockNode &Node) const;

  raw_ostream &printBlockFreq(raw_ostream &OS, const BlockNode &Node) const;
  raw_ostream &printBlockFreq(raw_ostream &OS,
                              const BlockFrequency &Freq) const;

  uint64_t getEntryFreq() const {
    assert(!Freqs.empty());
    return Freqs[0].Integer;
  }
  /// \brief Virtual destructor.
  ///
  /// Need a virtual destructor to mask the compiler warning about
  /// getBlockName().
  virtual ~BlockFrequencyInfoImplBase() {}
};

namespace bfi_detail {
template <class BlockT> struct TypeMap {};
template <> struct TypeMap<BasicBlock> {
  typedef BasicBlock BlockT;
  typedef Function FunctionT;
  typedef BranchProbabilityInfo BranchProbabilityInfoT;
  typedef Loop LoopT;
  typedef LoopInfo LoopInfoT;
};
template <> struct TypeMap<MachineBasicBlock> {
  typedef MachineBasicBlock BlockT;
  typedef MachineFunction FunctionT;
  typedef MachineBranchProbabilityInfo BranchProbabilityInfoT;
  typedef MachineLoop LoopT;
  typedef MachineLoopInfo LoopInfoT;
};

/// \brief Get the name of a MachineBasicBlock.
///
/// Get the name of a MachineBasicBlock.  It's templated so that including from
/// CodeGen is unnecessary (that would be a layering issue).
///
/// This is used mainly for debug output.  The name is similar to
/// MachineBasicBlock::getFullName(), but skips the name of the function.
template <class BlockT> std::string getBlockName(const BlockT *BB) {
  assert(BB && "Unexpected nullptr");
  auto MachineName = "BB" + Twine(BB->getNumber());
  if (BB->getBasicBlock())
    return (MachineName + "[" + BB->getName() + "]").str();
  return MachineName.str();
}
/// \brief Get the name of a BasicBlock.
template <> inline std::string getBlockName(const BasicBlock *BB) {
  assert(BB && "Unexpected nullptr");
  return BB->getName().str();
}

/// \brief Graph of irreducible control flow.
///
/// This graph is used for determining the SCCs in a loop (or top-level
/// function) that has irreducible control flow.
///
/// During the block frequency algorithm, the local graphs are defined in a
/// light-weight way, deferring to the \a BasicBlock or \a MachineBasicBlock
/// graphs for most edges, but getting others from \a LoopData::ExitMap.  The
/// latter only has successor information.
///
/// \a IrreducibleGraph makes this graph explicit.  It's in a form that can use
/// \a GraphTraits (so that \a analyzeIrreducible() can use \a scc_iterator),
/// and it explicitly lists predecessors and successors.  The initialization
/// that relies on \c MachineBasicBlock is defined in the header.
struct IrreducibleGraph {
  typedef BlockFrequencyInfoImplBase BFIBase;

  BFIBase &BFI;

  typedef BFIBase::BlockNode BlockNode;
  struct IrrNode {
    BlockNode Node;
    unsigned NumIn;
    std::deque<const IrrNode *> Edges;
    IrrNode(const BlockNode &Node) : Node(Node), NumIn(0) {}

    typedef std::deque<const IrrNode *>::const_iterator iterator;
    iterator pred_begin() const { return Edges.begin(); }
    iterator succ_begin() const { return Edges.begin() + NumIn; }
    iterator pred_end() const { return succ_begin(); }
    iterator succ_end() const { return Edges.end(); }
  };
  BlockNode Start;
  const IrrNode *StartIrr;
  std::vector<IrrNode> Nodes;
  SmallDenseMap<uint32_t, IrrNode *, 4> Lookup;

  /// \brief Construct an explicit graph containing irreducible control flow.
  ///
  /// Construct an explicit graph of the control flow in \c OuterLoop (or the
  /// top-level function, if \c OuterLoop is \c nullptr).  Uses \c
  /// addBlockEdges to add block successors that have not been packaged into
  /// loops.
  ///
  /// \a BlockFrequencyInfoImpl::computeIrreducibleMass() is the only expected
  /// user of this.
  template <class BlockEdgesAdder>
  IrreducibleGraph(BFIBase &BFI, const BFIBase::LoopData *OuterLoop,
                   BlockEdgesAdder addBlockEdges)
      : BFI(BFI), StartIrr(nullptr) {
    initialize(OuterLoop, addBlockEdges);
  }

  template <class BlockEdgesAdder>
  void initialize(const BFIBase::LoopData *OuterLoop,
                  BlockEdgesAdder addBlockEdges);
  void addNodesInLoop(const BFIBase::LoopData &OuterLoop);
  void addNodesInFunction();
  void addNode(const BlockNode &Node) {
    Nodes.emplace_back(Node);
    BFI.Working[Node.Index].getMass() = BlockMass::getEmpty();
  }
  void indexNodes();
  template <class BlockEdgesAdder>
  void addEdges(const BlockNode &Node, const BFIBase::LoopData *OuterLoop,
                BlockEdgesAdder addBlockEdges);
  void addEdge(IrrNode &Irr, const BlockNode &Succ,
               const BFIBase::LoopData *OuterLoop);
};
template <class BlockEdgesAdder>
void IrreducibleGraph::initialize(const BFIBase::LoopData *OuterLoop,
                                  BlockEdgesAdder addBlockEdges) {
  if (OuterLoop) {
    addNodesInLoop(*OuterLoop);
    for (auto N : OuterLoop->Nodes)
      addEdges(N, OuterLoop, addBlockEdges);
  } else {
    addNodesInFunction();
    for (uint32_t Index = 0; Index < BFI.Working.size(); ++Index)
      addEdges(Index, OuterLoop, addBlockEdges);
  }
  StartIrr = Lookup[Start.Index];
}
template <class BlockEdgesAdder>
void IrreducibleGraph::addEdges(const BlockNode &Node,
                                const BFIBase::LoopData *OuterLoop,
                                BlockEdgesAdder addBlockEdges) {
  auto L = Lookup.find(Node.Index);
  if (L == Lookup.end())
    return;
  IrrNode &Irr = *L->second;
  const auto &Working = BFI.Working[Node.Index];

  if (Working.isAPackage())
    for (const auto &I : Working.Loop->Exits)
      addEdge(Irr, I.first, OuterLoop);
  else
    addBlockEdges(*this, Irr, OuterLoop);
}
}

/// \brief Shared implementation for block frequency analysis.
///
/// This is a shared implementation of BlockFrequencyInfo and
/// MachineBlockFrequencyInfo, and calculates the relative frequencies of
/// blocks.
///
/// LoopInfo defines a loop as a "non-trivial" SCC dominated by a single block,
/// which is called the header.  A given loop, L, can have sub-loops, which are
/// loops within the subgraph of L that exclude its header.  (A "trivial" SCC
/// consists of a single block that does not have a self-edge.)
///
/// In addition to loops, this algorithm has limited support for irreducible
/// SCCs, which are SCCs with multiple entry blocks.  Irreducible SCCs are
/// discovered on they fly, and modelled as loops with multiple headers.
///
/// The headers of irreducible sub-SCCs consist of its entry blocks and all
/// nodes that are targets of a backedge within it (excluding backedges within
/// true sub-loops).  Block frequency calculations act as if a block is
/// inserted that intercepts all the edges to the headers.  All backedges and
/// entries point to this block.  Its successors are the headers, which split
/// the frequency evenly.
///
/// This algorithm leverages BlockMass and UnsignedFloat to maintain precision,
/// separates mass distribution from loop scaling, and dithers to eliminate
/// probability mass loss.
///
/// The implementation is split between BlockFrequencyInfoImpl, which knows the
/// type of graph being modelled (BasicBlock vs. MachineBasicBlock), and
/// BlockFrequencyInfoImplBase, which doesn't.  The base class uses \a
/// BlockNode, a wrapper around a uint32_t.  BlockNode is numbered from 0 in
/// reverse-post order.  This gives two advantages:  it's easy to compare the
/// relative ordering of two nodes, and maps keyed on BlockT can be represented
/// by vectors.
///
/// This algorithm is O(V+E), unless there is irreducible control flow, in
/// which case it's O(V*E) in the worst case.
///
/// These are the main stages:
///
///  0. Reverse post-order traversal (\a initializeRPOT()).
///
///     Run a single post-order traversal and save it (in reverse) in RPOT.
///     All other stages make use of this ordering.  Save a lookup from BlockT
///     to BlockNode (the index into RPOT) in Nodes.
///
///  1. Loop initialization (\a initializeLoops()).
///
///     Translate LoopInfo/MachineLoopInfo into a form suitable for the rest of
///     the algorithm.  In particular, store the immediate members of each loop
///     in reverse post-order.
///
///  2. Calculate mass and scale in loops (\a computeMassInLoops()).
///
///     For each loop (bottom-up), distribute mass through the DAG resulting
///     from ignoring backedges and treating sub-loops as a single pseudo-node.
///     Track the backedge mass distributed to the loop header, and use it to
///     calculate the loop scale (number of loop iterations).  Immediate
///     members that represent sub-loops will already have been visited and
///     packaged into a pseudo-node.
///
///     Distributing mass in a loop is a reverse-post-order traversal through
///     the loop.  Start by assigning full mass to the Loop header.  For each
///     node in the loop:
///
///         - Fetch and categorize the weight distribution for its successors.
///           If this is a packaged-subloop, the weight distribution is stored
///           in \a LoopData::Exits.  Otherwise, fetch it from
///           BranchProbabilityInfo.
///
///         - Each successor is categorized as \a Weight::Local, a local edge
///           within the current loop, \a Weight::Backedge, a backedge to the
///           loop header, or \a Weight::Exit, any successor outside the loop.
///           The weight, the successor, and its category are stored in \a
///           Distribution.  There can be multiple edges to each successor.
///
///         - If there's a backedge to a non-header, there's an irreducible SCC.
///           The usual flow is temporarily aborted.  \a
///           computeIrreducibleMass() finds the irreducible SCCs within the
///           loop, packages them up, and restarts the flow.
///
///         - Normalize the distribution:  scale weights down so that their sum
///           is 32-bits, and coalesce multiple edges to the same node.
///
///         - Distribute the mass accordingly, dithering to minimize mass loss,
///           as described in \a distributeMass().
///
///     Finally, calculate the loop scale from the accumulated backedge mass.
///
///  3. Distribute mass in the function (\a computeMassInFunction()).
///
///     Finally, distribute mass through the DAG resulting from packaging all
///     loops in the function.  This uses the same algorithm as distributing
///     mass in a loop, except that there are no exit or backedge edges.
///
///  4. Unpackage loops (\a unwrapLoops()).
///
///     Initialize each block's frequency to a floating point representation of
///     its mass.
///
///     Visit loops top-down, scaling the frequencies of its immediate members
///     by the loop's pseudo-node's frequency.
///
///  5. Convert frequencies to a 64-bit range (\a finalizeMetrics()).
///
///     Using the min and max frequencies as a guide, translate floating point
///     frequencies to an appropriate range in uint64_t.
///
/// It has some known flaws.
///
///   - Loop scale is limited to 4096 per loop (2^12) to avoid exhausting
///     BlockFrequency's 64-bit integer precision.
///
///   - The model of irreducible control flow is a rough approximation.
///
///     Modelling irreducible control flow exactly involves setting up and
///     solving a group of infinite geometric series.  Such precision is
///     unlikely to be worthwhile, since most of our algorithms give up on
///     irreducible control flow anyway.
///
///     Nevertheless, we might find that we need to get closer.  Here's a sort
///     of TODO list for the model with diminishing returns, to be completed as
///     necessary.
///
///       - The headers for the \a LoopData representing an irreducible SCC
///         include non-entry blocks.  When these extra blocks exist, they
///         indicate a self-contained irreducible sub-SCC.  We could treat them
///         as sub-loops, rather than arbitrarily shoving the problematic
///         blocks into the headers of the main irreducible SCC.
///
///       - Backedge frequencies are assumed to be evenly split between the
///         headers of a given irreducible SCC.  Instead, we could track the
///         backedge mass separately for each header, and adjust their relative
///         frequencies.
///
///       - Entry frequencies are assumed to be evenly split between the
///         headers of a given irreducible SCC, which is the only option if we
///         need to compute mass in the SCC before its parent loop.  Instead,
///         we could partially compute mass in the parent loop, and stop when
///         we get to the SCC.  Here, we have the correct ratio of entry
///         masses, which we can use to adjust their relative frequencies.
///         Compute mass in the SCC, and then continue propagation in the
///         parent.
///
///       - We can propagate mass iteratively through the SCC, for some fixed
///         number of iterations.  Each iteration starts by assigning the entry
///         blocks their backedge mass from the prior iteration.  The final
///         mass for each block (and each exit, and the total backedge mass
///         used for computing loop scale) is the sum of all iterations.
///         (Running this until fixed point would "solve" the geometric
///         series by simulation.)
template <class BT> class BlockFrequencyInfoImpl : BlockFrequencyInfoImplBase {
  typedef typename bfi_detail::TypeMap<BT>::BlockT BlockT;
  typedef typename bfi_detail::TypeMap<BT>::FunctionT FunctionT;
  typedef typename bfi_detail::TypeMap<BT>::BranchProbabilityInfoT
  BranchProbabilityInfoT;
  typedef typename bfi_detail::TypeMap<BT>::LoopT LoopT;
  typedef typename bfi_detail::TypeMap<BT>::LoopInfoT LoopInfoT;

  // This is part of a workaround for a GCC 4.7 crash on lambdas.
  friend struct bfi_detail::BlockEdgesAdder<BT>;

  typedef GraphTraits<const BlockT *> Successor;
  typedef GraphTraits<Inverse<const BlockT *>> Predecessor;

  const BranchProbabilityInfoT *BPI;
  const LoopInfoT *LI;
  const FunctionT *F;

  // All blocks in reverse postorder.
  std::vector<const BlockT *> RPOT;
  DenseMap<const BlockT *, BlockNode> Nodes;

  typedef typename std::vector<const BlockT *>::const_iterator rpot_iterator;

  rpot_iterator rpot_begin() const { return RPOT.begin(); }
  rpot_iterator rpot_end() const { return RPOT.end(); }

  size_t getIndex(const rpot_iterator &I) const { return I - rpot_begin(); }

  BlockNode getNode(const rpot_iterator &I) const {
    return BlockNode(getIndex(I));
  }
  BlockNode getNode(const BlockT *BB) const { return Nodes.lookup(BB); }

  const BlockT *getBlock(const BlockNode &Node) const {
    assert(Node.Index < RPOT.size());
    return RPOT[Node.Index];
  }

  /// \brief Run (and save) a post-order traversal.
  ///
  /// Saves a reverse post-order traversal of all the nodes in \a F.
  void initializeRPOT();

  /// \brief Initialize loop data.
  ///
  /// Build up \a Loops using \a LoopInfo.  \a LoopInfo gives us a mapping from
  /// each block to the deepest loop it's in, but we need the inverse.  For each
  /// loop, we store in reverse post-order its "immediate" members, defined as
  /// the header, the headers of immediate sub-loops, and all other blocks in
  /// the loop that are not in sub-loops.
  void initializeLoops();

  /// \brief Propagate to a block's successors.
  ///
  /// In the context of distributing mass through \c OuterLoop, divide the mass
  /// currently assigned to \c Node between its successors.
  ///
  /// \return \c true unless there's an irreducible backedge.
  bool propagateMassToSuccessors(LoopData *OuterLoop, const BlockNode &Node);

  /// \brief Compute mass in a particular loop.
  ///
  /// Assign mass to \c Loop's header, and then for each block in \c Loop in
  /// reverse post-order, distribute mass to its successors.  Only visits nodes
  /// that have not been packaged into sub-loops.
  ///
  /// \pre \a computeMassInLoop() has been called for each subloop of \c Loop.
  /// \return \c true unless there's an irreducible backedge.
  bool computeMassInLoop(LoopData &Loop);

  /// \brief Try to compute mass in the top-level function.
  ///
  /// Assign mass to the entry block, and then for each block in reverse
  /// post-order, distribute mass to its successors.  Skips nodes that have
  /// been packaged into loops.
  ///
  /// \pre \a computeMassInLoops() has been called.
  /// \return \c true unless there's an irreducible backedge.
  bool tryToComputeMassInFunction();

  /// \brief Compute mass in (and package up) irreducible SCCs.
  ///
  /// Find the irreducible SCCs in \c OuterLoop, add them to \a Loops (in front
  /// of \c Insert), and call \a computeMassInLoop() on each of them.
  ///
  /// If \c OuterLoop is \c nullptr, it refers to the top-level function.
  ///
  /// \pre \a computeMassInLoop() has been called for each subloop of \c
  /// OuterLoop.
  /// \pre \c Insert points at the the last loop successfully processed by \a
  /// computeMassInLoop().
  /// \pre \c OuterLoop has irreducible SCCs.
  void computeIrreducibleMass(LoopData *OuterLoop,
                              std::list<LoopData>::iterator Insert);

  /// \brief Compute mass in all loops.
  ///
  /// For each loop bottom-up, call \a computeMassInLoop().
  ///
  /// \a computeMassInLoop() aborts (and returns \c false) on loops that
  /// contain a irreducible sub-SCCs.  Use \a computeIrreducibleMass() and then
  /// re-enter \a computeMassInLoop().
  ///
  /// \post \a computeMassInLoop() has returned \c true for every loop.
  void computeMassInLoops();

  /// \brief Compute mass in the top-level function.
  ///
  /// Uses \a tryToComputeMassInFunction() and \a computeIrreducibleMass() to
  /// compute mass in the top-level function.
  ///
  /// \post \a tryToComputeMassInFunction() has returned \c true.
  void computeMassInFunction();

  std::string getBlockName(const BlockNode &Node) const override {
    return bfi_detail::getBlockName(getBlock(Node));
  }

public:
  const FunctionT *getFunction() const { return F; }

  void doFunction(const FunctionT *F, const BranchProbabilityInfoT *BPI,
                  const LoopInfoT *LI);
  BlockFrequencyInfoImpl() : BPI(nullptr), LI(nullptr), F(nullptr) {}

  using BlockFrequencyInfoImplBase::getEntryFreq;
  BlockFrequency getBlockFreq(const BlockT *BB) const {
    return BlockFrequencyInfoImplBase::getBlockFreq(getNode(BB));
  }
  Float getFloatingBlockFreq(const BlockT *BB) const {
    return BlockFrequencyInfoImplBase::getFloatingBlockFreq(getNode(BB));
  }

  /// \brief Print the frequencies for the current function.
  ///
  /// Prints the frequencies for the blocks in the current function.
  ///
  /// Blocks are printed in the natural iteration order of the function, rather
  /// than reverse post-order.  This provides two advantages:  writing -analyze
  /// tests is easier (since blocks come out in source order), and even
  /// unreachable blocks are printed.
  ///
  /// \a BlockFrequencyInfoImplBase::print() only knows reverse post-order, so
  /// we need to override it here.
  raw_ostream &print(raw_ostream &OS) const override;
  using BlockFrequencyInfoImplBase::dump;

  using BlockFrequencyInfoImplBase::printBlockFreq;
  raw_ostream &printBlockFreq(raw_ostream &OS, const BlockT *BB) const {
    return BlockFrequencyInfoImplBase::printBlockFreq(OS, getNode(BB));
  }
};

template <class BT>
void BlockFrequencyInfoImpl<BT>::doFunction(const FunctionT *F,
                                            const BranchProbabilityInfoT *BPI,
                                            const LoopInfoT *LI) {
  // Save the parameters.
  this->BPI = BPI;
  this->LI = LI;
  this->F = F;

  // Clean up left-over data structures.
  BlockFrequencyInfoImplBase::clear();
  RPOT.clear();
  Nodes.clear();

  // Initialize.
  DEBUG(dbgs() << "\nblock-frequency: " << F->getName() << "\n================="
               << std::string(F->getName().size(), '=') << "\n");
  initializeRPOT();
  initializeLoops();

  // Visit loops in post-order to find thelocal mass distribution, and then do
  // the full function.
  computeMassInLoops();
  computeMassInFunction();
  unwrapLoops();
  finalizeMetrics();
}

template <class BT> void BlockFrequencyInfoImpl<BT>::initializeRPOT() {
  const BlockT *Entry = F->begin();
  RPOT.reserve(F->size());
  std::copy(po_begin(Entry), po_end(Entry), std::back_inserter(RPOT));
  std::reverse(RPOT.begin(), RPOT.end());

  assert(RPOT.size() - 1 <= BlockNode::getMaxIndex() &&
         "More nodes in function than Block Frequency Info supports");

  DEBUG(dbgs() << "reverse-post-order-traversal\n");
  for (rpot_iterator I = rpot_begin(), E = rpot_end(); I != E; ++I) {
    BlockNode Node = getNode(I);
    DEBUG(dbgs() << " - " << getIndex(I) << ": " << getBlockName(Node) << "\n");
    Nodes[*I] = Node;
  }

  Working.reserve(RPOT.size());
  for (size_t Index = 0; Index < RPOT.size(); ++Index)
    Working.emplace_back(Index);
  Freqs.resize(RPOT.size());
}

template <class BT> void BlockFrequencyInfoImpl<BT>::initializeLoops() {
  DEBUG(dbgs() << "loop-detection\n");
  if (LI->empty())
    return;

  // Visit loops top down and assign them an index.
  std::deque<std::pair<const LoopT *, LoopData *>> Q;
  for (const LoopT *L : *LI)
    Q.emplace_back(L, nullptr);
  while (!Q.empty()) {
    const LoopT *Loop = Q.front().first;
    LoopData *Parent = Q.front().second;
    Q.pop_front();

    BlockNode Header = getNode(Loop->getHeader());
    assert(Header.isValid());

    Loops.emplace_back(Parent, Header);
    Working[Header.Index].Loop = &Loops.back();
    DEBUG(dbgs() << " - loop = " << getBlockName(Header) << "\n");

    for (const LoopT *L : *Loop)
      Q.emplace_back(L, &Loops.back());
  }

  // Visit nodes in reverse post-order and add them to their deepest containing
  // loop.
  for (size_t Index = 0; Index < RPOT.size(); ++Index) {
    // Loop headers have already been mostly mapped.
    if (Working[Index].isLoopHeader()) {
      LoopData *ContainingLoop = Working[Index].getContainingLoop();
      if (ContainingLoop)
        ContainingLoop->Nodes.push_back(Index);
      continue;
    }

    const LoopT *Loop = LI->getLoopFor(RPOT[Index]);
    if (!Loop)
      continue;

    // Add this node to its containing loop's member list.
    BlockNode Header = getNode(Loop->getHeader());
    assert(Header.isValid());
    const auto &HeaderData = Working[Header.Index];
    assert(HeaderData.isLoopHeader());

    Working[Index].Loop = HeaderData.Loop;
    HeaderData.Loop->Nodes.push_back(Index);
    DEBUG(dbgs() << " - loop = " << getBlockName(Header)
                 << ": member = " << getBlockName(Index) << "\n");
  }
}

template <class BT> void BlockFrequencyInfoImpl<BT>::computeMassInLoops() {
  // Visit loops with the deepest first, and the top-level loops last.
  for (auto L = Loops.rbegin(), E = Loops.rend(); L != E; ++L) {
    if (computeMassInLoop(*L))
      continue;
    auto Next = std::next(L);
    computeIrreducibleMass(&*L, L.base());
    L = std::prev(Next);
    if (computeMassInLoop(*L))
      continue;
    llvm_unreachable("unhandled irreducible control flow");
  }
}

template <class BT>
bool BlockFrequencyInfoImpl<BT>::computeMassInLoop(LoopData &Loop) {
  // Compute mass in loop.
  DEBUG(dbgs() << "compute-mass-in-loop: " << getLoopName(Loop) << "\n");

  if (Loop.isIrreducible()) {
    BlockMass Remaining = BlockMass::getFull();
    for (uint32_t H = 0; H < Loop.NumHeaders; ++H) {
      auto &Mass = Working[Loop.Nodes[H].Index].getMass();
      Mass = Remaining * BranchProbability(1, Loop.NumHeaders - H);
      Remaining -= Mass;
    }
    for (const BlockNode &M : Loop.Nodes)
      if (!propagateMassToSuccessors(&Loop, M))
        llvm_unreachable("unhandled irreducible control flow");
  } else {
    Working[Loop.getHeader().Index].getMass() = BlockMass::getFull();
    if (!propagateMassToSuccessors(&Loop, Loop.getHeader()))
      llvm_unreachable("irreducible control flow to loop header!?");
    for (const BlockNode &M : Loop.members())
      if (!propagateMassToSuccessors(&Loop, M))
        // Irreducible backedge.
        return false;
  }

  computeLoopScale(Loop);
  packageLoop(Loop);
  return true;
}

template <class BT>
bool BlockFrequencyInfoImpl<BT>::tryToComputeMassInFunction() {
  // Compute mass in function.
  DEBUG(dbgs() << "compute-mass-in-function\n");
  assert(!Working.empty() && "no blocks in function");
  assert(!Working[0].isLoopHeader() && "entry block is a loop header");

  Working[0].getMass() = BlockMass::getFull();
  for (rpot_iterator I = rpot_begin(), IE = rpot_end(); I != IE; ++I) {
    // Check for nodes that have been packaged.
    BlockNode Node = getNode(I);
    if (Working[Node.Index].isPackaged())
      continue;

    if (!propagateMassToSuccessors(nullptr, Node))
      return false;
  }
  return true;
}

template <class BT> void BlockFrequencyInfoImpl<BT>::computeMassInFunction() {
  if (tryToComputeMassInFunction())
    return;
  computeIrreducibleMass(nullptr, Loops.begin());
  if (tryToComputeMassInFunction())
    return;
  llvm_unreachable("unhandled irreducible control flow");
}

/// \note This should be a lambda, but that crashes GCC 4.7.
namespace bfi_detail {
template <class BT> struct BlockEdgesAdder {
  typedef BT BlockT;
  typedef BlockFrequencyInfoImplBase::LoopData LoopData;
  typedef GraphTraits<const BlockT *> Successor;

  const BlockFrequencyInfoImpl<BT> &BFI;
  explicit BlockEdgesAdder(const BlockFrequencyInfoImpl<BT> &BFI)
      : BFI(BFI) {}
  void operator()(IrreducibleGraph &G, IrreducibleGraph::IrrNode &Irr,
                  const LoopData *OuterLoop) {
    const BlockT *BB = BFI.RPOT[Irr.Node.Index];
    for (auto I = Successor::child_begin(BB), E = Successor::child_end(BB);
         I != E; ++I)
      G.addEdge(Irr, BFI.getNode(*I), OuterLoop);
  }
};
}
template <class BT>
void BlockFrequencyInfoImpl<BT>::computeIrreducibleMass(
    LoopData *OuterLoop, std::list<LoopData>::iterator Insert) {
  DEBUG(dbgs() << "analyze-irreducible-in-";
        if (OuterLoop) dbgs() << "loop: " << getLoopName(*OuterLoop) << "\n";
        else dbgs() << "function\n");

  using namespace bfi_detail;
  // Ideally, addBlockEdges() would be declared here as a lambda, but that
  // crashes GCC 4.7.
  BlockEdgesAdder<BT> addBlockEdges(*this);
  IrreducibleGraph G(*this, OuterLoop, addBlockEdges);

  for (auto &L : analyzeIrreducible(G, OuterLoop, Insert))
    computeMassInLoop(L);

  if (!OuterLoop)
    return;
  updateLoopWithIrreducible(*OuterLoop);
}

template <class BT>
bool
BlockFrequencyInfoImpl<BT>::propagateMassToSuccessors(LoopData *OuterLoop,
                                                      const BlockNode &Node) {
  DEBUG(dbgs() << " - node: " << getBlockName(Node) << "\n");
  // Calculate probability for successors.
  Distribution Dist;
  if (auto *Loop = Working[Node.Index].getPackagedLoop()) {
    assert(Loop != OuterLoop && "Cannot propagate mass in a packaged loop");
    if (!addLoopSuccessorsToDist(OuterLoop, *Loop, Dist))
      // Irreducible backedge.
      return false;
  } else {
    const BlockT *BB = getBlock(Node);
    for (auto SI = Successor::child_begin(BB), SE = Successor::child_end(BB);
         SI != SE; ++SI)
      // Do not dereference SI, or getEdgeWeight() is linear in the number of
      // successors.
      if (!addToDist(Dist, OuterLoop, Node, getNode(*SI),
                     BPI->getEdgeWeight(BB, SI)))
        // Irreducible backedge.
        return false;
  }

  // Distribute mass to successors, saving exit and backedge data in the
  // loop header.
  distributeMass(Node, OuterLoop, Dist);
  return true;
}

template <class BT>
raw_ostream &BlockFrequencyInfoImpl<BT>::print(raw_ostream &OS) const {
  if (!F)
    return OS;
  OS << "block-frequency-info: " << F->getName() << "\n";
  for (const BlockT &BB : *F)
    OS << " - " << bfi_detail::getBlockName(&BB)
       << ": float = " << getFloatingBlockFreq(&BB)
       << ", int = " << getBlockFreq(&BB).getFrequency() << "\n";

  // Add an extra newline for readability.
  OS << "\n";
  return OS;
}
}

#undef DEBUG_TYPE

#endif
