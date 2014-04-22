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
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

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
    typedef SmallVector<BlockNode, 4> MemberList;
    BlockNode Header;       ///< Header.
    bool IsPackaged;        ///< Whether this has been packaged.
    ExitMap Exits;          ///< Successor edges (and weights).
    MemberList Members;     ///< Members of the loop.
    BlockMass BackedgeMass; ///< Mass returned to loop header.
    BlockMass Mass;
    Float Scale;

    LoopData(const BlockNode &Header) : Header(Header), IsPackaged(false) {}
  };

  /// \brief Index of loop information.
  struct WorkingData {
    LoopData *Loop;           ///< The loop this block is the header of.
    LoopData *ContainingLoop; ///< The block whose loop this block is inside.
    BlockMass Mass;           ///< Mass distribution from the entry block.

    WorkingData() : Loop(nullptr), ContainingLoop(nullptr) {}

    bool hasLoopHeader() const { return ContainingLoop; }
    bool isLoopHeader() const { return Loop; }

    BlockNode getContainingHeader() const {
      if (ContainingLoop)
        return ContainingLoop->Header;
      return BlockNode();
    }

    /// \brief Has ContainingLoop been packaged up?
    bool isPackaged() const {
      return ContainingLoop && ContainingLoop->IsPackaged;
    }
    /// \brief Has Loop been packaged up?
    bool isAPackage() const { return Loop && Loop->IsPackaged; }
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
  /// the distribution.  It should never overflow twice.  There's no flag for
  /// whether \a ForwardTotal overflows, since when \a Total exceeds 32-bits
  /// they both get re-computed during \a normalize().
  struct Distribution {
    typedef SmallVector<Weight, 4> WeightList;
    WeightList Weights;    ///< Individual successor weights.
    uint64_t Total;        ///< Sum of all weights.
    bool DidOverflow;      ///< Whether \a Total did overflow.
    uint32_t ForwardTotal; ///< Total excluding backedges.

    Distribution() : Total(0), DidOverflow(false), ForwardTotal(0) {}
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
  std::vector<std::unique_ptr<LoopData>> PackagedLoops;

  /// \brief Add all edges out of a packaged loop to the distribution.
  ///
  /// Adds all edges from LocalLoopHead to Dist.  Calls addToDist() to add each
  /// successor edge.
  void addLoopSuccessorsToDist(const BlockNode &LoopHead,
                               const BlockNode &LocalLoopHead,
                               Distribution &Dist);

  /// \brief Add an edge to the distribution.
  ///
  /// Adds an edge to Succ to Dist.  If \c LoopHead.isValid(), then whether the
  /// edge is forward/exit/backedge is in the context of LoopHead.  Otherwise,
  /// every edge should be a forward edge (since all the loops are packaged
  /// up).
  void addToDist(Distribution &Dist, const BlockNode &LoopHead,
                 const BlockNode &Pred, const BlockNode &Succ, uint64_t Weight);

  LoopData &getLoopPackage(const BlockNode &Head) {
    assert(Head.Index < Working.size());
    assert(Working[Head.Index].Loop != nullptr);
    return *Working[Head.Index].Loop;
  }

  /// \brief Distribute mass according to a distribution.
  ///
  /// Distributes the mass in Source according to Dist.  If LoopHead.isValid(),
  /// backedges and exits are stored in its entry in PackagedLoops.
  ///
  /// Mass is distributed in parallel from two copies of the source mass.
  ///
  /// The first mass (forward) represents the distribution of mass through the
  /// local DAG.  This distribution should lose mass at loop exits and ignore
  /// backedges.
  ///
  /// The second mass (general) represents the behavior of the loop in the
  /// global context.  In a given distribution from the head, how much mass
  /// exits, and to where?  How much mass returns to the loop head?
  ///
  /// The forward mass should be split up between local successors and exits,
  /// but only actually distributed to the local successors.  The general mass
  /// should be split up between all three types of successors, but distributed
  /// only to exits and backedges.
  void distributeMass(const BlockNode &Source, const BlockNode &LoopHead,
                      Distribution &Dist);

  /// \brief Compute the loop scale for a loop.
  void computeLoopScale(const BlockNode &LoopHead);

  /// \brief Package up a loop.
  void packageLoop(const BlockNode &LoopHead);

  /// \brief Finalize frequency metrics.
  ///
  /// Unwraps loop packages, calculates final frequencies, and cleans up
  /// no-longer-needed data structures.
  void finalizeMetrics();

  /// \brief Clear all memory.
  void clear();

  virtual std::string getBlockName(const BlockNode &Node) const;

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
}

/// \brief Shared implementation for block frequency analysis.
///
/// This is a shared implementation of BlockFrequencyInfo and
/// MachineBlockFrequencyInfo, and calculates the relative frequencies of
/// blocks.
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
///  1. Loop indexing (\a initializeLoops()).
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
///     calculate the loop scale (number of loop iterations).
///
///     Visiting loops bottom-up is a post-order traversal of loop headers.
///     For each loop, immediate members that represent sub-loops will already
///     have been visited and packaged into a pseudo-node.
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
///         - Each successor is categorized as \a Weight::Local, a normal
///           forward edge within the current loop, \a Weight::Backedge, a
///           backedge to the loop header, or \a Weight::Exit, any successor
///           outside the loop.  The weight, the successor, and its category
///           are stored in \a Distribution.  There can be multiple edges to
///           each successor.
///
///         - Normalize the distribution:  scale weights down so that their sum
///           is 32-bits, and coalesce multiple edges to the same node.
///
///         - Distribute the mass accordingly, dithering to minimize mass loss,
///           as described in \a distributeMass().  Mass is distributed in
///           parallel in two ways: forward, and general.  Local successors
///           take their mass from the forward mass, while exit and backedge
///           successors take their mass from the general mass.  Additionally,
///           exit edges use up (ignored) mass from the forward mass, and local
///           edges use up (ignored) mass from the general distribution.
///
///     Finally, calculate the loop scale from the accumulated backedge mass.
///
///  3. Distribute mass in the function (\a computeMassInFunction()).
///
///     Finally, distribute mass through the DAG resulting from packaging all
///     loops in the function.  This uses the same algorithm as distributing
///     mass in a loop, except that there are no exit or backedge edges.
///
///  4. Loop unpackaging and cleanup (\a finalizeMetrics()).
///
///     Initialize the frequency to a floating point representation of its
///     mass.
///
///     Visit loops top-down (reverse post-order), scaling the loop header's
///     frequency by its psuedo-node's mass and loop scale.  Keep track of the
///     minimum and maximum final frequencies.
///
///     Using the min and max frequencies as a guide, translate floating point
///     frequencies to an appropriate range in uint64_t.
///
/// It has some known flaws.
///
///   - Irreducible control flow isn't modelled correctly.  In particular,
///     LoopInfo and MachineLoopInfo ignore irreducible backedges.  The main
///     result is that irreducible SCCs will under-scaled.  No mass is lost,
///     but the computed branch weights for the loop pseudo-node will be
///     incorrect.
///
///     Modelling irreducible control flow exactly involves setting up and
///     solving a group of infinite geometric series.  Such precision is
///     unlikely to be worthwhile, since most of our algorithms give up on
///     irreducible control flow anyway.
///
///     Nevertheless, we might find that we need to get closer.  If
///     LoopInfo/MachineLoopInfo flags loops with irreducible control flow
///     (and/or the function as a whole), we can find the SCCs, compute an
///     approximate exit frequency for the SCC as a whole, and scale up
///     accordingly.
///
///   - Loop scale is limited to 4096 per loop (2^12) to avoid exhausting
///     BlockFrequency's 64-bit integer precision.
template <class BT> class BlockFrequencyInfoImpl : BlockFrequencyInfoImplBase {
  typedef typename bfi_detail::TypeMap<BT>::BlockT BlockT;
  typedef typename bfi_detail::TypeMap<BT>::FunctionT FunctionT;
  typedef typename bfi_detail::TypeMap<BT>::BranchProbabilityInfoT
  BranchProbabilityInfoT;
  typedef typename bfi_detail::TypeMap<BT>::LoopT LoopT;
  typedef typename bfi_detail::TypeMap<BT>::LoopInfoT LoopInfoT;

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

  void initializeRPOT();
  void initializeLoops();
  void runOnFunction(const FunctionT *F);

  void propagateMassToSuccessors(const BlockNode &LoopHead,
                                 const BlockNode &Node);
  void computeMassInLoops();
  void computeMassInLoop(const BlockNode &LoopHead);
  void computeMassInFunction();

  std::string getBlockName(const BlockNode &Node) const override {
    return bfi_detail::getBlockName(getBlock(Node));
  }

public:
  const FunctionT *getFunction() const { return F; }

  void doFunction(const FunctionT *F, const BranchProbabilityInfoT *BPI,
                  const LoopInfoT *LI);
  BlockFrequencyInfoImpl() : BPI(0), LI(0), F(0) {}

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

  Working.resize(RPOT.size());
  Freqs.resize(RPOT.size());
}

template <class BT> void BlockFrequencyInfoImpl<BT>::initializeLoops() {
  DEBUG(dbgs() << "loop-detection\n");
  if (LI->empty())
    return;

  // Visit loops top down and assign them an index.
  std::deque<const LoopT *> Q;
  Q.insert(Q.end(), LI->begin(), LI->end());
  while (!Q.empty()) {
    const LoopT *Loop = Q.front();
    Q.pop_front();
    Q.insert(Q.end(), Loop->begin(), Loop->end());

    // Save the order this loop was visited.
    BlockNode Header = getNode(Loop->getHeader());
    assert(Header.isValid());

    PackagedLoops.emplace_back(new LoopData(Header));
    Working[Header.Index].Loop = PackagedLoops.back().get();
    DEBUG(dbgs() << " - loop = " << getBlockName(Header) << "\n");
  }

  // Visit nodes in reverse post-order and add them to their deepest containing
  // loop.
  for (size_t Index = 0; Index < RPOT.size(); ++Index) {
    const LoopT *Loop = LI->getLoopFor(RPOT[Index]);
    if (!Loop)
      continue;

    // If this is a loop header, find its parent loop (if any).
    if (Working[Index].isLoopHeader())
      if (!(Loop = Loop->getParentLoop()))
        continue;

    // Add this node to its containing loop's member list.
    BlockNode Header = getNode(Loop->getHeader());
    assert(Header.isValid());
    const auto &HeaderData = Working[Header.Index];
    assert(HeaderData.isLoopHeader());

    Working[Index].ContainingLoop = HeaderData.Loop;
    HeaderData.Loop->Members.push_back(Index);
    DEBUG(dbgs() << " - loop = " << getBlockName(Header)
                 << ": member = " << getBlockName(Index) << "\n");
  }
}

template <class BT> void BlockFrequencyInfoImpl<BT>::computeMassInLoops() {
  // Visit loops with the deepest first, and the top-level loops last.
  for (const auto &L : make_range(PackagedLoops.rbegin(), PackagedLoops.rend()))
    computeMassInLoop(L->Header);
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::computeMassInLoop(const BlockNode &LoopHead) {
  // Compute mass in loop.
  DEBUG(dbgs() << "compute-mass-in-loop: " << getBlockName(LoopHead) << "\n");

  Working[LoopHead.Index].Mass = BlockMass::getFull();
  propagateMassToSuccessors(LoopHead, LoopHead);

  for (const BlockNode &M : getLoopPackage(LoopHead).Members)
    propagateMassToSuccessors(LoopHead, M);

  computeLoopScale(LoopHead);
  packageLoop(LoopHead);
}

template <class BT> void BlockFrequencyInfoImpl<BT>::computeMassInFunction() {
  // Compute mass in function.
  DEBUG(dbgs() << "compute-mass-in-function\n");
  assert(!Working.empty() && "no blocks in function");
  assert(!Working[0].isLoopHeader() && "entry block is a loop header");

  Working[0].Mass = BlockMass::getFull();
  for (rpot_iterator I = rpot_begin(), IE = rpot_end(); I != IE; ++I) {
    // Check for nodes that have been packaged.
    BlockNode Node = getNode(I);
    if (Working[Node.Index].hasLoopHeader())
      continue;

    propagateMassToSuccessors(BlockNode(), Node);
  }
}

template <class BT>
void
BlockFrequencyInfoImpl<BT>::propagateMassToSuccessors(const BlockNode &LoopHead,
                                                      const BlockNode &Node) {
  DEBUG(dbgs() << " - node: " << getBlockName(Node) << "\n");
  // Calculate probability for successors.
  Distribution Dist;
  if (Node != LoopHead && Working[Node.Index].isLoopHeader())
    addLoopSuccessorsToDist(LoopHead, Node, Dist);
  else {
    const BlockT *BB = getBlock(Node);
    for (auto SI = Successor::child_begin(BB), SE = Successor::child_end(BB);
         SI != SE; ++SI)
      // Do not dereference SI, or getEdgeWeight() is linear in the number of
      // successors.
      addToDist(Dist, LoopHead, Node, getNode(*SI), BPI->getEdgeWeight(BB, SI));
  }

  // Distribute mass to successors, saving exit and backedge data in the
  // loop header.
  distributeMass(Node, LoopHead, Dist);
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
