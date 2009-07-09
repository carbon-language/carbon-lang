//===-- ConstantRange.cpp - ConstantRange implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Represent a range of possible values that may occur when the program is run
// for an integral value.  This keeps track of a lower and upper bound for the
// constant, which MAY wrap around the end of the numeric range.  To do this, it
// keeps track of a [lower, upper) bound, which specifies an interval just like
// STL iterators.  When used with boolean values, the following are important
// ranges (other integral ranges use min/max values for special range values):
//
//  [F, F) = {}     = Empty set
//  [T, F) = {T}
//  [F, T) = {F}
//  [T, T) = {F, T} = Full set
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// Initialize a range to hold the single specified value.
///
ConstantRangeBase::ConstantRangeBase(const APInt & V)
  : Lower(V), Upper(V + 1) {}

ConstantRangeBase::ConstantRangeBase(const APInt &L, const APInt &U)
  : Lower(L), Upper(U) {
  assert(L.getBitWidth() == U.getBitWidth() &&
         "ConstantRange with unequal bit widths");
}

/// print - Print out the bounds to a stream...
///
void ConstantRangeBase::print(raw_ostream &OS) const {
  OS << "[" << Lower << "," << Upper << ")";
}

/// dump - Allow printing from a debugger easily...
///
void ConstantRangeBase::dump() const {
  print(errs());
}

std::ostream &llvm::operator<<(std::ostream &o,
                               const ConstantRangeBase &CR) {
  raw_os_ostream OS(o);
  OS << CR;
  return o;
}

/// Initialize a full (the default) or empty set for the specified type.
///
ConstantRange::ConstantRange(uint32_t BitWidth, bool Full) :
  ConstantRangeBase(APInt(BitWidth, 0), APInt(BitWidth, 0)) {
  if (Full)
    Lower = Upper = APInt::getMaxValue(BitWidth);
  else
    Lower = Upper = APInt::getMinValue(BitWidth);
}

/// Initialize a range to hold the single specified value.
///
ConstantRange::ConstantRange(const APInt & V) : ConstantRangeBase(V) {}

ConstantRange::ConstantRange(const APInt &L, const APInt &U)
  : ConstantRangeBase(L, U) {
  assert((L != U || (L.isMaxValue() || L.isMinValue())) &&
         "Lower == Upper, but they aren't min or max value!");
}

/// isFullSet - Return true if this set contains all of the elements possible
/// for this data-type
bool ConstantRange::isFullSet() const {
  return Lower == Upper && Lower.isMaxValue();
}

/// isEmptySet - Return true if this set contains no members.
///
bool ConstantRange::isEmptySet() const {
  return Lower == Upper && Lower.isMinValue();
}

/// isWrappedSet - Return true if this set wraps around the top of the range,
/// for example: [100, 8)
///
bool ConstantRange::isWrappedSet() const {
  return Lower.ugt(Upper);
}

/// getSetSize - Return the number of elements in this set.
///
APInt ConstantRange::getSetSize() const {
  if (isEmptySet()) 
    return APInt(getBitWidth(), 0);
  if (getBitWidth() == 1) {
    if (Lower != Upper)  // One of T or F in the set...
      return APInt(2, 1);
    return APInt(2, 2);      // Must be full set...
  }

  // Simply subtract the bounds...
  return Upper - Lower;
}

/// getUnsignedMax - Return the largest unsigned value contained in the
/// ConstantRange.
///
APInt ConstantRange::getUnsignedMax() const {
  if (isFullSet() || isWrappedSet())
    return APInt::getMaxValue(getBitWidth());
  else
    return getUpper() - 1;
}

/// getUnsignedMin - Return the smallest unsigned value contained in the
/// ConstantRange.
///
APInt ConstantRange::getUnsignedMin() const {
  if (isFullSet() || (isWrappedSet() && getUpper() != 0))
    return APInt::getMinValue(getBitWidth());
  else
    return getLower();
}

/// getSignedMax - Return the largest signed value contained in the
/// ConstantRange.
///
APInt ConstantRange::getSignedMax() const {
  APInt SignedMax(APInt::getSignedMaxValue(getBitWidth()));
  if (!isWrappedSet()) {
    if (getLower().sle(getUpper() - 1))
      return getUpper() - 1;
    else
      return SignedMax;
  } else {
    if ((getUpper() - 1).slt(getLower())) {
      if (getLower() != SignedMax)
        return SignedMax;
      else
        return getUpper() - 1;
    } else {
      return getUpper() - 1;
    }
  }
}

/// getSignedMin - Return the smallest signed value contained in the
/// ConstantRange.
///
APInt ConstantRange::getSignedMin() const {
  APInt SignedMin(APInt::getSignedMinValue(getBitWidth()));
  if (!isWrappedSet()) {
    if (getLower().sle(getUpper() - 1))
      return getLower();
    else
      return SignedMin;
  } else {
    if ((getUpper() - 1).slt(getLower())) {
      if (getUpper() != SignedMin)
        return SignedMin;
      else
        return getLower();
    } else {
      return getLower();
    }
  }
}

/// contains - Return true if the specified value is in the set.
///
bool ConstantRange::contains(const APInt &V) const {
  if (Lower == Upper)
    return isFullSet();

  if (!isWrappedSet())
    return Lower.ule(V) && V.ult(Upper);
  else
    return Lower.ule(V) || V.ult(Upper);
}

/// subtract - Subtract the specified constant from the endpoints of this
/// constant range.
ConstantRange ConstantRange::subtract(const APInt &Val) const {
  assert(Val.getBitWidth() == getBitWidth() && "Wrong bit width");
  // If the set is empty or full, don't modify the endpoints.
  if (Lower == Upper) 
    return *this;
  return ConstantRange(Lower - Val, Upper - Val);
}


// intersect1Wrapped - This helper function is used to intersect two ranges when
// it is known that LHS is wrapped and RHS isn't.
//
ConstantRange 
ConstantRange::intersect1Wrapped(const ConstantRange &LHS,
                                 const ConstantRange &RHS) {
  assert(LHS.isWrappedSet() && !RHS.isWrappedSet());

  // Check to see if we overlap on the Left side of RHS...
  //
  if (RHS.Lower.ult(LHS.Upper)) {
    // We do overlap on the left side of RHS, see if we overlap on the right of
    // RHS...
    if (RHS.Upper.ugt(LHS.Lower)) {
      // Ok, the result overlaps on both the left and right sides.  See if the
      // resultant interval will be smaller if we wrap or not...
      //
      if (LHS.getSetSize().ult(RHS.getSetSize()))
        return LHS;
      else
        return RHS;

    } else {
      // No overlap on the right, just on the left.
      return ConstantRange(RHS.Lower, LHS.Upper);
    }
  } else {
    // We don't overlap on the left side of RHS, see if we overlap on the right
    // of RHS...
    if (RHS.Upper.ugt(LHS.Lower)) {
      // Simple overlap...
      return ConstantRange(LHS.Lower, RHS.Upper);
    } else {
      // No overlap...
      return ConstantRange(LHS.getBitWidth(), false);
    }
  }
}

/// intersectWith - Return the range that results from the intersection of this
/// range with another range.
///
ConstantRange ConstantRange::intersectWith(const ConstantRange &CR) const {
  assert(getBitWidth() == CR.getBitWidth() && 
         "ConstantRange types don't agree!");
  // Handle common special cases
  if (isEmptySet() || CR.isFullSet())  
    return *this;
  if (isFullSet()  || CR.isEmptySet()) 
    return CR;

  if (!isWrappedSet()) {
    if (!CR.isWrappedSet()) {
      APInt L = APIntOps::umax(Lower, CR.Lower);
      APInt U = APIntOps::umin(Upper, CR.Upper);

      if (L.ult(U)) // If range isn't empty...
        return ConstantRange(L, U);
      else
        return ConstantRange(getBitWidth(), false);// Otherwise, empty set
    } else
      return intersect1Wrapped(CR, *this);
  } else {   // We know "this" is wrapped...
    if (!CR.isWrappedSet())
      return intersect1Wrapped(*this, CR);
    else {
      // Both ranges are wrapped...
      APInt L = APIntOps::umax(Lower, CR.Lower);
      APInt U = APIntOps::umin(Upper, CR.Upper);
      return ConstantRange(L, U);
    }
  }
  return *this;
}

/// maximalIntersectWith - Return the range that results from the intersection
/// of this range with another range.  The resultant range is guaranteed to
/// include all elements contained in both input ranges, and to have the
/// smallest possible set size that does so.  Because there may be two
/// intersections with the same set size, A.maximalIntersectWith(B) might not
/// be equal to B.maximalIntersect(A).
ConstantRange
ConstantRange::maximalIntersectWith(const ConstantRange &CR) const {
  assert(getBitWidth() == CR.getBitWidth() && 
         "ConstantRange types don't agree!");

  // Handle common cases.
  if (   isEmptySet() || CR.isFullSet()) return *this;
  if (CR.isEmptySet() ||    isFullSet()) return CR;

  if (!isWrappedSet() && CR.isWrappedSet())
    return CR.maximalIntersectWith(*this);

  if (!isWrappedSet() && !CR.isWrappedSet()) {
    if (Lower.ult(CR.Lower)) {
      if (Upper.ule(CR.Lower))
        return ConstantRange(getBitWidth(), false);

      if (Upper.ult(CR.Upper))
        return ConstantRange(CR.Lower, Upper);

      return CR;
    } else {
      if (Upper.ult(CR.Upper))
        return *this;

      if (Lower.ult(CR.Upper))
        return ConstantRange(Lower, CR.Upper);

      return ConstantRange(getBitWidth(), false);
    }
  }

  if (isWrappedSet() && !CR.isWrappedSet()) {
    if (CR.Lower.ult(Upper)) {
      if (CR.Upper.ult(Upper))
        return CR;

      if (CR.Upper.ult(Lower))
        return ConstantRange(CR.Lower, Upper);

      if (getSetSize().ult(CR.getSetSize()))
        return *this;
      else
        return CR;
    } else if (CR.Lower.ult(Lower)) {
      if (CR.Upper.ule(Lower))
        return ConstantRange(getBitWidth(), false);

      return ConstantRange(Lower, CR.Upper);
    }
    return CR;
  }

  if (CR.Upper.ult(Upper)) {
    if (CR.Lower.ult(Upper)) {
      if (getSetSize().ult(CR.getSetSize()))
        return *this;
      else
        return CR;
    }

    if (CR.Lower.ult(Lower))
      return ConstantRange(Lower, CR.Upper);

    return CR;
  } else if (CR.Upper.ult(Lower)) {
    if (CR.Lower.ult(Lower))
      return *this;

    return ConstantRange(CR.Lower, Upper);
  }
  if (getSetSize().ult(CR.getSetSize()))
    return *this;
  else
    return CR;
}


/// unionWith - Return the range that results from the union of this range with
/// another range.  The resultant range is guaranteed to include the elements of
/// both sets, but may contain more.  For example, [3, 9) union [12,15) is
/// [3, 15), which includes 9, 10, and 11, which were not included in either
/// set before.
///
ConstantRange ConstantRange::unionWith(const ConstantRange &CR) const {
  assert(getBitWidth() == CR.getBitWidth() && 
         "ConstantRange types don't agree!");

  if (   isFullSet() || CR.isEmptySet()) return *this;
  if (CR.isFullSet() ||    isEmptySet()) return CR;

  if (!isWrappedSet() && CR.isWrappedSet()) return CR.unionWith(*this);

  APInt L = Lower, U = Upper;

  if (!isWrappedSet() && !CR.isWrappedSet()) {
    if (CR.Lower.ult(L))
      L = CR.Lower;

    if (CR.Upper.ugt(U))
      U = CR.Upper;
  }

  if (isWrappedSet() && !CR.isWrappedSet()) {
    if ((CR.Lower.ult(Upper) && CR.Upper.ult(Upper)) ||
        (CR.Lower.ugt(Lower) && CR.Upper.ugt(Lower))) {
      return *this;
    }

    if (CR.Lower.ule(Upper) && Lower.ule(CR.Upper)) {
      return ConstantRange(getBitWidth());
    }

    if (CR.Lower.ule(Upper) && CR.Upper.ule(Lower)) {
      APInt d1 = CR.Upper - Upper, d2 = Lower - CR.Upper;
      if (d1.ult(d2)) {
        U = CR.Upper;
      } else {
        L = CR.Upper;
      }
    }

    if (Upper.ult(CR.Lower) && CR.Upper.ult(Lower)) {
      APInt d1 = CR.Lower - Upper, d2 = Lower - CR.Upper;
      if (d1.ult(d2)) {
        U = CR.Lower + 1;
      } else {
        L = CR.Upper - 1;
      }
    }

    if (Upper.ult(CR.Lower) && Lower.ult(CR.Upper)) {
      APInt d1 = CR.Lower - Upper, d2 = Lower - CR.Lower;

      if (d1.ult(d2)) {
        U = CR.Lower + 1;
      } else {
        L = CR.Lower;
      }
    }
  }

  if (isWrappedSet() && CR.isWrappedSet()) {
    if (Lower.ult(CR.Upper) || CR.Lower.ult(Upper))
      return ConstantRange(getBitWidth());

    if (CR.Upper.ugt(U)) {
      U = CR.Upper;
    }

    if (CR.Lower.ult(L)) {
      L = CR.Lower;
    }

    if (L == U) return ConstantRange(getBitWidth());
  }

  return ConstantRange(L, U);
}

/// zeroExtend - Return a new range in the specified integer type, which must
/// be strictly larger than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// zero extended.
ConstantRange ConstantRange::zeroExtend(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize < DstTySize && "Not a value extension");
  if (isFullSet())
    // Change a source full set into [0, 1 << 8*numbytes)
    return ConstantRange(APInt(DstTySize,0), APInt(DstTySize,1).shl(SrcTySize));

  APInt L = Lower; L.zext(DstTySize);
  APInt U = Upper; U.zext(DstTySize);
  return ConstantRange(L, U);
}

/// signExtend - Return a new range in the specified integer type, which must
/// be strictly larger than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// sign extended.
ConstantRange ConstantRange::signExtend(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize < DstTySize && "Not a value extension");
  if (isFullSet()) {
    return ConstantRange(APInt::getHighBitsSet(DstTySize,DstTySize-SrcTySize+1),
                         APInt::getLowBitsSet(DstTySize, SrcTySize-1));
  }

  APInt L = Lower; L.sext(DstTySize);
  APInt U = Upper; U.sext(DstTySize);
  return ConstantRange(L, U);
}

/// truncate - Return a new range in the specified integer type, which must be
/// strictly smaller than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// truncated to the specified type.
ConstantRange ConstantRange::truncate(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize > DstTySize && "Not a value truncation");
  APInt Size(APInt::getLowBitsSet(SrcTySize, DstTySize));
  if (isFullSet() || getSetSize().ugt(Size))
    return ConstantRange(DstTySize);

  APInt L = Lower; L.trunc(DstTySize);
  APInt U = Upper; U.trunc(DstTySize);
  return ConstantRange(L, U);
}

ConstantRange
ConstantRange::add(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return ConstantRange(getBitWidth(), /*isFullSet=*/false);

  APInt Spread_X = getSetSize(), Spread_Y = Other.getSetSize();
  APInt NewLower = getLower() + Other.getLower();
  APInt NewUpper = getUpper() + Other.getUpper() - 1;
  if (NewLower == NewUpper)
    return ConstantRange(getBitWidth(), /*isFullSet=*/true);

  ConstantRange X = ConstantRange(NewLower, NewUpper);
  if (X.getSetSize().ult(Spread_X) || X.getSetSize().ult(Spread_Y))
    // We've wrapped, therefore, full set.
    return ConstantRange(getBitWidth(), /*isFullSet=*/true);

  return X;
}

ConstantRange
ConstantRange::multiply(const ConstantRange &Other) const {
  // TODO: Implement multiply.
  return ConstantRange(getBitWidth(),
                       !(isEmptySet() || Other.isEmptySet()));
}

ConstantRange
ConstantRange::smax(const ConstantRange &Other) const {
  // TODO: Implement smax.
  return ConstantRange(getBitWidth(),
                       !(isEmptySet() || Other.isEmptySet()));
}

ConstantRange
ConstantRange::umax(const ConstantRange &Other) const {
  // X umax Y is: range(umax(X_umin, Y_umin),
  //                    umax(X_umax, Y_umax))
  if (isEmptySet() || Other.isEmptySet())
    return ConstantRange(getBitWidth(), /*isFullSet=*/false);
  if (isFullSet() || Other.isFullSet())
    return ConstantRange(getBitWidth(), /*isFullSet=*/true);
  APInt NewL = APIntOps::umax(getUnsignedMin(), Other.getUnsignedMin());
  APInt NewU = APIntOps::umax(getUnsignedMax(), Other.getUnsignedMax()) + 1;
  if (NewU == NewL)
    return ConstantRange(getBitWidth(), /*isFullSet=*/true);
  return ConstantRange(NewL, NewU);
}

ConstantRange
ConstantRange::udiv(const ConstantRange &Other) const {
  // TODO: Implement udiv.
  return ConstantRange(getBitWidth(),
                       !(isEmptySet() || Other.isEmptySet()));
}

/// Initialize a full (the default) or empty set for the specified type.
///
ConstantSignedRange::ConstantSignedRange(uint32_t BitWidth, bool Full) :
  ConstantRangeBase(APInt(BitWidth, 0), APInt(BitWidth, 0)) {
  if (Full)
    Lower = Upper = APInt::getSignedMaxValue(BitWidth);
  else
    Lower = Upper = APInt::getSignedMinValue(BitWidth);
}

/// Initialize a range to hold the single specified value.
///
ConstantSignedRange::ConstantSignedRange(const APInt & V)
  : ConstantRangeBase(V) {}

ConstantSignedRange::ConstantSignedRange(const APInt &L, const APInt &U)
  : ConstantRangeBase(L, U) {
  assert((L != U || (L.isMaxSignedValue() || L.isMinSignedValue())) &&
         "Lower == Upper, but they aren't min or max value!");
}

/// isFullSet - Return true if this set contains all of the elements possible
/// for this data-type
bool ConstantSignedRange::isFullSet() const {
  return Lower == Upper && Lower.isMaxSignedValue();
}

/// isEmptySet - Return true if this set contains no members.
///
bool ConstantSignedRange::isEmptySet() const {
  return Lower == Upper && Lower.isMinSignedValue();
}

/// isWrappedSet - Return true if this set wraps around the top of the range,
/// for example: [100, 8)
///
bool ConstantSignedRange::isWrappedSet() const {
  return Lower.sgt(Upper);
}

/// getSetSize - Return the number of elements in this set.
///
APInt ConstantSignedRange::getSetSize() const {
  if (isEmptySet())
    return APInt(getBitWidth(), 0);
  if (getBitWidth() == 1) {
    if (Lower != Upper)  // One of T or F in the set...
      return APInt(2, 1);
    return APInt(2, 2);      // Must be full set...
  }

  // Simply subtract the bounds...
  return Upper - Lower;
}

/// getSignedMax - Return the largest signed value contained in the
/// ConstantSignedRange.
///
APInt ConstantSignedRange::getSignedMax() const {
  if (isFullSet() || isWrappedSet())
    return APInt::getSignedMaxValue(getBitWidth());
  else
    return getUpper() - 1;
}

/// getSignedMin - Return the smallest signed value contained in the
/// ConstantSignedRange.
///
APInt ConstantSignedRange::getSignedMin() const {
  if (isFullSet() || (isWrappedSet() &&
                      getUpper() != APInt::getSignedMinValue(getBitWidth())))
    return APInt::getSignedMinValue(getBitWidth());
  else
    return getLower();
}

/// getUnsignedMax - Return the largest unsigned value contained in the
/// ConstantSignedRange.
///
APInt ConstantSignedRange::getUnsignedMax() const {
  APInt UnsignedMax(APInt::getMaxValue(getBitWidth()));
  if (!isWrappedSet()) {
    if (getLower().ule(getUpper() - 1))
      return getUpper() - 1;
    else
      return UnsignedMax;
  } else {
    if ((getUpper() - 1).ult(getLower())) {
      if (getLower() != UnsignedMax)
        return UnsignedMax;
      else
        return getUpper() - 1;
    } else {
      return getUpper() - 1;
    }
  }
}

/// getUnsignedMin - Return the smallest unsigned value contained in the
/// ConstantSignedRange.
///
APInt ConstantSignedRange::getUnsignedMin() const {
  APInt UnsignedMin(APInt::getMinValue(getBitWidth()));
  if (!isWrappedSet()) {
    if (getLower().ule(getUpper() - 1))
      return getLower();
    else
      return UnsignedMin;
  } else {
    if ((getUpper() - 1).ult(getLower())) {
      if (getUpper() != UnsignedMin)
        return UnsignedMin;
      else
        return getLower();
    } else {
      return getLower();
    }
  }
}

/// contains - Return true if the specified value is in the set.
///
bool ConstantSignedRange::contains(const APInt &V) const {
  if (Lower == Upper)
    return isFullSet();

  if (!isWrappedSet())
    return Lower.sle(V) && V.slt(Upper);
  else
    return Lower.sle(V) || V.slt(Upper);
}

/// subtract - Subtract the specified constant from the endpoints of this
/// constant range.
ConstantSignedRange ConstantSignedRange::subtract(const APInt &Val) const {
  assert(Val.getBitWidth() == getBitWidth() && "Wrong bit width");
  // If the set is empty or full, don't modify the endpoints.
  if (Lower == Upper)
    return *this;
  return ConstantSignedRange(Lower - Val, Upper - Val);
}


// intersect1Wrapped - This helper function is used to intersect two ranges when
// it is known that LHS is wrapped and RHS isn't.
//
ConstantSignedRange
ConstantSignedRange::intersect1Wrapped(const ConstantSignedRange &LHS,
                                 const ConstantSignedRange &RHS) {
  assert(LHS.isWrappedSet() && !RHS.isWrappedSet());

  // Check to see if we overlap on the Left side of RHS...
  //
  if (RHS.Lower.slt(LHS.Upper)) {
    // We do overlap on the left side of RHS, see if we overlap on the right of
    // RHS...
    if (RHS.Upper.sgt(LHS.Lower)) {
      // Ok, the result overlaps on both the left and right sides.  See if the
      // resultant interval will be smaller if we wrap or not...
      //
      if (LHS.getSetSize().ult(RHS.getSetSize()))
        return LHS;
      else
        return RHS;

    } else {
      // No overlap on the right, just on the left.
      return ConstantSignedRange(RHS.Lower, LHS.Upper);
    }
  } else {
    // We don't overlap on the left side of RHS, see if we overlap on the right
    // of RHS...
    if (RHS.Upper.sgt(LHS.Lower)) {
      // Simple overlap...
      return ConstantSignedRange(LHS.Lower, RHS.Upper);
    } else {
      // No overlap...
      return ConstantSignedRange(LHS.getBitWidth(), false);
    }
  }
}

/// intersectWith - Return the range that results from the intersection of this
/// range with another range.
///
ConstantSignedRange
ConstantSignedRange::intersectWith(const ConstantSignedRange &CR) const {
  assert(getBitWidth() == CR.getBitWidth() &&
         "ConstantSignedRange types don't agree!");
  // Handle common special cases
  if (isEmptySet() || CR.isFullSet())
    return *this;
  if (isFullSet()  || CR.isEmptySet())
    return CR;

  if (!isWrappedSet()) {
    if (!CR.isWrappedSet()) {
      APInt L = APIntOps::smax(Lower, CR.Lower);
      APInt U = APIntOps::smin(Upper, CR.Upper);

      if (L.slt(U)) // If range isn't empty...
        return ConstantSignedRange(L, U);
      else
        return ConstantSignedRange(getBitWidth(), false);// Otherwise, empty set
    } else
      return intersect1Wrapped(CR, *this);
  } else {   // We know "this" is wrapped...
    if (!CR.isWrappedSet())
      return intersect1Wrapped(*this, CR);
    else {
      // Both ranges are wrapped...
      APInt L = APIntOps::smax(Lower, CR.Lower);
      APInt U = APIntOps::smin(Upper, CR.Upper);
      return ConstantSignedRange(L, U);
    }
  }
  return *this;
}

/// maximalIntersectWith - Return the range that results from the intersection
/// of this range with another range.  The resultant range is guaranteed to
/// include all elements contained in both input ranges, and to have the
/// smallest possible set size that does so.  Because there may be two
/// intersections with the same set size, A.maximalIntersectWith(B) might not
/// be equal to B.maximalIntersect(A).
ConstantSignedRange
ConstantSignedRange::maximalIntersectWith(const ConstantSignedRange &CR) const {
  assert(getBitWidth() == CR.getBitWidth() &&
         "ConstantSignedRange types don't agree!");

  // Handle common cases.
  if (   isEmptySet() || CR.isFullSet()) return *this;
  if (CR.isEmptySet() ||    isFullSet()) return CR;

  if (!isWrappedSet() && CR.isWrappedSet())
    return CR.maximalIntersectWith(*this);

  if (!isWrappedSet() && !CR.isWrappedSet()) {
    if (Lower.slt(CR.Lower)) {
      if (Upper.sle(CR.Lower))
        return ConstantSignedRange(getBitWidth(), false);

      if (Upper.slt(CR.Upper))
        return ConstantSignedRange(CR.Lower, Upper);

      return CR;
    } else {
      if (Upper.slt(CR.Upper))
        return *this;

      if (Lower.slt(CR.Upper))
        return ConstantSignedRange(Lower, CR.Upper);

      return ConstantSignedRange(getBitWidth(), false);
    }
  }

  if (isWrappedSet() && !CR.isWrappedSet()) {
    if (CR.Lower.slt(Upper)) {
      if (CR.Upper.slt(Upper))
        return CR;

      if (CR.Upper.slt(Lower))
        return ConstantSignedRange(CR.Lower, Upper);

      if (getSetSize().ult(CR.getSetSize()))
        return *this;
      else
        return CR;
    } else if (CR.Lower.slt(Lower)) {
      if (CR.Upper.sle(Lower))
        return ConstantSignedRange(getBitWidth(), false);

      return ConstantSignedRange(Lower, CR.Upper);
    }
    return CR;
  }

  if (CR.Upper.slt(Upper)) {
    if (CR.Lower.slt(Upper)) {
      if (getSetSize().ult(CR.getSetSize()))
        return *this;
      else
        return CR;
    }

    if (CR.Lower.slt(Lower))
      return ConstantSignedRange(Lower, CR.Upper);

    return CR;
  } else if (CR.Upper.slt(Lower)) {
    if (CR.Lower.slt(Lower))
      return *this;

    return ConstantSignedRange(CR.Lower, Upper);
  }
  if (getSetSize().ult(CR.getSetSize()))
    return *this;
  else
    return CR;
}


/// unionWith - Return the range that results from the union of this range with
/// another range.  The resultant range is guaranteed to include the elements of
/// both sets, but may contain more.  For example, [3, 9) union [12,15) is
/// [3, 15), which includes 9, 10, and 11, which were not included in either
/// set before.
///
ConstantSignedRange
ConstantSignedRange::unionWith(const ConstantSignedRange &CR) const {
  assert(getBitWidth() == CR.getBitWidth() &&
         "ConstantSignedRange types don't agree!");

  if (   isFullSet() || CR.isEmptySet()) return *this;
  if (CR.isFullSet() ||    isEmptySet()) return CR;

  if (!isWrappedSet() && CR.isWrappedSet()) return CR.unionWith(*this);

  APInt L = Lower, U = Upper;

  if (!isWrappedSet() && !CR.isWrappedSet()) {
    if (CR.Lower.slt(L))
      L = CR.Lower;

    if (CR.Upper.sgt(U))
      U = CR.Upper;
  }

  if (isWrappedSet() && !CR.isWrappedSet()) {
    if ((CR.Lower.slt(Upper) && CR.Upper.slt(Upper)) ||
        (CR.Lower.sgt(Lower) && CR.Upper.sgt(Lower))) {
      return *this;
    }

    if (CR.Lower.sle(Upper) && Lower.sle(CR.Upper)) {
      return ConstantSignedRange(getBitWidth());
    }

    if (CR.Lower.sle(Upper) && CR.Upper.sle(Lower)) {
      APInt d1 = CR.Upper - Upper, d2 = Lower - CR.Upper;
      if (d1.slt(d2)) {
        U = CR.Upper;
      } else {
        L = CR.Upper;
      }
    }

    if (Upper.slt(CR.Lower) && CR.Upper.slt(Lower)) {
      APInt d1 = CR.Lower - Upper, d2 = Lower - CR.Upper;
      if (d1.slt(d2)) {
        U = CR.Lower + 1;
      } else {
        L = CR.Upper - 1;
      }
    }

    if (Upper.slt(CR.Lower) && Lower.slt(CR.Upper)) {
      APInt d1 = CR.Lower - Upper, d2 = Lower - CR.Lower;

      if (d1.slt(d2)) {
        U = CR.Lower + 1;
      } else {
        L = CR.Lower;
      }
    }
  }

  if (isWrappedSet() && CR.isWrappedSet()) {
    if (Lower.slt(CR.Upper) || CR.Lower.slt(Upper))
      return ConstantSignedRange(getBitWidth());

    if (CR.Upper.sgt(U)) {
      U = CR.Upper;
    }

    if (CR.Lower.slt(L)) {
      L = CR.Lower;
    }

    if (L == U) return ConstantSignedRange(getBitWidth());
  }

  return ConstantSignedRange(L, U);
}

/// zeroExtend - Return a new range in the specified integer type, which must
/// be strictly larger than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// zero extended.
ConstantSignedRange ConstantSignedRange::zeroExtend(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize < DstTySize && "Not a value extension");
  if (isEmptySet())
    return ConstantSignedRange(SrcTySize, /*isFullSet=*/false);
  if (isFullSet())
    // Change a source full set into [0, 1 << 8*numbytes)
    return ConstantSignedRange(APInt(DstTySize,0),
                               APInt(DstTySize,1).shl(SrcTySize));

  APInt L, U;
  if (Lower.isNegative() && !Upper.isNegative()) {
    L = APInt(SrcTySize, 0);
    U = APInt::getSignedMinValue(SrcTySize);
  } else {
    L = Lower;
    U = Upper;
  }
  L.zext(DstTySize);
  U.zext(DstTySize);
  return ConstantSignedRange(L, U);
}

/// signExtend - Return a new range in the specified integer type, which must
/// be strictly larger than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// sign extended.
ConstantSignedRange ConstantSignedRange::signExtend(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize < DstTySize && "Not a value extension");
  if (isEmptySet())
    return ConstantSignedRange(SrcTySize, /*isFullSet=*/false);
  if (isFullSet())
    return ConstantSignedRange(APInt(getSignedMin()).sext(DstTySize),
                               APInt(getSignedMax()).sext(DstTySize)+1);

  APInt L = Lower; L.sext(DstTySize);
  APInt U = Upper; U.sext(DstTySize);
  return ConstantSignedRange(L, U);
}

/// truncate - Return a new range in the specified integer type, which must be
/// strictly smaller than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// truncated to the specified type.
ConstantSignedRange ConstantSignedRange::truncate(uint32_t DstTySize) const {
  // TODO: Implement truncate.
  return ConstantSignedRange(DstTySize, !isEmptySet());
}

ConstantSignedRange
ConstantSignedRange::add(const ConstantSignedRange &Other) const {
  // TODO: Implement add.
  return ConstantSignedRange(getBitWidth(),
                             !(isEmptySet() || Other.isEmptySet()));
}

ConstantSignedRange
ConstantSignedRange::multiply(const ConstantSignedRange &Other) const {
  // TODO: Implement multiply.
  return ConstantSignedRange(getBitWidth(),
                             !(isEmptySet() || Other.isEmptySet()));
}

ConstantSignedRange
ConstantSignedRange::smax(const ConstantSignedRange &Other) const {
  // X smax Y is: range(smax(X_smin, Y_smin),
  //                    smax(X_smax, Y_smax))
  if (isEmptySet() || Other.isEmptySet())
    return ConstantSignedRange(getBitWidth(), /*isFullSet=*/false);
  if (isFullSet() || Other.isFullSet())
    return ConstantSignedRange(getBitWidth(), /*isFullSet=*/true);
  APInt NewL = APIntOps::smax(getSignedMin(), Other.getSignedMin());
  APInt NewU = APIntOps::smax(getSignedMax(), Other.getSignedMax()) + 1;
  if (NewU == NewL)
    return ConstantSignedRange(getBitWidth(), /*isFullSet=*/true);
  return ConstantSignedRange(NewL, NewU);
}

ConstantSignedRange
ConstantSignedRange::umax(const ConstantSignedRange &Other) const {
  // TODO: Implement umax.
  return ConstantSignedRange(getBitWidth(),
                             !(isEmptySet() || Other.isEmptySet()));
}

ConstantSignedRange
ConstantSignedRange::udiv(const ConstantSignedRange &Other) const {
  // TODO: Implement udiv.
  return ConstantSignedRange(getBitWidth(),
                             !(isEmptySet() || Other.isEmptySet()));
}
