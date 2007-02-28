//===-- ConstantRange.cpp - ConstantRange implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "llvm/Support/Streams.h"
#include <ostream>
using namespace llvm;

/// Initialize a full (the default) or empty set for the specified type.
///
ConstantRange::ConstantRange(uint32_t BitWidth, bool Full) :
  Lower(BitWidth, 0), Upper(BitWidth, 0) {
  if (Full)
    Lower = Upper = APInt::getMaxValue(BitWidth);
  else
    Lower = Upper = APInt::getMinValue(BitWidth);
}

/// Initialize a range to hold the single specified value.
///
ConstantRange::ConstantRange(const APInt & V) : Lower(V), Upper(V + 1) { }

ConstantRange::ConstantRange(const APInt &L, const APInt &U) :
  Lower(L), Upper(U) {
  assert(L.getBitWidth() == U.getBitWidth() && 
         "ConstantRange with unequal bit widths");
  uint32_t BitWidth = L.getBitWidth();
  assert((L != U || (L == APInt::getMaxValue(BitWidth) ||
                     L == APInt::getMinValue(BitWidth))) &&
         "Lower == Upper, but they aren't min or max value!");
}

/// isFullSet - Return true if this set contains all of the elements possible
/// for this data-type
bool ConstantRange::isFullSet() const {
  return Lower == Upper && Lower == APInt::getMaxValue(getBitWidth());
}

/// isEmptySet - Return true if this set contains no members.
///
bool ConstantRange::isEmptySet() const {
  return Lower == Upper && Lower == APInt::getMinValue(getBitWidth());
}

/// isWrappedSet - Return true if this set wraps around the top of the range,
/// for example: [100, 8)
///
bool ConstantRange::isWrappedSet(bool isSigned) const {
  if (isSigned)
    return Lower.sgt(Upper);
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

/// contains - Return true if the specified value is in the set.
///
bool ConstantRange::contains(const APInt &V, bool isSigned) const {
  if (Lower == Upper) {
    if (isFullSet()) 
      return true;
    return false;
  }

  if (!isWrappedSet(isSigned))
    if (isSigned)
      return Lower.sle(V) && V.slt(Upper);
    else
      return Lower.ule(V) && V.ult(Upper);
  if (isSigned)
    return Lower.sle(V) || V.slt(Upper);
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
                                 const ConstantRange &RHS, bool isSigned) {
  assert(LHS.isWrappedSet(isSigned) && !RHS.isWrappedSet(isSigned));

  // Check to see if we overlap on the Left side of RHS...
  //
  bool LT = (isSigned ? RHS.Lower.slt(LHS.Upper) : RHS.Lower.ult(LHS.Upper));
  bool GT = (isSigned ? RHS.Upper.sgt(LHS.Lower) : RHS.Upper.ugt(LHS.Lower));
  if (LT) {
    // We do overlap on the left side of RHS, see if we overlap on the right of
    // RHS...
    if (GT) {
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
    if (GT) {
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
ConstantRange ConstantRange::intersectWith(const ConstantRange &CR,
                                           bool isSigned) const {
  assert(getBitWidth() == CR.getBitWidth() && 
         "ConstantRange types don't agree!");
  // Handle common special cases
  if (isEmptySet() || CR.isFullSet())  
    return *this;
  if (isFullSet()  || CR.isEmptySet()) 
    return CR;

  if (!isWrappedSet(isSigned)) {
    if (!CR.isWrappedSet(isSigned)) {
      using namespace APIntOps;
      APInt L = isSigned ? smax(Lower, CR.Lower) : umax(Lower, CR.Lower);
      APInt U = isSigned ? smin(Upper, CR.Upper) : umin(Upper, CR.Upper);

      if (isSigned ? L.slt(U) : L.ult(U)) // If range isn't empty...
        return ConstantRange(L, U);
      else
        return ConstantRange(getBitWidth(), false);// Otherwise, empty set
    } else
      return intersect1Wrapped(CR, *this, isSigned);
  } else {   // We know "this" is wrapped...
    if (!CR.isWrappedSet(isSigned))
      return intersect1Wrapped(*this, CR, isSigned);
    else {
      // Both ranges are wrapped...
      using namespace APIntOps;
      APInt L = isSigned ? smax(Lower, CR.Lower) : umax(Lower, CR.Lower);
      APInt U = isSigned ? smin(Upper, CR.Upper) : umin(Upper, CR.Upper);
      return ConstantRange(L, U);
    }
  }
  return *this;
}

/// unionWith - Return the range that results from the union of this range with
/// another range.  The resultant range is guaranteed to include the elements of
/// both sets, but may contain more.  For example, [3, 9) union [12,15) is [3,
/// 15), which includes 9, 10, and 11, which were not included in either set
/// before.
///
ConstantRange ConstantRange::unionWith(const ConstantRange &CR,
                                       bool isSigned) const {
  assert(getBitWidth() == CR.getBitWidth() && 
         "ConstantRange types don't agree!");

  assert(0 && "Range union not implemented yet!");

  return *this;
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

/// truncate - Return a new range in the specified integer type, which must be
/// strictly smaller than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// truncated to the specified type.
ConstantRange ConstantRange::truncate(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize > DstTySize && "Not a value truncation");
  APInt Size = APInt::getMaxValue(DstTySize).zext(SrcTySize);
  if (isFullSet() || getSetSize().ugt(Size))
    return ConstantRange(DstTySize);

  APInt L = Lower; L.trunc(DstTySize);
  APInt U = Upper; U.trunc(DstTySize);
  return ConstantRange(L, U);
}

/// print - Print out the bounds to a stream...
///
void ConstantRange::print(std::ostream &OS) const {
  OS << "[" << Lower.toStringSigned(10) << "," 
            << Upper.toStringSigned(10) << " )";
}

/// dump - Allow printing from a debugger easily...
///
void ConstantRange::dump() const {
  print(cerr);
}
