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
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Support/Streams.h"
#include <ostream>
using namespace llvm;

static ConstantInt *getMaxValue(const Type *Ty, bool isSigned = false) {
  if (Ty->isInteger()) {
    if (isSigned) {
      // Calculate 011111111111111...
      unsigned TypeBits = Ty->getPrimitiveSizeInBits();
      int64_t Val = INT64_MAX;             // All ones
      Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
      return ConstantInt::get(Ty, Val);
    }
    return ConstantInt::getAllOnesValue(Ty);
  }
  return 0;
}

// Static constructor to create the minimum constant for an integral type...
static ConstantInt *getMinValue(const Type *Ty, bool isSigned = false) {
  if (Ty->isInteger()) {
    if (isSigned) {
      // Calculate 1111111111000000000000
      unsigned TypeBits = Ty->getPrimitiveSizeInBits();
      int64_t Val = -1;                    // All ones
      Val <<= TypeBits-1;                  // Shift over to the right spot
      return ConstantInt::get(Ty, Val);
    }
    return ConstantInt::get(Ty, 0);
  }
  return 0;
}
static ConstantInt *Next(ConstantInt *CI) {
  Constant *Result = ConstantExpr::getAdd(CI,
                                          ConstantInt::get(CI->getType(), 1));
  return cast<ConstantInt>(Result);
}

static bool LT(ConstantInt *A, ConstantInt *B, bool isSigned) {
  Constant *C = ConstantExpr::getICmp(
    (isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT), A, B);
  assert(isa<ConstantInt>(C) && "Constant folding of integrals not impl??");
  return cast<ConstantInt>(C)->getZExtValue();
}

static bool LTE(ConstantInt *A, ConstantInt *B, bool isSigned) {
  Constant *C = ConstantExpr::getICmp(
    (isSigned ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE), A, B);
  assert(isa<ConstantInt>(C) && "Constant folding of integrals not impl??");
  return cast<ConstantInt>(C)->getZExtValue();
}

static bool GT(ConstantInt *A, ConstantInt *B, bool isSigned) { 
  return LT(B, A, isSigned); }

static ConstantInt *Min(ConstantInt *A, ConstantInt *B, 
                             bool isSigned) {
  return LT(A, B, isSigned) ? A : B;
}
static ConstantInt *Max(ConstantInt *A, ConstantInt *B,
                             bool isSigned) {
  return GT(A, B, isSigned) ? A : B;
}

/// Initialize a full (the default) or empty set for the specified type.
///
ConstantRange::ConstantRange(const Type *Ty, bool Full) {
  assert(Ty->isIntegral() &&
         "Cannot make constant range of non-integral type!");
  if (Full)
    Lower = Upper = getMaxValue(Ty);
  else
    Lower = Upper = getMinValue(Ty);
}

/// Initialize a range to hold the single specified value.
///
ConstantRange::ConstantRange(Constant *V) 
  : Lower(cast<ConstantInt>(V)), Upper(Next(cast<ConstantInt>(V))) { }

/// Initialize a range of values explicitly... this will assert out if
/// Lower==Upper and Lower != Min or Max for its type (or if the two constants
/// have different types)
///
ConstantRange::ConstantRange(Constant *L, Constant *U) 
  : Lower(cast<ConstantInt>(L)), Upper(cast<ConstantInt>(U)) {
  assert(Lower->getType() == Upper->getType() &&
         "Incompatible types for ConstantRange!");

  // Make sure that if L & U are equal that they are either Min or Max...
  assert((L != U || (L == getMaxValue(L->getType()) ||
                     L == getMinValue(L->getType())))
          && "Lower == Upper, but they aren't min or max for type!");
}

/// Initialize a set of values that all satisfy the condition with C.
///
ConstantRange::ConstantRange(unsigned short ICmpOpcode, ConstantInt *C) {
  switch (ICmpOpcode) {
  default: assert(0 && "Invalid ICmp opcode to ConstantRange ctor!");
  case ICmpInst::ICMP_EQ: Lower = C; Upper = Next(C); return;
  case ICmpInst::ICMP_NE: Upper = C; Lower = Next(C); return;
  case ICmpInst::ICMP_ULT:
    Lower = getMinValue(C->getType());
    Upper = C;
    return;
  case ICmpInst::ICMP_SLT:
    Lower = getMinValue(C->getType(), true);
    Upper = C;
    return;
  case ICmpInst::ICMP_UGT:
    Lower = Next(C);
    Upper = getMinValue(C->getType());        // Min = Next(Max)
    return;
  case ICmpInst::ICMP_SGT:
    Lower = Next(C);
    Upper = getMinValue(C->getType(), true);  // Min = Next(Max)
    return;
  case ICmpInst::ICMP_ULE:
    Lower = getMinValue(C->getType());
    Upper = Next(C);
    return;
  case ICmpInst::ICMP_SLE:
    Lower = getMinValue(C->getType(), true);
    Upper = Next(C);
    return;
  case ICmpInst::ICMP_UGE:
    Lower = C;
    Upper = getMinValue(C->getType());        // Min = Next(Max)
    return;
  case ICmpInst::ICMP_SGE:
    Lower = C;
    Upper = getMinValue(C->getType(), true);  // Min = Next(Max)
    return;
  }
}

/// getType - Return the LLVM data type of this range.
///
const Type *ConstantRange::getType() const { return Lower->getType(); }

/// isFullSet - Return true if this set contains all of the elements possible
/// for this data-type
bool ConstantRange::isFullSet() const {
  return Lower == Upper && Lower == getMaxValue(getType());
}

/// isEmptySet - Return true if this set contains no members.
///
bool ConstantRange::isEmptySet() const {
  return Lower == Upper && Lower == getMinValue(getType());
}

/// isWrappedSet - Return true if this set wraps around the top of the range,
/// for example: [100, 8)
///
bool ConstantRange::isWrappedSet(bool isSigned) const {
  return GT(Lower, Upper, isSigned);
}

/// getSingleElement - If this set contains a single element, return it,
/// otherwise return null.
ConstantInt *ConstantRange::getSingleElement() const {
  if (Upper == Next(Lower))  // Is it a single element range?
    return Lower;
  return 0;
}

/// getSetSize - Return the number of elements in this set.
///
uint64_t ConstantRange::getSetSize() const {
  if (isEmptySet()) return 0;
  if (getType() == Type::Int1Ty) {
    if (Lower != Upper)  // One of T or F in the set...
      return 1;
    return 2;            // Must be full set...
  }

  // Simply subtract the bounds...
  Constant *Result = ConstantExpr::getSub(Upper, Lower);
  return cast<ConstantInt>(Result)->getZExtValue();
}

/// contains - Return true if the specified value is in the set.
///
bool ConstantRange::contains(ConstantInt *Val, bool isSigned) const {
  if (Lower == Upper) {
    if (isFullSet()) return true;
    return false;
  }

  if (!isWrappedSet(isSigned))
    return LTE(Lower, Val, isSigned) && LT(Val, Upper, isSigned);
  return LTE(Lower, Val, isSigned) || LT(Val, Upper, isSigned);
}

/// subtract - Subtract the specified constant from the endpoints of this
/// constant range.
ConstantRange ConstantRange::subtract(ConstantInt *CI) const {
  assert(CI->getType() == getType() && getType()->isInteger() &&
         "Cannot subtract from different type range or non-integer!");
  // If the set is empty or full, don't modify the endpoints.
  if (Lower == Upper) return *this;
  return ConstantRange(ConstantExpr::getSub(Lower, CI),
                       ConstantExpr::getSub(Upper, CI));
}


// intersect1Wrapped - This helper function is used to intersect two ranges when
// it is known that LHS is wrapped and RHS isn't.
//
static ConstantRange intersect1Wrapped(const ConstantRange &LHS,
                                       const ConstantRange &RHS,
                                       bool isSigned) {
  assert(LHS.isWrappedSet(isSigned) && !RHS.isWrappedSet(isSigned));

  // Check to see if we overlap on the Left side of RHS...
  //
  if (LT(RHS.getLower(), LHS.getUpper(), isSigned)) {
    // We do overlap on the left side of RHS, see if we overlap on the right of
    // RHS...
    if (GT(RHS.getUpper(), LHS.getLower(), isSigned)) {
      // Ok, the result overlaps on both the left and right sides.  See if the
      // resultant interval will be smaller if we wrap or not...
      //
      if (LHS.getSetSize() < RHS.getSetSize())
        return LHS;
      else
        return RHS;

    } else {
      // No overlap on the right, just on the left.
      return ConstantRange(RHS.getLower(), LHS.getUpper());
    }
  } else {
    // We don't overlap on the left side of RHS, see if we overlap on the right
    // of RHS...
    if (GT(RHS.getUpper(), LHS.getLower(), isSigned)) {
      // Simple overlap...
      return ConstantRange(LHS.getLower(), RHS.getUpper());
    } else {
      // No overlap...
      return ConstantRange(LHS.getType(), false);
    }
  }
}

/// intersect - Return the range that results from the intersection of this
/// range with another range.
///
ConstantRange ConstantRange::intersectWith(const ConstantRange &CR,
                                           bool isSigned) const {
  assert(getType() == CR.getType() && "ConstantRange types don't agree!");
  // Handle common special cases
  if (isEmptySet() || CR.isFullSet())  return *this;
  if (isFullSet()  || CR.isEmptySet()) return CR;

  if (!isWrappedSet(isSigned)) {
    if (!CR.isWrappedSet(isSigned)) {
      ConstantInt *L = Max(Lower, CR.Lower, isSigned);
      ConstantInt *U = Min(Upper, CR.Upper, isSigned);

      if (LT(L, U, isSigned))  // If range isn't empty...
        return ConstantRange(L, U);
      else
        return ConstantRange(getType(), false);  // Otherwise, return empty set
    } else
      return intersect1Wrapped(CR, *this, isSigned);
  } else {   // We know "this" is wrapped...
    if (!CR.isWrappedSet(isSigned))
      return intersect1Wrapped(*this, CR, isSigned);
    else {
      // Both ranges are wrapped...
      ConstantInt *L = Max(Lower, CR.Lower, isSigned);
      ConstantInt *U = Min(Upper, CR.Upper, isSigned);
      return ConstantRange(L, U);
    }
  }
  return *this;
}

/// union - Return the range that results from the union of this range with
/// another range.  The resultant range is guaranteed to include the elements of
/// both sets, but may contain more.  For example, [3, 9) union [12,15) is [3,
/// 15), which includes 9, 10, and 11, which were not included in either set
/// before.
///
ConstantRange ConstantRange::unionWith(const ConstantRange &CR,
                                       bool isSigned) const {
  assert(getType() == CR.getType() && "ConstantRange types don't agree!");

  assert(0 && "Range union not implemented yet!");

  return *this;
}

/// zeroExtend - Return a new range in the specified integer type, which must
/// be strictly larger than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// zero extended.
ConstantRange ConstantRange::zeroExtend(const Type *Ty) const {
  unsigned SrcTySize = getLower()->getType()->getPrimitiveSizeInBits();
  assert(SrcTySize < Ty->getPrimitiveSizeInBits() && "Not a value extension");
  if (isFullSet()) {
    // Change a source full set into [0, 1 << 8*numbytes)
    return ConstantRange(Constant::getNullValue(Ty),
                         ConstantInt::get(Ty, 1ULL << SrcTySize));
  }

  Constant *Lower = getLower();
  Constant *Upper = getUpper();

  return ConstantRange(ConstantExpr::getZExt(Lower, Ty),
                       ConstantExpr::getZExt(Upper, Ty));
}

/// truncate - Return a new range in the specified integer type, which must be
/// strictly smaller than the current type.  The returned range will
/// correspond to the possible range of values as if the source range had been
/// truncated to the specified type.
ConstantRange ConstantRange::truncate(const Type *Ty) const {
  unsigned SrcTySize = getLower()->getType()->getPrimitiveSizeInBits();
  assert(SrcTySize > Ty->getPrimitiveSizeInBits() && "Not a value truncation");
  uint64_t Size = 1ULL << Ty->getPrimitiveSizeInBits();
  if (isFullSet() || getSetSize() >= Size)
    return ConstantRange(getType());

  return ConstantRange(
      ConstantExpr::getTrunc(getLower(), Ty),
      ConstantExpr::getTrunc(getUpper(), Ty));
}

/// print - Print out the bounds to a stream...
///
void ConstantRange::print(std::ostream &OS) const {
  OS << "[" << *Lower << "," << *Upper << " )";
}

/// dump - Allow printing from a debugger easily...
///
void ConstantRange::dump() const {
  print(cerr);
}
