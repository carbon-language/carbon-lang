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
#include "llvm/Type.h"
using namespace llvm;

static ConstantIntegral *Next(ConstantIntegral *CI) {
  if (CI->getType() == Type::BoolTy)
    return CI == ConstantBool::True ? ConstantBool::False : ConstantBool::True;
      
  Constant *Result = ConstantExpr::getAdd(CI,
                                          ConstantInt::get(CI->getType(), 1));
  return cast<ConstantIntegral>(Result);
}

static bool LT(ConstantIntegral *A, ConstantIntegral *B) {
  Constant *C = ConstantExpr::getSetLT(A, B);
  assert(isa<ConstantBool>(C) && "Constant folding of integrals not impl??");
  return cast<ConstantBool>(C)->getValue();
}

static bool LTE(ConstantIntegral *A, ConstantIntegral *B) {
  Constant *C = ConstantExpr::getSetLE(A, B);
  assert(isa<ConstantBool>(C) && "Constant folding of integrals not impl??");
  return cast<ConstantBool>(C)->getValue();
}

static bool GT(ConstantIntegral *A, ConstantIntegral *B) { return LT(B, A); }

static ConstantIntegral *Min(ConstantIntegral *A, ConstantIntegral *B) {
  return LT(A, B) ? A : B;
}
static ConstantIntegral *Max(ConstantIntegral *A, ConstantIntegral *B) {
  return GT(A, B) ? A : B;
}

/// Initialize a full (the default) or empty set for the specified type.
///
ConstantRange::ConstantRange(const Type *Ty, bool Full) {
  assert(Ty->isIntegral() &&
         "Cannot make constant range of non-integral type!");
  if (Full)
    Lower = Upper = ConstantIntegral::getMaxValue(Ty);
  else
    Lower = Upper = ConstantIntegral::getMinValue(Ty);
}

/// Initialize a range to hold the single specified value.
///
ConstantRange::ConstantRange(Constant *V)
  : Lower(cast<ConstantIntegral>(V)), Upper(Next(cast<ConstantIntegral>(V))) {
}

/// Initialize a range of values explicitly... this will assert out if
/// Lower==Upper and Lower != Min or Max for its type (or if the two constants
/// have different types)
///
ConstantRange::ConstantRange(Constant *L, Constant *U)
  : Lower(cast<ConstantIntegral>(L)), Upper(cast<ConstantIntegral>(U)) {
  assert(Lower->getType() == Upper->getType() &&
         "Incompatible types for ConstantRange!");
  
  // Make sure that if L & U are equal that they are either Min or Max...
  assert((L != U || (L == ConstantIntegral::getMaxValue(L->getType()) ||
                     L == ConstantIntegral::getMinValue(L->getType()))) &&
         "Lower == Upper, but they aren't min or max for type!");
}

/// Initialize a set of values that all satisfy the condition with C.
///
ConstantRange::ConstantRange(unsigned SetCCOpcode, ConstantIntegral *C) {
  switch (SetCCOpcode) {
  default: assert(0 && "Invalid SetCC opcode to ConstantRange ctor!");
  case Instruction::SetEQ: Lower = C; Upper = Next(C); return;
  case Instruction::SetNE: Upper = C; Lower = Next(C); return;
  case Instruction::SetLT:
    Lower = ConstantIntegral::getMinValue(C->getType());
    Upper = C;
    return;
  case Instruction::SetGT:
    Lower = Next(C);
    Upper = ConstantIntegral::getMinValue(C->getType());  // Min = Next(Max)
    return;
  case Instruction::SetLE:
    Lower = ConstantIntegral::getMinValue(C->getType());
    Upper = Next(C);
    return;
  case Instruction::SetGE:
    Lower = C;
    Upper = ConstantIntegral::getMinValue(C->getType());  // Min = Next(Max)
    return;
  }
}

/// getType - Return the LLVM data type of this range.
///
const Type *ConstantRange::getType() const { return Lower->getType(); }

/// isFullSet - Return true if this set contains all of the elements possible
/// for this data-type
bool ConstantRange::isFullSet() const {
  return Lower == Upper && Lower == ConstantIntegral::getMaxValue(getType());
}
  
/// isEmptySet - Return true if this set contains no members.
///
bool ConstantRange::isEmptySet() const {
  return Lower == Upper && Lower == ConstantIntegral::getMinValue(getType());
}

/// isWrappedSet - Return true if this set wraps around the top of the range,
/// for example: [100, 8)
///
bool ConstantRange::isWrappedSet() const {
  return GT(Lower, Upper);
}

  
/// getSingleElement - If this set contains a single element, return it,
/// otherwise return null.
ConstantIntegral *ConstantRange::getSingleElement() const {
  if (Upper == Next(Lower))  // Is it a single element range?
    return Lower;
  return 0;
}

/// getSetSize - Return the number of elements in this set.
///
uint64_t ConstantRange::getSetSize() const {
  if (isEmptySet()) return 0;
  if (getType() == Type::BoolTy) {
    if (Lower != Upper)  // One of T or F in the set...
      return 1;
    return 2;            // Must be full set...
  }
  
  // Simply subtract the bounds...
  Constant *Result = ConstantExpr::getSub(Upper, Lower);
  return cast<ConstantInt>(Result)->getRawValue();
}

/// contains - Return true if the specified value is in the set.
///
bool ConstantRange::contains(ConstantInt *Val) const {
  if (Lower == Upper) {
    if (isFullSet()) return true;
    return false;
  }

  if (!isWrappedSet())
    return LTE(Lower, Val) && LT(Val, Upper);
  return LTE(Lower, Val) || LT(Val, Upper);
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
                                       const ConstantRange &RHS) {
  assert(LHS.isWrappedSet() && !RHS.isWrappedSet());

  // Check to see if we overlap on the Left side of RHS...
  //
  if (LT(RHS.getLower(), LHS.getUpper())) {
    // We do overlap on the left side of RHS, see if we overlap on the right of
    // RHS...
    if (GT(RHS.getUpper(), LHS.getLower())) {
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
    if (GT(RHS.getUpper(), LHS.getLower())) {
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
ConstantRange ConstantRange::intersectWith(const ConstantRange &CR) const {
  assert(getType() == CR.getType() && "ConstantRange types don't agree!");
  // Handle common special cases
  if (isEmptySet() || CR.isFullSet())  return *this;
  if (isFullSet()  || CR.isEmptySet()) return CR;

  if (!isWrappedSet()) {
    if (!CR.isWrappedSet()) {
      ConstantIntegral *L = Max(Lower, CR.Lower);
      ConstantIntegral *U = Min(Upper, CR.Upper);

      if (LT(L, U))  // If range isn't empty...
        return ConstantRange(L, U);
      else
        return ConstantRange(getType(), false);  // Otherwise, return empty set
    } else
      return intersect1Wrapped(CR, *this);
  } else {   // We know "this" is wrapped...
    if (!CR.isWrappedSet())
      return intersect1Wrapped(*this, CR);
    else {
      // Both ranges are wrapped...
      ConstantIntegral *L = Max(Lower, CR.Lower);
      ConstantIntegral *U = Min(Upper, CR.Upper);
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
ConstantRange ConstantRange::unionWith(const ConstantRange &CR) const {
  assert(getType() == CR.getType() && "ConstantRange types don't agree!");

  assert(0 && "Range union not implemented yet!");

  return *this;
}

/// zeroExtend - Return a new range in the specified integer type, which must
/// be strictly larger than the current type.  The returned range will
/// correspond to the possible range of values if the source range had been
/// zero extended.
ConstantRange ConstantRange::zeroExtend(const Type *Ty) const {
  assert(getLower()->getType()->getPrimitiveSize() < Ty->getPrimitiveSize() &&
         "Not a value extension");
  if (isFullSet()) {
    // Change a source full set into [0, 1 << 8*numbytes)
    unsigned SrcTySize = getLower()->getType()->getPrimitiveSize();
    return ConstantRange(Constant::getNullValue(Ty),
                         ConstantUInt::get(Ty, 1ULL << SrcTySize*8));
  }

  Constant *Lower = getLower();
  Constant *Upper = getUpper();
  if (Lower->getType()->isInteger() && !Lower->getType()->isUnsigned()) {
    // Ensure we are doing a ZERO extension even if the input range is signed.
    Lower = ConstantExpr::getCast(Lower, Ty->getUnsignedVersion());
    Upper = ConstantExpr::getCast(Upper, Ty->getUnsignedVersion());
  }

  return ConstantRange(ConstantExpr::getCast(Lower, Ty),
                       ConstantExpr::getCast(Upper, Ty));
}

/// truncate - Return a new range in the specified integer type, which must be
/// strictly smaller than the current type.  The returned range will
/// correspond to the possible range of values if the source range had been
/// truncated to the specified type.
ConstantRange ConstantRange::truncate(const Type *Ty) const {
  assert(getLower()->getType()->getPrimitiveSize() > Ty->getPrimitiveSize() &&
         "Not a value truncation");
  uint64_t Size = 1ULL << Ty->getPrimitiveSize()*8;
  if (isFullSet() || getSetSize() >= Size)
    return ConstantRange(getType());

  return ConstantRange(ConstantExpr::getCast(getLower(), Ty),
                       ConstantExpr::getCast(getUpper(), Ty));
}


/// print - Print out the bounds to a stream...
///
void ConstantRange::print(std::ostream &OS) const {
  OS << "[" << Lower << "," << Upper << " )";
}

/// dump - Allow printing from a debugger easily...
///
void ConstantRange::dump() const {
  print(std::cerr);
}
