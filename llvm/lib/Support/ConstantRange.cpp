//===-- ConstantRange.cpp - ConstantRange implementation ------------------===//
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
#include "llvm/Type.h"
#include "llvm/Instruction.h"
#include "llvm/ConstantHandling.h"

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

/// Initialize a range of values explicitly... this will assert out if
/// Lower==Upper and Lower != Min or Max for its type (or if the two constants
/// have different types)
///
ConstantRange::ConstantRange(ConstantIntegral *L,
                             ConstantIntegral *U) : Lower(L), Upper(U) {
  assert(Lower->getType() == Upper->getType() &&
         "Incompatible types for ConstantRange!");
  
  // Make sure that if L & U are equal that they are either Min or Max...
  assert((L != U || (L == ConstantIntegral::getMaxValue(L->getType()) ||
                     L == ConstantIntegral::getMinValue(L->getType()))) &&
         "Lower == Upper, but they aren't min or max for type!");
}

static ConstantIntegral *Next(ConstantIntegral *CI) {
  if (CI->getType() == Type::BoolTy)
    return CI == ConstantBool::True ? ConstantBool::False : ConstantBool::True;
      
  // Otherwise use operator+ in the ConstantHandling Library.
  Constant *Result = *ConstantInt::get(CI->getType(), 1) + *CI;
  assert(Result && "ConstantHandling not implemented for integral plus!?");
  return cast<ConstantIntegral>(Result);
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
  return (*(Constant*)Lower > *(Constant*)Upper)->getValue();
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
  Constant *Result = *(Constant*)Upper - *(Constant*)Lower;
  assert(Result && "Subtraction of constant integers not implemented?");
  return cast<ConstantInt>(Result)->getRawValue();
}




// intersect1Wrapped - This helper function is used to intersect two ranges when
// it is known that LHS is wrapped and RHS isn't.
//
static ConstantRange intersect1Wrapped(const ConstantRange &LHS,
                                       const ConstantRange &RHS) {
  assert(LHS.isWrappedSet() && !RHS.isWrappedSet());

  // Check to see if we overlap on the Left side of RHS...
  //
  if ((*(Constant*)RHS.getLower() < *(Constant*)LHS.getUpper())->getValue()) {
    // We do overlap on the left side of RHS, see if we overlap on the right of
    // RHS...
    if ((*(Constant*)RHS.getUpper() > *(Constant*)LHS.getLower())->getValue()) {
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
    if ((*(Constant*)RHS.getUpper() > *(Constant*)LHS.getLower())->getValue()) {
      // Simple overlap...
      return ConstantRange(LHS.getLower(), RHS.getUpper());
    } else {
      // No overlap...
      return ConstantRange(LHS.getType(), false);
    }
  }
}

static ConstantIntegral *Min(ConstantIntegral *A, ConstantIntegral *B) {
  if ((*(Constant*)A < *(Constant*)B)->getValue())
    return A;
  return B;
}
static ConstantIntegral *Max(ConstantIntegral *A, ConstantIntegral *B) {
  if ((*(Constant*)A > *(Constant*)B)->getValue())
    return A;
  return B;
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

      if ((*L < *U)->getValue())  // If range isn't empty...
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
