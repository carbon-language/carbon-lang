//===- TransformInternals.cpp - Implement shared functions for transforms -===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file defines shared functions used by the different components of the
//  Transforms library.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Expressions.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"

namespace llvm {

static const Type *getStructOffsetStep(const StructType *STy, uint64_t &Offset,
                                       std::vector<Value*> &Indices,
                                       const TargetData &TD) {
  assert(Offset < TD.getTypeSize(STy) && "Offset not in composite!");
  const StructLayout *SL = TD.getStructLayout(STy);

  // This loop terminates always on a 0 <= i < MemberOffsets.size()
  unsigned i;
  for (i = 0; i < SL->MemberOffsets.size()-1; ++i)
    if (Offset >= SL->MemberOffsets[i] && Offset < SL->MemberOffsets[i+1])
      break;
  
  assert(Offset >= SL->MemberOffsets[i] &&
         (i == SL->MemberOffsets.size()-1 || Offset < SL->MemberOffsets[i+1]));
  
  // Make sure to save the current index...
  Indices.push_back(ConstantUInt::get(Type::UByteTy, i));
  Offset = SL->MemberOffsets[i];
  return STy->getContainedType(i);
}


// getStructOffsetType - Return a vector of offsets that are to be used to index
// into the specified struct type to get as close as possible to index as we
// can.  Note that it is possible that we cannot get exactly to Offset, in which
// case we update offset to be the offset we actually obtained.  The resultant
// leaf type is returned.
//
// If StopEarly is set to true (the default), the first object with the
// specified type is returned, even if it is a struct type itself.  In this
// case, this routine will not drill down to the leaf type.  Set StopEarly to
// false if you want a leaf
//
const Type *getStructOffsetType(const Type *Ty, unsigned &Offset,
                                std::vector<Value*> &Indices,
                                const TargetData &TD, bool StopEarly) {
  if (Offset == 0 && StopEarly && !Indices.empty())
    return Ty;    // Return the leaf type

  uint64_t ThisOffset;
  const Type *NextType;
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    if (STy->getElementTypes().empty()) {
      Offset = 0;
      return STy;
    }

    ThisOffset = Offset;
    NextType = getStructOffsetStep(STy, ThisOffset, Indices, TD);
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    assert(Offset == 0 || Offset < TD.getTypeSize(ATy) &&
           "Offset not in composite!");

    NextType = ATy->getElementType();
    unsigned ChildSize = TD.getTypeSize(NextType);
    Indices.push_back(ConstantSInt::get(Type::LongTy, Offset/ChildSize));
    ThisOffset = (Offset/ChildSize)*ChildSize;
  } else {
    Offset = 0;   // Return the offset that we were able to achieve
    return Ty;    // Return the leaf type
  }

  unsigned SubOffs = Offset - ThisOffset;
  const Type *LeafTy = getStructOffsetType(NextType, SubOffs,
                                           Indices, TD, StopEarly);
  Offset = ThisOffset + SubOffs;
  return LeafTy;
}

// ConvertibleToGEP - This function returns true if the specified value V is
// a valid index into a pointer of type Ty.  If it is valid, Idx is filled in
// with the values that would be appropriate to make this a getelementptr
// instruction.  The type returned is the root type that the GEP would point to
//
const Type *ConvertibleToGEP(const Type *Ty, Value *OffsetVal,
                             std::vector<Value*> &Indices,
                             const TargetData &TD,
                             BasicBlock::iterator *BI) {
  const CompositeType *CompTy = dyn_cast<CompositeType>(Ty);
  if (CompTy == 0) return 0;

  // See if the cast is of an integer expression that is either a constant,
  // or a value scaled by some amount with a possible offset.
  //
  ExprType Expr = ClassifyExpr(OffsetVal);

  // Get the offset and scale values if they exists...
  // A scale of zero with Expr.Var != 0 means a scale of 1.
  //
  int64_t Offset = Expr.Offset ? getConstantValue(Expr.Offset) : 0;
  int64_t Scale  = Expr.Scale  ? getConstantValue(Expr.Scale)  : 0;

  if (Expr.Var && Scale == 0) Scale = 1;   // Scale != 0 if Expr.Var != 0
 
  // Loop over the Scale and Offset values, filling in the Indices vector for
  // our final getelementptr instruction.
  //
  const Type *NextTy = CompTy;
  do {
    if (!isa<CompositeType>(NextTy))
      return 0;  // Type must not be ready for processing...
    CompTy = cast<CompositeType>(NextTy);

    if (const StructType *StructTy = dyn_cast<StructType>(CompTy)) {
      // Step into the appropriate element of the structure...
      uint64_t ActualOffset = (Offset < 0) ? 0 : (uint64_t)Offset;
      NextTy = getStructOffsetStep(StructTy, ActualOffset, Indices, TD);
      Offset -= ActualOffset;
    } else {
      const Type *ElTy = cast<SequentialType>(CompTy)->getElementType();
      if (!ElTy->isSized() || (isa<PointerType>(CompTy) && !Indices.empty()))
        return 0; // Type is unreasonable... escape!
      unsigned ElSize = TD.getTypeSize(ElTy);
      if (ElSize == 0) return 0;   // Avoid division by zero...
      int64_t ElSizeS = ElSize;

      // See if the user is indexing into a different cell of this array...
      if (Scale && (Scale >= ElSizeS || -Scale >= ElSizeS)) {
        // A scale n*ElSize might occur if we are not stepping through
        // array by one.  In this case, we will have to insert math to munge
        // the index.
        //
        int64_t ScaleAmt = Scale/ElSizeS;
        if (Scale-ScaleAmt*ElSizeS)
          return 0;  // Didn't scale by a multiple of element size, bail out
        Scale = 0;   // Scale is consumed

        int64_t Index = Offset/ElSize;        // is zero unless Offset > ElSize
        Offset -= Index*ElSize;               // Consume part of the offset

        if (BI) {              // Generate code?
          BasicBlock *BB = (*BI)->getParent();
          if (Expr.Var->getType() != Type::LongTy)
            Expr.Var = new CastInst(Expr.Var, Type::LongTy,
                                    Expr.Var->getName()+"-idxcast", *BI);

          if (ScaleAmt && ScaleAmt != 1) {
            // If we have to scale up our index, do so now
            Value *ScaleAmtVal = ConstantSInt::get(Type::LongTy, ScaleAmt);
            Expr.Var = BinaryOperator::create(Instruction::Mul, Expr.Var,
                                              ScaleAmtVal,
                                              Expr.Var->getName()+"-scale",*BI);
          }

          if (Index) {  // Add an offset to the index
            Value *IndexAmt = ConstantSInt::get(Type::LongTy, Index);
            Expr.Var = BinaryOperator::create(Instruction::Add, Expr.Var,
                                              IndexAmt,
                                              Expr.Var->getName()+"-offset",
                                              *BI);
          }
        }

        Indices.push_back(Expr.Var);
        Expr.Var = 0;
      } else if (Offset >= (int64_t)ElSize || -Offset >= (int64_t)ElSize) {
        // Calculate the index that we are entering into the array cell with
        uint64_t Index = Offset/ElSize;
        Indices.push_back(ConstantSInt::get(Type::LongTy, Index));
        Offset -= (int64_t)(Index*ElSize);        // Consume part of the offset

      } else if (isa<ArrayType>(CompTy) || Indices.empty()) {
        // Must be indexing a small amount into the first cell of the array
        // Just index into element zero of the array here.
        //
        Indices.push_back(ConstantSInt::get(Type::LongTy, 0));
      } else {
        return 0;  // Hrm. wierd, can't handle this case.  Bail
      }
      NextTy = ElTy;
    }
  } while (Offset || Scale);    // Go until we're done!

  return NextTy;
}

} // End llvm namespace
