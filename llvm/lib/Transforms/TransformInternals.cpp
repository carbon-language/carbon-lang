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
using namespace llvm;

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
  Indices.push_back(ConstantUInt::get(Type::UIntTy, i));
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
const Type *llvm::getStructOffsetType(const Type *Ty, unsigned &Offset,
                                      std::vector<Value*> &Indices,
                                      const TargetData &TD, bool StopEarly) {
  if (Offset == 0 && StopEarly && !Indices.empty())
    return Ty;    // Return the leaf type

  uint64_t ThisOffset;
  const Type *NextType;
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    if (STy->getNumElements()) {
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
    if (ConstantSInt::isValueValidForType(Type::IntTy, Offset/ChildSize))
      Indices.push_back(ConstantSInt::get(Type::IntTy, Offset/ChildSize));
    else
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
const Type *llvm::ConvertibleToGEP(const Type *Ty, Value *OffsetVal,
                                   std::vector<Value*> &Indices,
                                   const TargetData &TD,
                                   BasicBlock::iterator *BI) {
  return 0;
}

