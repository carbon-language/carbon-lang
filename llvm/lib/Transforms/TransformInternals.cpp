//===-- TransformInternals.cpp - Implement shared functions for transforms --=//
//
//  This file defines shared functions used by the different components of the
//  Transforms library.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Method.h"
#include "llvm/Type.h"
#include "llvm/ConstantVals.h"
#include "llvm/Analysis/Expressions.h"
#include "llvm/iOther.h"
#include <algorithm>

// TargetData Hack: Eventually we will have annotations given to us by the
// backend so that we know stuff about type size and alignments.  For now
// though, just use this, because it happens to match the model that GCC uses.
//
const TargetData TD("LevelRaise: Should be GCC though!");

// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                          BasicBlock::iterator &BI, Value *V) {
  Instruction *I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I->replaceAllUsesWith(V);

  // Remove the unneccesary instruction now...
  BIL.remove(BI);

  // Make sure to propogate a name if there is one already...
  if (I->hasName() && !V->hasName())
    V->setName(I->getName(), BIL.getParent()->getSymbolTable());

  // Remove the dead instruction now...
  delete I;
}


// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                         BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BI = BIL.insert(BI, I)+1;

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Reexamine the instruction just inserted next time around the cleanup pass
  // loop.
  --BI;
}

void ReplaceInstWithInst(Instruction *From, Instruction *To) {
  BasicBlock *BB = From->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  BasicBlock::iterator BI = find(BIL.begin(), BIL.end(), From);
  assert(BI != BIL.end() && "Inst not in it's parents BB!");
  ReplaceInstWithInst(BIL, BI, To);
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
                                std::vector<Value*> &Offsets,
                                bool StopEarly = true) {
  if (Offset == 0 && StopEarly && !Offsets.empty())
    return Ty;    // Return the leaf type

  unsigned ThisOffset;
  const Type *NextType;
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    assert(Offset < TD.getTypeSize(STy) && "Offset not in composite!");
    const StructLayout *SL = TD.getStructLayout(STy);

    // This loop terminates always on a 0 <= i < MemberOffsets.size()
    unsigned i;
    for (i = 0; i < SL->MemberOffsets.size()-1; ++i)
      if (Offset >= SL->MemberOffsets[i] && Offset <  SL->MemberOffsets[i+1])
        break;
  
    assert(Offset >= SL->MemberOffsets[i] &&
           (i == SL->MemberOffsets.size()-1 || Offset <SL->MemberOffsets[i+1]));
    
    // Make sure to save the current index...
    Offsets.push_back(ConstantUInt::get(Type::UByteTy, i));
    ThisOffset = SL->MemberOffsets[i];
    NextType = STy->getElementTypes()[i];
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    assert(Offset < TD.getTypeSize(ATy) && "Offset not in composite!");

    NextType = ATy->getElementType();
    unsigned ChildSize = TD.getTypeSize(NextType);
    Offsets.push_back(ConstantUInt::get(Type::UIntTy, Offset/ChildSize));
    ThisOffset = (Offset/ChildSize)*ChildSize;
  } else {
    Offset = 0;   // Return the offset that we were able to acheive
    return Ty;    // Return the leaf type
  }

  unsigned SubOffs = Offset - ThisOffset;
  const Type *LeafTy = getStructOffsetType(NextType, SubOffs, Offsets);
  Offset = ThisOffset + SubOffs;
  return LeafTy;
}

// ConvertableToGEP - This function returns true if the specified value V is
// a valid index into a pointer of type Ty.  If it is valid, Idx is filled in
// with the values that would be appropriate to make this a getelementptr
// instruction.  The type returned is the root type that the GEP would point to
//
const Type *ConvertableToGEP(const Type *Ty, Value *OffsetVal,
                             std::vector<Value*> &Indices,
                             BasicBlock::iterator *BI = 0) {
  const CompositeType *CompTy = dyn_cast<CompositeType>(Ty);
  if (CompTy == 0) return 0;

  // See if the cast is of an integer expression that is either a constant,
  // or a value scaled by some amount with a possible offset.
  //
  analysis::ExprType Expr = analysis::ClassifyExpression(OffsetVal);

  // Get the offset and scale now...
  unsigned Offset = 0, Scale = Expr.Var != 0;

  // Get the offset value if it exists...
  if (Expr.Offset) {
    int Val = getConstantValue(Expr.Offset);
    if (Val < 0) return false;  // Don't mess with negative offsets
    Offset = (unsigned)Val;
  }

  // Get the scale value if it exists...
  if (Expr.Scale) {
    int Val = getConstantValue(Expr.Scale);
    if (Val < 0) return false;  // Don't mess with negative scales
    Scale = (unsigned)Val;
  }
  
  // Loop over the Scale and Offset values, filling in the Indices vector for
  // our final getelementptr instruction.
  //
  const Type *NextTy = CompTy;
  do {
    if (!isa<CompositeType>(NextTy))
      return 0;  // Type must not be ready for processing...
    CompTy = cast<CompositeType>(NextTy);

    if (const StructType *StructTy = dyn_cast<StructType>(CompTy)) {
      const StructLayout *SL = TD.getStructLayout(StructTy);
      unsigned ActualOffset = Offset;
      NextTy = getStructOffsetType(StructTy, ActualOffset, Indices);
      if (StructTy == NextTy && ActualOffset == 0) return 0; // No progress.  :(
      Offset -= ActualOffset;
    } else {
      const Type *ElTy = cast<SequentialType>(CompTy)->getElementType();
      if (!ElTy->isSized()) return 0; // Type is unreasonable... escape!
      unsigned ElSize = TD.getTypeSize(ElTy);

      // See if the user is indexing into a different cell of this array...
      if (Scale && Scale >= ElSize) {
        // A scale n*ElSize might occur if we are not stepping through
        // array by one.  In this case, we will have to insert math to munge
        // the index.
        //
        unsigned ScaleAmt = Scale/ElSize;
        if (Scale-ScaleAmt*ElSize)
          return 0;  // Didn't scale by a multiple of element size, bail out
        Scale = 0;   // Scale is consumed

        unsigned Index = Offset/ElSize;       // is zero unless Offset > ElSize
        Offset -= Index*ElSize;               // Consume part of the offset

        if (BI) {              // Generate code?
          BasicBlock *BB = (**BI)->getParent();
          if (Expr.Var->getType() != Type::UIntTy) {
            CastInst *IdxCast = new CastInst(Expr.Var, Type::UIntTy);
            if (Expr.Var->hasName())
              IdxCast->setName(Expr.Var->getName()+"-idxcast");
            *BI = BB->getInstList().insert(*BI, IdxCast)+1;
            Expr.Var = IdxCast;
          }

          if (ScaleAmt && ScaleAmt != 1) {
            // If we have to scale up our index, do so now
            Value *ScaleAmtVal = ConstantUInt::get(Type::UIntTy, ScaleAmt);
            Instruction *Scaler = BinaryOperator::create(Instruction::Mul,
                                                         Expr.Var,ScaleAmtVal);
            if (Expr.Var->hasName())
              Scaler->setName(Expr.Var->getName()+"-scale");

            *BI = BB->getInstList().insert(*BI, Scaler)+1;
            Expr.Var = Scaler;
          }

          if (Index) {  // Add an offset to the index
            Value *IndexAmt = ConstantUInt::get(Type::UIntTy, Index);
            Instruction *Offseter = BinaryOperator::create(Instruction::Add,
                                                           Expr.Var, IndexAmt);
            if (Expr.Var->hasName())
              Offseter->setName(Expr.Var->getName()+"-offset");
            *BI = BB->getInstList().insert(*BI, Offseter)+1;
            Expr.Var = Offseter;
          }
        }

        Indices.push_back(Expr.Var);
      } else if (Offset >= ElSize) {
        // Calculate the index that we are entering into the array cell with
        unsigned Index = Offset/ElSize;
        Indices.push_back(ConstantUInt::get(Type::UIntTy, Index));
        Offset -= Index*ElSize;               // Consume part of the offset

      } else if (!isa<PointerType>(CompTy) || CompTy == Ty) {
        // Must be indexing a small amount into the first cell of the array
        // Just index into element zero of the array here.
        //
        Indices.push_back(ConstantUInt::get(Type::UIntTy, 0));
      } else {
        return 0;  // Hrm. wierd, can't handle this case.  Bail
      }
      NextTy = ElTy;
    }
  } while (Offset || Scale);    // Go until we're done!

  return NextTy;
}
