//===-- TransformInternals.cpp - Implement shared functions for transforms --=//
//
//  This file defines shared functions used by the different components of the
//  Transforms library.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Method.h"
#include "llvm/Type.h"

// TargetData Hack: Eventually we will have annotations given to us by the
// backend so that we know stuff about type size and alignments.  For now
// though, just use this, because it happens to match the model that GCC uses.
//
const TargetData TD("LevelRaise: Should be GCC though!");

// losslessCastableTypes - Return true if the types are bitwise equivalent.
// This predicate returns true if it is possible to cast from one type to
// another without gaining or losing precision, or altering the bits in any way.
//
bool losslessCastableTypes(const Type *T1, const Type *T2) {
  if (!T1->isPrimitiveType() && !T1->isPointerType()) return false;
  if (!T2->isPrimitiveType() && !T2->isPointerType()) return false;

  if (T1->getPrimitiveID() == T2->getPrimitiveID())
    return true;  // Handles identity cast, and cast of differing pointer types

  // Now we know that they are two differing primitive or pointer types
  switch (T1->getPrimitiveID()) {
  case Type::UByteTyID:   return T2 == Type::SByteTy;
  case Type::SByteTyID:   return T2 == Type::UByteTy;
  case Type::UShortTyID:  return T2 == Type::ShortTy;
  case Type::ShortTyID:   return T2 == Type::UShortTy;
  case Type::UIntTyID:    return T2 == Type::IntTy;
  case Type::IntTyID:     return T2 == Type::UIntTy;
  case Type::ULongTyID:
  case Type::LongTyID:
  case Type::PointerTyID:
    return T2 == Type::ULongTy || T2 == Type::LongTy ||
           T2->getPrimitiveID() == Type::PointerTyID;
  default:
    return false;  // Other types have no identity values
  }
}


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


