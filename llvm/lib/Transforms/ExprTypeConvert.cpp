//===- ExprTypeConvert.cpp - Code to change an LLVM Expr Type ---------------=//
//
// This file implements the part of level raising that checks to see if it is
// possible to coerce an entire expression tree into a different type.  If
// convertable, other routines from this file will do the conversion.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Method.h"
#include "llvm/Support/STLExtras.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Optimizations/DCE.h"
#include <map>
#include <algorithm>

#include "llvm/Assembly/Writer.h"

//#define DEBUG_EXPR_CONVERT 1

static inline const Type *getTy(const Value *V, ValueTypeCache &CT) {
  ValueTypeCache::iterator I = CT.find(V);
  if (I == CT.end()) return V->getType();
  return I->second;
}


static bool OperandConvertableToType(User *U, Value *V, const Type *Ty,
                                     ValueTypeCache &ConvertedTypes);

static void ConvertOperandToType(User *U, Value *OldVal, Value *NewVal,
                                 ValueMapCache &VMC);


// ExpressionConvertableToType - Return true if it is possible
static bool ExpressionConvertableToType(Value *V, const Type *Ty,
                                        ValueTypeCache &CTMap) {
  ValueTypeCache::iterator CTMI = CTMap.find(V);
  if (CTMI != CTMap.end()) return CTMI->second == Ty;
  CTMap[V] = Ty;

  // Expressions are only convertable if all of the users of the expression can
  // have this value converted.  This makes use of the map to avoid infinite
  // recursion.
  //
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I)
    if (!OperandConvertableToType(*I, V, Ty, CTMap))
      return false;

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) {
    // It's not an instruction, check to see if it's a constant... all constants
    // can be converted to an equivalent value (except pointers, they can't be
    // const prop'd in general).
    //
    if (isa<ConstPoolVal>(V) &&
        !isa<PointerType>(V->getType()) && !isa<PointerType>(Ty)) return true;

    return false;              // Otherwise, we can't convert!
  }
  if (I->getType() == Ty) return false;  // Expression already correct type!

  switch (I->getOpcode()) {
  case Instruction::Cast:
    // We can convert the expr if the cast destination type is losslessly
    // convertable to the requested type.
    return losslessCastableTypes(Ty, I->getType());

  case Instruction::Add:
  case Instruction::Sub:
    return ExpressionConvertableToType(I->getOperand(0), Ty, CTMap) &&
           ExpressionConvertableToType(I->getOperand(1), Ty, CTMap);
  case Instruction::Shr:
    if (Ty->isSigned() != V->getType()->isSigned()) return false;
    // FALL THROUGH
  case Instruction::Shl:
    return ExpressionConvertableToType(I->getOperand(0), Ty, CTMap);

  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);
    if (LI->hasIndices()) return false;
    return ExpressionConvertableToType(LI->getPtrOperand(),
                                       PointerType::get(Ty), CTMap);
  }
  case Instruction::GetElementPtr: {
    // GetElementPtr's are directly convertable to a pointer type if they have
    // a number of zeros at the end.  Because removing these values does not
    // change the logical offset of the GEP, it is okay and fair to remove them.
    // This can change this:
    //   %t1 = getelementptr %Hosp * %hosp, ubyte 4, ubyte 0  ; <%List **>
    //   %t2 = cast %List * * %t1 to %List *
    // into
    //   %t2 = getelementptr %Hosp * %hosp, ubyte 4           ; <%List *>
    // 
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    if (!PTy) return false;

    // Check to see if there are zero elements that we can remove from the
    // index array.  If there are, check to see if removing them causes us to
    // get to the right type...
    //
    vector<ConstPoolVal*> Indices = GEP->getIndices();
    const Type *BaseType = GEP->getPtrOperand()->getType();

    while (Indices.size() &&
           cast<ConstPoolUInt>(Indices.back())->getValue() == 0) {
      Indices.pop_back();
      const Type *ElTy = GetElementPtrInst::getIndexedType(BaseType, Indices,
                                                           true);
      if (ElTy == PTy->getValueType())
        return true;  // Found a match!!
    }
    break;   // No match, maybe next time.
  }
  }
  return false;
}




static Value *ConvertExpressionToType(Value *V, const Type *Ty,
                                      ValueMapCache &VMC) {
  ValueMapCache::ExprMapTy::iterator VMCI = VMC.ExprMap.find(V);
  if (VMCI != VMC.ExprMap.end())
    return VMCI->second;

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0)
    if (ConstPoolVal *CPV = cast<ConstPoolVal>(V)) {
      // Constants are converted by constant folding the cast that is required.
      // We assume here that all casts are implemented for constant prop.
      Value *Result = opt::ConstantFoldCastInstruction(CPV, Ty);
      if (!Result) cerr << "Couldn't fold " << CPV << " to " << Ty << endl;
      assert(Result && "ConstantFoldCastInstruction Failed!!!");

      // Add the instruction to the expression map
      VMC.ExprMap[V] = Result;
      return Result;
    }


  BasicBlock *BB = I->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  string Name = I->getName();  if (!Name.empty()) I->setName("");
  Instruction *Res;     // Result of conversion

  //cerr << endl << endl << "Type:\t" << Ty << "\nInst: " << I << "BB Before: " << BB << endl;

  switch (I->getOpcode()) {
  case Instruction::Cast:
    Res = new CastInst(I->getOperand(0), Ty, Name);
    break;
    
  case Instruction::Add:
  case Instruction::Sub:
    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                             ConvertExpressionToType(I->getOperand(0), Ty, VMC),
                             ConvertExpressionToType(I->getOperand(1), Ty, VMC),
                                 Name);
    break;

  case Instruction::Shl:
  case Instruction::Shr:
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(),
                        ConvertExpressionToType(I->getOperand(0), Ty, VMC),
                        I->getOperand(1), Name);
    break;

  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);
    assert(!LI->hasIndices());
    Res = new LoadInst(ConvertExpressionToType(LI->getPtrOperand(),
                                               PointerType::get(Ty), VMC),
                       Name);
    break;
  }

  case Instruction::GetElementPtr: {
    // GetElementPtr's are directly convertable to a pointer type if they have
    // a number of zeros at the end.  Because removing these values does not
    // change the logical offset of the GEP, it is okay and fair to remove them.
    // This can change this:
    //   %t1 = getelementptr %Hosp * %hosp, ubyte 4, ubyte 0  ; <%List **>
    //   %t2 = cast %List * * %t1 to %List *
    // into
    //   %t2 = getelementptr %Hosp * %hosp, ubyte 4           ; <%List *>
    // 
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);

    // Check to see if there are zero elements that we can remove from the
    // index array.  If there are, check to see if removing them causes us to
    // get to the right type...
    //
    vector<ConstPoolVal*> Indices = GEP->getIndices();
    const Type *BaseType = GEP->getPtrOperand()->getType();
    const Type *PVTy = cast<PointerType>(Ty)->getValueType();
    Res = 0;
    while (Indices.size() &&
           cast<ConstPoolUInt>(Indices.back())->getValue() == 0) {
      Indices.pop_back();
      if (GetElementPtrInst::getIndexedType(BaseType, Indices, true) == PVTy) {
        if (Indices.size() == 0) {
          Res = new CastInst(GEP->getPtrOperand(), BaseType); // NOOP
        } else {
          Res = new GetElementPtrInst(GEP->getPtrOperand(), Indices, Name);
        }
        break;
      }
    }
    assert(Res && "Didn't find match!");
    break;   // No match, maybe next time.
  }

  default:
    assert(0 && "Expression convertable, but don't know how to convert?");
    return 0;
  }

  BasicBlock::iterator It = find(BIL.begin(), BIL.end(), I);
  assert(It != BIL.end() && "Instruction not in own basic block??");
  BIL.insert(It, Res);

  // Add the instruction to the expression map
  VMC.ExprMap[I] = Res;

  // Expressions are only convertable if all of the users of the expression can
  // have this value converted.  This makes use of the map to avoid infinite
  // recursion.
  //
  unsigned NumUses = I->use_size();
  for (unsigned It = 0; It < NumUses; ) {
    unsigned OldSize = NumUses;
    ConvertOperandToType(*(I->use_begin()+It), I, Res, VMC);
    NumUses = I->use_size();
    if (NumUses == OldSize) ++It;
  }

#ifdef DEBUG_EXPR_CONVERT
  cerr << "ExpIn: " << I << "ExpOut: " << Res;
#endif

  return Res;
}



// RetValConvertableToType - Return true if it is possible
bool RetValConvertableToType(Value *V, const Type *Ty,
                             ValueTypeCache &ConvertedTypes) {
  ValueTypeCache::iterator I = ConvertedTypes.find(V);
  if (I != ConvertedTypes.end()) return I->second == Ty;
  ConvertedTypes[V] = Ty;

  // It is safe to convert the specified value to the specified type IFF all of
  // the uses of the value can be converted to accept the new typed value.
  //
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I)
    if (!OperandConvertableToType(*I, V, Ty, ConvertedTypes))
      return false;

  return true;
}







// OperandConvertableToType - Return true if it is possible to convert operand
// V of User (instruction) U to the specified type.  This is true iff it is
// possible to change the specified instruction to accept this.  CTMap is a map
// of converted types, so that circular definitions will see the future type of
// the expression, not the static current type.
//
static bool OperandConvertableToType(User *U, Value *V, const Type *Ty,
                                     ValueTypeCache &CTMap) {
  assert(V->getType() != Ty &&
         "OperandConvertableToType: Operand is already right type!");
  Instruction *I = dyn_cast<Instruction>(U);
  if (I == 0) return false;              // We can't convert!

  switch (I->getOpcode()) {
  case Instruction::Cast:
    assert(I->getOperand(0) == V);
    // We can convert the expr if the cast destination type is losslessly
    // convertable to the requested type.
    return losslessCastableTypes(Ty, I->getOperand(0)->getType());

  case Instruction::Add:
  case Instruction::Sub: {
    Value *OtherOp = I->getOperand((V == I->getOperand(0)) ? 1 : 0);
    return RetValConvertableToType(I, Ty, CTMap) &&
           ExpressionConvertableToType(OtherOp, Ty, CTMap);
  }
  case Instruction::SetEQ:
  case Instruction::SetNE: {
    Value *OtherOp = I->getOperand((V == I->getOperand(0)) ? 1 : 0);
    return ExpressionConvertableToType(OtherOp, Ty, CTMap);
  }
  case Instruction::Shr:
    if (Ty->isSigned() != V->getType()->isSigned()) return false;
    // FALL THROUGH
  case Instruction::Shl:
    assert(I->getOperand(0) == V);
    return RetValConvertableToType(I, Ty, CTMap);

  case Instruction::Load:
    assert(I->getOperand(0) == V);
    if (const PointerType *PT = dyn_cast<PointerType>(Ty)) {
      LoadInst *LI = cast<LoadInst>(I);
      if (LI->hasIndices() || 
          TD.getTypeSize(PT->getValueType()) != TD.getTypeSize(LI->getType()))
        return false;

      return RetValConvertableToType(LI, PT->getValueType(), CTMap);
    }
    return false;

  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(I);
    if (SI->hasIndices()) return false;

    if (V == I->getOperand(0)) {
      // Can convert the store if we can convert the pointer operand to match
      // the new  value type...
      return ExpressionConvertableToType(I->getOperand(1), PointerType::get(Ty),
                                         CTMap);
    } else if (const PointerType *PT = dyn_cast<PointerType>(Ty)) {
      if (isa<ArrayType>(PT->getValueType()))
        return false;  // Avoid getDataSize on unsized array type!
      assert(V == I->getOperand(1));

      // Must move the same amount of data...
      if (TD.getTypeSize(PT->getValueType()) != 
          TD.getTypeSize(I->getOperand(0)->getType())) return false;

      // Can convert store if the incoming value is convertable...
      return ExpressionConvertableToType(I->getOperand(0), PT->getValueType(),
                                         CTMap);
    }
    return false;
  }


#if 0
  case Instruction::GetElementPtr: {
    // GetElementPtr's are directly convertable to a pointer type if they have
    // a number of zeros at the end.  Because removing these values does not
    // change the logical offset of the GEP, it is okay and fair to remove them.
    // This can change this:
    //   %t1 = getelementptr %Hosp * %hosp, ubyte 4, ubyte 0  ; <%List **>
    //   %t2 = cast %List * * %t1 to %List *
    // into
    //   %t2 = getelementptr %Hosp * %hosp, ubyte 4           ; <%List *>
    // 
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    if (!PTy) return false;

    // Check to see if there are zero elements that we can remove from the
    // index array.  If there are, check to see if removing them causes us to
    // get to the right type...
    //
    vector<ConstPoolVal*> Indices = GEP->getIndices();
    const Type *BaseType = GEP->getPtrOperand()->getType();

    while (Indices.size() &&
           cast<ConstPoolUInt>(Indices.back())->getValue() == 0) {
      Indices.pop_back();
      const Type *ElTy = GetElementPtrInst::getIndexedType(BaseType, Indices,
                                                           true);
      if (ElTy == PTy->getValueType())
        return true;  // Found a match!!
    }
    break;   // No match, maybe next time.
  }
#endif
  }
  return false;
}


void ConvertUsersType(Value *V, Value *NewVal, ValueMapCache &VMC) {
  unsigned NumUses = V->use_size();
  for (unsigned It = 0; It < NumUses; ) {
    unsigned OldSize = NumUses;
    ConvertOperandToType(*(V->use_begin()+It), V, NewVal, VMC);
    NumUses = V->use_size();
    if (NumUses == OldSize) ++It;
  }

  if (NumUses == 0)
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      BasicBlock *BB = I->getParent();

      // Now we just need to remove the old instruction so we don't get infinite
      // loops.  Note that we cannot use DCE because DCE won't remove a store
      // instruction, for example.
      //
#ifdef DEBUG_EXPR_CONVERT
      cerr << "DELETING: " << (void*)I << " " << I;
#endif
      BB->getInstList().remove(I);
      delete I;
    }
}



static void ConvertOperandToType(User *U, Value *OldVal, Value *NewVal,
                                 ValueMapCache &VMC) {
  if (isa<ValueHandle>(U)) return;  // Valuehandles don't let go of operands...

  if (VMC.OperandsMapped.count(make_pair(U, OldVal))) return;

  VMC.OperandsMapped.insert(make_pair(U, OldVal));

  Instruction *I = cast<Instruction>(U);  // Only Instructions convertable

  BasicBlock *BB = I->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  string Name = I->getName();  if (!Name.empty()) I->setName("");
  Instruction *Res;     // Result of conversion

  //cerr << endl << endl << "Type:\t" << Ty << "\nInst: " << I << "BB Before: " << BB << endl;

  // Prevent I from being removed...
  ValueHandle IHandle(I);
#ifdef DEBUG_EXPR_CONVERT
  cerr << "VH AQUIRING: " << I;
#endif

  switch (I->getOpcode()) {
  case Instruction::Cast:
    assert(I->getOperand(0) == OldVal);
    Res = new CastInst(NewVal, I->getType(), Name);
    break;

  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::SetEQ:
  case Instruction::SetNE: {
    unsigned OtherIdx = (OldVal == I->getOperand(0)) ? 1 : 0;
    Value *OtherOp    = I->getOperand(OtherIdx);
    Value *NewOther   = ConvertExpressionToType(OtherOp, NewVal->getType(),VMC);

    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                                 OtherIdx == 0 ? NewOther : NewVal,
                                 OtherIdx == 1 ? NewOther : NewVal,
                                 Name);
    break;
  }
  case Instruction::Shl:
  case Instruction::Shr:
    assert(I->getOperand(0) == OldVal);
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(), NewVal,
                        I->getOperand(1), Name);
    break;

  case Instruction::Load:
    assert(I->getOperand(0) == OldVal);
    Res = new LoadInst(NewVal, Name);
    break;

  case Instruction::Store: {
    if (I->getOperand(0) == OldVal) {  // Replace the source value
      Value *NewPtr =
        ConvertExpressionToType(I->getOperand(1),
                                PointerType::get(NewVal->getType()), VMC);
      Res = new StoreInst(NewVal, NewPtr);
    } else {                           // Replace the source pointer
      const Type *ValType =cast<PointerType>(NewVal->getType())->getValueType();
      Value *NewV = ConvertExpressionToType(I->getOperand(0), ValType, VMC);
      Res = new StoreInst(NewV, NewVal);
    }
    break;
  }

#if 0
  case Instruction::GetElementPtr: {
    // GetElementPtr's are directly convertable to a pointer type if they have
    // a number of zeros at the end.  Because removing these values does not
    // change the logical offset of the GEP, it is okay and fair to remove them.
    // This can change this:
    //   %t1 = getelementptr %Hosp * %hosp, ubyte 4, ubyte 0  ; <%List **>
    //   %t2 = cast %List * * %t1 to %List *
    // into
    //   %t2 = getelementptr %Hosp * %hosp, ubyte 4           ; <%List *>
    // 
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);

    // Check to see if there are zero elements that we can remove from the
    // index array.  If there are, check to see if removing them causes us to
    // get to the right type...
    //
    vector<ConstPoolVal*> Indices = GEP->getIndices();
    const Type *BaseType = GEP->getPtrOperand()->getType();
    const Type *PVTy = cast<PointerType>(Ty)->getValueType();
    Res = 0;
    while (Indices.size() &&
           cast<ConstPoolUInt>(Indices.back())->getValue() == 0) {
      Indices.pop_back();
      if (GetElementPtrInst::getIndexedType(BaseType, Indices, true) == PVTy) {
        if (Indices.size() == 0) {
          Res = new CastInst(GEP->getPtrOperand(), BaseType); // NOOP
        } else {
          Res = new GetElementPtrInst(GEP->getPtrOperand(), Indices, Name);
        }
        break;
      }
    }
    assert(Res && "Didn't find match!");
    break;   // No match, maybe next time.
  }
#endif

  default:
    assert(0 && "Expression convertable, but don't know how to convert?");
    return;
  }

  BasicBlock::iterator It = find(BIL.begin(), BIL.end(), I);
  assert(It != BIL.end() && "Instruction not in own basic block??");
  BIL.insert(It, Res);   // Keep It pointing to old instruction

#ifdef DEBUG_EXPR_CONVERT
  cerr << "In: " << I << "Out: " << Res;
#endif

  if (I->getType() != Res->getType())
    ConvertUsersType(I, Res, VMC);
  else {
    for (unsigned It = 0; It < I->use_size(); ) {
      User *Use = *(I->use_begin()+It);
      if (isa<ValueHandle>(Use))            // Don't remove ValueHandles!
        ++It;
      else
        Use->replaceUsesOfWith(I, Res);
    }

    if (I->use_size() == 0) {
      // Now we just need to remove the old instruction so we don't get infinite
      // loops.  Note that we cannot use DCE because DCE won't remove a store
      // instruction, for example.
      //
#ifdef DEBUG_EXPR_CONVERT
      cerr << "DELETING: " << (void*)I << " " << I;
#endif
      BIL.remove(I);
      delete I;
    } else {
      for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI)
        assert(isa<ValueHandle>((Value*)*UI) && "Uses of Instruction remain!!!");
    }
  }
}


ValueHandle::~ValueHandle() {
  if (Operands[0]->use_size() == 1) {
    Value *V = Operands[0];
    Operands.clear();   // Drop use!

    // Now we just need to remove the old instruction so we don't get infinite
    // loops.  Note that we cannot use DCE because DCE won't remove a store
    // instruction, for example.
    //
    Instruction *I = cast<Instruction>(V);
    assert(I->getParent() && "Inst not in basic block!");

#ifdef DEBUG_EXPR_CONVERT
    cerr << "VH DELETING: " << (void*)I << " " << I;
#endif
    I->getParent()->getInstList().remove(I);
    delete I;
  } else {
#ifdef DEBUG_EXPR_CONVERT
    cerr << "VH RELEASING: " << Operands[0];
#endif
  }
}
