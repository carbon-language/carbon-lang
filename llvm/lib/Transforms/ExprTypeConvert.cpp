//===- ExprTypeConvert.cpp - Code to change an LLVM Expr Type ---------------=//
//
// This file implements the part of level raising that checks to see if it is
// possible to coerce an entire expression tree into a different type.  If
// convertable, other routines from this file will do the conversion.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Method.h"
#include "llvm/iOther.h"
#include "llvm/iPHINode.h"
#include "llvm/iMemory.h"
#include "llvm/ConstantVals.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Optimizations/DCE.h"
#include "llvm/Analysis/Expressions.h"
#include "Support/STLExtras.h"
#include <map>
#include <algorithm>

#include "llvm/Assembly/Writer.h"

//#define DEBUG_EXPR_CONVERT 1

static bool OperandConvertableToType(User *U, Value *V, const Type *Ty,
                                     ValueTypeCache &ConvertedTypes);

static void ConvertOperandToType(User *U, Value *OldVal, Value *NewVal,
                                 ValueMapCache &VMC);

// AllIndicesZero - Return true if all of the indices of the specified memory
// access instruction are zero, indicating an effectively nil offset to the 
// pointer value.
//
static bool AllIndicesZero(const MemAccessInst *MAI) {
  for (User::const_op_iterator S = MAI->idx_begin(), E = MAI->idx_end();
       S != E; ++S)
    if (!isa<Constant>(*S) || !cast<Constant>(*S)->isNullValue())
      return false;
  return true;
}

static unsigned getBaseTypeSize(const Type *Ty) {
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty))
    if (ATy->isUnsized())
      return getBaseTypeSize(ATy->getElementType());
  return TD.getTypeSize(Ty);
}


// Peephole Malloc instructions: we take a look at the use chain of the
// malloc instruction, and try to find out if the following conditions hold:
//   1. The malloc is of the form: 'malloc [sbyte], uint <constant>'
//   2. The only users of the malloc are cast & add instructions
//   3. Of the cast instructions, there is only one destination pointer type
//      [RTy] where the size of the pointed to object is equal to the number
//      of bytes allocated.
//
// If these conditions hold, we convert the malloc to allocate an [RTy]
// element.  TODO: This comment is out of date WRT arrays
//
static bool MallocConvertableToType(MallocInst *MI, const Type *Ty,
                                    ValueTypeCache &CTMap) {
  if (!MI->isArrayAllocation() ||            // No array allocation?
      !isa<PointerType>(Ty)) return false;   // Malloc always returns pointers

  // Deal with the type to allocate, not the pointer type...
  Ty = cast<PointerType>(Ty)->getElementType();

  // Analyze the number of bytes allocated...
  analysis::ExprType Expr = analysis::ClassifyExpression(MI->getArraySize());

  // Must have a scale or offset to analyze it...
  if (!Expr.Offset && !Expr.Scale) return false;

  if (Expr.Offset && (Expr.Scale || Expr.Var)) {
    // This is wierd, shouldn't happen, but if it does, I wanna know about it!
    cerr << "LevelRaise.cpp: Crazy allocation detected!\n";
    return false;    
  }

  // Get the number of bytes allocated...
  int SizeVal = getConstantValue(Expr.Offset ? Expr.Offset : Expr.Scale);
  if (SizeVal <= 0) {
    cerr << "malloc of a negative number???\n";
    return false;
  }
  unsigned Size = (unsigned)SizeVal;
  unsigned ReqTypeSize = getBaseTypeSize(Ty);

  // Does the size of the allocated type match the number of bytes
  // allocated?
  //
  if (ReqTypeSize == Size)
    return true;

  // If not, it's possible that an array of constant size is being allocated.
  // In this case, the Size will be a multiple of the data size.
  //
  if (!Expr.Offset) return false;  // Offset must be set, not scale...

#if 1
  return false;
#else   // THIS CAN ONLY BE RUN VERY LATE, after several passes to make sure
        // things are adequately raised!
  // See if the allocated amount is a multiple of the type size...
  if (Size/ReqTypeSize*ReqTypeSize != Size)
    return false;   // Nope.

  // Unfortunately things tend to be powers of two, so there may be
  // many false hits.  We don't want to optimistically assume that we
  // have the right type on the first try, so scan the use list of the
  // malloc instruction, looking for the cast to the biggest type...
  //
  for (Value::use_iterator I = MI->use_begin(), E = MI->use_end(); I != E; ++I)
    if (CastInst *CI = dyn_cast<CastInst>(*I))
      if (const PointerType *PT = 
          dyn_cast<PointerType>(CI->getOperand(0)->getType()))
        if (getBaseTypeSize(PT->getElementType()) > ReqTypeSize)
          return false;     // We found a type bigger than this one!
  
  return true;
#endif
}

static Instruction *ConvertMallocToType(MallocInst *MI, const Type *Ty,
                                        const string &Name, ValueMapCache &VMC){
  BasicBlock *BB = MI->getParent();
  BasicBlock::iterator It = BB->end();

  // Analyze the number of bytes allocated...
  analysis::ExprType Expr = analysis::ClassifyExpression(MI->getArraySize());

  const PointerType *AllocTy = cast<PointerType>(Ty);
  const Type *ElType = AllocTy->getElementType();

  if (Expr.Var && !isa<ArrayType>(ElType)) {
    ElType = ArrayType::get(AllocTy->getElementType());
    AllocTy = PointerType::get(ElType);
  }

  // If the array size specifier is not an unsigned integer, insert a cast now.
  if (Expr.Var && Expr.Var->getType() != Type::UIntTy) {
    It = find(BB->getInstList().begin(), BB->getInstList().end(), MI);
    CastInst *SizeCast = new CastInst(Expr.Var, Type::UIntTy);
    It = BB->getInstList().insert(It, SizeCast)+1;
    Expr.Var = SizeCast;
  }

  // Check to see if they are allocating a constant sized array of a type...
#if 0   // THIS CAN ONLY BE RUN VERY LATE
  if (!Expr.Var) {
    unsigned OffsetAmount  = (unsigned)getConstantValue(Expr.Offset);
    unsigned DataSize = TD.getTypeSize(ElType);
    
    if (OffsetAmount > DataSize) // Allocate a sized array amount...
      Expr.Var = ConstantUInt::get(Type::UIntTy, OffsetAmount/DataSize);
  }
#endif

  Instruction *NewI = new MallocInst(AllocTy, Expr.Var, Name);

  if (AllocTy != Ty) { // Create a cast instruction to cast it to the correct ty
    if (It == BB->end())
      It = find(BB->getInstList().begin(), BB->getInstList().end(), MI);
                
    // Insert the new malloc directly into the code ourselves
    assert(It != BB->getInstList().end());
    It = BB->getInstList().insert(It, NewI)+1;

    // Return the cast as the value to use...
    NewI = new CastInst(NewI, Ty);
  }

  return NewI;
}


// ExpressionConvertableToType - Return true if it is possible
bool ExpressionConvertableToType(Value *V, const Type *Ty,
                                 ValueTypeCache &CTMap) {
  if (V->getType() == Ty) return true;  // Expression already correct type!

  // Expression type must be holdable in a register.
  if (!isFirstClassType(Ty))
    return false;
  
  ValueTypeCache::iterator CTMI = CTMap.find(V);
  if (CTMI != CTMap.end()) return CTMI->second == Ty;

  CTMap[V] = Ty;

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) {
    // It's not an instruction, check to see if it's a constant... all constants
    // can be converted to an equivalent value (except pointers, they can't be
    // const prop'd in general).  We just ask the constant propogator to see if
    // it can convert the value...
    //
    if (Constant *CPV = dyn_cast<Constant>(V))
      if (opt::ConstantFoldCastInstruction(CPV, Ty))
        return true;  // Don't worry about deallocating, it's a constant.

    return false;              // Otherwise, we can't convert!
  }

  switch (I->getOpcode()) {
  case Instruction::Cast:
    // We can convert the expr if the cast destination type is losslessly
    // convertable to the requested type.
    if (!Ty->isLosslesslyConvertableTo(I->getType())) return false;
#if 1
    // We also do not allow conversion of a cast that casts from a ptr to array
    // of X to a *X.  For example: cast [4 x %List *] * %val to %List * *
    //
    if (PointerType *SPT = dyn_cast<PointerType>(I->getOperand(0)->getType()))
      if (PointerType *DPT = dyn_cast<PointerType>(I->getType()))
        if (ArrayType *AT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (AT->getElementType() == DPT->getElementType())
            return false;
#endif
    break;

  case Instruction::Add:
  case Instruction::Sub:
    if (!ExpressionConvertableToType(I->getOperand(0), Ty, CTMap) ||
        !ExpressionConvertableToType(I->getOperand(1), Ty, CTMap))
      return false;
    break;
  case Instruction::Shr:
    if (Ty->isSigned() != V->getType()->isSigned()) return false;
    // FALL THROUGH
  case Instruction::Shl:
    if (!ExpressionConvertableToType(I->getOperand(0), Ty, CTMap))
      return false;
    break;

  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);
    if (LI->hasIndices() && !AllIndicesZero(LI)) {
      // We can't convert a load expression if it has indices... unless they are
      // all zero.
      return false;
    }

    if (!ExpressionConvertableToType(LI->getPointerOperand(),
                                     PointerType::get(Ty), CTMap))
      return false;
    break;                                     
  }
  case Instruction::PHINode: {
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i)
      if (!ExpressionConvertableToType(PN->getIncomingValue(i), Ty, CTMap))
        return false;
    break;
  }

  case Instruction::Malloc:
    if (!MallocConvertableToType(cast<MallocInst>(I), Ty, CTMap))
      return false;
    break;

#if 1
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
    vector<Value*> Indices = GEP->copyIndices();
    const Type *BaseType = GEP->getPointerOperand()->getType();
    const Type *ElTy = 0;

    while (!Indices.empty() && isa<ConstantUInt>(Indices.back()) &&
           cast<ConstantUInt>(Indices.back())->getValue() == 0) {
      Indices.pop_back();
      ElTy = GetElementPtrInst::getIndexedType(BaseType, Indices,
                                                           true);
      if (ElTy == PTy->getElementType())
        break;  // Found a match!!
      ElTy = 0;
    }

    if (ElTy) break;
    return false;   // No match, maybe next time.
  }
#endif

  default:
    return false;
  }

  // Expressions are only convertable if all of the users of the expression can
  // have this value converted.  This makes use of the map to avoid infinite
  // recursion.
  //
  for (Value::use_iterator It = I->use_begin(), E = I->use_end(); It != E; ++It)
    if (!OperandConvertableToType(*It, I, Ty, CTMap))
      return false;

  return true;
}


Value *ConvertExpressionToType(Value *V, const Type *Ty, ValueMapCache &VMC) {
  ValueMapCache::ExprMapTy::iterator VMCI = VMC.ExprMap.find(V);
  if (VMCI != VMC.ExprMap.end()) {
    assert(VMCI->second->getType() == Ty);
    return VMCI->second;
  }

#ifdef DEBUG_EXPR_CONVERT
  cerr << "CETT: " << (void*)V << " " << V;
#endif

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0)
    if (Constant *CPV = cast<Constant>(V)) {
      // Constants are converted by constant folding the cast that is required.
      // We assume here that all casts are implemented for constant prop.
      Value *Result = opt::ConstantFoldCastInstruction(CPV, Ty);
      assert(Result && "ConstantFoldCastInstruction Failed!!!");
      assert(Result->getType() == Ty && "Const prop of cast failed!");

      // Add the instruction to the expression map
      VMC.ExprMap[V] = Result;
      return Result;
    }


  BasicBlock *BB = I->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  string Name = I->getName();  if (!Name.empty()) I->setName("");
  Instruction *Res;     // Result of conversion

  ValueHandle IHandle(VMC, I);  // Prevent I from being removed!
  
  Constant *Dummy = Constant::getNullConstant(Ty);

  //cerr << endl << endl << "Type:\t" << Ty << "\nInst: " << I << "BB Before: " << BB << endl;

  switch (I->getOpcode()) {
  case Instruction::Cast:
    Res = new CastInst(I->getOperand(0), Ty, Name);
    break;
    
  case Instruction::Add:
  case Instruction::Sub:
    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                                 Dummy, Dummy, Name);
    VMC.ExprMap[I] = Res;   // Add node to expression eagerly

    Res->setOperand(0, ConvertExpressionToType(I->getOperand(0), Ty, VMC));
    Res->setOperand(1, ConvertExpressionToType(I->getOperand(1), Ty, VMC));
    break;

  case Instruction::Shl:
  case Instruction::Shr:
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(), Dummy,
                        I->getOperand(1), Name);
    VMC.ExprMap[I] = Res;
    Res->setOperand(0, ConvertExpressionToType(I->getOperand(0), Ty, VMC));
    break;

  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);
    assert(!LI->hasIndices() || AllIndicesZero(LI));

    Res = new LoadInst(Constant::getNullConstant(PointerType::get(Ty)), Name);
    VMC.ExprMap[I] = Res;
    Res->setOperand(0, ConvertExpressionToType(LI->getPointerOperand(),
                                               PointerType::get(Ty), VMC));
    assert(Res->getOperand(0)->getType() == PointerType::get(Ty));
    assert(Ty == Res->getType());
    assert(isFirstClassType(Res->getType()) && "Load of structure or array!");
    break;
  }

  case Instruction::PHINode: {
    PHINode *OldPN = cast<PHINode>(I);
    PHINode *NewPN = new PHINode(Ty, Name);

    VMC.ExprMap[I] = NewPN;   // Add node to expression eagerly
    while (OldPN->getNumOperands()) {
      BasicBlock *BB = OldPN->getIncomingBlock(0);
      Value *OldVal = OldPN->getIncomingValue(0);
      ValueHandle OldValHandle(VMC, OldVal);
      OldPN->removeIncomingValue(BB);
      Value *V = ConvertExpressionToType(OldVal, Ty, VMC);
      NewPN->addIncoming(V, BB);
    }
    Res = NewPN;
    break;
  }

  case Instruction::Malloc: {
    Res = ConvertMallocToType(cast<MallocInst>(I), Ty, Name, VMC);
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
    vector<Value*> Indices = GEP->copyIndices();
    const Type *BaseType = GEP->getPointerOperand()->getType();
    const Type *PVTy = cast<PointerType>(Ty)->getElementType();
    Res = 0;
    while (!Indices.empty() && isa<ConstantUInt>(Indices.back()) &&
           cast<ConstantUInt>(Indices.back())->getValue() == 0) {
      Indices.pop_back();
      if (GetElementPtrInst::getIndexedType(BaseType, Indices, true) == PVTy) {
        if (Indices.size() == 0) {
          Res = new CastInst(GEP->getPointerOperand(), BaseType); // NOOP
        } else {
          Res = new GetElementPtrInst(GEP->getPointerOperand(), Indices, Name);
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

  assert(Res->getType() == Ty && "Didn't convert expr to correct type!");

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
  cerr << "ExpIn: " << (void*)I << " " << I
       << "ExpOut: " << (void*)Res << " " << Res;
#endif

  if (I->use_empty()) {
#ifdef DEBUG_EXPR_CONVERT
    cerr << "EXPR DELETING: " << (void*)I << " " << I;
#endif
    BIL.remove(I);
    VMC.OperandsMapped.erase(I);
    VMC.ExprMap.erase(I);
    delete I;
  }

  return Res;
}



// ValueConvertableToType - Return true if it is possible
bool ValueConvertableToType(Value *V, const Type *Ty,
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
  if (V->getType() == Ty) return true;   // Operand already the right type?

  // Expression type must be holdable in a register.
  if (!isFirstClassType(Ty))
    return false;

  Instruction *I = dyn_cast<Instruction>(U);
  if (I == 0) return false;              // We can't convert!

  switch (I->getOpcode()) {
  case Instruction::Cast:
    assert(I->getOperand(0) == V);
    // We can convert the expr if the cast destination type is losslessly
    // convertable to the requested type.
    if (!Ty->isLosslesslyConvertableTo(I->getOperand(0)->getType()))
      return false;
#if 1
    // We also do not allow conversion of a cast that casts from a ptr to array
    // of X to a *X.  For example: cast [4 x %List *] * %val to %List * *
    //
    if (PointerType *SPT = dyn_cast<PointerType>(I->getOperand(0)->getType()))
      if (PointerType *DPT = dyn_cast<PointerType>(I->getType()))
        if (ArrayType *AT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (AT->getElementType() == DPT->getElementType())
            return false;
#endif
    return true;

  case Instruction::Add:
    if (isa<PointerType>(Ty)) {
      Value *IndexVal = I->getOperand(V == I->getOperand(0) ? 1 : 0);
      vector<Value*> Indices;
      if (const Type *ETy = ConvertableToGEP(Ty, IndexVal, Indices)) {
        const Type *RetTy = PointerType::get(ETy);

        // Only successful if we can convert this type to the required type
        if (ValueConvertableToType(I, RetTy, CTMap)) {
          CTMap[I] = RetTy;
          return true;
        }
      }
    }
    // FALLTHROUGH
  case Instruction::Sub: {
    Value *OtherOp = I->getOperand((V == I->getOperand(0)) ? 1 : 0);
    return ValueConvertableToType(I, Ty, CTMap) &&
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
    return ValueConvertableToType(I, Ty, CTMap);

  case Instruction::Load:
    // Cannot convert the types of any subscripts...
    if (I->getOperand(0) != V) return false;

    if (const PointerType *PT = dyn_cast<PointerType>(Ty)) {
      LoadInst *LI = cast<LoadInst>(I);
      
      if (LI->hasIndices() && !AllIndicesZero(LI))
        return false;

      const Type *LoadedTy = PT->getElementType();

      // They could be loading the first element of a composite type...
      if (const CompositeType *CT = dyn_cast<CompositeType>(LoadedTy)) {
        unsigned Offset = 0;     // No offset, get first leaf.
        vector<Value*> Indices;  // Discarded...
        LoadedTy = getStructOffsetType(CT, Offset, Indices, false);
        assert(Offset == 0 && "Offset changed from zero???");
      }

      if (!isFirstClassType(LoadedTy))
        return false;

      if (TD.getTypeSize(LoadedTy) != TD.getTypeSize(LI->getType()))
        return false;

      return ValueConvertableToType(LI, LoadedTy, CTMap);
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
      if (isa<ArrayType>(PT->getElementType()))
        return false;  // Avoid getDataSize on unsized array type!
      assert(V == I->getOperand(1));

      // Must move the same amount of data...
      if (TD.getTypeSize(PT->getElementType()) != 
          TD.getTypeSize(I->getOperand(0)->getType())) return false;

      // Can convert store if the incoming value is convertable...
      return ExpressionConvertableToType(I->getOperand(0), PT->getElementType(),
                                         CTMap);
    }
    return false;
  }

  case Instruction::GetElementPtr:
    // Convert a getelementptr [sbyte] * %reg111, uint 16 freely back to
    // anything that is a pointer type...
    //
    if (I->getType() != PointerType::get(Type::SByteTy) ||
        I->getNumOperands() != 2 || V != I->getOperand(0) ||
        I->getOperand(1)->getType() != Type::UIntTy || !isa<PointerType>(Ty))
      return false;
    return true;

  case Instruction::PHINode: {
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i)
      if (!ExpressionConvertableToType(PN->getIncomingValue(i), Ty, CTMap))
        return false;
    return ValueConvertableToType(PN, Ty, CTMap);
  }

  case Instruction::Call: {
    User::op_iterator OI = find(I->op_begin(), I->op_end(), V);
    assert (OI != I->op_end() && "Not using value!");
    unsigned OpNum = OI - I->op_begin();

    if (OpNum == 0)
      return false; // Can't convert method pointer type yet.  FIXME
    
    const PointerType *MPtr = cast<PointerType>(I->getOperand(0)->getType());
    const MethodType *MTy = cast<MethodType>(MPtr->getElementType());
    if (!MTy->isVarArg()) return false;

    if ((OpNum-1) < MTy->getParamTypes().size())
      return false;  // It's not in the varargs section...

    // If we get this far, we know the value is in the varargs section of the
    // method!  We can convert if we don't reinterpret the value...
    //
    return Ty->isLosslesslyConvertableTo(V->getType());
  }
  }
  return false;
}


void ConvertValueToNewType(Value *V, Value *NewVal, ValueMapCache &VMC) {
  ValueHandle VH(VMC, V);

  unsigned NumUses = V->use_size();
  for (unsigned It = 0; It < NumUses; ) {
    unsigned OldSize = NumUses;
    ConvertOperandToType(*(V->use_begin()+It), V, NewVal, VMC);
    NumUses = V->use_size();
    if (NumUses == OldSize) ++It;
  }
}



static void ConvertOperandToType(User *U, Value *OldVal, Value *NewVal,
                                 ValueMapCache &VMC) {
  if (isa<ValueHandle>(U)) return;  // Valuehandles don't let go of operands...

  if (VMC.OperandsMapped.count(U)) return;
  VMC.OperandsMapped.insert(U);

  ValueMapCache::ExprMapTy::iterator VMCI = VMC.ExprMap.find(U);
  if (VMCI != VMC.ExprMap.end())
    return;


  Instruction *I = cast<Instruction>(U);  // Only Instructions convertable

  BasicBlock *BB = I->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  string Name = I->getName();  if (!Name.empty()) I->setName("");
  Instruction *Res;     // Result of conversion

  //cerr << endl << endl << "Type:\t" << Ty << "\nInst: " << I << "BB Before: " << BB << endl;

  // Prevent I from being removed...
  ValueHandle IHandle(VMC, I);

  const Type *NewTy = NewVal->getType();
  Constant *Dummy = (NewTy != Type::VoidTy) ? 
                  Constant::getNullConstant(NewTy) : 0;

  switch (I->getOpcode()) {
  case Instruction::Cast:
    assert(I->getOperand(0) == OldVal);
    Res = new CastInst(NewVal, I->getType(), Name);
    break;

  case Instruction::Add:
    if (isa<PointerType>(NewTy)) {
      Value *IndexVal = I->getOperand(OldVal == I->getOperand(0) ? 1 : 0);
      vector<Value*> Indices;
      BasicBlock::iterator It = find(BIL.begin(), BIL.end(), I);

      if (const Type *ETy = ConvertableToGEP(NewTy, IndexVal, Indices, &It)) {
        // If successful, convert the add to a GEP
        const Type *RetTy = PointerType::get(ETy);
        // First operand is actually the given pointer...
        Res = new GetElementPtrInst(NewVal, Indices, Name);
        assert(cast<PointerType>(Res->getType())->getElementType() == ETy &&
               "ConvertableToGEP broken!");
        break;
      }
    }
    // FALLTHROUGH

  case Instruction::Sub:
  case Instruction::SetEQ:
  case Instruction::SetNE: {
    Res = BinaryOperator::create(cast<BinaryOperator>(I)->getOpcode(),
                                 Dummy, Dummy, Name);
    VMC.ExprMap[I] = Res;   // Add node to expression eagerly

    unsigned OtherIdx = (OldVal == I->getOperand(0)) ? 1 : 0;
    Value *OtherOp    = I->getOperand(OtherIdx);
    Value *NewOther   = ConvertExpressionToType(OtherOp, NewTy, VMC);

    Res->setOperand(OtherIdx, NewOther);
    Res->setOperand(!OtherIdx, NewVal);
    break;
  }
  case Instruction::Shl:
  case Instruction::Shr:
    assert(I->getOperand(0) == OldVal);
    Res = new ShiftInst(cast<ShiftInst>(I)->getOpcode(), NewVal,
                        I->getOperand(1), Name);
    break;

  case Instruction::Load: {
    assert(I->getOperand(0) == OldVal && isa<PointerType>(NewVal->getType()));
    const Type *LoadedTy = cast<PointerType>(NewVal->getType())->getElementType();

    vector<Value*> Indices;

    if (const CompositeType *CT = dyn_cast<CompositeType>(LoadedTy)) {
      unsigned Offset = 0;   // No offset, get first leaf.
      LoadedTy = getStructOffsetType(CT, Offset, Indices, false);
    }
    assert(isFirstClassType(LoadedTy));

    Res = new LoadInst(NewVal, Indices, Name);
    assert(isFirstClassType(Res->getType()) && "Load of structure or array!");
    break;
  }

  case Instruction::Store: {
    if (I->getOperand(0) == OldVal) {  // Replace the source value
      const PointerType *NewPT = PointerType::get(NewTy);
      Res = new StoreInst(NewVal, Constant::getNullConstant(NewPT));
      VMC.ExprMap[I] = Res;
      Res->setOperand(1, ConvertExpressionToType(I->getOperand(1), NewPT, VMC));
    } else {                           // Replace the source pointer
      const Type *ValTy = cast<PointerType>(NewTy)->getElementType();
      Res = new StoreInst(Constant::getNullConstant(ValTy), NewVal);
      VMC.ExprMap[I] = Res;
      Res->setOperand(0, ConvertExpressionToType(I->getOperand(0), ValTy, VMC));
    }
    break;
  }


  case Instruction::GetElementPtr: {
    // Convert a getelementptr [sbyte] * %reg111, uint 16 freely back to
    // anything that is a pointer type...
    //
    BasicBlock::iterator It = find(BIL.begin(), BIL.end(), I);
    
    // Insert a cast right before this instruction of the index value...
    CastInst *CIdx = new CastInst(I->getOperand(1), NewTy);
    It = BIL.insert(It, CIdx)+1;
    
    // Insert an add right before this instruction 
    Instruction *AddInst = BinaryOperator::create(Instruction::Add, NewVal,
                                                  CIdx, Name);
    It = BIL.insert(It, AddInst)+1;

    // Finally, cast the result back to our previous type...
    Res = new CastInst(AddInst, I->getType());
    break;
  }

  case Instruction::PHINode: {
    PHINode *OldPN = cast<PHINode>(I);
    PHINode *NewPN = new PHINode(NewTy, Name);
    VMC.ExprMap[I] = NewPN;

    while (OldPN->getNumOperands()) {
      BasicBlock *BB = OldPN->getIncomingBlock(0);
      Value *OldVal = OldPN->getIncomingValue(0);
      OldPN->removeIncomingValue(BB);
      Value *V = ConvertExpressionToType(OldVal, NewTy, VMC);
      NewPN->addIncoming(V, BB);
    }
    Res = NewPN;
    break;
  }

  case Instruction::Call: {
    Value *Meth = I->getOperand(0);
    vector<Value*> Params(I->op_begin()+1, I->op_end());

    vector<Value*>::iterator OI = find(Params.begin(), Params.end(), OldVal);
    assert (OI != Params.end() && "Not using value!");

    *OI = NewVal;
    Res = new CallInst(Meth, Params, Name);
    break;
  }
  default:
    assert(0 && "Expression convertable, but don't know how to convert?");
    return;
  }

  BasicBlock::iterator It = find(BIL.begin(), BIL.end(), I);
  assert(It != BIL.end() && "Instruction not in own basic block??");
  BIL.insert(It, Res);   // Keep It pointing to old instruction

#ifdef DEBUG_EXPR_CONVERT
  cerr << "COT CREATED: "  << (void*)Res << " " << Res;
  cerr << "In: " << (void*)I << " " << I << "Out: " << (void*)Res << " " << Res;
#endif

  // Add the instruction to the expression map
  VMC.ExprMap[I] = Res;

  if (I->getType() != Res->getType())
    ConvertValueToNewType(I, Res, VMC);
  else {
    for (unsigned It = 0; It < I->use_size(); ) {
      User *Use = *(I->use_begin()+It);
      if (isa<ValueHandle>(Use))            // Don't remove ValueHandles!
        ++It;
      else
        Use->replaceUsesOfWith(I, Res);
    }

    if (I->use_empty()) {
      // Now we just need to remove the old instruction so we don't get infinite
      // loops.  Note that we cannot use DCE because DCE won't remove a store
      // instruction, for example.
      //
#ifdef DEBUG_EXPR_CONVERT
      cerr << "DELETING: " << (void*)I << " " << I;
#endif
      BIL.remove(I);
      VMC.OperandsMapped.erase(I);
      VMC.ExprMap.erase(I);
      delete I;
    } else {
      for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI)
        assert(isa<ValueHandle>((Value*)*UI) &&"Uses of Instruction remain!!!");
    }
  }
}


ValueHandle::ValueHandle(ValueMapCache &VMC, Value *V)
  : Instruction(Type::VoidTy, UserOp1, ""), Cache(VMC) {
#ifdef DEBUG_EXPR_CONVERT
  //cerr << "VH AQUIRING: " << (void*)V << " " << V;
#endif
  Operands.push_back(Use(V, this));
}

static void RecursiveDelete(ValueMapCache &Cache, Instruction *I) {
  if (!I || !I->use_empty()) return;

  assert(I->getParent() && "Inst not in basic block!");

#ifdef DEBUG_EXPR_CONVERT
  //cerr << "VH DELETING: " << (void*)I << " " << I;
#endif

  for (User::op_iterator OI = I->op_begin(), OE = I->op_end(); 
       OI != OE; ++OI) {
    Instruction *U = dyn_cast<Instruction>(*OI);
    if (U) {
      *OI = 0;
      RecursiveDelete(Cache, dyn_cast<Instruction>(U));
    }
  }

  I->getParent()->getInstList().remove(I);

  Cache.OperandsMapped.erase(I);
  Cache.ExprMap.erase(I);
  delete I;
}

ValueHandle::~ValueHandle() {
  if (Operands[0]->use_size() == 1) {
    Value *V = Operands[0];
    Operands[0] = 0;   // Drop use!

    // Now we just need to remove the old instruction so we don't get infinite
    // loops.  Note that we cannot use DCE because DCE won't remove a store
    // instruction, for example.
    //
    RecursiveDelete(Cache, dyn_cast<Instruction>(V));
  } else {
#ifdef DEBUG_EXPR_CONVERT
    //cerr << "VH RELEASING: " << (void*)Operands[0].get() << " " << Operands[0]->use_size() << " " << Operands[0];
#endif
  }
}
