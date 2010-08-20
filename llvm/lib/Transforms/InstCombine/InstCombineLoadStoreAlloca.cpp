//===- InstCombineLoadStoreAlloca.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for load, store and alloca.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;
using namespace PatternMatch;

STATISTIC(NumDeadStore, "Number of dead stores eliminated");

Instruction *InstCombiner::visitAllocaInst(AllocaInst &AI) {
  // Ensure that the alloca array size argument has type intptr_t, so that
  // any casting is exposed early.
  if (TD) {
    const Type *IntPtrTy = TD->getIntPtrType(AI.getContext());
    if (AI.getArraySize()->getType() != IntPtrTy) {
      Value *V = Builder->CreateIntCast(AI.getArraySize(),
                                        IntPtrTy, false);
      AI.setOperand(0, V);
      return &AI;
    }
  }

  // Convert: alloca Ty, C - where C is a constant != 1 into: alloca [C x Ty], 1
  if (AI.isArrayAllocation()) {  // Check C != 1
    if (const ConstantInt *C = dyn_cast<ConstantInt>(AI.getArraySize())) {
      const Type *NewTy = 
        ArrayType::get(AI.getAllocatedType(), C->getZExtValue());
      assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
      AllocaInst *New = Builder->CreateAlloca(NewTy, 0, AI.getName());
      New->setAlignment(AI.getAlignment());

      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...also skip interleaved debug info
      //
      BasicBlock::iterator It = New;
      while (isa<AllocaInst>(*It) || isa<DbgInfoIntrinsic>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      Value *NullIdx =Constant::getNullValue(Type::getInt32Ty(AI.getContext()));
      Value *Idx[2];
      Idx[0] = NullIdx;
      Idx[1] = NullIdx;
      Value *V = GetElementPtrInst::CreateInBounds(New, Idx, Idx + 2,
                                                   New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
    }
  }

  if (TD && isa<AllocaInst>(AI) && AI.getAllocatedType()->isSized()) {
    // If alloca'ing a zero byte object, replace the alloca with a null pointer.
    // Note that we only do this for alloca's, because malloc should allocate
    // and return a unique pointer, even for a zero byte allocation.
    if (TD->getTypeAllocSize(AI.getAllocatedType()) == 0)
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));

    // If the alignment is 0 (unspecified), assign it the preferred alignment.
    if (AI.getAlignment() == 0)
      AI.setAlignment(TD->getPrefTypeAlignment(AI.getAllocatedType()));
  }

  return 0;
}


/// InstCombineLoadCast - Fold 'load (cast P)' -> cast (load P)' when possible.
static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI,
                                        const TargetData *TD) {
  User *CI = cast<User>(LI.getOperand(0));
  Value *CastOp = CI->getOperand(0);

  const PointerType *DestTy = cast<PointerType>(CI->getType());
  const Type *DestPTy = DestTy->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {

    // If the address spaces don't match, don't eliminate the cast.
    if (DestTy->getAddressSpace() != SrcTy->getAddressSpace())
      return 0;

    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isIntegerTy() || DestPTy->isPointerTy() || 
         DestPTy->isVectorTy()) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            Value *Idxs[2];
            Idxs[0] = Constant::getNullValue(Type::getInt32Ty(LI.getContext()));
            Idxs[1] = Idxs[0];
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs, 2);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if (IC.getTargetData() &&
          (SrcPTy->isIntegerTy() || SrcPTy->isPointerTy() || 
            SrcPTy->isVectorTy()) &&
          // Do not allow turning this into a load of an integer, which is then
          // casted to a pointer, this pessimizes pointer analysis a lot.
          (SrcPTy->isPointerTy() == LI.getType()->isPointerTy()) &&
          IC.getTargetData()->getTypeSizeInBits(SrcPTy) ==
               IC.getTargetData()->getTypeSizeInBits(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the load, cast
        // the result of the loaded value.
        LoadInst *NewLoad = 
          IC.Builder->CreateLoad(CastOp, LI.isVolatile(), CI->getName());
        NewLoad->setAlignment(LI.getAlignment());
        // Now cast the result of the load.
        return new BitCastInst(NewLoad, LI.getType());
      }
    }
  }
  return 0;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // Attempt to improve the alignment.
  if (TD) {
    unsigned KnownAlign =
      GetOrEnforceKnownAlignment(Op, TD->getPrefTypeAlignment(LI.getType()));
    unsigned LoadAlign = LI.getAlignment();
    unsigned EffectiveLoadAlign = LoadAlign != 0 ? LoadAlign :
      TD->getABITypeAlignment(LI.getType());

    if (KnownAlign > EffectiveLoadAlign)
      LI.setAlignment(KnownAlign);
    else if (LoadAlign == 0)
      LI.setAlignment(EffectiveLoadAlign);
  }

  // load (cast X) --> cast (load X) iff safe.
  if (isa<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
      return Res;

  // None of the following transforms are legal for volatile loads.
  if (LI.isVolatile()) return 0;
  
  // Do really simple store-to-load forwarding and load CSE, to catch cases
  // where there are several consequtive memory accesses to the same location,
  // separated by a few arithmetic operations.
  BasicBlock::iterator BBI = &LI;
  if (Value *AvailableVal = FindAvailableLoadedValue(Op, LI.getParent(), BBI,6))
    return ReplaceInstUsesWith(LI, AvailableVal);

  // load(gep null, ...) -> unreachable
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op)) {
    const Value *GEPI0 = GEPI->getOperand(0);
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<ConstantPointerNull>(GEPI0) && GEPI->getPointerAddressSpace() == 0){
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }
  } 

  // load null/undef -> unreachable
  // TODO: Consider a target hook for valid address spaces for this xform.
  if (isa<UndefValue>(Op) ||
      (isa<ConstantPointerNull>(Op) && LI.getPointerAddressSpace() == 0)) {
    // Insert a new store to null instruction before the load to indicate that
    // this code is not reachable.  We do this instead of inserting an
    // unreachable instruction directly because we cannot modify the CFG.
    new StoreInst(UndefValue::get(LI.getType()),
                  Constant::getNullValue(Op->getType()), &LI);
    return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
  }

  // Instcombine load (constantexpr_cast global) -> cast (load global)
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
    if (CE->isCast())
      if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
        return Res;
  
  if (Op->hasOneUse()) {
    // Change select and PHI nodes to select values instead of addresses: this
    // helps alias analysis out a lot, allows many others simplifications, and
    // exposes redundancy in the code.
    //
    // Note that we cannot do the transformation unless we know that the
    // introduced loads cannot trap!  Something like this is valid as long as
    // the condition is always false: load (select bool %C, int* null, int* %G),
    // but it would not be valid if we transformed it to load from null
    // unconditionally.
    //
    if (SelectInst *SI = dyn_cast<SelectInst>(Op)) {
      // load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2).
      unsigned Align = LI.getAlignment();
      if (isSafeToLoadUnconditionally(SI->getOperand(1), SI, Align, TD) &&
          isSafeToLoadUnconditionally(SI->getOperand(2), SI, Align, TD)) {
        LoadInst *V1 = Builder->CreateLoad(SI->getOperand(1),
                                           SI->getOperand(1)->getName()+".val");
        LoadInst *V2 = Builder->CreateLoad(SI->getOperand(2),
                                           SI->getOperand(2)->getName()+".val");
        V1->setAlignment(Align);
        V2->setAlignment(Align);
        return SelectInst::Create(SI->getCondition(), V1, V2);
      }

      // load (select (cond, null, P)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(1)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(2));
          return &LI;
        }

      // load (select (cond, P, null)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(2)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(1));
          return &LI;
        }
    }
  }
  return 0;
}

/// InstCombineStoreToCast - Fold store V, (cast P) -> store (cast V), P
/// when possible.  This makes it generally easy to do alias analysis and/or
/// SROA/mem2reg of the memory object.
static Instruction *InstCombineStoreToCast(InstCombiner &IC, StoreInst &SI) {
  User *CI = cast<User>(SI.getOperand(1));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType());
  if (SrcTy == 0) return 0;
  
  const Type *SrcPTy = SrcTy->getElementType();

  if (!DestPTy->isIntegerTy() && !DestPTy->isPointerTy())
    return 0;
  
  /// NewGEPIndices - If SrcPTy is an aggregate type, we can emit a "noop gep"
  /// to its first element.  This allows us to handle things like:
  ///   store i32 xxx, (bitcast {foo*, float}* %P to i32*)
  /// on 32-bit hosts.
  SmallVector<Value*, 4> NewGEPIndices;
  
  // If the source is an array, the code below will not succeed.  Check to
  // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
  // constants.
  if (SrcPTy->isArrayTy() || SrcPTy->isStructTy()) {
    // Index through pointer.
    Constant *Zero = Constant::getNullValue(Type::getInt32Ty(SI.getContext()));
    NewGEPIndices.push_back(Zero);
    
    while (1) {
      if (const StructType *STy = dyn_cast<StructType>(SrcPTy)) {
        if (!STy->getNumElements()) /* Struct can be empty {} */
          break;
        NewGEPIndices.push_back(Zero);
        SrcPTy = STy->getElementType(0);
      } else if (const ArrayType *ATy = dyn_cast<ArrayType>(SrcPTy)) {
        NewGEPIndices.push_back(Zero);
        SrcPTy = ATy->getElementType();
      } else {
        break;
      }
    }
    
    SrcTy = PointerType::get(SrcPTy, SrcTy->getAddressSpace());
  }

  if (!SrcPTy->isIntegerTy() && !SrcPTy->isPointerTy())
    return 0;
  
  // If the pointers point into different address spaces or if they point to
  // values with different sizes, we can't do the transformation.
  if (!IC.getTargetData() ||
      SrcTy->getAddressSpace() != 
        cast<PointerType>(CI->getType())->getAddressSpace() ||
      IC.getTargetData()->getTypeSizeInBits(SrcPTy) !=
      IC.getTargetData()->getTypeSizeInBits(DestPTy))
    return 0;

  // Okay, we are casting from one integer or pointer type to another of
  // the same size.  Instead of casting the pointer before 
  // the store, cast the value to be stored.
  Value *NewCast;
  Value *SIOp0 = SI.getOperand(0);
  Instruction::CastOps opcode = Instruction::BitCast;
  const Type* CastSrcTy = SIOp0->getType();
  const Type* CastDstTy = SrcPTy;
  if (CastDstTy->isPointerTy()) {
    if (CastSrcTy->isIntegerTy())
      opcode = Instruction::IntToPtr;
  } else if (CastDstTy->isIntegerTy()) {
    if (SIOp0->getType()->isPointerTy())
      opcode = Instruction::PtrToInt;
  }
  
  // SIOp0 is a pointer to aggregate and this is a store to the first field,
  // emit a GEP to index into its first field.
  if (!NewGEPIndices.empty())
    CastOp = IC.Builder->CreateInBoundsGEP(CastOp, NewGEPIndices.begin(),
                                           NewGEPIndices.end());
  
  NewCast = IC.Builder->CreateCast(opcode, SIOp0, CastDstTy,
                                   SIOp0->getName()+".c");
  return new StoreInst(NewCast, CastOp);
}

/// equivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
///
static bool equivalentAddressValues(Value *A, Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;
  
  // Test if the values come form identical arithmetic instructions.
  // This uses isIdenticalToWhenDefined instead of isIdenticalTo because
  // its only used to compare two uses within the same basic block, which
  // means that they'll always either have the same value or one of them
  // will have an undefined value.
  if (isa<BinaryOperator>(A) ||
      isa<CastInst>(A) ||
      isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;
  
  // Otherwise they may not be equivalent.
  return false;
}

// If this instruction has two uses, one of which is a llvm.dbg.declare,
// return the llvm.dbg.declare.
DbgDeclareInst *InstCombiner::hasOneUsePlusDeclare(Value *V) {
  if (!V->hasNUses(2))
    return 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    User *U = *UI;
    if (DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(U))
      return DI;
    if (isa<BitCastInst>(U) && U->hasOneUse()) {
      if (DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(*U->use_begin()))
        return DI;
      }
  }
  return 0;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  // If the RHS is an alloca with a single use, zapify the store, making the
  // alloca dead.
  // If the RHS is an alloca with a two uses, the other one being a 
  // llvm.dbg.declare, zapify the store and the declare, making the
  // alloca dead.  We must do this to prevent declares from affecting
  // codegen.
  if (!SI.isVolatile()) {
    if (Ptr->hasOneUse()) {
      if (isa<AllocaInst>(Ptr)) 
        return EraseInstFromFunction(SI);
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        if (isa<AllocaInst>(GEP->getOperand(0))) {
          if (GEP->getOperand(0)->hasOneUse())
            return EraseInstFromFunction(SI);
          if (DbgDeclareInst *DI = hasOneUsePlusDeclare(GEP->getOperand(0))) {
            EraseInstFromFunction(*DI);
            return EraseInstFromFunction(SI);
          }
        }
      }
    }
    if (DbgDeclareInst *DI = hasOneUsePlusDeclare(Ptr)) {
      EraseInstFromFunction(*DI);
      return EraseInstFromFunction(SI);
    }
  }

  // Attempt to improve the alignment.
  if (TD) {
    unsigned KnownAlign =
      GetOrEnforceKnownAlignment(Ptr, TD->getPrefTypeAlignment(Val->getType()));
    unsigned StoreAlign = SI.getAlignment();
    unsigned EffectiveStoreAlign = StoreAlign != 0 ? StoreAlign :
      TD->getABITypeAlignment(Val->getType());

    if (KnownAlign > EffectiveStoreAlign)
      SI.setAlignment(KnownAlign);
    else if (StoreAlign == 0)
      SI.setAlignment(EffectiveStoreAlign);
  }

  // Do really simple DSE, to catch cases where there are several consecutive
  // stores to the same location, separated by a few arithmetic operations. This
  // situation often occurs with bitfield accesses.
  BasicBlock::iterator BBI = &SI;
  for (unsigned ScanInsts = 6; BBI != SI.getParent()->begin() && ScanInsts;
       --ScanInsts) {
    --BBI;
    // Don't count debug info directives, lest they affect codegen,
    // and we skip pointer-to-pointer bitcasts, which are NOPs.
    if (isa<DbgInfoIntrinsic>(BBI) ||
        (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy())) {
      ScanInsts++;
      continue;
    }    
    
    if (StoreInst *PrevSI = dyn_cast<StoreInst>(BBI)) {
      // Prev store isn't volatile, and stores to the same location?
      if (!PrevSI->isVolatile() &&equivalentAddressValues(PrevSI->getOperand(1),
                                                          SI.getOperand(1))) {
        ++NumDeadStore;
        ++BBI;
        EraseInstFromFunction(*PrevSI);
        continue;
      }
      break;
    }
    
    // If this is a load, we have to stop.  However, if the loaded value is from
    // the pointer we're loading and is producing the pointer we're storing,
    // then *this* store is dead (X = load P; store X -> P).
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI == Val && equivalentAddressValues(LI->getOperand(0), Ptr) &&
          !SI.isVolatile())
        return EraseInstFromFunction(SI);
      
      // Otherwise, this is a load from some other location.  Stores before it
      // may not be dead.
      break;
    }
    
    // Don't skip over loads or things that can modify memory.
    if (BBI->mayWriteToMemory() || BBI->mayReadFromMemory())
      break;
  }
  
  
  if (SI.isVolatile()) return 0;  // Don't hack volatile stores.

  // Attempt to narrow sequences where we load a wide value, perform bitmasks
  // that only affect the low bits of it, and then store it back.  This 
  // typically arises from bitfield initializers in C++.
  ConstantInt *CI1 =0, *CI2 = 0;
  Value *Ld = 0;
  if (getTargetData() &&
      match(SI.getValueOperand(),
            m_And(m_Or(m_Value(Ld), m_ConstantInt(CI1)), m_ConstantInt(CI2))) &&
      isa<LoadInst>(Ld) &&
      equivalentAddressValues(cast<LoadInst>(Ld)->getPointerOperand(), Ptr)) {
    APInt OrMask = CI1->getValue();
    APInt AndMask = CI2->getValue();
    
    // Compute the prefix of the value that is unmodified by the bitmasking.
    unsigned LeadingAndOnes = AndMask.countLeadingOnes();
    unsigned LeadingOrZeros = OrMask.countLeadingZeros();
    unsigned Prefix = std::min(LeadingAndOnes, LeadingOrZeros);
    uint64_t NewWidth = AndMask.getBitWidth() - Prefix;
    while (NewWidth < AndMask.getBitWidth() &&
           getTargetData()->isIllegalInteger(NewWidth))
      NewWidth = NextPowerOf2(NewWidth);
    
    // If we can find a power-of-2 prefix (and if the values we're working with
    // are themselves POT widths), then we can narrow the store.  We rely on
    // later iterations of instcombine to propagate the demanded bits to narrow
    // the other computations in the chain.
    if (NewWidth < AndMask.getBitWidth() &&
        getTargetData()->isLegalInteger(NewWidth)) {
      const Type *NewType = IntegerType::get(Ptr->getContext(), NewWidth);
      const Type *NewPtrType = PointerType::getUnqual(NewType);
      
      Value *NewVal = Builder->CreateTrunc(SI.getValueOperand(), NewType);
      Value *NewPtr = Builder->CreateBitCast(Ptr, NewPtrType);
      
      // On big endian targets, we need to offset from the original pointer
      // in order to store to the low-bit suffix.
      if (getTargetData()->isBigEndian()) {
        uint64_t GEPOffset = (AndMask.getBitWidth() - NewWidth) / 8;
        NewPtr = Builder->CreateConstGEP1_64(NewPtr, GEPOffset);
      }
      
      return new StoreInst(NewVal, NewPtr);
    }
  }

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr) && SI.getPointerAddressSpace() == 0) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        Worklist.Add(U);  // Dropped a use.
    }
    return 0;  // Do not modify these!
  }

  // store undef, Ptr -> noop
  if (isa<UndefValue>(Val))
    return EraseInstFromFunction(SI);

  // If the pointer destination is a cast, see if we can fold the cast into the
  // source instead.
  if (isa<CastInst>(Ptr))
    if (Instruction *Res = InstCombineStoreToCast(*this, SI))
      return Res;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->isCast())
      if (Instruction *Res = InstCombineStoreToCast(*this, SI))
        return Res;

  
  // If this store is the last instruction in the basic block (possibly
  // excepting debug info instructions), and if the block ends with an
  // unconditional branch, try to move it to the successor block.
  BBI = &SI; 
  do {
    ++BBI;
  } while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy()));
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional())
      if (SimplifyStoreAtEndOfBlock(SI))
        return 0;  // xform done!
  
  return 0;
}

/// SimplifyStoreAtEndOfBlock - Turn things like:
///   if () { *P = v1; } else { *P = v2 }
/// into a phi node with a store in the successor.
///
/// Simplify things like:
///   *P = v1; if () { *P = v2; }
/// into a phi node with a store in the successor.
///
bool InstCombiner::SimplifyStoreAtEndOfBlock(StoreInst &SI) {
  BasicBlock *StoreBB = SI.getParent();
  
  // Check to see if the successor block has exactly two incoming edges.  If
  // so, see if the other predecessor contains a store to the same location.
  // if so, insert a PHI node (if needed) and move the stores down.
  BasicBlock *DestBB = StoreBB->getTerminator()->getSuccessor(0);
  
  // Determine whether Dest has exactly two predecessors and, if so, compute
  // the other predecessor.
  pred_iterator PI = pred_begin(DestBB);
  BasicBlock *P = *PI;
  BasicBlock *OtherBB = 0;

  if (P != StoreBB)
    OtherBB = P;

  if (++PI == pred_end(DestBB))
    return false;
  
  P = *PI;
  if (P != StoreBB) {
    if (OtherBB)
      return false;
    OtherBB = P;
  }
  if (++PI != pred_end(DestBB))
    return false;

  // Bail out if all the relevant blocks aren't distinct (this can happen,
  // for example, if SI is in an infinite loop)
  if (StoreBB == DestBB || OtherBB == DestBB)
    return false;

  // Verify that the other block ends in a branch and is not otherwise empty.
  BasicBlock::iterator BBI = OtherBB->getTerminator();
  BranchInst *OtherBr = dyn_cast<BranchInst>(BBI);
  if (!OtherBr || BBI == OtherBB->begin())
    return false;
  
  // If the other block ends in an unconditional branch, check for the 'if then
  // else' case.  there is an instruction before the branch.
  StoreInst *OtherStore = 0;
  if (OtherBr->isUnconditional()) {
    --BBI;
    // Skip over debugging info.
    while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy())) {
      if (BBI==OtherBB->begin())
        return false;
      --BBI;
    }
    // If this isn't a store, isn't a store to the same location, or if the
    // alignments differ, bail out.
    OtherStore = dyn_cast<StoreInst>(BBI);
    if (!OtherStore || OtherStore->getOperand(1) != SI.getOperand(1) ||
        OtherStore->getAlignment() != SI.getAlignment())
      return false;
  } else {
    // Otherwise, the other block ended with a conditional branch. If one of the
    // destinations is StoreBB, then we have the if/then case.
    if (OtherBr->getSuccessor(0) != StoreBB && 
        OtherBr->getSuccessor(1) != StoreBB)
      return false;
    
    // Okay, we know that OtherBr now goes to Dest and StoreBB, so this is an
    // if/then triangle.  See if there is a store to the same ptr as SI that
    // lives in OtherBB.
    for (;; --BBI) {
      // Check to see if we find the matching store.
      if ((OtherStore = dyn_cast<StoreInst>(BBI))) {
        if (OtherStore->getOperand(1) != SI.getOperand(1) ||
            OtherStore->getAlignment() != SI.getAlignment())
          return false;
        break;
      }
      // If we find something that may be using or overwriting the stored
      // value, or if we run out of instructions, we can't do the xform.
      if (BBI->mayReadFromMemory() || BBI->mayWriteToMemory() ||
          BBI == OtherBB->begin())
        return false;
    }
    
    // In order to eliminate the store in OtherBr, we have to
    // make sure nothing reads or overwrites the stored value in
    // StoreBB.
    for (BasicBlock::iterator I = StoreBB->begin(); &*I != &SI; ++I) {
      // FIXME: This should really be AA driven.
      if (I->mayReadFromMemory() || I->mayWriteToMemory())
        return false;
    }
  }
  
  // Insert a PHI node now if we need it.
  Value *MergedVal = OtherStore->getOperand(0);
  if (MergedVal != SI.getOperand(0)) {
    PHINode *PN = PHINode::Create(MergedVal->getType(), "storemerge");
    PN->reserveOperandSpace(2);
    PN->addIncoming(SI.getOperand(0), SI.getParent());
    PN->addIncoming(OtherStore->getOperand(0), OtherBB);
    MergedVal = InsertNewInstBefore(PN, DestBB->front());
  }
  
  // Advance to a place where it is safe to insert the new store and
  // insert it.
  BBI = DestBB->getFirstNonPHI();
  InsertNewInstBefore(new StoreInst(MergedVal, SI.getOperand(1),
                                    OtherStore->isVolatile(),
                                    SI.getAlignment()), *BBI);
  
  // Nuke the old stores.
  EraseInstFromFunction(SI);
  EraseInstFromFunction(*OtherStore);
  return true;
}
