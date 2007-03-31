//===- CodeGenPrepare.cpp - Prepare a function for code generation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass munges the code in the input function to better prepare it for
// SelectionDAG-based code generation.  This works around limitations in it's
// basic-block-at-a-time approach.  It should eventually be removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "codegenprepare"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/SmallSet.h"
using namespace llvm;

namespace {  
  class VISIBILITY_HIDDEN CodeGenPrepare : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// transformation profitability.
    const TargetLowering *TLI;
  public:
    CodeGenPrepare(const TargetLowering *tli = 0) : TLI(tli) {}
    bool runOnFunction(Function &F);
    
  private:
    bool OptimizeBlock(BasicBlock &BB);
    bool OptimizeGEPExpression(GetElementPtrInst *GEPI);
  };
}
static RegisterPass<CodeGenPrepare> X("codegenprepare",
                                      "Optimize for code generation");

FunctionPass *llvm::createCodeGenPreparePass(const TargetLowering *TLI) {
  return new CodeGenPrepare(TLI);
}


bool CodeGenPrepare::runOnFunction(Function &F) {
  bool MadeChange = true;
  bool EverMadeChange = false;
  while (MadeChange) {
    MadeChange = false;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      MadeChange |= OptimizeBlock(*BB);
    EverMadeChange |= MadeChange;
  }
  return EverMadeChange;
}

/// SplitEdgeNicely - Split the critical edge from TI to it's specified
/// successor if it will improve codegen.  We only do this if the successor has
/// phi nodes (otherwise critical edges are ok).  If there is already another
/// predecessor of the succ that is empty (and thus has no phi nodes), use it
/// instead of introducing a new block.
static void SplitEdgeNicely(TerminatorInst *TI, unsigned SuccNum, Pass *P) {
  BasicBlock *TIBB = TI->getParent();
  BasicBlock *Dest = TI->getSuccessor(SuccNum);
  assert(isa<PHINode>(Dest->begin()) &&
         "This should only be called if Dest has a PHI!");
  
  /// TIPHIValues - This array is lazily computed to determine the values of
  /// PHIs in Dest that TI would provide.
  std::vector<Value*> TIPHIValues;
  
  // Check to see if Dest has any blocks that can be used as a split edge for
  // this terminator.
  for (pred_iterator PI = pred_begin(Dest), E = pred_end(Dest); PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    // To be usable, the pred has to end with an uncond branch to the dest.
    BranchInst *PredBr = dyn_cast<BranchInst>(Pred->getTerminator());
    if (!PredBr || !PredBr->isUnconditional() ||
        // Must be empty other than the branch.
        &Pred->front() != PredBr)
      continue;
    
    // Finally, since we know that Dest has phi nodes in it, we have to make
    // sure that jumping to Pred will have the same affect as going to Dest in
    // terms of PHI values.
    PHINode *PN;
    unsigned PHINo = 0;
    bool FoundMatch = true;
    for (BasicBlock::iterator I = Dest->begin();
         (PN = dyn_cast<PHINode>(I)); ++I, ++PHINo) {
      if (PHINo == TIPHIValues.size())
        TIPHIValues.push_back(PN->getIncomingValueForBlock(TIBB));
      
      // If the PHI entry doesn't work, we can't use this pred.
      if (TIPHIValues[PHINo] != PN->getIncomingValueForBlock(Pred)) {
        FoundMatch = false;
        break;
      }
    }
    
    // If we found a workable predecessor, change TI to branch to Succ.
    if (FoundMatch) {
      Dest->removePredecessor(TIBB);
      TI->setSuccessor(SuccNum, Pred);
      return;
    }
  }
  
  SplitCriticalEdge(TI, SuccNum, P, true);  
}


/// InsertGEPComputeCode - Insert code into BB to compute Ptr+PtrOffset,
/// casting to the type of GEPI.
static Instruction *InsertGEPComputeCode(Instruction *&V, BasicBlock *BB,
                                         Instruction *GEPI, Value *Ptr,
                                         Value *PtrOffset) {
  if (V) return V;   // Already computed.
  
  // Figure out the insertion point
  BasicBlock::iterator InsertPt;
  if (BB == GEPI->getParent()) {
    // If GEP is already inserted into BB, insert right after the GEP.
    InsertPt = GEPI;
    ++InsertPt;
  } else {
    // Otherwise, insert at the top of BB, after any PHI nodes
    InsertPt = BB->begin();
    while (isa<PHINode>(InsertPt)) ++InsertPt;
  }
  
  // If Ptr is itself a cast, but in some other BB, emit a copy of the cast into
  // BB so that there is only one value live across basic blocks (the cast 
  // operand).
  if (CastInst *CI = dyn_cast<CastInst>(Ptr))
    if (CI->getParent() != BB && isa<PointerType>(CI->getOperand(0)->getType()))
      Ptr = CastInst::create(CI->getOpcode(), CI->getOperand(0), CI->getType(),
                             "", InsertPt);
  
  // Add the offset, cast it to the right type.
  Ptr = BinaryOperator::createAdd(Ptr, PtrOffset, "", InsertPt);
  // Ptr is an integer type, GEPI is pointer type ==> IntToPtr
  return V = CastInst::create(Instruction::IntToPtr, Ptr, GEPI->getType(), 
                              "", InsertPt);
}

/// ReplaceUsesOfGEPInst - Replace all uses of RepPtr with inserted code to
/// compute its value.  The RepPtr value can be computed with Ptr+PtrOffset. One
/// trivial way of doing this would be to evaluate Ptr+PtrOffset in RepPtr's
/// block, then ReplaceAllUsesWith'ing everything.  However, we would prefer to
/// sink PtrOffset into user blocks where doing so will likely allow us to fold
/// the constant add into a load or store instruction.  Additionally, if a user
/// is a pointer-pointer cast, we look through it to find its users.
static void ReplaceUsesOfGEPInst(Instruction *RepPtr, Value *Ptr, 
                                 Constant *PtrOffset, BasicBlock *DefBB,
                                 GetElementPtrInst *GEPI,
                           std::map<BasicBlock*,Instruction*> &InsertedExprs) {
  while (!RepPtr->use_empty()) {
    Instruction *User = cast<Instruction>(RepPtr->use_back());
    
    // If the user is a Pointer-Pointer cast, recurse. Only BitCast can be
    // used for a Pointer-Pointer cast.
    if (isa<BitCastInst>(User)) {
      ReplaceUsesOfGEPInst(User, Ptr, PtrOffset, DefBB, GEPI, InsertedExprs);
      
      // Drop the use of RepPtr. The cast is dead.  Don't delete it now, else we
      // could invalidate an iterator.
      User->setOperand(0, UndefValue::get(RepPtr->getType()));
      continue;
    }
    
    // If this is a load of the pointer, or a store through the pointer, emit
    // the increment into the load/store block.
    Instruction *NewVal;
    if (isa<LoadInst>(User) ||
        (isa<StoreInst>(User) && User->getOperand(0) != RepPtr)) {
      NewVal = InsertGEPComputeCode(InsertedExprs[User->getParent()], 
                                    User->getParent(), GEPI,
                                    Ptr, PtrOffset);
    } else {
      // If this use is not foldable into the addressing mode, use a version 
      // emitted in the GEP block.
      NewVal = InsertGEPComputeCode(InsertedExprs[DefBB], DefBB, GEPI, 
                                    Ptr, PtrOffset);
    }
    
    if (GEPI->getType() != RepPtr->getType()) {
      BasicBlock::iterator IP = NewVal;
      ++IP;
      // NewVal must be a GEP which must be pointer type, so BitCast
      NewVal = new BitCastInst(NewVal, RepPtr->getType(), "", IP);
    }
    User->replaceUsesOfWith(RepPtr, NewVal);
  }
}

/// OptimizeGEPExpression - Since we are doing basic-block-at-a-time instruction
/// selection, we want to be a bit careful about some things.  In particular, if
/// we have a GEP instruction that is used in a different block than it is
/// defined, the addressing expression of the GEP cannot be folded into loads or
/// stores that use it.  In this case, decompose the GEP and move constant
/// indices into blocks that use it.
bool CodeGenPrepare::OptimizeGEPExpression(GetElementPtrInst *GEPI) {
  // If this GEP is only used inside the block it is defined in, there is no
  // need to rewrite it.
  bool isUsedOutsideDefBB = false;
  BasicBlock *DefBB = GEPI->getParent();
  for (Value::use_iterator UI = GEPI->use_begin(), E = GEPI->use_end(); 
       UI != E; ++UI) {
    if (cast<Instruction>(*UI)->getParent() != DefBB) {
      isUsedOutsideDefBB = true;
      break;
    }
  }
  if (!isUsedOutsideDefBB) return false;

  // If this GEP has no non-zero constant indices, there is nothing we can do,
  // ignore it.
  bool hasConstantIndex = false;
  bool hasVariableIndex = false;
  for (GetElementPtrInst::op_iterator OI = GEPI->op_begin()+1,
       E = GEPI->op_end(); OI != E; ++OI) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(*OI)) {
      if (!CI->isZero()) {
        hasConstantIndex = true;
        break;
      }
    } else {
      hasVariableIndex = true;
    }
  }
  
  // If this is a "GEP X, 0, 0, 0", turn this into a cast.
  if (!hasConstantIndex && !hasVariableIndex) {
    /// The GEP operand must be a pointer, so must its result -> BitCast
    Value *NC = new BitCastInst(GEPI->getOperand(0), GEPI->getType(), 
                                GEPI->getName(), GEPI);
    GEPI->replaceAllUsesWith(NC);
    GEPI->eraseFromParent();
    return true;
  }
  
  // If this is a GEP &Alloca, 0, 0, forward subst the frame index into uses.
  if (!hasConstantIndex && !isa<AllocaInst>(GEPI->getOperand(0)))
    return false;

  // If we don't have target lowering info, we can't lower the GEP.
  if (!TLI) return false;
  const TargetData *TD = TLI->getTargetData();

  // Otherwise, decompose the GEP instruction into multiplies and adds.  Sum the
  // constant offset (which we now know is non-zero) and deal with it later.
  uint64_t ConstantOffset = 0;
  const Type *UIntPtrTy = TD->getIntPtrType();
  Value *Ptr = new PtrToIntInst(GEPI->getOperand(0), UIntPtrTy, "", GEPI);
  const Type *Ty = GEPI->getOperand(0)->getType();

  for (GetElementPtrInst::op_iterator OI = GEPI->op_begin()+1,
       E = GEPI->op_end(); OI != E; ++OI) {
    Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field)
        ConstantOffset += TD->getStructLayout(StTy)->getElementOffset(Field);
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // Handle constant subscripts.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getZExtValue() == 0) continue;
        ConstantOffset += (int64_t)TD->getTypeSize(Ty)*CI->getSExtValue();
        continue;
      }
      
      // Ptr = Ptr + Idx * ElementSize;
      
      // Cast Idx to UIntPtrTy if needed.
      Idx = CastInst::createIntegerCast(Idx, UIntPtrTy, true/*SExt*/, "", GEPI);
      
      uint64_t ElementSize = TD->getTypeSize(Ty);
      // Mask off bits that should not be set.
      ElementSize &= ~0ULL >> (64-UIntPtrTy->getPrimitiveSizeInBits());
      Constant *SizeCst = ConstantInt::get(UIntPtrTy, ElementSize);

      // Multiply by the element size and add to the base.
      Idx = BinaryOperator::createMul(Idx, SizeCst, "", GEPI);
      Ptr = BinaryOperator::createAdd(Ptr, Idx, "", GEPI);
    }
  }
  
  // Make sure that the offset fits in uintptr_t.
  ConstantOffset &= ~0ULL >> (64-UIntPtrTy->getPrimitiveSizeInBits());
  Constant *PtrOffset = ConstantInt::get(UIntPtrTy, ConstantOffset);
  
  // Okay, we have now emitted all of the variable index parts to the BB that
  // the GEP is defined in.  Loop over all of the using instructions, inserting
  // an "add Ptr, ConstantOffset" into each block that uses it and update the
  // instruction to use the newly computed value, making GEPI dead.  When the
  // user is a load or store instruction address, we emit the add into the user
  // block, otherwise we use a canonical version right next to the gep (these 
  // won't be foldable as addresses, so we might as well share the computation).
  
  std::map<BasicBlock*,Instruction*> InsertedExprs;
  ReplaceUsesOfGEPInst(GEPI, Ptr, PtrOffset, DefBB, GEPI, InsertedExprs);
  
  // Finally, the GEP is dead, remove it.
  GEPI->eraseFromParent();
  
  return true;
}

/// SinkInvariantGEPIndex - If a GEP instruction has a variable index that has
/// been hoisted out of the loop by LICM pass, sink it back into the use BB
/// if it can be determined that the index computation can be folded into the
/// addressing mode of the load / store uses.
static bool SinkInvariantGEPIndex(BinaryOperator *BinOp,
                                  const TargetLowering &TLI) {
  // Only look at Add.
  if (BinOp->getOpcode() != Instruction::Add)
    return false;

  // DestBBs - These are the blocks where a copy of BinOp will be inserted.
  SmallSet<BasicBlock*, 8> DestBBs;
  BasicBlock *DefBB = BinOp->getParent();
  bool MadeChange = false;
  for (Value::use_iterator UI = BinOp->use_begin(), E = BinOp->use_end(); 
       UI != E; ++UI) {
    Instruction *GEPI = cast<Instruction>(*UI);
    // Only look for GEP use in another block.
    if (GEPI->getParent() == DefBB) continue;

    if (isa<GetElementPtrInst>(GEPI)) {
      // If the GEP has another variable index, abondon.
      bool hasVariableIndex = false;
      for (GetElementPtrInst::op_iterator OI = GEPI->op_begin()+1,
             OE = GEPI->op_end(); OI != OE; ++OI)
        if (*OI != BinOp && !isa<ConstantInt>(*OI)) {
          hasVariableIndex = true;
          break;
        }
      if (hasVariableIndex)
        break;

      BasicBlock *GEPIBB = GEPI->getParent();
      for (Value::use_iterator UUI = GEPI->use_begin(), UE = GEPI->use_end(); 
           UUI != UE; ++UUI) {
        Instruction *GEPIUser = cast<Instruction>(*UUI);
        const Type *UseTy = NULL;
        if (LoadInst *Load = dyn_cast<LoadInst>(GEPIUser))
          UseTy = Load->getType();
        else if (StoreInst *Store = dyn_cast<StoreInst>(GEPIUser))
          UseTy = Store->getOperand(0)->getType();

        // Check if it is possible to fold the expression to address mode.
        if (UseTy && isa<ConstantInt>(BinOp->getOperand(1))) {
          uint64_t Scale = TLI.getTargetData()->getTypeSize(UseTy);
          int64_t Cst = cast<ConstantInt>(BinOp->getOperand(1))->getSExtValue();
          // e.g. load (gep i32 * %P, (X+42)) => load (%P + X*4 + 168).
          if (TLI.isLegalAddressImmediate(Cst*Scale, UseTy) &&
              (Scale == 1 || TLI.isLegalAddressScale(Scale, UseTy))) {
            DestBBs.insert(GEPIBB);
            MadeChange = true;
            break;
          }
        }
      }
    }
  }

  // Nothing to do.
  if (!MadeChange)
    return false;

  /// InsertedOps - Only insert a duplicate in each block once.
  std::map<BasicBlock*, BinaryOperator*> InsertedOps;
  for (Value::use_iterator UI = BinOp->use_begin(), E = BinOp->use_end(); 
       UI != E; ) {
    Instruction *User = cast<Instruction>(*UI);
    BasicBlock *UserBB = User->getParent();

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // If any user in this BB wants it, replace all the uses in the BB.
    if (DestBBs.count(UserBB)) {
      // Sink it into user block.
      BinaryOperator *&InsertedOp = InsertedOps[UserBB];
      if (!InsertedOp) {
        BasicBlock::iterator InsertPt = UserBB->begin();
        while (isa<PHINode>(InsertPt)) ++InsertPt;
      
        InsertedOp =
          BinaryOperator::create(BinOp->getOpcode(), BinOp->getOperand(0),
                                 BinOp->getOperand(1), "", InsertPt);
      }

      User->replaceUsesOfWith(BinOp, InsertedOp);
    }
  }

  if (BinOp->use_empty())
      BinOp->eraseFromParent();

  return true;
}

/// OptimizeNoopCopyExpression - We have determined that the specified cast
/// instruction is a noop copy (e.g. it's casting from one pointer type to
/// another, int->uint, or int->sbyte on PPC.
///
/// Return true if any changes are made.
static bool OptimizeNoopCopyExpression(CastInst *CI) {
  BasicBlock *DefBB = CI->getParent();
  
  /// InsertedCasts - Only insert a cast in each block once.
  std::map<BasicBlock*, CastInst*> InsertedCasts;
  
  bool MadeChange = false;
  for (Value::use_iterator UI = CI->use_begin(), E = CI->use_end(); 
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);
    
    // Figure out which BB this cast is used in.  For PHI's this is the
    // appropriate predecessor block.
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User)) {
      unsigned OpVal = UI.getOperandNo()/2;
      UserBB = PN->getIncomingBlock(OpVal);
    }
    
    // Preincrement use iterator so we don't invalidate it.
    ++UI;
    
    // If this user is in the same block as the cast, don't change the cast.
    if (UserBB == DefBB) continue;
    
    // If we have already inserted a cast into this block, use it.
    CastInst *&InsertedCast = InsertedCasts[UserBB];

    if (!InsertedCast) {
      BasicBlock::iterator InsertPt = UserBB->begin();
      while (isa<PHINode>(InsertPt)) ++InsertPt;
      
      InsertedCast = 
        CastInst::create(CI->getOpcode(), CI->getOperand(0), CI->getType(), "", 
                         InsertPt);
      MadeChange = true;
    }
    
    // Replace a use of the cast with a use of the new casat.
    TheUse = InsertedCast;
  }
  
  // If we removed all uses, nuke the cast.
  if (CI->use_empty())
    CI->eraseFromParent();
  
  return MadeChange;
}



// In this pass we look for GEP and cast instructions that are used
// across basic blocks and rewrite them to improve basic-block-at-a-time
// selection.
bool CodeGenPrepare::OptimizeBlock(BasicBlock &BB) {
  bool MadeChange = false;
  
  // Split all critical edges where the dest block has a PHI and where the phi
  // has shared immediate operands.
  TerminatorInst *BBTI = BB.getTerminator();
  if (BBTI->getNumSuccessors() > 1) {
    for (unsigned i = 0, e = BBTI->getNumSuccessors(); i != e; ++i)
      if (isa<PHINode>(BBTI->getSuccessor(i)->begin()) &&
          isCriticalEdge(BBTI, i, true))
        SplitEdgeNicely(BBTI, i, this);
  }
  
  
  for (BasicBlock::iterator BBI = BB.begin(), E = BB.end(); BBI != E; ) {
    Instruction *I = BBI++;
    
    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      // If we found an inline asm expession, and if the target knows how to
      // lower it to normal LLVM code, do so now.
      if (TLI && isa<InlineAsm>(CI->getCalledValue()))
        if (const TargetAsmInfo *TAI = 
            TLI->getTargetMachine().getTargetAsmInfo()) {
          if (TAI->ExpandInlineAsm(CI))
            BBI = BB.begin();
        }
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
      MadeChange |= OptimizeGEPExpression(GEPI);
    } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
      // If the source of the cast is a constant, then this should have
      // already been constant folded.  The only reason NOT to constant fold
      // it is if something (e.g. LSR) was careful to place the constant
      // evaluation in a block other than then one that uses it (e.g. to hoist
      // the address of globals out of a loop).  If this is the case, we don't
      // want to forward-subst the cast.
      if (isa<Constant>(CI->getOperand(0)))
        continue;
      
      if (!TLI) continue;
      
      // If this is a noop copy, sink it into user blocks to reduce the number
      // of virtual registers that must be created and coallesced.
      MVT::ValueType SrcVT = TLI->getValueType(CI->getOperand(0)->getType());
      MVT::ValueType DstVT = TLI->getValueType(CI->getType());
      
      // This is an fp<->int conversion?
      if (MVT::isInteger(SrcVT) != MVT::isInteger(DstVT))
        continue;
      
      // If this is an extension, it will be a zero or sign extension, which
      // isn't a noop.
      if (SrcVT < DstVT) continue;
      
      // If these values will be promoted, find out what they will be promoted
      // to.  This helps us consider truncates on PPC as noop copies when they
      // are.
      if (TLI->getTypeAction(SrcVT) == TargetLowering::Promote)
        SrcVT = TLI->getTypeToTransformTo(SrcVT);
      if (TLI->getTypeAction(DstVT) == TargetLowering::Promote)
        DstVT = TLI->getTypeToTransformTo(DstVT);
      
      // If, after promotion, these are the same types, this is a noop copy.
      if (SrcVT == DstVT)
        MadeChange |= OptimizeNoopCopyExpression(CI);
    } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(I)) {
      if (TLI)
        MadeChange |= SinkInvariantGEPIndex(BinOp, *TLI);
    }
  }
  return MadeChange;
}

