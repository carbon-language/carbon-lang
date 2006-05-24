//===-- Local.cpp - Functions to perform local transformations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include <cerrno>
#include <cmath>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Local constant propagation...
//

/// doConstantPropagation - If an instruction references constants, try to fold
/// them together...
///
bool llvm::doConstantPropagation(BasicBlock::iterator &II) {
  if (Constant *C = ConstantFoldInstruction(II)) {
    // Replaces all of the uses of a variable with uses of the constant.
    II->replaceAllUsesWith(C);

    // Remove the instruction from the basic block...
    II = II->getParent()->getInstList().erase(II);
    return true;
  }

  return false;
}

/// ConstantFoldInstruction - Attempt to constant fold the specified
/// instruction.  If successful, the constant result is returned, if not, null
/// is returned.  Note that this function can only fail when attempting to fold
/// instructions like loads and stores, which have no constant expression form.
///
Constant *llvm::ConstantFoldInstruction(Instruction *I) {
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (PN->getNumIncomingValues() == 0)
      return Constant::getNullValue(PN->getType());

    Constant *Result = dyn_cast<Constant>(PN->getIncomingValue(0));
    if (Result == 0) return 0;

    // Handle PHI nodes specially here...
    for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) != Result && PN->getIncomingValue(i) != PN)
        return 0;   // Not all the same incoming constants...

    // If we reach here, all incoming values are the same constant.
    return Result;
  } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (Function *F = CI->getCalledFunction())
      if (canConstantFoldCallTo(F)) {
        std::vector<Constant*> Args;
        for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
          if (Constant *Op = dyn_cast<Constant>(CI->getOperand(i)))
            Args.push_back(Op);
          else
            return 0;
        return ConstantFoldCall(F, Args);
      }
    return 0;
  }

  Constant *Op0 = 0, *Op1 = 0;
  switch (I->getNumOperands()) {
  default:
  case 2:
    Op1 = dyn_cast<Constant>(I->getOperand(1));
    if (Op1 == 0) return 0;        // Not a constant?, can't fold
  case 1:
    Op0 = dyn_cast<Constant>(I->getOperand(0));
    if (Op0 == 0) return 0;        // Not a constant?, can't fold
    break;
  case 0: return 0;
  }

  if (isa<BinaryOperator>(I) || isa<ShiftInst>(I))
    return ConstantExpr::get(I->getOpcode(), Op0, Op1);

  switch (I->getOpcode()) {
  default: return 0;
  case Instruction::Cast:
    return ConstantExpr::getCast(Op0, I->getType());
  case Instruction::Select:
    if (Constant *Op2 = dyn_cast<Constant>(I->getOperand(2)))
      return ConstantExpr::getSelect(Op0, Op1, Op2);
    return 0;
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Op0, Op1);
  case Instruction::InsertElement:
    if (Constant *Op2 = dyn_cast<Constant>(I->getOperand(2)))
      return ConstantExpr::getInsertElement(Op0, Op1, Op2);
    return 0;
  case Instruction::ShuffleVector:
    if (Constant *Op2 = dyn_cast<Constant>(I->getOperand(2)))
      return ConstantExpr::getShuffleVector(Op0, Op1, Op2);
    return 0;
  case Instruction::GetElementPtr:
    std::vector<Constant*> IdxList;
    IdxList.reserve(I->getNumOperands()-1);
    if (Op1) IdxList.push_back(Op1);
    for (unsigned i = 2, e = I->getNumOperands(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>(I->getOperand(i)))
        IdxList.push_back(C);
      else
        return 0;  // Non-constant operand
    return ConstantExpr::getGetElementPtr(Op0, IdxList);
  }
}

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool llvm::ConstantFoldTerminator(BasicBlock *BB) {
  TerminatorInst *T = BB->getTerminator();

  // Branch - See if we are conditional jumping on constant
  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = cast<BasicBlock>(BI->getOperand(0));
    BasicBlock *Dest2 = cast<BasicBlock>(BI->getOperand(1));

    if (ConstantBool *Cond = dyn_cast<ConstantBool>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getValue() ? Dest2 : Dest1;

      //cerr << "Function: " << T->getParent()->getParent()
      //     << "\nRemoving branch from " << T->getParent()
      //     << "\n\nTo: " << OldDest << endl;

      // Let the basic block know that we are letting go of it.  Based on this,
      // it will adjust it's PHI nodes.
      assert(BI->getParent() && "Terminator not inserted in block!");
      OldDest->removePredecessor(BI->getParent());

      // Set the unconditional destination, and change the insn to be an
      // unconditional branch.
      BI->setUnconditionalDest(Destination);
      return true;
    } else if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Change a conditional branch to unconditional.
      BI->setUnconditionalDest(Dest1);
      return true;
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch into a
    // single branch instruction!
    ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *TheOnlyDest = SI->getSuccessor(0);  // The default dest
    BasicBlock *DefaultDest = TheOnlyDest;
    assert(TheOnlyDest == SI->getDefaultDest() &&
           "Default destination is not successor #0?");

    // Figure out which case it goes to...
    for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
      // Found case matching a constant operand?
      if (SI->getSuccessorValue(i) == CI) {
        TheOnlyDest = SI->getSuccessor(i);
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (SI->getSuccessor(i) == DefaultDest) {
        // Remove this entry...
        DefaultDest->removePredecessor(SI->getParent());
        SI->removeCase(i);
        --i; --e;  // Don't skip an entry...
        continue;
      }

      // Otherwise, check to see if the switch only branches to one destination.
      // We do this by reseting "TheOnlyDest" to null when we find two non-equal
      // destinations.
      if (SI->getSuccessor(i) != TheOnlyDest) TheOnlyDest = 0;
    }

    if (CI && !TheOnlyDest) {
      // Branching on a constant, but not any of the cases, go to the default
      // successor.
      TheOnlyDest = SI->getDefaultDest();
    }

    // If we found a single destination that we can fold the switch into, do so
    // now.
    if (TheOnlyDest) {
      // Insert the new branch..
      new BranchInst(TheOnlyDest, SI);
      BasicBlock *BB = SI->getParent();

      // Remove entries from PHI nodes which we no longer branch to...
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        // Found case matching a constant operand?
        BasicBlock *Succ = SI->getSuccessor(i);
        if (Succ == TheOnlyDest)
          TheOnlyDest = 0;  // Don't modify the first branch to TheOnlyDest
        else
          Succ->removePredecessor(BB);
      }

      // Delete the old switch...
      BB->getInstList().erase(SI);
      return true;
    } else if (SI->getNumSuccessors() == 2) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      Value *Cond = new SetCondInst(Instruction::SetEQ, SI->getCondition(),
                                    SI->getSuccessorValue(1), "cond", SI);
      // Insert the new branch...
      new BranchInst(SI->getSuccessor(1), SI->getSuccessor(0), Cond, SI);

      // Delete the old switch...
      SI->getParent()->getInstList().erase(SI);
      return true;
    }
  }
  return false;
}

/// ConstantFoldLoadThroughGEPConstantExpr - Given a constant and a
/// getelementptr constantexpr, return the constant value being addressed by the
/// constant expression, or null if something is funny and we can't decide.
Constant *llvm::ConstantFoldLoadThroughGEPConstantExpr(Constant *C, 
                                                       ConstantExpr *CE) {
  if (CE->getOperand(1) != Constant::getNullValue(CE->getOperand(1)->getType()))
    return 0;  // Do not allow stepping over the value!
  
  // Loop over all of the operands, tracking down which value we are
  // addressing...
  gep_type_iterator I = gep_type_begin(CE), E = gep_type_end(CE);
  for (++I; I != E; ++I)
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      ConstantUInt *CU = cast<ConstantUInt>(I.getOperand());
      assert(CU->getValue() < STy->getNumElements() &&
             "Struct index out of range!");
      unsigned El = (unsigned)CU->getValue();
      if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
        C = CS->getOperand(El);
      } else if (isa<ConstantAggregateZero>(C)) {
        C = Constant::getNullValue(STy->getElementType(El));
      } else if (isa<UndefValue>(C)) {
        C = UndefValue::get(STy->getElementType(El));
      } else {
        return 0;
      }
    } else if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand())) {
      if (const ArrayType *ATy = dyn_cast<ArrayType>(*I)) {
        if ((uint64_t)CI->getRawValue() >= ATy->getNumElements())
          C = UndefValue::get(ATy->getElementType());
        if (ConstantArray *CA = dyn_cast<ConstantArray>(C))
          C = CA->getOperand((unsigned)CI->getRawValue());
        else if (isa<ConstantAggregateZero>(C))
          C = Constant::getNullValue(ATy->getElementType());
        else if (isa<UndefValue>(C))
          C = UndefValue::get(ATy->getElementType());
        else
          return 0;
      } else if (const PackedType *PTy = dyn_cast<PackedType>(*I)) {
        if ((uint64_t)CI->getRawValue() >= PTy->getNumElements())
          C = UndefValue::get(PTy->getElementType());
        if (ConstantPacked *CP = dyn_cast<ConstantPacked>(C))
          C = CP->getOperand((unsigned)CI->getRawValue());
        else if (isa<ConstantAggregateZero>(C))
          C = Constant::getNullValue(PTy->getElementType());
        else if (isa<UndefValue>(C))
          C = UndefValue::get(PTy->getElementType());
        else
          return 0;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  return C;
}


//===----------------------------------------------------------------------===//
//  Local dead code elimination...
//

bool llvm::isInstructionTriviallyDead(Instruction *I) {
  if (!I->use_empty() || isa<TerminatorInst>(I)) return false;

  if (!I->mayWriteToMemory()) return true;

  if (CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction()) {
      unsigned IntrinsicID = F->getIntrinsicID();
#define GET_SIDE_EFFECT_INFO
#include "llvm/Intrinsics.gen"
#undef GET_SIDE_EFFECT_INFO
    }
  return false;
}

// dceInstruction - Inspect the instruction at *BBI and figure out if it's
// [trivially] dead.  If so, remove the instruction and update the iterator
// to point to the instruction that immediately succeeded the original
// instruction.
//
bool llvm::dceInstruction(BasicBlock::iterator &BBI) {
  // Look for un"used" definitions...
  if (isInstructionTriviallyDead(BBI)) {
    BBI = BBI->getParent()->getInstList().erase(BBI);   // Bye bye
    return true;
  }
  return false;
}
