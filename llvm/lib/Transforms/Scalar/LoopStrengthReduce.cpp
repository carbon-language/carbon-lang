//===- LoopStrengthReduce.cpp - Strength Reduce GEPs in Loops -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass performs a strength reduction on array references inside loops that
// have as one or more of their components the loop induction variable.  This is
// accomplished by creating a new Value to hold the initial value of the array
// access for the first iteration, and then creating a new GEP instruction in
// the loop to increment the value by the appropriate amount.
//
// There are currently several deficiencies in the implementation, marked with
// FIXME in the code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumReduced ("loop-reduce", "Number of GEPs strength reduced");

  class LoopStrengthReduce : public FunctionPass {
    LoopInfo *LI;
    DominatorSet *DS;
    bool Changed;
  public:
    virtual bool runOnFunction(Function &) {
      LI = &getAnalysis<LoopInfo>();
      DS = &getAnalysis<DominatorSet>();
      Changed = false;

      for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
        runOnLoop(*I);
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorSet>();
    }
  private:
    void runOnLoop(Loop *L);
    void strengthReduceGEP(GetElementPtrInst *GEPI, Loop *L,
                           Instruction *InsertBefore,
                           std::set<Instruction*> &DeadInsts);
    void DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts);
  };
  RegisterOpt<LoopStrengthReduce> X("loop-reduce", 
                                    "Strength Reduce GEP Uses of Ind. Vars");
}

FunctionPass *llvm::createLoopStrengthReducePass() {
  return new LoopStrengthReduce();
}

/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
void LoopStrengthReduce::
DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts) {
  while (!Insts.empty()) {
    Instruction *I = *Insts.begin();
    Insts.erase(Insts.begin());
    if (isInstructionTriviallyDead(I)) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *U = dyn_cast<Instruction>(I->getOperand(i)))
          Insts.insert(U);
      I->getParent()->getInstList().erase(I);
      Changed = true;
    }
  }
}

void LoopStrengthReduce::strengthReduceGEP(GetElementPtrInst *GEPI, Loop *L,
                                           Instruction *InsertBefore,
                                           std::set<Instruction*> &DeadInsts) {
  // We will strength reduce the GEP by splitting it into two parts.  The first
  // is a GEP to hold the initial value of the non-strength-reduced GEP upon
  // entering the loop, which we will insert at the end of the loop preheader.
  // The second is a GEP to hold the incremented value of the initial GEP.
  // The LoopIndVarSimplify pass guarantees that loop counts start at zero, so
  // we will replace the indvar with a constant zero value to create the first
  // GEP.
  //
  // We currently only handle GEP instructions that consist of zero or more
  // constants and one instance of the canonical induction variable.
  bool foundIndvar = false;
  bool indvarLast = false;
  std::vector<Value *> pre_op_vector;
  std::vector<Value *> inc_op_vector;
  Value *CanonicalIndVar = L->getCanonicalInductionVariable();
  for (unsigned op = 1, e = GEPI->getNumOperands(); op != e; ++op) {
    Value *operand = GEPI->getOperand(op);
    if (operand == CanonicalIndVar) {
      // FIXME: We currently only support strength reducing GEP instructions
      // with one instance of the canonical induction variable.  This means that 
      // we can't deal with statements of the form A[i][i].
      if (foundIndvar == true)
        return;
        
      // FIXME: use getCanonicalInductionVariableIncrement to choose between
      // one and neg one maybe?  We need to support int *foo = GEP base, -1
      const Type *Ty = CanonicalIndVar->getType();
      pre_op_vector.push_back(Constant::getNullValue(Ty));
      inc_op_vector.push_back(ConstantInt::get(Ty, 1));
      foundIndvar = true;
      indvarLast = true;
    } else if (isa<Constant>(operand)) {
      pre_op_vector.push_back(operand);
      if (indvarLast == true) indvarLast = false;
    } else
      return;
  }
  // FIXME: handle GEPs where the indvar is not the last element of the index
  // array.
  if (indvarLast == false)
    return;
  assert(true == foundIndvar && "Indvar used by GEP not found in operand list");
  
  // FIXME: Being able to hoist the definition of the initial pointer value
  // would allow us to strength reduce more loops.  For example, %tmp.32 in the
  // following loop:
  // entry:
  //   br label %no_exit.0
  // no_exit.0:		; preds = %entry, %no_exit.0
  //   %init.1.0 = phi uint [ 0, %entry ], [ %indvar.next, %no_exit.0 ]
  //   %tmp.32 = load uint** %CROSSING
  //   %tmp.35 = getelementptr uint* %tmp.32, uint %init.1.0
  //   br label %no_exit.0
  BasicBlock *Header = L->getHeader();
  if (Instruction *GepPtrOp = dyn_cast<Instruction>(GEPI->getOperand(0)))
    if (!DS->dominates(GepPtrOp, Header->begin()))
      return;
  
  // If all operands of the GEP we are going to insert into the preheader
  // are constants, generate a GEP ConstantExpr instead. 
  //
  // If there is only one operand after the initial non-constant one, we know
  // that it was the induction variable, and has been replaced by a constant
  // null value.  In this case, replace the GEP with a use of pointer directly.
  //
  // 
  BasicBlock *Preheader = L->getLoopPreheader();
  Value *PreGEP;
  if (isa<Constant>(GEPI->getOperand(0))) {
    Constant *C = dyn_cast<Constant>(GEPI->getOperand(0));
    PreGEP = ConstantExpr::getGetElementPtr(C, pre_op_vector);
  } else if (pre_op_vector.size() == 1) {
    PreGEP = GEPI->getOperand(0);
  } else {
    PreGEP = new GetElementPtrInst(GEPI->getOperand(0),
                                   pre_op_vector, GEPI->getName(), 
                                   Preheader->getTerminator());
  }

  // The next step of the strength reduction is to create a PHI that will choose
  // between the initial GEP we created and inserted into the preheader, and 
  // the incremented GEP that we will create below and insert into the loop body
  PHINode *NewPHI = new PHINode(PreGEP->getType(), 
                                GEPI->getName()+".str", InsertBefore);
  NewPHI->addIncoming(PreGEP, Preheader);
  
  // Now, create the GEP instruction to increment the value selected by the PHI
  // instruction we just created above by one, and add it as the second incoming
  // Value and BasicBlock pair to the PHINode.
  Instruction *IncrInst = 
    const_cast<Instruction*>(L->getCanonicalInductionVariableIncrement());
  GetElementPtrInst *StrGEP = new GetElementPtrInst(NewPHI, inc_op_vector,
                                                    GEPI->getName()+".inc",
                                                    IncrInst);
  NewPHI->addIncoming(StrGEP, IncrInst->getParent());
  
  // Replace all uses of the old GEP instructions with the new PHI
  GEPI->replaceAllUsesWith(NewPHI);
  
  // The old GEP is now dead.
  DeadInsts.insert(GEPI);
  ++NumReduced;
}

void LoopStrengthReduce::runOnLoop(Loop *L) {
  // First step, transform all loops nesting inside of this loop.
  for (LoopInfo::iterator I = L->begin(), E = L->end(); I != E; ++I)
    runOnLoop(*I);

  // Next, get the first PHINode since it is guaranteed to be the canonical
  // induction variable for the loop by the preceding IndVarSimplify pass.
  PHINode *PN = L->getCanonicalInductionVariable();
  if (0 == PN)
    return;

  // Insert secondary PHI nodes after the canonical induction variable's PHI
  // for the strength reduced pointers that we will be creating.
  Instruction *InsertBefore = PN->getNext();

  // FIXME: Need to use SCEV to detect GEP uses of the indvar, since indvars
  // pass creates code like this, which we can't currently detect:
  //  %tmp.1 = sub uint 2000, %indvar
  //  %tmp.8 = getelementptr int* %y, uint %tmp.1
  
  // Strength reduce all GEPs in the Loop
  std::set<Instruction*> DeadInsts;
  for (Value::use_iterator UI = PN->use_begin(), UE = PN->use_end();
       UI != UE; ++UI)
    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(*UI))
      strengthReduceGEP(GEPI, L, InsertBefore, DeadInsts);

  // Clean up after ourselves
  if (!DeadInsts.empty()) {
    DeleteTriviallyDeadInstructions(DeadInsts);

    // At this point, we know that we have killed one or more GEP instructions.
    // It is worth checking to see if the cann indvar is also dead, so that we
    // can remove it as well.  The requirements for the cann indvar to be
    // considered dead are:
    // 1. the cann indvar has one use
    // 2. the use is an add instruction
    // 3. the add has one use
    // 4. the add is used by the cann indvar
    // If all four cases above are true, then we can remove both the add and
    // the cann indvar.
    if (PN->hasOneUse()) {
      BinaryOperator *BO = dyn_cast<BinaryOperator>(*(PN->use_begin()));
      if (BO && BO->getOpcode() == Instruction::Add)
        if (BO->hasOneUse()) {
          PHINode *PotentialIndvar = dyn_cast<PHINode>(*(BO->use_begin()));
          if (PotentialIndvar && PN == PotentialIndvar) {
            PN->dropAllReferences();
            DeadInsts.insert(BO);
            DeadInsts.insert(PN);
            DeleteTriviallyDeadInstructions(DeadInsts);
          }
        }
    }
  }
}
