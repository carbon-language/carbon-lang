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
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumReduced ("loop-reduce", "Number of GEPs strength reduced");

  class GEPCache {
  public:
    GEPCache() : CachedPHINode(0), Map() {}

    GEPCache *get(Value *v) {
      std::map<Value *, GEPCache>::iterator I = Map.find(v);
      if (I == Map.end())
        I = Map.insert(std::pair<Value *, GEPCache>(v, GEPCache())).first;
      return &I->second;
    }

    PHINode *CachedPHINode;
    std::map<Value *, GEPCache> Map;
  };

  class LoopStrengthReduce : public FunctionPass {
    LoopInfo *LI;
    DominatorSet *DS;
    bool Changed;
    unsigned MaxTargetAMSize;
  public:
    LoopStrengthReduce(unsigned MTAMS = 1)
      : MaxTargetAMSize(MTAMS) {
    }

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
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorSet>();
      AU.addRequired<TargetData>();
    }
  private:
    void runOnLoop(Loop *L);
    void strengthReduceGEP(GetElementPtrInst *GEPI, Loop *L,
                           GEPCache* GEPCache,
                           Instruction *InsertBefore,
                           std::set<Instruction*> &DeadInsts);
    void DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts);
  };
  RegisterOpt<LoopStrengthReduce> X("loop-reduce", 
                                    "Strength Reduce GEP Uses of Ind. Vars");
}

FunctionPass *llvm::createLoopStrengthReducePass(unsigned MaxTargetAMSize) {
  return new LoopStrengthReduce(MaxTargetAMSize);
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
                                           GEPCache *Cache,
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
  // constants or loop invariable expressions prior to an instance of the
  // canonical induction variable.
  unsigned indvar = 0;
  std::vector<Value *> pre_op_vector;
  std::vector<Value *> inc_op_vector;
  const Type *ty = GEPI->getOperand(0)->getType();
  Value *CanonicalIndVar = L->getCanonicalInductionVariable();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  bool AllConstantOperands = true;
  Cache = Cache->get(GEPI->getOperand(0));

  for (unsigned op = 1, e = GEPI->getNumOperands(); op != e; ++op) {
    Value *operand = GEPI->getOperand(op);
    if (ty->getTypeID() == Type::StructTyID) {
      assert(isa<ConstantUInt>(operand));
      ConstantUInt *c = dyn_cast<ConstantUInt>(operand);
      ty = ty->getContainedType(unsigned(c->getValue()));
    } else {
      ty = ty->getContainedType(0);
    }

    if (operand == CanonicalIndVar) {
      // FIXME: use getCanonicalInductionVariableIncrement to choose between
      // one and neg one maybe?  We need to support int *foo = GEP base, -1
      const Type *Ty = CanonicalIndVar->getType();
      pre_op_vector.push_back(Constant::getNullValue(Ty));
      inc_op_vector.push_back(ConstantInt::get(Ty, 1));
      indvar = op;
      break;
    } else if (isa<Constant>(operand) || isa<Argument>(operand)) {
      pre_op_vector.push_back(operand);
    } else if (Instruction *inst = dyn_cast<Instruction>(operand)) {
      if (!DS->dominates(inst, Preheader->getTerminator()))
        return;
      pre_op_vector.push_back(operand);
      AllConstantOperands = false;
    } else
      return;
    Cache = Cache->get(operand);
  }
  assert(indvar > 0 && "Indvar used by GEP not found in operand list");
  
  // Ensure the pointer base is loop invariant.  While strength reduction
  // makes sense even if the pointer changed on every iteration, there is no
  // realistic way of handling it unless GEPs were completely decomposed into
  // their constituent operations so we have explicit multiplications to work
  // with.
  if (Instruction *GepPtrOp = dyn_cast<Instruction>(GEPI->getOperand(0)))
    if (!DS->dominates(GepPtrOp, Preheader->getTerminator()))
      return;

  // Don't reduce multiplies that the target can handle via addressing modes.
  uint64_t sz = getAnalysis<TargetData>().getTypeSize(ty);
  if (sz && (sz & (sz-1)) == 0)   // Power of two?
    if (sz <= (1ULL << (MaxTargetAMSize-1)))
      return;
  
  // If all operands of the GEP we are going to insert into the preheader
  // are constants, generate a GEP ConstantExpr instead. 
  //
  // If there is only one operand after the initial non-constant one, we know
  // that it was the induction variable, and has been replaced by a constant
  // null value.  In this case, replace the GEP with a use of pointer directly.
  PHINode *NewPHI;
  if (Cache->CachedPHINode == 0) {
    Value *PreGEP;
    if (AllConstantOperands && isa<Constant>(GEPI->getOperand(0))) {
      Constant *C = dyn_cast<Constant>(GEPI->getOperand(0));
      PreGEP = ConstantExpr::getGetElementPtr(C, pre_op_vector);
    } else if (pre_op_vector.size() == 1) {
      PreGEP = GEPI->getOperand(0);
    } else {
      PreGEP = new GetElementPtrInst(GEPI->getOperand(0),
                                    pre_op_vector, GEPI->getName()+".pre", 
                                    Preheader->getTerminator());
    }

    // The next step of the strength reduction is to create a PHI that will
    // choose between the initial GEP we created and inserted into the
    // preheader, and the incremented GEP that we will create below and insert
    // into the loop body.
    NewPHI = new PHINode(PreGEP->getType(), 
                                  GEPI->getName()+".str", InsertBefore);
    NewPHI->addIncoming(PreGEP, Preheader);
    
    // Now, create the GEP instruction to increment by one the value selected
    // by the PHI instruction we just created above, and add it as the second
    // incoming Value/BasicBlock pair to the PHINode.  It is inserted before
    // the increment of the canonical induction variable.
    Instruction *IncrInst = 
      const_cast<Instruction*>(L->getCanonicalInductionVariableIncrement());
    GetElementPtrInst *StrGEP = new GetElementPtrInst(NewPHI, inc_op_vector,
                                                      GEPI->getName()+".inc",
                                                      IncrInst);
    pred_iterator PI = pred_begin(Header);
    if (*PI == Preheader)
      ++PI;
    NewPHI->addIncoming(StrGEP, *PI);
    Cache->CachedPHINode = NewPHI;
  } else {
    // Reuse previously created pointer, as it is identical to the one we were
    // about to create.
    NewPHI = Cache->CachedPHINode;
  }
  
  if (GEPI->getNumOperands() - 1 == indvar) {
    // If there were no operands following the induction variable, replace all
    // uses of the old GEP instruction with the new PHI.
    GEPI->replaceAllUsesWith(NewPHI);
  } else {
    // Create a new GEP instruction using the new PHI as the base.  The
    // operands of the original GEP past the induction variable become
    // operands of this new GEP.
    std::vector<Value *> op_vector;
    const Type *Ty = CanonicalIndVar->getType();
    op_vector.push_back(Constant::getNullValue(Ty));
    for (unsigned op = indvar + 1; op < GEPI->getNumOperands(); op++)
      op_vector.push_back(GEPI->getOperand(op));
    GetElementPtrInst *newGEP = new GetElementPtrInst(NewPHI, op_vector,
                                                      GEPI->getName() + ".lsr",
                                                      GEPI);
    GEPI->replaceAllUsesWith(newGEP);
  }
  
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

  // FIXME: Need to use SCEV to detect GEP uses of the indvar, since indvars
  // pass creates code like this, which we can't currently detect:
  //  %tmp.1 = sub uint 2000, %indvar
  //  %tmp.8 = getelementptr int* %y, uint %tmp.1
  
  // Strength reduce all GEPs in the Loop.  Insert secondary PHI nodes for the
  // strength reduced pointers we'll be creating after the canonical induction
  // variable's PHI.
  std::set<Instruction*> DeadInsts;
  GEPCache Cache;
  for (Value::use_iterator UI = PN->use_begin(), UE = PN->use_end();
       UI != UE; ++UI)
    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(*UI))
      strengthReduceGEP(GEPI, L, &Cache, PN->getNext(), DeadInsts);

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
    // FIXME: this needs to eliminate an induction variable even if it's being
    // compared against some value to decide loop termination.
    if (PN->hasOneUse()) {
      BinaryOperator *BO = dyn_cast<BinaryOperator>(*(PN->use_begin()));
      if (BO && BO->getOpcode() == Instruction::Add)
        if (BO->hasOneUse()) {
          if (PN == dyn_cast<PHINode>(*(BO->use_begin()))) {
            DeadInsts.insert(BO);
            // Break the cycle, then delete the PHI.
            PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
            PN->eraseFromParent();
            DeleteTriviallyDeadInstructions(DeadInsts);
          }
        }
    }
  }
}
