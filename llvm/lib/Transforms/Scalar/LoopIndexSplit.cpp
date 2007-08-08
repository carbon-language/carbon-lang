//===- LoopIndexSplit.cpp - Loop Index Splitting Pass ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Index Splitting Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-index-split"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

STATISTIC(NumIndexSplit, "Number of loops index split");

namespace {

  class VISIBILITY_HIDDEN LoopIndexSplit : public LoopPass {

  public:
    static char ID; // Pass ID, replacement for typeid
    LoopIndexSplit() : LoopPass((intptr_t)&ID) {}

    // Index split Loop L. Return true if loop is split.
    bool runOnLoop(Loop *L, LPPassManager &LPM);

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<DominatorTree>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }

  private:

    class SplitInfo {
    public:
      SplitInfo() : IndVar(NULL), SplitValue(NULL), ExitValue(NULL),
                    SplitCondition(NULL), ExitCondition(NULL) {}
      // Induction variable whose range is being split by this transformation.
      PHINode *IndVar;
      
      // Induction variable's range is split at this value.
      Value *SplitValue;
      
      // Induction variable's final loop exit value.
      Value *ExitValue;
      
      // This compare instruction compares IndVar against SplitValue.
      ICmpInst *SplitCondition;

      // Loop exit condition.
      ICmpInst *ExitCondition;

      // Clear split info.
      void clear() {
        IndVar = NULL;
        SplitValue = NULL;
        ExitValue = NULL;
        SplitCondition = NULL;
        ExitCondition = NULL;
      }
    };

  private:
    /// Find condition inside a loop that is suitable candidate for index split.
    void findSplitCondition();

    /// processOneIterationLoop - Current loop L contains compare instruction
    /// that compares induction variable, IndVar, agains loop invariant. If
    /// entire (i.e. meaningful) loop body is dominated by this compare
    /// instruction then loop body is executed only for one iteration. In
    /// such case eliminate loop structure surrounding this loop body. For
    bool processOneIterationLoop(SplitInfo &SD, LPPassManager &LPM);
    
    /// If loop header includes loop variant instruction operands then
    /// this loop may not be eliminated.
    bool safeHeader(SplitInfo &SD,  BasicBlock *BB);

    /// If Exit block includes loop variant instructions then this
    /// loop may not be eliminated.
    bool safeExitBlock(SplitInfo &SD, BasicBlock *BB);

    /// Find cost of spliting loop L.
    unsigned findSplitCost(Loop *L, SplitInfo &SD);
    bool splitLoop(SplitInfo &SD);

  private:

    // Current Loop.
    Loop *L;
    ScalarEvolution *SE;
    DominatorTree *DT;
    SmallVector<SplitInfo, 4> SplitData;
  };

  char LoopIndexSplit::ID = 0;
  RegisterPass<LoopIndexSplit> X ("loop-index-split", "Index Split Loops");
}

LoopPass *llvm::createLoopIndexSplitPass() {
  return new LoopIndexSplit();
}

// Index split Loop L. Return true if loop is split.
bool LoopIndexSplit::runOnLoop(Loop *IncomingLoop, LPPassManager &LPM) {
  bool Changed = false;
  L = IncomingLoop;

  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();

  findSplitCondition();

  if (SplitData.empty())
    return false;

  // First see if it is possible to eliminate loop itself or not.
  for (SmallVector<SplitInfo, 4>::iterator SI = SplitData.begin(),
         E = SplitData.end(); SI != E; ++SI) {
    SplitInfo &SD = *SI;
    if (SD.SplitCondition->getPredicate() == ICmpInst::ICMP_EQ) {
      Changed = processOneIterationLoop(SD,LPM);
      if (Changed) {
        ++NumIndexSplit;
        // If is loop is eliminated then nothing else to do here.
        return Changed;
      }
    }
  }

  unsigned MaxCost = 99;
  unsigned Index = 0;
  unsigned MostProfitableSDIndex = 0;
  for (SmallVector<SplitInfo, 4>::iterator SI = SplitData.begin(),
         E = SplitData.end(); SI != E; ++SI, ++Index) {
    SplitInfo SD = *SI;

    // ICM_EQs are already handled above.
    if (SD.SplitCondition->getPredicate() == ICmpInst::ICMP_EQ)
      continue;
    
    unsigned Cost = findSplitCost(L, SD);
    if (Cost < MaxCost)
      MostProfitableSDIndex = Index;
  }

  // Split most profitiable condition.
  Changed = splitLoop(SplitData[MostProfitableSDIndex]);

  if (Changed)
    ++NumIndexSplit;
  
  return Changed;
}

/// Find condition inside a loop that is suitable candidate for index split.
void LoopIndexSplit::findSplitCondition() {

  SplitInfo SD;
  BasicBlock *Header = L->getHeader();

  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);

    if (!PN->getType()->isInteger())
      continue;

    SCEVHandle SCEV = SE->getSCEV(PN);
    if (!isa<SCEVAddRecExpr>(SCEV)) 
      continue;

    // If this phi node is used in a compare instruction then it is a
    // split condition candidate.
    for (Value::use_iterator UI = PN->use_begin(), E = PN->use_end(); 
         UI != E; ++UI) {
      if (ICmpInst *CI = dyn_cast<ICmpInst>(*UI)) {
        SD.SplitCondition = CI;
        break;
      }
    }

    // Valid SplitCondition's one operand is phi node and the other operand
    // is loop invariant.
    if (SD.SplitCondition) {
      if (SD.SplitCondition->getOperand(0) != PN)
        SD.SplitValue = SD.SplitCondition->getOperand(0);
      else
        SD.SplitValue = SD.SplitCondition->getOperand(1);
      SCEVHandle ValueSCEV = SE->getSCEV(SD.SplitValue);

      // If SplitValue is not invariant then SplitCondition is not appropriate.
      if (!ValueSCEV->isLoopInvariant(L))
        SD.SplitCondition = NULL;
    }

    // We are looking for only one split condition.
    if (SD.SplitCondition) {
      SD.IndVar = PN;
      SplitData.push_back(SD);
      // Before reusing SD for next split condition clear its content.
      SD.clear();
    }
  }
}

/// processOneIterationLoop - Current loop L contains compare instruction
/// that compares induction variable, IndVar, against loop invariant. If
/// entire (i.e. meaningful) loop body is dominated by this compare
/// instruction then loop body is executed only once. In such case eliminate 
/// loop structure surrounding this loop body. For example,
///     for (int i = start; i < end; ++i) {
///         if ( i == somevalue) {
///           loop_body
///         }
///     }
/// can be transformed into
///     if (somevalue >= start && somevalue < end) {
///        i = somevalue;
///        loop_body
///     }
bool LoopIndexSplit::processOneIterationLoop(SplitInfo &SD, LPPassManager &LPM) {

  BasicBlock *Header = L->getHeader();

  // First of all, check if SplitCondition dominates entire loop body
  // or not.
  
  // If SplitCondition is not in loop header then this loop is not suitable
  // for this transformation.
  if (SD.SplitCondition->getParent() != Header)
    return false;
  
  // If one of the Header block's successor is not an exit block then this
  // loop is not a suitable candidate.
  BasicBlock *ExitBlock = NULL;
  for (succ_iterator SI = succ_begin(Header), E = succ_end(Header); SI != E; ++SI) {
    if (L->isLoopExit(*SI)) {
      ExitBlock = *SI;
      break;
    }
  }

  if (!ExitBlock)
    return false;

  // If loop header includes loop variant instruction operands then
  // this loop may not be eliminated.
  if (!safeHeader(SD, Header)) 
    return false;

  // If Exit block includes loop variant instructions then this
  // loop may not be eliminated.
  if (!safeExitBlock(SD, ExitBlock)) 
    return false;

  // Update CFG.

  // As a first step to break this loop, remove Latch to Header edge.
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *LatchSucc = NULL;
  BranchInst *BR = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!BR)
    return false;
  Header->removePredecessor(Latch);
  for (succ_iterator SI = succ_begin(Latch), E = succ_end(Latch);
       SI != E; ++SI) {
    if (Header != *SI)
      LatchSucc = *SI;
  }
  BR->setUnconditionalDest(LatchSucc);

  BasicBlock *Preheader = L->getLoopPreheader();
  Instruction *Terminator = Header->getTerminator();
  Value *StartValue = SD.IndVar->getIncomingValueForBlock(Preheader);

  // Replace split condition in header.
  // Transform 
  //      SplitCondition : icmp eq i32 IndVar, SplitValue
  // into
  //      c1 = icmp uge i32 SplitValue, StartValue
  //      c2 = icmp ult i32 vSplitValue, ExitValue
  //      and i32 c1, c2 
  bool SignedPredicate = SD.ExitCondition->isSignedPredicate();
  Instruction *C1 = new ICmpInst(SignedPredicate ? 
                                 ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE,
                                 SD.SplitValue, StartValue, "lisplit", 
                                 Terminator);
  Instruction *C2 = new ICmpInst(SignedPredicate ? 
                                 ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                                 SD.SplitValue, SD.ExitValue, "lisplit", 
                                 Terminator);
  Instruction *NSplitCond = BinaryOperator::createAnd(C1, C2, "lisplit", 
                                                      Terminator);
  SD.SplitCondition->replaceAllUsesWith(NSplitCond);
  SD.SplitCondition->eraseFromParent();

  // Now, clear latch block. Remove instructions that are responsible
  // to increment induction variable. 
  Instruction *LTerminator = Latch->getTerminator();
  for (BasicBlock::iterator LB = Latch->begin(), LE = Latch->end();
       LB != LE; ) {
    Instruction *I = LB;
    ++LB;
    if (isa<PHINode>(I) || I == LTerminator)
      continue;

    I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
  }

  LPM.deleteLoopFromQueue(L);

  // Update Dominator Info.
  // Only CFG change done is to remove Latch to Header edge. This
  // does not change dominator tree because Latch did not dominate
  // Header.
  if (DominanceFrontier *DF = getAnalysisToUpdate<DominanceFrontier>()) {
    DominanceFrontier::iterator HeaderDF = DF->find(Header);
    if (HeaderDF != DF->end()) 
      DF->removeFromFrontier(HeaderDF, Header);

    DominanceFrontier::iterator LatchDF = DF->find(Latch);
    if (LatchDF != DF->end()) 
      DF->removeFromFrontier(LatchDF, Header);
  }
  return true;
}

// If loop header includes loop variant instruction operands then
// this loop can not be eliminated. This is used by processOneIterationLoop().
bool LoopIndexSplit::safeHeader(SplitInfo &SD, BasicBlock *Header) {

  Instruction *Terminator = Header->getTerminator();
  for(BasicBlock::iterator BI = Header->begin(), BE = Header->end(); 
      BI != BE; ++BI) {
    Instruction *I = BI;

    // PHI Nodes are OK. FIXME : Handle last value assignments.
    if (isa<PHINode>(I))
      continue;

    // SplitCondition itself is OK.
    if (I == SD.SplitCondition)
      continue;

    // Terminator is also harmless.
    if (I == Terminator)
      continue;

    // Otherwise we have a instruction that may not be safe.
    return false;
  }
  
  return true;
}

// If Exit block includes loop variant instructions then this
// loop may not be eliminated. This is used by processOneIterationLoop().
bool LoopIndexSplit::safeExitBlock(SplitInfo &SD, BasicBlock *ExitBlock) {

  Instruction *IndVarIncrement = NULL;

  for (BasicBlock::iterator BI = ExitBlock->begin(), BE = ExitBlock->end();
       BI != BE; ++BI) {
    Instruction *I = BI;

    // PHI Nodes are OK. FIXME : Handle last value assignments.
    if (isa<PHINode>(I))
      continue;

    // Check if I is induction variable increment instruction.
    if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(I)) {
      if (BOp->getOpcode() != Instruction::Add)
        return false;

      Value *Op0 = BOp->getOperand(0);
      Value *Op1 = BOp->getOperand(1);
      PHINode *PN = NULL;
      ConstantInt *CI = NULL;

      if ((PN = dyn_cast<PHINode>(Op0))) {
        if ((CI = dyn_cast<ConstantInt>(Op1)))
          IndVarIncrement = I;
      } else 
        if ((PN = dyn_cast<PHINode>(Op1))) {
          if ((CI = dyn_cast<ConstantInt>(Op0)))
            IndVarIncrement = I;
      }
          
      if (IndVarIncrement && PN == SD.IndVar && CI->isOne())
        continue;
    }

    // I is an Exit condition if next instruction is block terminator.
    // Exit condition is OK if it compares loop invariant exit value,
    // which is checked below.
    else if (ICmpInst *EC = dyn_cast<ICmpInst>(I)) {
      ++BI;
      Instruction *N = BI;
      if (N == ExitBlock->getTerminator()) {
        SD.ExitCondition = EC;
        continue;
      }
    }

    // Otherwise we have instruction that may not be safe.
    return false;
  }

  // Check if Exit condition is comparing induction variable against 
  // loop invariant value. If one operand is induction variable and 
  // the other operand is loop invaraint then Exit condition is safe.
  if (SD.ExitCondition) {
    Value *Op0 = SD.ExitCondition->getOperand(0);
    Value *Op1 = SD.ExitCondition->getOperand(1);

    Instruction *Insn0 = dyn_cast<Instruction>(Op0);
    Instruction *Insn1 = dyn_cast<Instruction>(Op1);
    
    if (Insn0 && Insn0 == IndVarIncrement)
      SD.ExitValue = Op1;
    else if (Insn1 && Insn1 == IndVarIncrement)
      SD.ExitValue = Op0;

    SCEVHandle ValueSCEV = SE->getSCEV(SD.ExitValue);
    if (!ValueSCEV->isLoopInvariant(L))
      return false;
  }

  // We could not find any reason to consider ExitBlock unsafe.
  return true;
}

/// Find cost of spliting loop L. Cost is measured in terms of size growth.
/// Size is growth is calculated based on amount of code duplicated in second
/// loop.
unsigned LoopIndexSplit::findSplitCost(Loop *L, SplitInfo &SD) {

  unsigned Cost = 0;
  BasicBlock *SDBlock = SD.SplitCondition->getParent();
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    // If a block is not dominated by split condition block then
    // it must be duplicated in both loops.
    if (!DT->dominates(SDBlock, BB))
      Cost += BB->size();
  }

  return Cost;
}

bool LoopIndexSplit::splitLoop(SplitInfo &SD) {
  // FIXME :)
  return false;
}
