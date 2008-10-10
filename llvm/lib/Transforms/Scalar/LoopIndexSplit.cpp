//===- LoopIndexSplit.cpp - Loop Index Splitting Pass ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Index Splitting Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-index-split"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

STATISTIC(NumIndexSplit, "Number of loops index split");

namespace {

  class VISIBILITY_HIDDEN LoopIndexSplit : public LoopPass {

  public:
    static char ID; // Pass ID, replacement for typeid
    LoopIndexSplit() : LoopPass(&ID) {}

    // Index split Loop L. Return true if loop is split.
    bool runOnLoop(Loop *L, LPPassManager &LPM);

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }

  private:

    class SplitInfo {
    public:
      SplitInfo() : SplitValue(NULL), SplitCondition(NULL), 
                    UseTrueBranchFirst(true), A_ExitValue(NULL), 
                    B_StartValue(NULL) {}

      // Induction variable's range is split at this value.
      Value *SplitValue;
      
      // This instruction compares IndVar against SplitValue.
      Instruction *SplitCondition;

      // True if after loop index split, first loop will execute split condition's
      // true branch.
      bool UseTrueBranchFirst;

      // Exit value for first loop after loop split.
      Value *A_ExitValue;

      // Start value for second loop after loop split.
      Value *B_StartValue;

      // Clear split info.
      void clear() {
        SplitValue = NULL;
        SplitCondition = NULL;
        UseTrueBranchFirst = true;
        A_ExitValue = NULL;
        B_StartValue = NULL;
      }

    };
    
  private:

    // safeIcmpInst - CI is considered safe instruction if one of the operand
    // is SCEVAddRecExpr based on induction variable and other operand is
    // loop invariant. If CI is safe then populate SplitInfo object SD appropriately
    // and return true;
    bool safeICmpInst(ICmpInst *CI, SplitInfo &SD);

    /// Find condition inside a loop that is suitable candidate for index split.
    void findSplitCondition();

    /// Find loop's exit condition.
    void findLoopConditionals();

    /// Return induction variable associated with value V.
    void findIndVar(Value *V, Loop *L);

    /// processOneIterationLoop - Current loop L contains compare instruction
    /// that compares induction variable, IndVar, agains loop invariant. If
    /// entire (i.e. meaningful) loop body is dominated by this compare
    /// instruction then loop body is executed only for one iteration. In
    /// such case eliminate loop structure surrounding this loop body. For
    bool processOneIterationLoop(SplitInfo &SD);
    
    /// isOneIterationLoop - Return true if split condition is EQ and 
    /// the IV is not used outside the loop.
    bool isOneIterationLoop(ICmpInst *CI);

    void updateLoopBounds(ICmpInst *CI);
    /// updateLoopIterationSpace - Current loop body is covered by an AND
    /// instruction whose operands compares induction variables with loop
    /// invariants. If possible, hoist this check outside the loop by
    /// updating appropriate start and end values for induction variable.
    bool updateLoopIterationSpace(SplitInfo &SD);

    /// If loop header includes loop variant instruction operands then
    /// this loop may not be eliminated.
    bool safeHeader(SplitInfo &SD,  BasicBlock *BB);

    /// If Exiting block includes loop variant instructions then this
    /// loop may not be eliminated.
    bool safeExitingBlock(SplitInfo &SD, BasicBlock *BB);

    /// removeBlocks - Remove basic block DeadBB and all blocks dominated by DeadBB.
    /// This routine is used to remove split condition's dead branch, dominated by
    /// DeadBB. LiveBB dominates split conidition's other branch.
    void removeBlocks(BasicBlock *DeadBB, Loop *LP, BasicBlock *LiveBB);

    /// safeSplitCondition - Return true if it is possible to
    /// split loop using given split condition.
    bool safeSplitCondition(SplitInfo &SD);

    /// calculateLoopBounds - ALoop exit value and BLoop start values are calculated
    /// based on split value. 
    void calculateLoopBounds(SplitInfo &SD);

    /// updatePHINodes - CFG has been changed. 
    /// Before 
    ///   - ExitBB's single predecessor was Latch
    ///   - Latch's second successor was Header
    /// Now
    ///   - ExitBB's single predecessor was Header
    ///   - Latch's one and only successor was Header
    ///
    /// Update ExitBB PHINodes' to reflect this change.
    void updatePHINodes(BasicBlock *ExitBB, BasicBlock *Latch, 
                        BasicBlock *Header,
                        PHINode *IV, Instruction *IVIncrement, Loop *LP);

    /// moveExitCondition - Move exit condition EC into split condition block CondBB.
    void moveExitCondition(BasicBlock *CondBB, BasicBlock *ActiveBB,
                           BasicBlock *ExitBB, ICmpInst *EC, ICmpInst *SC,
                           PHINode *IV, Instruction *IVAdd, Loop *LP);

    /// splitLoop - Split current loop L in two loops using split information
    /// SD. Update dominator information. Maintain LCSSA form.
    bool splitLoop(SplitInfo &SD);

    void initialize() {
      IndVar = NULL; 
      IndVarIncrement = NULL;
      ExitCondition = NULL;
      StartValue = NULL;
      ExitValueNum = 0;
      SplitData.clear();
    }

  private:

    // Current Loop.
    Loop *L;
    LPPassManager *LPM;
    LoopInfo *LI;
    ScalarEvolution *SE;
    DominatorTree *DT;
    DominanceFrontier *DF;
    SmallVector<SplitInfo, 4> SplitData;

    // Induction variable whose range is being split by this transformation.
    PHINode *IndVar;
    Instruction *IndVarIncrement;
      
    // Loop exit condition.
    ICmpInst *ExitCondition;

    // Induction variable's initial value.
    Value *StartValue;

    // Induction variable's final loop exit value operand number in exit condition..
    unsigned ExitValueNum;
  };
}

char LoopIndexSplit::ID = 0;
static RegisterPass<LoopIndexSplit>
X("loop-index-split", "Index Split Loops");

LoopPass *llvm::createLoopIndexSplitPass() {
  return new LoopIndexSplit();
}

// Index split Loop L. Return true if loop is split.
bool LoopIndexSplit::runOnLoop(Loop *IncomingLoop, LPPassManager &LPM_Ref) {
  bool Changed = false;
  L = IncomingLoop;
  LPM = &LPM_Ref;

  // FIXME - Nested loops make dominator info updates tricky. 
  if (!L->getSubLoops().empty())
    return false;

  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  DF = &getAnalysis<DominanceFrontier>();

  initialize();

  findLoopConditionals();

  if (!ExitCondition)
    return false;

  findSplitCondition();

  if (SplitData.empty())
    return false;

  // First see if it is possible to eliminate loop itself or not.
  for (SmallVector<SplitInfo, 4>::iterator SI = SplitData.begin();
       SI != SplitData.end();) {
    SplitInfo &SD = *SI;
    ICmpInst *CI = dyn_cast<ICmpInst>(SD.SplitCondition);
    if (SD.SplitCondition->getOpcode() == Instruction::And) {
      Changed = updateLoopIterationSpace(SD);
      if (Changed) {
        ++NumIndexSplit;
        // If is loop is eliminated then nothing else to do here.
        return Changed;
      } else {
        SmallVector<SplitInfo, 4>::iterator Delete_SI = SI;
        SI = SplitData.erase(Delete_SI);
      }
    }
    else if (isOneIterationLoop(CI)) {
      Changed = processOneIterationLoop(SD);
      if (Changed) {
        ++NumIndexSplit;
        // If is loop is eliminated then nothing else to do here.
        return Changed;
      } else {
        SmallVector<SplitInfo, 4>::iterator Delete_SI = SI;
        SI = SplitData.erase(Delete_SI);
      }
    } else
      ++SI;
  }

  if (SplitData.empty())
    return false;

  // Split most profitiable condition.
  // FIXME : Implement cost analysis.
  unsigned MostProfitableSDIndex = 0;
  Changed = splitLoop(SplitData[MostProfitableSDIndex]);

  if (Changed)
    ++NumIndexSplit;
  
  return Changed;
}

/// isOneIterationLoop - Return true if split condition is EQ and 
/// the IV is not used outside the loop.
bool LoopIndexSplit::isOneIterationLoop(ICmpInst *CI) {
  if (!CI)
    return false;
  if (CI->getPredicate() != ICmpInst::ICMP_EQ)
    return false;

  Value *Incr = IndVar->getIncomingValueForBlock(L->getLoopLatch());
  for (Value::use_iterator UI = Incr->use_begin(), E = Incr->use_end(); 
       UI != E; ++UI)
    if (!L->contains(cast<Instruction>(*UI)->getParent()))
      return false;

  return true;
}
/// Return true if V is a induction variable or induction variable's
/// increment for loop L.
void LoopIndexSplit::findIndVar(Value *V, Loop *L) {
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return;

  // Check if I is a phi node from loop header or not.
  if (PHINode *PN = dyn_cast<PHINode>(V)) {
    if (PN->getParent() == L->getHeader()) {
      IndVar = PN;
      return;
    }
  }
 
  // Check if I is a add instruction whose one operand is
  // phi node from loop header and second operand is constant.
  if (I->getOpcode() != Instruction::Add)
    return;
  
  Value *Op0 = I->getOperand(0);
  Value *Op1 = I->getOperand(1);
  
  if (PHINode *PN = dyn_cast<PHINode>(Op0)) 
    if (PN->getParent() == L->getHeader()) 
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) 
        if (CI->isOne()) {
          IndVar = PN;
          IndVarIncrement = I;
          return;
        }

  if (PHINode *PN = dyn_cast<PHINode>(Op1)) 
    if (PN->getParent() == L->getHeader()) 
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0)) 
        if (CI->isOne()) {
          IndVar = PN;
          IndVarIncrement = I;
          return;
        }
  
  return;
}

// Find loop's exit condition and associated induction variable.
void LoopIndexSplit::findLoopConditionals() {

  BasicBlock *ExitingBlock = NULL;

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (!L->isLoopExit(BB))
      continue;
    if (ExitingBlock)
      return;
    ExitingBlock = BB;
  }

  if (!ExitingBlock)
    return;

  // If exiting block is neither loop header nor loop latch then this loop is
  // not suitable. 
  if (ExitingBlock != L->getHeader() && ExitingBlock != L->getLoopLatch())
    return;

  // If exit block's terminator is conditional branch inst then we have found
  // exit condition.
  BranchInst *BR = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (!BR || BR->isUnconditional())
    return;
  
  ICmpInst *CI = dyn_cast<ICmpInst>(BR->getCondition());
  if (!CI)
    return;

  // FIXME 
  if (CI->getPredicate() == ICmpInst::ICMP_EQ
      || CI->getPredicate() == ICmpInst::ICMP_NE)
    return;

  ExitCondition = CI;

  // Exit condition's one operand is loop invariant exit value and second 
  // operand is SCEVAddRecExpr based on induction variable.
  Value *V0 = CI->getOperand(0);
  Value *V1 = CI->getOperand(1);
  
  SCEVHandle SH0 = SE->getSCEV(V0);
  SCEVHandle SH1 = SE->getSCEV(V1);
  
  if (SH0->isLoopInvariant(L) && isa<SCEVAddRecExpr>(SH1)) {
    ExitValueNum = 0;
    findIndVar(V1, L);
  }
  else if (SH1->isLoopInvariant(L) && isa<SCEVAddRecExpr>(SH0)) {
    ExitValueNum =  1;
    findIndVar(V0, L);
  }

  if (!IndVar) 
    ExitCondition = NULL;
  else if (IndVar) {
    BasicBlock *Preheader = L->getLoopPreheader();
    StartValue = IndVar->getIncomingValueForBlock(Preheader);
  }

  // If start value is more then exit value where induction variable
  // increments by 1 then we are potentially dealing with an infinite loop.
  // Do not index split this loop.
  if (ExitCondition) {
    ConstantInt *SV = dyn_cast<ConstantInt>(StartValue);
    ConstantInt *EV = 
      dyn_cast<ConstantInt>(ExitCondition->getOperand(ExitValueNum));
    if (SV && EV && SV->getSExtValue() > EV->getSExtValue())
      ExitCondition = NULL;
    else if (EV && EV->isZero())
      ExitCondition = NULL;
  }
}

/// Find condition inside a loop that is suitable candidate for index split.
void LoopIndexSplit::findSplitCondition() {

  SplitInfo SD;
  // Check all basic block's terminators.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    SD.clear();
    BasicBlock *BB = *I;

    // If this basic block does not terminate in a conditional branch
    // then terminator is not a suitable split condition.
    BranchInst *BR = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BR)
      continue;
    
    if (BR->isUnconditional())
      continue;

    if (Instruction *AndI = dyn_cast<Instruction>(BR->getCondition())) {
      if (AndI->getOpcode() == Instruction::And) {
        ICmpInst *Op0 = dyn_cast<ICmpInst>(AndI->getOperand(0));
        ICmpInst *Op1 = dyn_cast<ICmpInst>(AndI->getOperand(1));

        if (!Op0 || !Op1)
          continue;

        if (!safeICmpInst(Op0, SD))
          continue;
        SD.clear();
        if (!safeICmpInst(Op1, SD))
          continue;
        SD.clear();
        SD.SplitCondition = AndI;
        SplitData.push_back(SD);
        continue;
      }
    }
    ICmpInst *CI = dyn_cast<ICmpInst>(BR->getCondition());
    if (!CI || CI == ExitCondition)
      continue;

    if (CI->getPredicate() == ICmpInst::ICMP_NE)
      continue;

    // If split condition predicate is GT or GE then first execute
    // false branch of split condition.
    if (CI->getPredicate() == ICmpInst::ICMP_UGT
        || CI->getPredicate() == ICmpInst::ICMP_SGT
        || CI->getPredicate() == ICmpInst::ICMP_UGE
        || CI->getPredicate() == ICmpInst::ICMP_SGE)
      SD.UseTrueBranchFirst = false;

    // If one operand is loop invariant and second operand is SCEVAddRecExpr
    // based on induction variable then CI is a candidate split condition.
    if (safeICmpInst(CI, SD))
      SplitData.push_back(SD);
  }
}

// safeIcmpInst - CI is considered safe instruction if one of the operand
// is SCEVAddRecExpr based on induction variable and other operand is
// loop invariant. If CI is safe then populate SplitInfo object SD appropriately
// and return true;
bool LoopIndexSplit::safeICmpInst(ICmpInst *CI, SplitInfo &SD) {

  Value *V0 = CI->getOperand(0);
  Value *V1 = CI->getOperand(1);
  
  SCEVHandle SH0 = SE->getSCEV(V0);
  SCEVHandle SH1 = SE->getSCEV(V1);
  
  if (SH0->isLoopInvariant(L) && isa<SCEVAddRecExpr>(SH1)) {
    SD.SplitValue = V0;
    SD.SplitCondition = CI;
    if (PHINode *PN = dyn_cast<PHINode>(V1)) {
      if (PN == IndVar)
        return true;
    }
    else  if (Instruction *Insn = dyn_cast<Instruction>(V1)) {
      if (IndVarIncrement && IndVarIncrement == Insn)
        return true;
    }
  }
  else if (SH1->isLoopInvariant(L) && isa<SCEVAddRecExpr>(SH0)) {
    SD.SplitValue =  V1;
    SD.SplitCondition = CI;
    if (PHINode *PN = dyn_cast<PHINode>(V0)) {
      if (PN == IndVar)
        return true;
    }
    else  if (Instruction *Insn = dyn_cast<Instruction>(V0)) {
      if (IndVarIncrement && IndVarIncrement == Insn)
        return true;
    }
  }

  return false;
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
bool LoopIndexSplit::processOneIterationLoop(SplitInfo &SD) {

  BasicBlock *Header = L->getHeader();

  // First of all, check if SplitCondition dominates entire loop body
  // or not.
  
  // If SplitCondition is not in loop header then this loop is not suitable
  // for this transformation.
  if (SD.SplitCondition->getParent() != Header)
    return false;
  
  // If loop header includes loop variant instruction operands then
  // this loop may not be eliminated.
  if (!safeHeader(SD, Header)) 
    return false;

  // If Exiting block includes loop variant instructions then this
  // loop may not be eliminated.
  if (!safeExitingBlock(SD, ExitCondition->getParent())) 
    return false;

  // Filter loops where split condition's false branch is not empty.
  if (ExitCondition->getParent() != Header->getTerminator()->getSuccessor(1))
    return false;

  // If split condition is not safe then do not process this loop.
  // For example,
  // for(int i = 0; i < N; i++) {
  //    if ( i == XYZ) {
  //      A;
  //    else
  //      B;
  //    }
  //   C;
  //   D;
  // }
  if (!safeSplitCondition(SD))
    return false;

  BasicBlock *Latch = L->getLoopLatch();
  BranchInst *BR = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!BR)
    return false;

  // Update CFG.

  // Replace index variable with split value in loop body. Loop body is executed
  // only when index variable is equal to split value.
  IndVar->replaceAllUsesWith(SD.SplitValue);

  Instruction *LTerminator = Latch->getTerminator();
  Instruction *Terminator = Header->getTerminator();
  Value *ExitValue = ExitCondition->getOperand(ExitValueNum);

  // Replace split condition in header.
  // Transform 
  //      SplitCondition : icmp eq i32 IndVar, SplitValue
  // into
  //      c1 = icmp uge i32 SplitValue, StartValue
  //      c2 = icmp ult i32 SplitValue, ExitValue
  //      and i32 c1, c2 
  bool SignedPredicate = ExitCondition->isSignedPredicate();
  CmpInst::Predicate C2Predicate = ExitCondition->getPredicate();
  if (LTerminator->getOperand(0) != Header)
    C2Predicate = CmpInst::getInversePredicate(C2Predicate);
  Instruction *C1 = new ICmpInst(SignedPredicate ? 
                                 ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE,
                                 SD.SplitValue, StartValue, "lisplit", 
                                 Terminator);
  Instruction *C2 = new ICmpInst(C2Predicate,
                                 SD.SplitValue, ExitValue, "lisplit", 
                                 Terminator);
  Instruction *NSplitCond = BinaryOperator::CreateAnd(C1, C2, "lisplit", 
                                                      Terminator);
  SD.SplitCondition->replaceAllUsesWith(NSplitCond);
  SD.SplitCondition->eraseFromParent();

  // Remove Latch to Header edge.
  BasicBlock *LatchSucc = NULL;
  Header->removePredecessor(Latch);
  for (succ_iterator SI = succ_begin(Latch), E = succ_end(Latch);
       SI != E; ++SI) {
    if (Header != *SI)
      LatchSucc = *SI;
  }
  BR->setUnconditionalDest(LatchSucc);

  // Now, clear latch block. Remove instructions that are responsible
  // to increment induction variable. 
  for (BasicBlock::iterator LB = Latch->begin(), LE = Latch->end();
       LB != LE; ) {
    Instruction *I = LB;
    ++LB;
    if (isa<PHINode>(I) || I == LTerminator)
      continue;

    if (I == IndVarIncrement) {
      // Replace induction variable increment if it is not used outside 
      // the loop.
      bool UsedOutsideLoop = false;
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); 
           UI != E; ++UI) {
        if (Instruction *Use = dyn_cast<Instruction>(UI)) 
          if (!L->contains(Use->getParent())) {
            UsedOutsideLoop = true;
            break;
          }
      }
      if (!UsedOutsideLoop) {
        I->replaceAllUsesWith(ExitValue);
        I->eraseFromParent();
      }
    }
    else {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  LPM->deleteLoopFromQueue(L);

  // Update Dominator Info.
  // Only CFG change done is to remove Latch to Header edge. This
  // does not change dominator tree because Latch did not dominate
  // Header.
  if (DF) {
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

    // PHI Nodes are OK.
    if (isa<PHINode>(I))
      continue;

    // SplitCondition itself is OK.
    if (I == SD.SplitCondition)
      continue;

    // Induction variable is OK.
    if (I == IndVar)
      continue;

    // Induction variable increment is OK.
    if (I == IndVarIncrement)
      continue;

    // Terminator is also harmless.
    if (I == Terminator)
      continue;

    // Otherwise we have a instruction that may not be safe.
    return false;
  }
  
  return true;
}

// If Exiting block includes loop variant instructions then this
// loop may not be eliminated. This is used by processOneIterationLoop().
bool LoopIndexSplit::safeExitingBlock(SplitInfo &SD, 
                                       BasicBlock *ExitingBlock) {

  for (BasicBlock::iterator BI = ExitingBlock->begin(), 
         BE = ExitingBlock->end(); BI != BE; ++BI) {
    Instruction *I = BI;

    // PHI Nodes are OK.
    if (isa<PHINode>(I))
      continue;

    // Induction variable increment is OK.
    if (IndVarIncrement && IndVarIncrement == I)
      continue;

    // Check if I is induction variable increment instruction.
    if (I->getOpcode() == Instruction::Add) {

      Value *Op0 = I->getOperand(0);
      Value *Op1 = I->getOperand(1);
      PHINode *PN = NULL;
      ConstantInt *CI = NULL;

      if ((PN = dyn_cast<PHINode>(Op0))) {
        if ((CI = dyn_cast<ConstantInt>(Op1)))
          if (CI->isOne()) {
            if (!IndVarIncrement && PN == IndVar)
              IndVarIncrement = I;
            // else this is another loop induction variable
            continue;
          }
      } else 
        if ((PN = dyn_cast<PHINode>(Op1))) {
          if ((CI = dyn_cast<ConstantInt>(Op0)))
            if (CI->isOne()) {
              if (!IndVarIncrement && PN == IndVar)
                IndVarIncrement = I;
              // else this is another loop induction variable
              continue;
            }
      }
    } 

    // I is an Exit condition if next instruction is block terminator.
    // Exit condition is OK if it compares loop invariant exit value,
    // which is checked below.
    else if (ICmpInst *EC = dyn_cast<ICmpInst>(I)) {
      if (EC == ExitCondition)
        continue;
    }

    if (I == ExitingBlock->getTerminator())
      continue;

    // Otherwise we have instruction that may not be safe.
    return false;
  }

  // We could not find any reason to consider ExitingBlock unsafe.
  return true;
}

void LoopIndexSplit::updateLoopBounds(ICmpInst *CI) {

  Value *V0 = CI->getOperand(0);
  Value *V1 = CI->getOperand(1);
  Value *NV = NULL;

  SCEVHandle SH0 = SE->getSCEV(V0);
  
  if (SH0->isLoopInvariant(L))
    NV = V0;
  else
    NV = V1;

  if (ExitCondition->getPredicate() == ICmpInst::ICMP_SGT
      || ExitCondition->getPredicate() == ICmpInst::ICMP_UGT
      || ExitCondition->getPredicate() == ICmpInst::ICMP_SGE
      || ExitCondition->getPredicate() == ICmpInst::ICMP_UGE)  {
    ExitCondition->swapOperands();
    if (ExitValueNum)
      ExitValueNum = 0;
    else
      ExitValueNum = 1;
  }

  Value *NUB = NULL;
  Value *NLB = NULL;
  Value *UB = ExitCondition->getOperand(ExitValueNum);
  const Type *Ty = NV->getType();
  bool Sign = ExitCondition->isSignedPredicate();
  BasicBlock *Preheader = L->getLoopPreheader();
  Instruction *PHTerminator = Preheader->getTerminator();

  assert (NV && "Unexpected value");

  switch (CI->getPredicate()) {
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE:
    // for (i = LB; i < UB; ++i)
    //   if (i <= NV && ...)
    //      LOOP_BODY
    // 
    // is transformed into
    // NUB = min (NV+1, UB)
    // for (i = LB; i < NUB ; ++i)
    //   LOOP_BODY
    //
    if (ExitCondition->getPredicate() == ICmpInst::ICMP_SLT
        || ExitCondition->getPredicate() == ICmpInst::ICMP_ULT) {
      Value *A = BinaryOperator::CreateAdd(NV, ConstantInt::get(Ty, 1, Sign),
                                           "lsplit.add", PHTerminator);
      Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              A, UB,"lsplit,c", PHTerminator);
      NUB = SelectInst::Create(C, A, UB, "lsplit.nub", PHTerminator);
    }
    
    // for (i = LB; i <= UB; ++i)
    //   if (i <= NV && ...)
    //      LOOP_BODY
    // 
    // is transformed into
    // NUB = min (NV, UB)
    // for (i = LB; i <= NUB ; ++i)
    //   LOOP_BODY
    //
    else if (ExitCondition->getPredicate() == ICmpInst::ICMP_SLE
             || ExitCondition->getPredicate() == ICmpInst::ICMP_ULE) {
      Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              NV, UB, "lsplit.c", PHTerminator);
      NUB = SelectInst::Create(C, NV, UB, "lsplit.nub", PHTerminator);
    }
    break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
    // for (i = LB; i < UB; ++i)
    //   if (i < NV && ...)
    //      LOOP_BODY
    // 
    // is transformed into
    // NUB = min (NV, UB)
    // for (i = LB; i < NUB ; ++i)
    //   LOOP_BODY
    //
    if (ExitCondition->getPredicate() == ICmpInst::ICMP_SLT
        || ExitCondition->getPredicate() == ICmpInst::ICMP_ULT) {
      Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              NV, UB, "lsplit.c", PHTerminator);
      NUB = SelectInst::Create(C, NV, UB, "lsplit.nub", PHTerminator);
    }

    // for (i = LB; i <= UB; ++i)
    //   if (i < NV && ...)
    //      LOOP_BODY
    // 
    // is transformed into
    // NUB = min (NV -1 , UB)
    // for (i = LB; i <= NUB ; ++i)
    //   LOOP_BODY
    //
    else if (ExitCondition->getPredicate() == ICmpInst::ICMP_SLE
             || ExitCondition->getPredicate() == ICmpInst::ICMP_ULE) {
      Value *S = BinaryOperator::CreateSub(NV, ConstantInt::get(Ty, 1, Sign),
                                           "lsplit.add", PHTerminator);
      Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              S, UB, "lsplit.c", PHTerminator);
      NUB = SelectInst::Create(C, S, UB, "lsplit.nub", PHTerminator);
    }
    break;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
    // for (i = LB; i (< or <=) UB; ++i)
    //   if (i >= NV && ...)
    //      LOOP_BODY
    // 
    // is transformed into
    // NLB = max (NV, LB)
    // for (i = NLB; i (< or <=) UB ; ++i)
    //   LOOP_BODY
    //
    {
      Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              NV, StartValue, "lsplit.c", PHTerminator);
      NLB = SelectInst::Create(C, StartValue, NV, "lsplit.nlb", PHTerminator);
    }
    break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
    // for (i = LB; i (< or <=) UB; ++i)
    //   if (i > NV && ...)
    //      LOOP_BODY
    // 
    // is transformed into
    // NLB = max (NV+1, LB)
    // for (i = NLB; i (< or <=) UB ; ++i)
    //   LOOP_BODY
    //
    {
      Value *A = BinaryOperator::CreateAdd(NV, ConstantInt::get(Ty, 1, Sign),
                                           "lsplit.add", PHTerminator);
      Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              A, StartValue, "lsplit.c", PHTerminator);
      NLB = SelectInst::Create(C, StartValue, A, "lsplit.nlb", PHTerminator);
    }
    break;
  default:
    assert ( 0 && "Unexpected split condition predicate");
  }

  if (NLB) {
    unsigned i = IndVar->getBasicBlockIndex(Preheader);
    IndVar->setIncomingValue(i, NLB);
  }

  if (NUB) {
    ExitCondition->setOperand(ExitValueNum, NUB);
  }
}
/// updateLoopIterationSpace - Current loop body is covered by an AND
/// instruction whose operands compares induction variables with loop
/// invariants. If possible, hoist this check outside the loop by
/// updating appropriate start and end values for induction variable.
bool LoopIndexSplit::updateLoopIterationSpace(SplitInfo &SD) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *ExitingBlock = ExitCondition->getParent();
  BasicBlock *SplitCondBlock = SD.SplitCondition->getParent();

  ICmpInst *Op0 = cast<ICmpInst>(SD.SplitCondition->getOperand(0));
  ICmpInst *Op1 = cast<ICmpInst>(SD.SplitCondition->getOperand(1));

  if (Op0->getPredicate() == ICmpInst::ICMP_EQ 
      || Op0->getPredicate() == ICmpInst::ICMP_NE
      || Op1->getPredicate() == ICmpInst::ICMP_EQ 
      || Op1->getPredicate() == ICmpInst::ICMP_NE)
    return false;

  // Check if SplitCondition dominates entire loop body
  // or not.
  
  // If SplitCondition is not in loop header then this loop is not suitable
  // for this transformation.
  if (SD.SplitCondition->getParent() != Header)
    return false;
  
  // If loop header includes loop variant instruction operands then
  // this loop may not be eliminated.
  Instruction *Terminator = Header->getTerminator();
  for(BasicBlock::iterator BI = Header->begin(), BE = Header->end(); 
      BI != BE; ++BI) {
    Instruction *I = BI;

    // PHI Nodes are OK.
    if (isa<PHINode>(I))
      continue;

    // SplitCondition itself is OK.
    if (I == SD.SplitCondition)
      continue;
    if (I == Op0 || I == Op1)
      continue;

    // Induction variable is OK.
    if (I == IndVar)
      continue;

    // Induction variable increment is OK.
    if (I == IndVarIncrement)
      continue;

    // Terminator is also harmless.
    if (I == Terminator)
      continue;

    // Otherwise we have a instruction that may not be safe.
    return false;
  }

  // If Exiting block includes loop variant instructions then this
  // loop may not be eliminated.
  if (!safeExitingBlock(SD, ExitCondition->getParent())) 
    return false;
  
  // Verify that loop exiting block has only two predecessor, where one predecessor
  // is split condition block. The other predecessor will become exiting block's
  // dominator after CFG is updated. TODO : Handle CFG's where exiting block has
  // more then two predecessors. This requires extra work in updating dominator
  // information.
  BasicBlock *ExitingBBPred = NULL;
  for (pred_iterator PI = pred_begin(ExitingBlock), PE = pred_end(ExitingBlock);
       PI != PE; ++PI) {
    BasicBlock *BB = *PI;
    if (SplitCondBlock == BB) 
      continue;
    if (ExitingBBPred)
      return false;
    else
      ExitingBBPred = BB;
  }
  
  // Update loop bounds to absorb Op0 check.
  updateLoopBounds(Op0);
  // Update loop bounds to absorb Op1 check.
  updateLoopBounds(Op1);

  // Update CFG

  // Unconditionally connect split block to its remaining successor. 
  BranchInst *SplitTerminator = 
    cast<BranchInst>(SplitCondBlock->getTerminator());
  BasicBlock *Succ0 = SplitTerminator->getSuccessor(0);
  BasicBlock *Succ1 = SplitTerminator->getSuccessor(1);
  if (Succ0 == ExitCondition->getParent())
    SplitTerminator->setUnconditionalDest(Succ1);
  else
    SplitTerminator->setUnconditionalDest(Succ0);

  // Remove split condition.
  SD.SplitCondition->eraseFromParent();
  if (Op0->use_empty())
    Op0->eraseFromParent();
  if (Op1->use_empty())
    Op1->eraseFromParent();
      
  BranchInst *ExitInsn =
    dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  assert (ExitInsn && "Unable to find suitable loop exit branch");
  BasicBlock *ExitBlock = ExitInsn->getSuccessor(1);
  if (L->contains(ExitBlock))
    ExitBlock = ExitInsn->getSuccessor(0);

  // Update domiantor info. Now, ExitingBlock has only one predecessor, 
  // ExitingBBPred, and it is ExitingBlock's immediate domiantor.
  DT->changeImmediateDominator(ExitingBlock, ExitingBBPred);
  
  // If ExitingBlock is a member of loop BB's DF list then replace it with
  // loop header and exit block.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (BB == Header || BB == ExitingBlock)
      continue;
    DominanceFrontier::iterator BBDF = DF->find(BB);
    DominanceFrontier::DomSetType::iterator DomSetI = BBDF->second.begin();
    DominanceFrontier::DomSetType::iterator DomSetE = BBDF->second.end();
    while (DomSetI != DomSetE) {
      DominanceFrontier::DomSetType::iterator CurrentItr = DomSetI;
      ++DomSetI;
      BasicBlock *DFBB = *CurrentItr;
      if (DFBB == ExitingBlock) {
        BBDF->second.erase(DFBB);
        BBDF->second.insert(Header);
        if (Header != ExitingBlock)
          BBDF->second.insert(ExitBlock);
      }
    }
  }

  return true;
}


/// removeBlocks - Remove basic block DeadBB and all blocks dominated by DeadBB.
/// This routine is used to remove split condition's dead branch, dominated by
/// DeadBB. LiveBB dominates split conidition's other branch.
void LoopIndexSplit::removeBlocks(BasicBlock *DeadBB, Loop *LP, 
                                  BasicBlock *LiveBB) {

  // First update DeadBB's dominance frontier. 
  SmallVector<BasicBlock *, 8> FrontierBBs;
  DominanceFrontier::iterator DeadBBDF = DF->find(DeadBB);
  if (DeadBBDF != DF->end()) {
    SmallVector<BasicBlock *, 8> PredBlocks;
    
    DominanceFrontier::DomSetType DeadBBSet = DeadBBDF->second;
    for (DominanceFrontier::DomSetType::iterator DeadBBSetI = DeadBBSet.begin(),
           DeadBBSetE = DeadBBSet.end(); DeadBBSetI != DeadBBSetE; ++DeadBBSetI) {
      BasicBlock *FrontierBB = *DeadBBSetI;
      FrontierBBs.push_back(FrontierBB);

      // Rremove any PHI incoming edge from blocks dominated by DeadBB.
      PredBlocks.clear();
      for(pred_iterator PI = pred_begin(FrontierBB), PE = pred_end(FrontierBB);
          PI != PE; ++PI) {
        BasicBlock *P = *PI;
        if (P == DeadBB || DT->dominates(DeadBB, P))
          PredBlocks.push_back(P);
      }

      for(BasicBlock::iterator FBI = FrontierBB->begin(), FBE = FrontierBB->end();
          FBI != FBE; ++FBI) {
        if (PHINode *PN = dyn_cast<PHINode>(FBI)) {
          for(SmallVector<BasicBlock *, 8>::iterator PI = PredBlocks.begin(),
                PE = PredBlocks.end(); PI != PE; ++PI) {
            BasicBlock *P = *PI;
            PN->removeIncomingValue(P);
          }
        }
        else
          break;
      }      
    }
  }
  
  // Now remove DeadBB and all nodes dominated by DeadBB in df order.
  SmallVector<BasicBlock *, 32> WorkList;
  DomTreeNode *DN = DT->getNode(DeadBB);
  for (df_iterator<DomTreeNode*> DI = df_begin(DN),
         E = df_end(DN); DI != E; ++DI) {
    BasicBlock *BB = DI->getBlock();
    WorkList.push_back(BB);
    BB->replaceAllUsesWith(UndefValue::get(Type::LabelTy));
  }

  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.back(); WorkList.pop_back();
    for(BasicBlock::iterator BBI = BB->begin(), BBE = BB->end(); 
        BBI != BBE; ) {
      Instruction *I = BBI;
      ++BBI;
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
    LPM->deleteSimpleAnalysisValue(BB, LP);
    DT->eraseNode(BB);
    DF->removeBlock(BB);
    LI->removeBlock(BB);
    BB->eraseFromParent();
  }

  // Update Frontier BBs' dominator info.
  while (!FrontierBBs.empty()) {
    BasicBlock *FBB = FrontierBBs.back(); FrontierBBs.pop_back();
    BasicBlock *NewDominator = FBB->getSinglePredecessor();
    if (!NewDominator) {
      pred_iterator PI = pred_begin(FBB), PE = pred_end(FBB);
      NewDominator = *PI;
      ++PI;
      if (NewDominator != LiveBB) {
        for(; PI != PE; ++PI) {
          BasicBlock *P = *PI;
          if (P == LiveBB) {
            NewDominator = LiveBB;
            break;
          }
          NewDominator = DT->findNearestCommonDominator(NewDominator, P);
        }
      }
    }
    assert (NewDominator && "Unable to fix dominator info.");
    DT->changeImmediateDominator(FBB, NewDominator);
    DF->changeImmediateDominator(FBB, NewDominator, DT);
  }

}

/// safeSplitCondition - Return true if it is possible to
/// split loop using given split condition.
bool LoopIndexSplit::safeSplitCondition(SplitInfo &SD) {

  BasicBlock *SplitCondBlock = SD.SplitCondition->getParent();
  BasicBlock *Latch = L->getLoopLatch();  
  BranchInst *SplitTerminator = 
    cast<BranchInst>(SplitCondBlock->getTerminator());
  BasicBlock *Succ0 = SplitTerminator->getSuccessor(0);
  BasicBlock *Succ1 = SplitTerminator->getSuccessor(1);

  // If split block does not dominate the latch then this is not a diamond.
  // Such loop may not benefit from index split.
  if (!DT->dominates(SplitCondBlock, Latch))
    return false;

  // Finally this split condition is safe only if merge point for
  // split condition branch is loop latch. This check along with previous
  // check, to ensure that exit condition is in either loop latch or header,
  // filters all loops with non-empty loop body between merge point
  // and exit condition.
  DominanceFrontier::iterator Succ0DF = DF->find(Succ0);
  assert (Succ0DF != DF->end() && "Unable to find Succ0 dominance frontier");
  if (Succ0DF->second.count(Latch))
    return true;

  DominanceFrontier::iterator Succ1DF = DF->find(Succ1);
  assert (Succ1DF != DF->end() && "Unable to find Succ1 dominance frontier");
  if (Succ1DF->second.count(Latch))
    return true;
  
  return false;
}

/// calculateLoopBounds - ALoop exit value and BLoop start values are calculated
/// based on split value. 
void LoopIndexSplit::calculateLoopBounds(SplitInfo &SD) {

  ICmpInst *SC = cast<ICmpInst>(SD.SplitCondition);
  ICmpInst::Predicate SP = SC->getPredicate();
  const Type *Ty = SD.SplitValue->getType();
  bool Sign = ExitCondition->isSignedPredicate();
  BasicBlock *Preheader = L->getLoopPreheader();
  Instruction *PHTerminator = Preheader->getTerminator();

  // Initially use split value as upper loop bound for first loop and lower loop
  // bound for second loop.
  Value *AEV = SD.SplitValue;
  Value *BSV = SD.SplitValue;

  if (ExitCondition->getPredicate() == ICmpInst::ICMP_SGT
      || ExitCondition->getPredicate() == ICmpInst::ICMP_UGT
      || ExitCondition->getPredicate() == ICmpInst::ICMP_SGE
      || ExitCondition->getPredicate() == ICmpInst::ICMP_UGE) {
    ExitCondition->swapOperands();
    if (ExitValueNum)
      ExitValueNum = 0;
    else
      ExitValueNum = 1;
  }

  switch (ExitCondition->getPredicate()) {
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGE:
  case ICmpInst::ICMP_UGE:
  default:
    assert (0 && "Unexpected exit condition predicate");

  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_ULT:
    {
      switch (SP) {
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_ULT:
        //
        // for (i = LB; i < UB; ++i) { if (i < SV) A; else B; }
        //
        // is transformed into
        // AEV = BSV = SV
        // for (i = LB; i < min(UB, AEV); ++i)
        //    A;
        // for (i = max(LB, BSV); i < UB; ++i);
        //    B;
        break;
      case ICmpInst::ICMP_SLE:
      case ICmpInst::ICMP_ULE:
        {
          //
          // for (i = LB; i < UB; ++i) { if (i <= SV) A; else B; }
          //
          // is transformed into
          //
          // AEV = SV + 1
          // BSV = SV + 1
          // for (i = LB; i < min(UB, AEV); ++i) 
          //       A;
          // for (i = max(LB, BSV); i < UB; ++i) 
          //       B;
          BSV = BinaryOperator::CreateAdd(SD.SplitValue,
                                          ConstantInt::get(Ty, 1, Sign),
                                          "lsplit.add", PHTerminator);
          AEV = BSV;
        }
        break;
      case ICmpInst::ICMP_SGE:
      case ICmpInst::ICMP_UGE: 
        //
        // for (i = LB; i < UB; ++i) { if (i >= SV) A; else B; }
        // 
        // is transformed into
        // AEV = BSV = SV
        // for (i = LB; i < min(UB, AEV); ++i)
        //    B;
        // for (i = max(BSV, LB); i < UB; ++i)
        //    A;
        break;
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_UGT: 
        {
          //
          // for (i = LB; i < UB; ++i) { if (i > SV) A; else B; }
          //
          // is transformed into
          //
          // BSV = AEV = SV + 1
          // for (i = LB; i < min(UB, AEV); ++i) 
          //       B;
          // for (i = max(LB, BSV); i < UB; ++i) 
          //       A;
          BSV = BinaryOperator::CreateAdd(SD.SplitValue,
                                          ConstantInt::get(Ty, 1, Sign),
                                          "lsplit.add", PHTerminator);
          AEV = BSV;
        }
        break;
      default:
        assert (0 && "Unexpected split condition predicate");
        break;
      } // end switch (SP)
    }
    break;
  case ICmpInst::ICMP_SLE:
  case ICmpInst::ICMP_ULE:
    {
      switch (SP) {
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_ULT:
        //
        // for (i = LB; i <= UB; ++i) { if (i < SV) A; else B; }
        //
        // is transformed into
        // AEV = SV - 1;
        // BSV = SV;
        // for (i = LB; i <= min(UB, AEV); ++i) 
        //       A;
        // for (i = max(LB, BSV); i <= UB; ++i) 
        //       B;
        AEV = BinaryOperator::CreateSub(SD.SplitValue,
                                        ConstantInt::get(Ty, 1, Sign),
                                        "lsplit.sub", PHTerminator);
        break;
      case ICmpInst::ICMP_SLE:
      case ICmpInst::ICMP_ULE:
        //
        // for (i = LB; i <= UB; ++i) { if (i <= SV) A; else B; }
        //
        // is transformed into
        // AEV = SV;
        // BSV = SV + 1;
        // for (i = LB; i <= min(UB, AEV); ++i) 
        //       A;
        // for (i = max(LB, BSV); i <= UB; ++i) 
        //       B;
        BSV = BinaryOperator::CreateAdd(SD.SplitValue,
                                        ConstantInt::get(Ty, 1, Sign),
                                        "lsplit.add", PHTerminator);
        break;
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_UGT: 
        //
        // for (i = LB; i <= UB; ++i) { if (i > SV) A; else B; }
        //
        // is transformed into
        // AEV = SV;
        // BSV = SV + 1;
        // for (i = LB; i <= min(AEV, UB); ++i)
        //      B;
        // for (i = max(LB, BSV); i <= UB; ++i)
        //      A;
        BSV = BinaryOperator::CreateAdd(SD.SplitValue,
                                        ConstantInt::get(Ty, 1, Sign),
                                        "lsplit.add", PHTerminator);
        break;
      case ICmpInst::ICMP_SGE:
      case ICmpInst::ICMP_UGE: 
        // ** TODO **
        //
        // for (i = LB; i <= UB; ++i) { if (i >= SV) A; else B; }
        //
        // is transformed into
        // AEV = SV - 1;
        // BSV = SV;
        // for (i = LB; i <= min(AEV, UB); ++i)
        //      B;
        // for (i = max(LB, BSV); i <= UB; ++i)
        //      A;
        AEV = BinaryOperator::CreateSub(SD.SplitValue,
                                        ConstantInt::get(Ty, 1, Sign),
                                        "lsplit.sub", PHTerminator);
        break;
      default:
        assert (0 && "Unexpected split condition predicate");
        break;
      } // end switch (SP)
    }
    break;
  }

  // Calculate ALoop induction variable's new exiting value and
  // BLoop induction variable's new starting value. Calculuate these
  // values in original loop's preheader.
  //      A_ExitValue = min(SplitValue, OrignalLoopExitValue)
  //      B_StartValue = max(SplitValue, OriginalLoopStartValue)
  Instruction *InsertPt = L->getHeader()->getFirstNonPHI();

  // If ExitValue operand is also defined in Loop header then
  // insert new ExitValue after this operand definition.
  if (Instruction *EVN = 
      dyn_cast<Instruction>(ExitCondition->getOperand(ExitValueNum))) {
    if (!isa<PHINode>(EVN))
      if (InsertPt->getParent() == EVN->getParent()) {
        BasicBlock::iterator LHBI = L->getHeader()->begin();
        BasicBlock::iterator LHBE = L->getHeader()->end();  
        for(;LHBI != LHBE; ++LHBI) {
          Instruction *I = LHBI;
          if (I == EVN) 
            break;
        }
        InsertPt = ++LHBI;
      }
  }
  Value *C1 = new ICmpInst(Sign ?
                           ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                           AEV,
                           ExitCondition->getOperand(ExitValueNum), 
                           "lsplit.ev", InsertPt);

  SD.A_ExitValue = SelectInst::Create(C1, AEV,
                                      ExitCondition->getOperand(ExitValueNum), 
                                      "lsplit.ev", InsertPt);

  Value *C2 = new ICmpInst(Sign ?
                           ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                           BSV, StartValue, "lsplit.sv",
                           PHTerminator);
  SD.B_StartValue = SelectInst::Create(C2, StartValue, BSV,
                                       "lsplit.sv", PHTerminator);
}

/// splitLoop - Split current loop L in two loops using split information
/// SD. Update dominator information. Maintain LCSSA form.
bool LoopIndexSplit::splitLoop(SplitInfo &SD) {

  if (!safeSplitCondition(SD))
    return false;

  // If split condition EQ is not handled.
  if (ICmpInst *ICMP = dyn_cast<ICmpInst>(SD.SplitCondition)) {
    if (ICMP->getPredicate() == ICmpInst::ICMP_EQ)
      return false;
  }
  
  BasicBlock *SplitCondBlock = SD.SplitCondition->getParent();
  
  // Unable to handle triangle loops at the moment.
  // In triangle loop, split condition is in header and one of the
  // the split destination is loop latch. If split condition is EQ
  // then such loops are already handle in processOneIterationLoop().
  BasicBlock *Latch = L->getLoopLatch();
  BranchInst *SplitTerminator = 
    cast<BranchInst>(SplitCondBlock->getTerminator());
  BasicBlock *Succ0 = SplitTerminator->getSuccessor(0);
  BasicBlock *Succ1 = SplitTerminator->getSuccessor(1);
  if (L->getHeader() == SplitCondBlock 
      && (Latch == Succ0 || Latch == Succ1))
    return false;

  // If split condition branches heads do not have single predecessor, 
  // SplitCondBlock, then is not possible to remove inactive branch.
  if (!Succ0->getSinglePredecessor() || !Succ1->getSinglePredecessor())
    return false;

  // If Exiting block includes loop variant instructions then this
  // loop may not be split safely.
  if (!safeExitingBlock(SD, ExitCondition->getParent())) 
    return false;

  // After loop is cloned there are two loops.
  //
  // First loop, referred as ALoop, executes first part of loop's iteration
  // space split.  Second loop, referred as BLoop, executes remaining
  // part of loop's iteration space. 
  //
  // ALoop's exit edge enters BLoop's header through a forwarding block which 
  // acts as a BLoop's preheader.
  BasicBlock *Preheader = L->getLoopPreheader();

  // Calculate ALoop induction variable's new exiting value and
  // BLoop induction variable's new starting value.
  calculateLoopBounds(SD);

  //[*] Clone loop.
  DenseMap<const Value *, Value *> ValueMap;
  Loop *BLoop = CloneLoop(L, LPM, LI, ValueMap, this);
  Loop *ALoop = L;
  BasicBlock *B_Header = BLoop->getHeader();

  //[*] ALoop's exiting edge BLoop's header.
  //    ALoop's original exit block becomes BLoop's exit block.
  PHINode *B_IndVar = cast<PHINode>(ValueMap[IndVar]);
  BasicBlock *A_ExitingBlock = ExitCondition->getParent();
  BranchInst *A_ExitInsn =
    dyn_cast<BranchInst>(A_ExitingBlock->getTerminator());
  assert (A_ExitInsn && "Unable to find suitable loop exit branch");
  BasicBlock *B_ExitBlock = A_ExitInsn->getSuccessor(1);
  if (L->contains(B_ExitBlock)) {
    B_ExitBlock = A_ExitInsn->getSuccessor(0);
    A_ExitInsn->setSuccessor(0, B_Header);
  } else
    A_ExitInsn->setSuccessor(1, B_Header);

  //[*] Update ALoop's exit value using new exit value.
  ExitCondition->setOperand(ExitValueNum, SD.A_ExitValue);
  
  // [*] Update BLoop's header phi nodes. Remove incoming PHINode's from
  //     original loop's preheader. Add incoming PHINode values from
  //     ALoop's exiting block. Update BLoop header's domiantor info.

  // Collect inverse map of Header PHINodes.
  DenseMap<Value *, Value *> InverseMap;
  for (BasicBlock::iterator BI = L->getHeader()->begin(), 
         BE = L->getHeader()->end(); BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PHINode *PNClone = cast<PHINode>(ValueMap[PN]);
      InverseMap[PNClone] = PN;
    } else
      break;
  }

  for (BasicBlock::iterator BI = B_Header->begin(), BE = B_Header->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      // Remove incoming value from original preheader.
      PN->removeIncomingValue(Preheader);

      // Add incoming value from A_ExitingBlock.
      if (PN == B_IndVar)
        PN->addIncoming(SD.B_StartValue, A_ExitingBlock);
      else { 
        PHINode *OrigPN = cast<PHINode>(InverseMap[PN]);
        Value *V2 = NULL;
        // If loop header is also loop exiting block then
        // OrigPN is incoming value for B loop header.
        if (A_ExitingBlock == L->getHeader())
          V2 = OrigPN;
        else
          V2 = OrigPN->getIncomingValueForBlock(A_ExitingBlock);
        PN->addIncoming(V2, A_ExitingBlock);
      }
    } else
      break;
  }
  DT->changeImmediateDominator(B_Header, A_ExitingBlock);
  DF->changeImmediateDominator(B_Header, A_ExitingBlock, DT);
  
  // [*] Update BLoop's exit block. Its new predecessor is BLoop's exit
  //     block. Remove incoming PHINode values from ALoop's exiting block.
  //     Add new incoming values from BLoop's incoming exiting value.
  //     Update BLoop exit block's dominator info..
  BasicBlock *B_ExitingBlock = cast<BasicBlock>(ValueMap[A_ExitingBlock]);
  for (BasicBlock::iterator BI = B_ExitBlock->begin(), BE = B_ExitBlock->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PN->addIncoming(ValueMap[PN->getIncomingValueForBlock(A_ExitingBlock)], 
                                                            B_ExitingBlock);
      PN->removeIncomingValue(A_ExitingBlock);
    } else
      break;
  }

  DT->changeImmediateDominator(B_ExitBlock, B_ExitingBlock);
  DF->changeImmediateDominator(B_ExitBlock, B_ExitingBlock, DT);

  //[*] Split ALoop's exit edge. This creates a new block which
  //    serves two purposes. First one is to hold PHINode defnitions
  //    to ensure that ALoop's LCSSA form. Second use it to act
  //    as a preheader for BLoop.
  BasicBlock *A_ExitBlock = SplitEdge(A_ExitingBlock, B_Header, this);

  //[*] Preserve ALoop's LCSSA form. Create new forwarding PHINodes
  //    in A_ExitBlock to redefine outgoing PHI definitions from ALoop.
  for(BasicBlock::iterator BI = B_Header->begin(), BE = B_Header->end();
      BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      Value *V1 = PN->getIncomingValueForBlock(A_ExitBlock);
      PHINode *newPHI = PHINode::Create(PN->getType(), PN->getName());
      newPHI->addIncoming(V1, A_ExitingBlock);
      A_ExitBlock->getInstList().push_front(newPHI);
      PN->removeIncomingValue(A_ExitBlock);
      PN->addIncoming(newPHI, A_ExitBlock);
    } else
      break;
  }

  //[*] Eliminate split condition's inactive branch from ALoop.
  BasicBlock *A_SplitCondBlock = SD.SplitCondition->getParent();
  BranchInst *A_BR = cast<BranchInst>(A_SplitCondBlock->getTerminator());
  BasicBlock *A_InactiveBranch = NULL;
  BasicBlock *A_ActiveBranch = NULL;
  if (SD.UseTrueBranchFirst) {
    A_ActiveBranch = A_BR->getSuccessor(0);
    A_InactiveBranch = A_BR->getSuccessor(1);
  } else {
    A_ActiveBranch = A_BR->getSuccessor(1);
    A_InactiveBranch = A_BR->getSuccessor(0);
  }
  A_BR->setUnconditionalDest(A_ActiveBranch);
  removeBlocks(A_InactiveBranch, L, A_ActiveBranch);

  //[*] Eliminate split condition's inactive branch in from BLoop.
  BasicBlock *B_SplitCondBlock = cast<BasicBlock>(ValueMap[A_SplitCondBlock]);
  BranchInst *B_BR = cast<BranchInst>(B_SplitCondBlock->getTerminator());
  BasicBlock *B_InactiveBranch = NULL;
  BasicBlock *B_ActiveBranch = NULL;
  if (SD.UseTrueBranchFirst) {
    B_ActiveBranch = B_BR->getSuccessor(1);
    B_InactiveBranch = B_BR->getSuccessor(0);
  } else {
    B_ActiveBranch = B_BR->getSuccessor(0);
    B_InactiveBranch = B_BR->getSuccessor(1);
  }
  B_BR->setUnconditionalDest(B_ActiveBranch);
  removeBlocks(B_InactiveBranch, BLoop, B_ActiveBranch);

  BasicBlock *A_Header = L->getHeader();
  if (A_ExitingBlock == A_Header)
    return true;

  //[*] Move exit condition into split condition block to avoid
  //    executing dead loop iteration.
  ICmpInst *B_ExitCondition = cast<ICmpInst>(ValueMap[ExitCondition]);
  Instruction *B_IndVarIncrement = cast<Instruction>(ValueMap[IndVarIncrement]);
  ICmpInst *B_SplitCondition = cast<ICmpInst>(ValueMap[SD.SplitCondition]);

  moveExitCondition(A_SplitCondBlock, A_ActiveBranch, A_ExitBlock, ExitCondition,
                    cast<ICmpInst>(SD.SplitCondition), IndVar, IndVarIncrement, 
                    ALoop);

  moveExitCondition(B_SplitCondBlock, B_ActiveBranch, B_ExitBlock, B_ExitCondition,
                    B_SplitCondition, B_IndVar, B_IndVarIncrement, BLoop);

  return true;
}

// moveExitCondition - Move exit condition EC into split condition block CondBB.
void LoopIndexSplit::moveExitCondition(BasicBlock *CondBB, BasicBlock *ActiveBB,
                                       BasicBlock *ExitBB, ICmpInst *EC, ICmpInst *SC,
                                       PHINode *IV, Instruction *IVAdd, Loop *LP) {

  BasicBlock *ExitingBB = EC->getParent();
  Instruction *CurrentBR = CondBB->getTerminator();

  // Move exit condition into split condition block.
  EC->moveBefore(CurrentBR);
  EC->setOperand(ExitValueNum == 0 ? 1 : 0, IV);

  // Move exiting block's branch into split condition block. Update its branch
  // destination.
  BranchInst *ExitingBR = cast<BranchInst>(ExitingBB->getTerminator());
  ExitingBR->moveBefore(CurrentBR);
  BasicBlock *OrigDestBB = NULL;
  if (ExitingBR->getSuccessor(0) == ExitBB) {
    OrigDestBB = ExitingBR->getSuccessor(1);
    ExitingBR->setSuccessor(1, ActiveBB);
  }
  else {
    OrigDestBB = ExitingBR->getSuccessor(0);
    ExitingBR->setSuccessor(0, ActiveBB);
  }
    
  // Remove split condition and current split condition branch.
  SC->eraseFromParent();
  CurrentBR->eraseFromParent();

  // Connect exiting block to original destination.
  BranchInst::Create(OrigDestBB, ExitingBB);

  // Update PHINodes
  updatePHINodes(ExitBB, ExitingBB, CondBB, IV, IVAdd, LP);

  // Fix dominator info.
  // ExitBB is now dominated by CondBB
  DT->changeImmediateDominator(ExitBB, CondBB);
  DF->changeImmediateDominator(ExitBB, CondBB, DT);
  
  // Basicblocks dominated by ActiveBB may have ExitingBB or
  // a basic block outside the loop in their DF list. If so,
  // replace it with CondBB.
  DomTreeNode *Node = DT->getNode(ActiveBB);
  for (df_iterator<DomTreeNode *> DI = df_begin(Node), DE = df_end(Node);
       DI != DE; ++DI) {
    BasicBlock *BB = DI->getBlock();
    DominanceFrontier::iterator BBDF = DF->find(BB);
    DominanceFrontier::DomSetType::iterator DomSetI = BBDF->second.begin();
    DominanceFrontier::DomSetType::iterator DomSetE = BBDF->second.end();
    while (DomSetI != DomSetE) {
      DominanceFrontier::DomSetType::iterator CurrentItr = DomSetI;
      ++DomSetI;
      BasicBlock *DFBB = *CurrentItr;
      if (DFBB == ExitingBB || !L->contains(DFBB)) {
        BBDF->second.erase(DFBB);
        BBDF->second.insert(CondBB);
      }
    }
  }
}

/// updatePHINodes - CFG has been changed. 
/// Before 
///   - ExitBB's single predecessor was Latch
///   - Latch's second successor was Header
/// Now
///   - ExitBB's single predecessor is Header
///   - Latch's one and only successor is Header
///
/// Update ExitBB PHINodes' to reflect this change.
void LoopIndexSplit::updatePHINodes(BasicBlock *ExitBB, BasicBlock *Latch, 
                                    BasicBlock *Header,
                                    PHINode *IV, Instruction *IVIncrement,
                                    Loop *LP) {

  for (BasicBlock::iterator BI = ExitBB->begin(), BE = ExitBB->end(); 
       BI != BE; ) {
    PHINode *PN = dyn_cast<PHINode>(BI);
    ++BI;
    if (!PN)
      break;

    Value *V = PN->getIncomingValueForBlock(Latch);
    if (PHINode *PHV = dyn_cast<PHINode>(V)) {
      // PHV is in Latch. PHV has one use is in ExitBB PHINode. And one use
      // in Header which is new incoming value for PN.
      Value *NewV = NULL;
      for (Value::use_iterator UI = PHV->use_begin(), E = PHV->use_end(); 
           UI != E; ++UI) 
        if (PHINode *U = dyn_cast<PHINode>(*UI)) 
          if (LP->contains(U->getParent())) {
            NewV = U;
            break;
          }

      // Add incoming value from header only if PN has any use inside the loop.
      if (NewV)
        PN->addIncoming(NewV, Header);

    } else if (Instruction *PHI = dyn_cast<Instruction>(V)) {
      // If this instruction is IVIncrement then IV is new incoming value 
      // from header otherwise this instruction must be incoming value from 
      // header because loop is in LCSSA form.
      if (PHI == IVIncrement)
        PN->addIncoming(IV, Header);
      else
        PN->addIncoming(V, Header);
    } else
      // Otherwise this is an incoming value from header because loop is in 
      // LCSSA form.
      PN->addIncoming(V, Header);
    
    // Remove incoming value from Latch.
    PN->removeIncomingValue(Latch);
  }
}
