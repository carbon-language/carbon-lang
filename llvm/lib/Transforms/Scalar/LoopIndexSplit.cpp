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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
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
      AU.addRequired<LoopInfo>();
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
      SplitInfo() : SplitValue(NULL), SplitCondition(NULL) {}

      // Induction variable's range is split at this value.
      Value *SplitValue;
      
      // This compare instruction compares IndVar against SplitValue.
      ICmpInst *SplitCondition;

      // Clear split info.
      void clear() {
        SplitValue = NULL;
        SplitCondition = NULL;
      }

    };
    
  private:
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
    
    /// If loop header includes loop variant instruction operands then
    /// this loop may not be eliminated.
    bool safeHeader(SplitInfo &SD,  BasicBlock *BB);

    /// If Exit block includes loop variant instructions then this
    /// loop may not be eliminated.
    bool safeExitBlock(SplitInfo &SD, BasicBlock *BB);

    /// removeBlocks - Remove basic block BB and all blocks dominated by BB.
    void removeBlocks(BasicBlock *InBB);

    /// Find cost of spliting loop L.
    unsigned findSplitCost(Loop *L, SplitInfo &SD);
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

  char LoopIndexSplit::ID = 0;
  RegisterPass<LoopIndexSplit> X ("loop-index-split", "Index Split Loops");
}

LoopPass *llvm::createLoopIndexSplitPass() {
  return new LoopIndexSplit();
}

// Index split Loop L. Return true if loop is split.
bool LoopIndexSplit::runOnLoop(Loop *IncomingLoop, LPPassManager &LPM_Ref) {
  bool Changed = false;
  L = IncomingLoop;
  LPM = &LPM_Ref;

  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  DF = getAnalysisToUpdate<DominanceFrontier>();

  initialize();

  findLoopConditionals();

  if (!ExitCondition)
    return false;

  findSplitCondition();

  if (SplitData.empty())
    return false;

  // First see if it is possible to eliminate loop itself or not.
  for (SmallVector<SplitInfo, 4>::iterator SI = SplitData.begin(),
         E = SplitData.end(); SI != E; ++SI) {
    SplitInfo &SD = *SI;
    if (SD.SplitCondition->getPredicate() == ICmpInst::ICMP_EQ) {
      Changed = processOneIterationLoop(SD);
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
  
  if (PHINode *PN = dyn_cast<PHINode>(Op0)) {
    if (PN->getParent() == L->getHeader()
        && isa<ConstantInt>(Op1)) {
      IndVar = PN;
      IndVarIncrement = I;
      return;
    }
  }
  
  if (PHINode *PN = dyn_cast<PHINode>(Op1)) {
    if (PN->getParent() == L->getHeader()
        && isa<ConstantInt>(Op0)) {
      IndVar = PN;
      IndVarIncrement = I;
      return;
    }
  }
  
  return;
}

// Find loop's exit condition and associated induction variable.
void LoopIndexSplit::findLoopConditionals() {

  BasicBlock *ExitBlock = NULL;

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (!L->isLoopExit(BB))
      continue;
    if (ExitBlock)
      return;
    ExitBlock = BB;
  }

  if (!ExitBlock)
    return;
  
  // If exit block's terminator is conditional branch inst then we have found
  // exit condition.
  BranchInst *BR = dyn_cast<BranchInst>(ExitBlock->getTerminator());
  if (!BR || BR->isUnconditional())
    return;
  
  ICmpInst *CI = dyn_cast<ICmpInst>(BR->getCondition());
  if (!CI)
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
}

/// Find condition inside a loop that is suitable candidate for index split.
void LoopIndexSplit::findSplitCondition() {

  SplitInfo SD;
  // Check all basic block's terminators.

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;

    // If this basic block does not terminate in a conditional branch
    // then terminator is not a suitable split condition.
    BranchInst *BR = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BR)
      continue;
    
    if (BR->isUnconditional())
      continue;

    ICmpInst *CI = dyn_cast<ICmpInst>(BR->getCondition());
    if (!CI || CI == ExitCondition)
      return;

    // If one operand is loop invariant and second operand is SCEVAddRecExpr
    // based on induction variable then CI is a candidate split condition.
    Value *V0 = CI->getOperand(0);
    Value *V1 = CI->getOperand(1);

    SCEVHandle SH0 = SE->getSCEV(V0);
    SCEVHandle SH1 = SE->getSCEV(V1);

    if (SH0->isLoopInvariant(L) && isa<SCEVAddRecExpr>(SH1)) {
      SD.SplitValue = V0;
      SD.SplitCondition = CI;
      if (PHINode *PN = dyn_cast<PHINode>(V1)) {
        if (PN == IndVar)
          SplitData.push_back(SD);
      }
      else  if (Instruction *Insn = dyn_cast<Instruction>(V1)) {
        if (IndVarIncrement && IndVarIncrement == Insn)
          SplitData.push_back(SD);
      }
    }
    else if (SH1->isLoopInvariant(L) && isa<SCEVAddRecExpr>(SH0)) {
      SD.SplitValue =  V1;
      SD.SplitCondition = CI;
      if (PHINode *PN = dyn_cast<PHINode>(V0)) {
        if (PN == IndVar)
          SplitData.push_back(SD);
      }
      else  if (Instruction *Insn = dyn_cast<Instruction>(V0)) {
        if (IndVarIncrement && IndVarIncrement == Insn)
          SplitData.push_back(SD);
      }
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

  // If Exit block includes loop variant instructions then this
  // loop may not be eliminated.
  if (!safeExitBlock(SD, ExitCondition->getParent())) 
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
  StartValue = IndVar->getIncomingValueForBlock(Preheader);

  // Replace split condition in header.
  // Transform 
  //      SplitCondition : icmp eq i32 IndVar, SplitValue
  // into
  //      c1 = icmp uge i32 SplitValue, StartValue
  //      c2 = icmp ult i32 vSplitValue, ExitValue
  //      and i32 c1, c2 
  bool SignedPredicate = ExitCondition->isSignedPredicate();
  Instruction *C1 = new ICmpInst(SignedPredicate ? 
                                 ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE,
                                 SD.SplitValue, StartValue, "lisplit", 
                                 Terminator);
  Instruction *C2 = new ICmpInst(SignedPredicate ? 
                                 ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                                 SD.SplitValue, 
                                 ExitCondition->getOperand(ExitValueNum), "lisplit", 
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

    // PHI Nodes are OK. FIXME : Handle last value assignments.
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

// If Exit block includes loop variant instructions then this
// loop may not be eliminated. This is used by processOneIterationLoop().
bool LoopIndexSplit::safeExitBlock(SplitInfo &SD, BasicBlock *ExitBlock) {

  for (BasicBlock::iterator BI = ExitBlock->begin(), BE = ExitBlock->end();
       BI != BE; ++BI) {
    Instruction *I = BI;

    // PHI Nodes are OK. FIXME : Handle last value assignments.
    if (isa<PHINode>(I))
      continue;

    // Induction variable increment is OK.
    if (IndVarIncrement && IndVarIncrement == I)
      continue;

    // Check if I is induction variable increment instruction.
    if (!IndVarIncrement && I->getOpcode() == Instruction::Add) {

      Value *Op0 = I->getOperand(0);
      Value *Op1 = I->getOperand(1);
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
          
      if (IndVarIncrement && PN == IndVar && CI->isOne())
        continue;
    }

    // I is an Exit condition if next instruction is block terminator.
    // Exit condition is OK if it compares loop invariant exit value,
    // which is checked below.
    else if (ICmpInst *EC = dyn_cast<ICmpInst>(I)) {
      if (EC == ExitCondition)
        continue;
    }

    if (I == ExitBlock->getTerminator())
      continue;

    // Otherwise we have instruction that may not be safe.
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

/// removeBlocks - Remove basic block BB and all blocks dominated by BB.
void LoopIndexSplit::removeBlocks(BasicBlock *InBB) {

  SmallVector<std::pair<BasicBlock *, succ_iterator>, 8> WorkList;
  WorkList.push_back(std::make_pair(InBB, succ_begin(InBB)));
  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.back(). first; 
    succ_iterator SIter =WorkList.back().second;

    // If all successor's are processed then remove this block.
    if (SIter == succ_end(BB)) {
      WorkList.pop_back();
      for(BasicBlock::iterator BBI = BB->begin(), BBE = BB->end(); 
          BBI != BBE; ++BBI) {
        Instruction *I = BBI;
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
        I->eraseFromParent();
      }
      DT->eraseNode(BB);
      DF->removeBlock(BB);
      LI->removeBlock(BB);
      BB->eraseFromParent();
    } else {
      BasicBlock *SuccBB = *SIter;
      ++WorkList.back().second;
      
      if (DT->dominates(BB, SuccBB)) {
        WorkList.push_back(std::make_pair(SuccBB, succ_begin(SuccBB)));
        continue;
      } else {
        // If SuccBB is not dominated by BB then it is not removed, however remove
        // any PHI incoming edge from BB.
        for(BasicBlock::iterator SBI = SuccBB->begin(), SBE = SuccBB->end();
            SBI != SBE; ++SBI) {
          if (PHINode *PN = dyn_cast<PHINode>(SBI)) 
            PN->removeIncomingValue(BB);
          else
            break;
        }

        // If BB is not dominating SuccBB then SuccBB is in BB's dominance
        // frontiner. 
        DominanceFrontier::iterator BBDF = DF->find(BB);
        DF->removeFromFrontier(BBDF, SuccBB);
      }
    }
  }
}

bool LoopIndexSplit::splitLoop(SplitInfo &SD) {

  BasicBlock *Preheader = L->getLoopPreheader();

  // True loop is original loop. False loop is cloned loop.

  bool SignedPredicate = ExitCondition->isSignedPredicate();  
  //[*] Calculate True loop's new Exit Value in loop preheader.
  //      TLExitValue = min(SplitValue, ExitValue)
  //[*] Calculate False loop's new Start Value in loop preheader.
  //      FLStartValue = min(SplitValue, TrueLoop.StartValue)
  Value *TLExitValue = NULL;
  Value *FLStartValue = NULL;
  if (isa<ConstantInt>(SD.SplitValue)) {
    TLExitValue = SD.SplitValue;
    FLStartValue = SD.SplitValue;
  }
  else {
    Value *C1 = new ICmpInst(SignedPredicate ? 
                            ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                            SD.SplitValue, 
                             ExitCondition->getOperand(ExitValueNum), 
                             "lsplit.ev",
                            Preheader->getTerminator());
    TLExitValue = new SelectInst(C1, SD.SplitValue, 
                                 ExitCondition->getOperand(ExitValueNum), 
                                 "lsplit.ev", Preheader->getTerminator());

    Value *C2 = new ICmpInst(SignedPredicate ? 
                             ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                             SD.SplitValue, StartValue, "lsplit.sv",
                             Preheader->getTerminator());
    FLStartValue = new SelectInst(C2, SD.SplitValue, StartValue,
                                  "lsplit.sv", Preheader->getTerminator());
  }

  //[*] Clone loop. Avoid true destination of split condition and 
  //    the blocks dominated by true destination. 
  DenseMap<const Value *, Value *> ValueMap;
  Loop *FalseLoop = CloneLoop(L, LPM, LI, ValueMap, this);
  BasicBlock *FalseHeader = FalseLoop->getHeader();

  //[*] True loop's exit edge enters False loop.
  PHINode *IndVarClone = cast<PHINode>(ValueMap[IndVar]);
  BasicBlock *ExitBlock = ExitCondition->getParent();
  BranchInst *ExitInsn = dyn_cast<BranchInst>(ExitBlock->getTerminator());
  assert (ExitInsn && "Unable to find suitable loop exit branch");
  BasicBlock *ExitDest = ExitInsn->getSuccessor(1);

  for (BasicBlock::iterator BI = FalseHeader->begin(), BE = FalseHeader->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PN->removeIncomingValue(Preheader);
      if (PN == IndVarClone)
        PN->addIncoming(FLStartValue, ExitBlock);
      // else { FIXME : Handl last value assignments.}
    }
    else
      break;
  }

  if (L->contains(ExitDest)) {
    ExitDest = ExitInsn->getSuccessor(0);
    ExitInsn->setSuccessor(0, FalseHeader);
  } else
    ExitInsn->setSuccessor(1, FalseHeader);

  if (DT) {
    DT->changeImmediateDominator(FalseHeader, ExitBlock);
    DT->changeImmediateDominator(ExitDest, cast<BasicBlock>(ValueMap[ExitBlock]));
  }

  assert (!L->contains(ExitDest) && " Unable to find exit edge destination");

  //[*] Split Exit Edge. 
  SplitEdge(ExitBlock, FalseHeader, this);

  //[*] Eliminate split condition's false branch from True loop.
  BasicBlock *SplitBlock = SD.SplitCondition->getParent();
  BranchInst *BR = cast<BranchInst>(SplitBlock->getTerminator());
  BasicBlock *FBB = BR->getSuccessor(1);
  BR->setUnconditionalDest(BR->getSuccessor(0));
  removeBlocks(FBB);

  //[*] Update True loop's exit value using new exit value.
  ExitCondition->setOperand(ExitValueNum, TLExitValue);

  //[*] Eliminate split condition's  true branch in False loop CFG.
  BasicBlock *FSplitBlock = cast<BasicBlock>(ValueMap[SplitBlock]);
  BranchInst *FBR = cast<BranchInst>(FSplitBlock->getTerminator());
  BasicBlock *TBB = FBR->getSuccessor(0);
  FBR->setUnconditionalDest(FBR->getSuccessor(1));
  removeBlocks(TBB);

  return true;
}

