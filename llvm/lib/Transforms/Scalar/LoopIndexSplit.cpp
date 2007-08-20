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
      AU.addRequired<DominanceFrontier>();
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

    /// If Exiting block includes loop variant instructions then this
    /// loop may not be eliminated.
    bool safeExitingBlock(SplitInfo &SD, BasicBlock *BB);

    /// removeBlocks - Remove basic block DeadBB and all blocks dominated by DeadBB.
    /// This routine is used to remove split condition's dead branch, dominated by
    /// DeadBB. LiveBB dominates split conidition's other branch.
    void removeBlocks(BasicBlock *DeadBB, Loop *LP, BasicBlock *LiveBB);

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
  for (SmallVector<SplitInfo, 4>::iterator SI = SplitData.begin(),
         E = SplitData.end(); SI != E;) {
    SplitInfo &SD = *SI;
    if (SD.SplitCondition->getPredicate() == ICmpInst::ICMP_EQ) {
      Changed = processOneIterationLoop(SD);
      if (Changed) {
        ++NumIndexSplit;
        // If is loop is eliminated then nothing else to do here.
        return Changed;
      } else {
        SmallVector<SplitInfo, 4>::iterator Delete_SI = SI;
        ++SI;
        SplitData.erase(Delete_SI);
      }
    } else
      ++SI;
  }

  unsigned MaxCost = 99;
  unsigned Index = 0;
  unsigned MostProfitableSDIndex = 0;
  for (SmallVector<SplitInfo, 4>::iterator SI = SplitData.begin(),
         E = SplitData.end(); SI != E; ++SI, ++Index) {
    SplitInfo SD = *SI;

    // ICM_EQs are already handled above.
    assert (SD.SplitCondition->getPredicate() != ICmpInst::ICMP_EQ &&
            "Unexpected split condition predicate");
    
    unsigned Cost = findSplitCost(L, SD);
    if (Cost < MaxCost)
      MostProfitableSDIndex = Index;
  }

  // Split most profitiable condition.
  if (!SplitData.empty())
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
  
  // If exit block's terminator is conditional branch inst then we have found
  // exit condition.
  BranchInst *BR = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
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

  // If Exiting block includes loop variant instructions then this
  // loop may not be eliminated.
  if (!safeExitingBlock(SD, ExitCondition->getParent())) 
    return false;

  // Update CFG.

  // Replace index variable with split value in loop body. Loop body is executed
  // only when index variable is equal to split value.
  IndVar->replaceAllUsesWith(SD.SplitValue);

  // Remove Latch to Header edge.
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

  Instruction *Terminator = Header->getTerminator();
  Value *ExitValue = ExitCondition->getOperand(ExitValueNum);

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
                                 SD.SplitValue, ExitValue, "lisplit", 
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

    if (I == IndVarIncrement) 
      I->replaceAllUsesWith(ExitValue);
    else
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

    if (I == ExitingBlock->getTerminator())
      continue;

    // Otherwise we have instruction that may not be safe.
    return false;
  }

  // We could not find any reason to consider ExitingBlock unsafe.
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

      BasicBlock *NewDominator = NULL;
      for(BasicBlock::iterator FBI = FrontierBB->begin(), FBE = FrontierBB->end();
          FBI != FBE; ++FBI) {
        if (PHINode *PN = dyn_cast<PHINode>(FBI)) {
          for(SmallVector<BasicBlock *, 8>::iterator PI = PredBlocks.begin(),
                PE = PredBlocks.end(); PI != PE; ++PI) {
            BasicBlock *P = *PI;
            PN->removeIncomingValue(P);
          }
          // If we have not identified new dominator then see if we can identify
          // one based on remaining incoming PHINode values.
          if (NewDominator == NULL && PN->getNumIncomingValues() == 1)
            NewDominator = PN->getIncomingBlock(0);
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
        BBI != BBE; ++BBI) {
      Instruction *I = BBI;
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

bool LoopIndexSplit::splitLoop(SplitInfo &SD) {

  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *SplitBlock = SD.SplitCondition->getParent();
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Header = L->getHeader();
  BranchInst *SplitTerminator = cast<BranchInst>(SplitBlock->getTerminator());

  // FIXME - Unable to handle triange loops at the moment.
  // In triangle loop, split condition is in header and one of the
  // the split destination is loop latch. If split condition is EQ
  // then such loops are already handle in processOneIterationLoop().
  if (Header == SplitBlock 
      && (Latch == SplitTerminator->getSuccessor(0) 
          || Latch == SplitTerminator->getSuccessor(1)))
    return false;

  // If one of the split condition branch is post dominating other then loop 
  // index split is not appropriate.
  BasicBlock *Succ0 = SplitTerminator->getSuccessor(0);
  BasicBlock *Succ1 = SplitTerminator->getSuccessor(1);
  if (DT->dominates(Succ0, Latch) || DT->dominates(Succ1, Latch))
    return false;

  // If one of the split condition branch is a predecessor of the other
  // split condition branch head then do not split loop on this condition.
  for(pred_iterator PI = pred_begin(Succ0), PE = pred_end(Succ0); PI != PE; ++PI)
    if (Succ1 == *PI)
      return false;
  for(pred_iterator PI = pred_begin(Succ1), PE = pred_end(Succ1); PI != PE; ++PI)
    if (Succ0 == *PI)
      return false;

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
  BasicBlock *ExitingBlock = ExitCondition->getParent();
  BranchInst *ExitInsn = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  assert (ExitInsn && "Unable to find suitable loop exit branch");
  BasicBlock *ExitDest = ExitInsn->getSuccessor(1);

  if (L->contains(ExitDest)) {
    ExitDest = ExitInsn->getSuccessor(0);
    ExitInsn->setSuccessor(0, FalseHeader);
  } else
    ExitInsn->setSuccessor(1, FalseHeader);

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

  // Update False loop's header
  for (BasicBlock::iterator BI = FalseHeader->begin(), BE = FalseHeader->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PN->removeIncomingValue(Preheader);
      if (PN == IndVarClone)
        PN->addIncoming(FLStartValue, ExitingBlock);
      else { 
        PHINode *OrigPN = cast<PHINode>(InverseMap[PN]);
        Value *V2 = OrigPN->getIncomingValueForBlock(ExitingBlock);
        PN->addIncoming(V2, ExitingBlock);
      }
    } else
      break;
  }

  // Update ExitDest. Now it's predecessor is False loop's exit block.
  BasicBlock *ExitingBlockClone = cast<BasicBlock>(ValueMap[ExitingBlock]);
  for (BasicBlock::iterator BI = ExitDest->begin(), BE = ExitDest->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PN->addIncoming(ValueMap[PN->getIncomingValueForBlock(ExitingBlock)], ExitingBlockClone);
      PN->removeIncomingValue(ExitingBlock);
    } else
      break;
  }

  if (DT) {
    DT->changeImmediateDominator(FalseHeader, ExitingBlock);
    DT->changeImmediateDominator(ExitDest, cast<BasicBlock>(ValueMap[ExitingBlock]));
  }

  assert (!L->contains(ExitDest) && " Unable to find exit edge destination");

  //[*] Split Exit Edge. 
  SplitEdge(ExitingBlock, FalseHeader, this);

  //[*] Eliminate split condition's false branch from True loop.
  BranchInst *BR = cast<BranchInst>(SplitBlock->getTerminator());
  BasicBlock *FBB = BR->getSuccessor(1);
  BR->setUnconditionalDest(BR->getSuccessor(0));
  removeBlocks(FBB, L, BR->getSuccessor(0));

  //[*] Update True loop's exit value using new exit value.
  ExitCondition->setOperand(ExitValueNum, TLExitValue);

  //[*] Eliminate split condition's  true branch in False loop CFG.
  BasicBlock *FSplitBlock = cast<BasicBlock>(ValueMap[SplitBlock]);
  BranchInst *FBR = cast<BranchInst>(FSplitBlock->getTerminator());
  BasicBlock *TBB = FBR->getSuccessor(0);
  FBR->setUnconditionalDest(FBR->getSuccessor(1));
  removeBlocks(TBB, FalseLoop, cast<BasicBlock>(FBR->getSuccessor(0)));

  return true;
}

