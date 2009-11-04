//===- JumpThreading.cpp - Thread control through conditional blocks ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Jump Threading pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jump-threading"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumThreads, "Number of jumps threaded");
STATISTIC(NumFolds,   "Number of terminators folded");
STATISTIC(NumDupes,   "Number of branch blocks duplicated to eliminate phi");

static cl::opt<unsigned>
Threshold("jump-threading-threshold", 
          cl::desc("Max block size to duplicate for jump threading"),
          cl::init(6), cl::Hidden);

namespace {
  /// This pass performs 'jump threading', which looks at blocks that have
  /// multiple predecessors and multiple successors.  If one or more of the
  /// predecessors of the block can be proven to always jump to one of the
  /// successors, we forward the edge from the predecessor to the successor by
  /// duplicating the contents of this block.
  ///
  /// An example of when this can occur is code like this:
  ///
  ///   if () { ...
  ///     X = 4;
  ///   }
  ///   if (X < 3) {
  ///
  /// In this case, the unconditional branch at the end of the first if can be
  /// revectored to the false side of the second if.
  ///
  class JumpThreading : public FunctionPass {
    TargetData *TD;
#ifdef NDEBUG
    SmallPtrSet<BasicBlock*, 16> LoopHeaders;
#else
    SmallSet<AssertingVH<BasicBlock>, 16> LoopHeaders;
#endif
  public:
    static char ID; // Pass identification
    JumpThreading() : FunctionPass(&ID) {}

    bool runOnFunction(Function &F);
    void FindLoopHeaders(Function &F);
    
    bool ProcessBlock(BasicBlock *BB);
    bool ThreadEdge(BasicBlock *BB, BasicBlock *PredBB, BasicBlock *SuccBB);
    bool DuplicateCondBranchOnPHIIntoPred(BasicBlock *BB,
                                          BasicBlock *PredBB);

    BasicBlock *FactorCommonPHIPreds(PHINode *PN, Value *Val);
    bool ProcessBranchOnDuplicateCond(BasicBlock *PredBB, BasicBlock *DestBB);
    bool ProcessSwitchOnDuplicateCond(BasicBlock *PredBB, BasicBlock *DestBB);

    bool ProcessJumpOnPHI(PHINode *PN);
    bool ProcessBranchOnLogical(Value *V, BasicBlock *BB, bool isAnd);
    bool ProcessBranchOnCompare(CmpInst *Cmp, BasicBlock *BB);
    
    bool SimplifyPartiallyRedundantLoad(LoadInst *LI);
  };
}

char JumpThreading::ID = 0;
static RegisterPass<JumpThreading>
X("jump-threading", "Jump Threading");

// Public interface to the Jump Threading pass
FunctionPass *llvm::createJumpThreadingPass() { return new JumpThreading(); }

/// runOnFunction - Top level algorithm.
///
bool JumpThreading::runOnFunction(Function &F) {
  DEBUG(errs() << "Jump threading on function '" << F.getName() << "'\n");
  TD = getAnalysisIfAvailable<TargetData>();
  
  FindLoopHeaders(F);
  
  bool AnotherIteration = true, EverChanged = false;
  while (AnotherIteration) {
    AnotherIteration = false;
    bool Changed = false;
    for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
      BasicBlock *BB = I;
      while (ProcessBlock(BB))
        Changed = true;
      
      ++I;
      
      // If the block is trivially dead, zap it.  This eliminates the successor
      // edges which simplifies the CFG.
      if (pred_begin(BB) == pred_end(BB) &&
          BB != &BB->getParent()->getEntryBlock()) {
        DEBUG(errs() << "  JT: Deleting dead block '" << BB->getName()
              << "' with terminator: " << *BB->getTerminator() << '\n');
        LoopHeaders.erase(BB);
        DeleteDeadBlock(BB);
        Changed = true;
      }
    }
    AnotherIteration = Changed;
    EverChanged |= Changed;
  }
  
  LoopHeaders.clear();
  return EverChanged;
}

/// getJumpThreadDuplicationCost - Return the cost of duplicating this block to
/// thread across it.
static unsigned getJumpThreadDuplicationCost(const BasicBlock *BB) {
  /// Ignore PHI nodes, these will be flattened when duplication happens.
  BasicBlock::const_iterator I = BB->getFirstNonPHI();
  
  // Sum up the cost of each instruction until we get to the terminator.  Don't
  // include the terminator because the copy won't include it.
  unsigned Size = 0;
  for (; !isa<TerminatorInst>(I); ++I) {
    // Debugger intrinsics don't incur code size.
    if (isa<DbgInfoIntrinsic>(I)) continue;
    
    // If this is a pointer->pointer bitcast, it is free.
    if (isa<BitCastInst>(I) && isa<PointerType>(I->getType()))
      continue;
    
    // All other instructions count for at least one unit.
    ++Size;
    
    // Calls are more expensive.  If they are non-intrinsic calls, we model them
    // as having cost of 4.  If they are a non-vector intrinsic, we model them
    // as having cost of 2 total, and if they are a vector intrinsic, we model
    // them as having cost 1.
    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      if (!isa<IntrinsicInst>(CI))
        Size += 3;
      else if (!isa<VectorType>(CI->getType()))
        Size += 1;
    }
  }
  
  // Threading through a switch statement is particularly profitable.  If this
  // block ends in a switch, decrease its cost to make it more likely to happen.
  if (isa<SwitchInst>(I))
    Size = Size > 6 ? Size-6 : 0;
  
  return Size;
}



/// FindLoopHeaders - We do not want jump threading to turn proper loop
/// structures into irreducible loops.  Doing this breaks up the loop nesting
/// hierarchy and pessimizes later transformations.  To prevent this from
/// happening, we first have to find the loop headers.  Here we approximate this
/// by finding targets of backedges in the CFG.
///
/// Note that there definitely are cases when we want to allow threading of
/// edges across a loop header.  For example, threading a jump from outside the
/// loop (the preheader) to an exit block of the loop is definitely profitable.
/// It is also almost always profitable to thread backedges from within the loop
/// to exit blocks, and is often profitable to thread backedges to other blocks
/// within the loop (forming a nested loop).  This simple analysis is not rich
/// enough to track all of these properties and keep it up-to-date as the CFG
/// mutates, so we don't allow any of these transformations.
///
void JumpThreading::FindLoopHeaders(Function &F) {
  SmallVector<std::pair<const BasicBlock*,const BasicBlock*>, 32> Edges;
  FindFunctionBackedges(F, Edges);
  
  for (unsigned i = 0, e = Edges.size(); i != e; ++i)
    LoopHeaders.insert(const_cast<BasicBlock*>(Edges[i].second));
}


/// FactorCommonPHIPreds - If there are multiple preds with the same incoming
/// value for the PHI, factor them together so we get one block to thread for
/// the whole group.
/// This is important for things like "phi i1 [true, true, false, true, x]"
/// where we only need to clone the block for the true blocks once.
///
BasicBlock *JumpThreading::FactorCommonPHIPreds(PHINode *PN, Value *Val) {
  SmallVector<BasicBlock*, 16> CommonPreds;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingValue(i) == Val)
      CommonPreds.push_back(PN->getIncomingBlock(i));
  
  if (CommonPreds.size() == 1)
    return CommonPreds[0];
    
  DEBUG(errs() << "  Factoring out " << CommonPreds.size()
        << " common predecessors.\n");
  return SplitBlockPredecessors(PN->getParent(),
                                &CommonPreds[0], CommonPreds.size(),
                                ".thr_comm", this);
}
  

/// GetBestDestForBranchOnUndef - If we determine that the specified block ends
/// in an undefined jump, decide which block is best to revector to.
///
/// Since we can pick an arbitrary destination, we pick the successor with the
/// fewest predecessors.  This should reduce the in-degree of the others.
///
static unsigned GetBestDestForJumpOnUndef(BasicBlock *BB) {
  TerminatorInst *BBTerm = BB->getTerminator();
  unsigned MinSucc = 0;
  BasicBlock *TestBB = BBTerm->getSuccessor(MinSucc);
  // Compute the successor with the minimum number of predecessors.
  unsigned MinNumPreds = std::distance(pred_begin(TestBB), pred_end(TestBB));
  for (unsigned i = 1, e = BBTerm->getNumSuccessors(); i != e; ++i) {
    TestBB = BBTerm->getSuccessor(i);
    unsigned NumPreds = std::distance(pred_begin(TestBB), pred_end(TestBB));
    if (NumPreds < MinNumPreds)
      MinSucc = i;
  }
  
  return MinSucc;
}

/// ProcessBlock - If there are any predecessors whose control can be threaded
/// through to a successor, transform them now.
bool JumpThreading::ProcessBlock(BasicBlock *BB) {
  // If this block has a single predecessor, and if that pred has a single
  // successor, merge the blocks.  This encourages recursive jump threading
  // because now the condition in this block can be threaded through
  // predecessors of our predecessor block.
  if (BasicBlock *SinglePred = BB->getSinglePredecessor())
    if (SinglePred->getTerminator()->getNumSuccessors() == 1 &&
        SinglePred != BB) {
      // If SinglePred was a loop header, BB becomes one.
      if (LoopHeaders.erase(SinglePred))
        LoopHeaders.insert(BB);
      
      // Remember if SinglePred was the entry block of the function.  If so, we
      // will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      MergeBasicBlockIntoOnlyPred(BB);
      
      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());
      return true;
    }
  
  // See if this block ends with a branch or switch.  If so, see if the
  // condition is a phi node.  If so, and if an entry of the phi node is a
  // constant, we can thread the block.
  Value *Condition;
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
    // Can't thread an unconditional jump.
    if (BI->isUnconditional()) return false;
    Condition = BI->getCondition();
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator()))
    Condition = SI->getCondition();
  else
    return false; // Must be an invoke.
  
  // If the terminator of this block is branching on a constant, simplify the
  // terminator to an unconditional branch.  This can occur due to threading in
  // other blocks.
  if (isa<ConstantInt>(Condition)) {
    DEBUG(errs() << "  In block '" << BB->getName()
          << "' folding terminator: " << *BB->getTerminator() << '\n');
    ++NumFolds;
    ConstantFoldTerminator(BB);
    return true;
  }
  
  // If the terminator is branching on an undef, we can pick any of the
  // successors to branch to.  Let GetBestDestForJumpOnUndef decide.
  if (isa<UndefValue>(Condition)) {
    unsigned BestSucc = GetBestDestForJumpOnUndef(BB);
    
    // Fold the branch/switch.
    TerminatorInst *BBTerm = BB->getTerminator();
    for (unsigned i = 0, e = BBTerm->getNumSuccessors(); i != e; ++i) {
      if (i == BestSucc) continue;
      BBTerm->getSuccessor(i)->removePredecessor(BB);
    }
    
    DEBUG(errs() << "  In block '" << BB->getName()
          << "' folding undef terminator: " << *BBTerm << '\n');
    BranchInst::Create(BBTerm->getSuccessor(BestSucc), BBTerm);
    BBTerm->eraseFromParent();
    return true;
  }
  
  Instruction *CondInst = dyn_cast<Instruction>(Condition);

  // If the condition is an instruction defined in another block, see if a
  // predecessor has the same condition:
  //     br COND, BBX, BBY
  //  BBX:
  //     br COND, BBZ, BBW
  if (!Condition->hasOneUse() && // Multiple uses.
      (CondInst == 0 || CondInst->getParent() != BB)) { // Non-local definition.
    pred_iterator PI = pred_begin(BB), E = pred_end(BB);
    if (isa<BranchInst>(BB->getTerminator())) {
      for (; PI != E; ++PI)
        if (BranchInst *PBI = dyn_cast<BranchInst>((*PI)->getTerminator()))
          if (PBI->isConditional() && PBI->getCondition() == Condition &&
              ProcessBranchOnDuplicateCond(*PI, BB))
            return true;
    } else {
      assert(isa<SwitchInst>(BB->getTerminator()) && "Unknown jump terminator");
      for (; PI != E; ++PI)
        if (SwitchInst *PSI = dyn_cast<SwitchInst>((*PI)->getTerminator()))
          if (PSI->getCondition() == Condition &&
              ProcessSwitchOnDuplicateCond(*PI, BB))
            return true;
    }
  }

  // All the rest of our checks depend on the condition being an instruction.
  if (CondInst == 0)
    return false;
  
  // See if this is a phi node in the current block.
  if (PHINode *PN = dyn_cast<PHINode>(CondInst))
    if (PN->getParent() == BB)
      return ProcessJumpOnPHI(PN);
  
  // If this is a conditional branch whose condition is and/or of a phi, try to
  // simplify it.
  if ((CondInst->getOpcode() == Instruction::And || 
       CondInst->getOpcode() == Instruction::Or) &&
      isa<BranchInst>(BB->getTerminator()) &&
      ProcessBranchOnLogical(CondInst, BB,
                             CondInst->getOpcode() == Instruction::And))
    return true;
  
  if (CmpInst *CondCmp = dyn_cast<CmpInst>(CondInst)) {
    if (isa<PHINode>(CondCmp->getOperand(0))) {
      // If we have "br (phi != 42)" and the phi node has any constant values
      // as operands, we can thread through this block.
      // 
      // If we have "br (cmp phi, x)" and the phi node contains x such that the
      // comparison uniquely identifies the branch target, we can thread
      // through this block.

      if (ProcessBranchOnCompare(CondCmp, BB))
        return true;      
    }
    
    // If we have a comparison, loop over the predecessors to see if there is
    // a condition with the same value.
    pred_iterator PI = pred_begin(BB), E = pred_end(BB);
    for (; PI != E; ++PI)
      if (BranchInst *PBI = dyn_cast<BranchInst>((*PI)->getTerminator()))
        if (PBI->isConditional() && *PI != BB) {
          if (CmpInst *CI = dyn_cast<CmpInst>(PBI->getCondition())) {
            if (CI->getOperand(0) == CondCmp->getOperand(0) &&
                CI->getOperand(1) == CondCmp->getOperand(1) &&
                CI->getPredicate() == CondCmp->getPredicate()) {
              // TODO: Could handle things like (x != 4) --> (x == 17)
              if (ProcessBranchOnDuplicateCond(*PI, BB))
                return true;
            }
          }
        }
  }

  // Check for some cases that are worth simplifying.  Right now we want to look
  // for loads that are used by a switch or by the condition for the branch.  If
  // we see one, check to see if it's partially redundant.  If so, insert a PHI
  // which can then be used to thread the values.
  //
  // This is particularly important because reg2mem inserts loads and stores all
  // over the place, and this blocks jump threading if we don't zap them.
  Value *SimplifyValue = CondInst;
  if (CmpInst *CondCmp = dyn_cast<CmpInst>(SimplifyValue))
    if (isa<Constant>(CondCmp->getOperand(1)))
      SimplifyValue = CondCmp->getOperand(0);
  
  if (LoadInst *LI = dyn_cast<LoadInst>(SimplifyValue))
    if (SimplifyPartiallyRedundantLoad(LI))
      return true;
  
  // TODO: If we have: "br (X > 0)"  and we have a predecessor where we know
  // "(X == 4)" thread through this block.
  
  return false;
}

/// ProcessBranchOnDuplicateCond - We found a block and a predecessor of that
/// block that jump on exactly the same condition.  This means that we almost
/// always know the direction of the edge in the DESTBB:
///  PREDBB:
///     br COND, DESTBB, BBY
///  DESTBB:
///     br COND, BBZ, BBW
///
/// If DESTBB has multiple predecessors, we can't just constant fold the branch
/// in DESTBB, we have to thread over it.
bool JumpThreading::ProcessBranchOnDuplicateCond(BasicBlock *PredBB,
                                                 BasicBlock *BB) {
  BranchInst *PredBI = cast<BranchInst>(PredBB->getTerminator());
  
  // If both successors of PredBB go to DESTBB, we don't know anything.  We can
  // fold the branch to an unconditional one, which allows other recursive
  // simplifications.
  bool BranchDir;
  if (PredBI->getSuccessor(1) != BB)
    BranchDir = true;
  else if (PredBI->getSuccessor(0) != BB)
    BranchDir = false;
  else {
    DEBUG(errs() << "  In block '" << PredBB->getName()
          << "' folding terminator: " << *PredBB->getTerminator() << '\n');
    ++NumFolds;
    ConstantFoldTerminator(PredBB);
    return true;
  }
   
  BranchInst *DestBI = cast<BranchInst>(BB->getTerminator());

  // If the dest block has one predecessor, just fix the branch condition to a
  // constant and fold it.
  if (BB->getSinglePredecessor()) {
    DEBUG(errs() << "  In block '" << BB->getName()
          << "' folding condition to '" << BranchDir << "': "
          << *BB->getTerminator() << '\n');
    ++NumFolds;
    Value *OldCond = DestBI->getCondition();
    DestBI->setCondition(ConstantInt::get(Type::getInt1Ty(BB->getContext()),
                                          BranchDir));
    ConstantFoldTerminator(BB);
    RecursivelyDeleteTriviallyDeadInstructions(OldCond);
    return true;
  }
 
  
  // Next, figure out which successor we are threading to.
  BasicBlock *SuccBB = DestBI->getSuccessor(!BranchDir);
  
  // Ok, try to thread it!
  return ThreadEdge(BB, PredBB, SuccBB);
}

/// ProcessSwitchOnDuplicateCond - We found a block and a predecessor of that
/// block that switch on exactly the same condition.  This means that we almost
/// always know the direction of the edge in the DESTBB:
///  PREDBB:
///     switch COND [... DESTBB, BBY ... ]
///  DESTBB:
///     switch COND [... BBZ, BBW ]
///
/// Optimizing switches like this is very important, because simplifycfg builds
/// switches out of repeated 'if' conditions.
bool JumpThreading::ProcessSwitchOnDuplicateCond(BasicBlock *PredBB,
                                                 BasicBlock *DestBB) {
  // Can't thread edge to self.
  if (PredBB == DestBB)
    return false;
  
  SwitchInst *PredSI = cast<SwitchInst>(PredBB->getTerminator());
  SwitchInst *DestSI = cast<SwitchInst>(DestBB->getTerminator());

  // There are a variety of optimizations that we can potentially do on these
  // blocks: we order them from most to least preferable.
  
  // If DESTBB *just* contains the switch, then we can forward edges from PREDBB
  // directly to their destination.  This does not introduce *any* code size
  // growth.  Skip debug info first.
  BasicBlock::iterator BBI = DestBB->begin();
  while (isa<DbgInfoIntrinsic>(BBI))
    BBI++;
  
  // FIXME: Thread if it just contains a PHI.
  if (isa<SwitchInst>(BBI)) {
    bool MadeChange = false;
    // Ignore the default edge for now.
    for (unsigned i = 1, e = DestSI->getNumSuccessors(); i != e; ++i) {
      ConstantInt *DestVal = DestSI->getCaseValue(i);
      BasicBlock *DestSucc = DestSI->getSuccessor(i);
      
      // Okay, DestSI has a case for 'DestVal' that goes to 'DestSucc'.  See if
      // PredSI has an explicit case for it.  If so, forward.  If it is covered
      // by the default case, we can't update PredSI.
      unsigned PredCase = PredSI->findCaseValue(DestVal);
      if (PredCase == 0) continue;
      
      // If PredSI doesn't go to DestBB on this value, then it won't reach the
      // case on this condition.
      if (PredSI->getSuccessor(PredCase) != DestBB &&
          DestSI->getSuccessor(i) != DestBB)
        continue;

      // Otherwise, we're safe to make the change.  Make sure that the edge from
      // DestSI to DestSucc is not critical and has no PHI nodes.
      DEBUG(errs() << "FORWARDING EDGE " << *DestVal << "   FROM: " << *PredSI);
      DEBUG(errs() << "THROUGH: " << *DestSI);

      // If the destination has PHI nodes, just split the edge for updating
      // simplicity.
      if (isa<PHINode>(DestSucc->begin()) && !DestSucc->getSinglePredecessor()){
        SplitCriticalEdge(DestSI, i, this);
        DestSucc = DestSI->getSuccessor(i);
      }
      FoldSingleEntryPHINodes(DestSucc);
      PredSI->setSuccessor(PredCase, DestSucc);
      MadeChange = true;
    }
    
    if (MadeChange)
      return true;
  }
  
  return false;
}


/// SimplifyPartiallyRedundantLoad - If LI is an obviously partially redundant
/// load instruction, eliminate it by replacing it with a PHI node.  This is an
/// important optimization that encourages jump threading, and needs to be run
/// interlaced with other jump threading tasks.
bool JumpThreading::SimplifyPartiallyRedundantLoad(LoadInst *LI) {
  // Don't hack volatile loads.
  if (LI->isVolatile()) return false;
  
  // If the load is defined in a block with exactly one predecessor, it can't be
  // partially redundant.
  BasicBlock *LoadBB = LI->getParent();
  if (LoadBB->getSinglePredecessor())
    return false;
  
  Value *LoadedPtr = LI->getOperand(0);

  // If the loaded operand is defined in the LoadBB, it can't be available.
  // FIXME: Could do PHI translation, that would be fun :)
  if (Instruction *PtrOp = dyn_cast<Instruction>(LoadedPtr))
    if (PtrOp->getParent() == LoadBB)
      return false;
  
  // Scan a few instructions up from the load, to see if it is obviously live at
  // the entry to its block.
  BasicBlock::iterator BBIt = LI;

  if (Value *AvailableVal = FindAvailableLoadedValue(LoadedPtr, LoadBB, 
                                                     BBIt, 6)) {
    // If the value if the load is locally available within the block, just use
    // it.  This frequently occurs for reg2mem'd allocas.
    //cerr << "LOAD ELIMINATED:\n" << *BBIt << *LI << "\n";
    
    // If the returned value is the load itself, replace with an undef. This can
    // only happen in dead loops.
    if (AvailableVal == LI) AvailableVal = UndefValue::get(LI->getType());
    LI->replaceAllUsesWith(AvailableVal);
    LI->eraseFromParent();
    return true;
  }

  // Otherwise, if we scanned the whole block and got to the top of the block,
  // we know the block is locally transparent to the load.  If not, something
  // might clobber its value.
  if (BBIt != LoadBB->begin())
    return false;
  
  
  SmallPtrSet<BasicBlock*, 8> PredsScanned;
  typedef SmallVector<std::pair<BasicBlock*, Value*>, 8> AvailablePredsTy;
  AvailablePredsTy AvailablePreds;
  BasicBlock *OneUnavailablePred = 0;
  
  // If we got here, the loaded value is transparent through to the start of the
  // block.  Check to see if it is available in any of the predecessor blocks.
  for (pred_iterator PI = pred_begin(LoadBB), PE = pred_end(LoadBB);
       PI != PE; ++PI) {
    BasicBlock *PredBB = *PI;

    // If we already scanned this predecessor, skip it.
    if (!PredsScanned.insert(PredBB))
      continue;

    // Scan the predecessor to see if the value is available in the pred.
    BBIt = PredBB->end();
    Value *PredAvailable = FindAvailableLoadedValue(LoadedPtr, PredBB, BBIt, 6);
    if (!PredAvailable) {
      OneUnavailablePred = PredBB;
      continue;
    }
    
    // If so, this load is partially redundant.  Remember this info so that we
    // can create a PHI node.
    AvailablePreds.push_back(std::make_pair(PredBB, PredAvailable));
  }
  
  // If the loaded value isn't available in any predecessor, it isn't partially
  // redundant.
  if (AvailablePreds.empty()) return false;
  
  // Okay, the loaded value is available in at least one (and maybe all!)
  // predecessors.  If the value is unavailable in more than one unique
  // predecessor, we want to insert a merge block for those common predecessors.
  // This ensures that we only have to insert one reload, thus not increasing
  // code size.
  BasicBlock *UnavailablePred = 0;
  
  // If there is exactly one predecessor where the value is unavailable, the
  // already computed 'OneUnavailablePred' block is it.  If it ends in an
  // unconditional branch, we know that it isn't a critical edge.
  if (PredsScanned.size() == AvailablePreds.size()+1 &&
      OneUnavailablePred->getTerminator()->getNumSuccessors() == 1) {
    UnavailablePred = OneUnavailablePred;
  } else if (PredsScanned.size() != AvailablePreds.size()) {
    // Otherwise, we had multiple unavailable predecessors or we had a critical
    // edge from the one.
    SmallVector<BasicBlock*, 8> PredsToSplit;
    SmallPtrSet<BasicBlock*, 8> AvailablePredSet;

    for (unsigned i = 0, e = AvailablePreds.size(); i != e; ++i)
      AvailablePredSet.insert(AvailablePreds[i].first);

    // Add all the unavailable predecessors to the PredsToSplit list.
    for (pred_iterator PI = pred_begin(LoadBB), PE = pred_end(LoadBB);
         PI != PE; ++PI)
      if (!AvailablePredSet.count(*PI))
        PredsToSplit.push_back(*PI);
    
    // Split them out to their own block.
    UnavailablePred =
      SplitBlockPredecessors(LoadBB, &PredsToSplit[0], PredsToSplit.size(),
                             "thread-split", this);
  }
  
  // If the value isn't available in all predecessors, then there will be
  // exactly one where it isn't available.  Insert a load on that edge and add
  // it to the AvailablePreds list.
  if (UnavailablePred) {
    assert(UnavailablePred->getTerminator()->getNumSuccessors() == 1 &&
           "Can't handle critical edge here!");
    Value *NewVal = new LoadInst(LoadedPtr, LI->getName()+".pr",
                                 UnavailablePred->getTerminator());
    AvailablePreds.push_back(std::make_pair(UnavailablePred, NewVal));
  }
  
  // Now we know that each predecessor of this block has a value in
  // AvailablePreds, sort them for efficient access as we're walking the preds.
  array_pod_sort(AvailablePreds.begin(), AvailablePreds.end());
  
  // Create a PHI node at the start of the block for the PRE'd load value.
  PHINode *PN = PHINode::Create(LI->getType(), "", LoadBB->begin());
  PN->takeName(LI);
  
  // Insert new entries into the PHI for each predecessor.  A single block may
  // have multiple entries here.
  for (pred_iterator PI = pred_begin(LoadBB), E = pred_end(LoadBB); PI != E;
       ++PI) {
    AvailablePredsTy::iterator I = 
      std::lower_bound(AvailablePreds.begin(), AvailablePreds.end(),
                       std::make_pair(*PI, (Value*)0));
    
    assert(I != AvailablePreds.end() && I->first == *PI &&
           "Didn't find entry for predecessor!");
    
    PN->addIncoming(I->second, I->first);
  }
  
  //cerr << "PRE: " << *LI << *PN << "\n";
  
  LI->replaceAllUsesWith(PN);
  LI->eraseFromParent();
  
  return true;
}


/// ProcessJumpOnPHI - We have a conditional branch or switch on a PHI node in
/// the current block.  See if there are any simplifications we can do based on
/// inputs to the phi node.
/// 
bool JumpThreading::ProcessJumpOnPHI(PHINode *PN) {
  BasicBlock *BB = PN->getParent();
  
  // See if the phi node has any constant integer or undef values.  If so, we
  // can determine where the corresponding predecessor will branch.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *PredVal = PN->getIncomingValue(i);
    
    // Check to see if this input is a constant integer.  If so, the direction
    // of the branch is predictable.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(PredVal)) {
      // Merge any common predecessors that will act the same.
      BasicBlock *PredBB = FactorCommonPHIPreds(PN, CI);
      
      BasicBlock *SuccBB;
      if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
        SuccBB = BI->getSuccessor(CI->isZero());
      else {
        SwitchInst *SI = cast<SwitchInst>(BB->getTerminator());
        SuccBB = SI->getSuccessor(SI->findCaseValue(CI));
      }
      
      // Ok, try to thread it!
      return ThreadEdge(BB, PredBB, SuccBB);
    }
    
    // If the input is an undef, then it doesn't matter which way it will go.
    // Pick an arbitrary dest and thread the edge.
    if (UndefValue *UV = dyn_cast<UndefValue>(PredVal)) {
      // Merge any common predecessors that will act the same.
      BasicBlock *PredBB = FactorCommonPHIPreds(PN, UV);
      BasicBlock *SuccBB =
        BB->getTerminator()->getSuccessor(GetBestDestForJumpOnUndef(BB));
      
      // Ok, try to thread it!
      return ThreadEdge(BB, PredBB, SuccBB);
    }
  }
  
  // If the incoming values are all variables, we don't know the destination of
  // any predecessors.  However, if any of the predecessor blocks end in an
  // unconditional branch, we can *duplicate* the jump into that block in order
  // to further encourage jump threading and to eliminate cases where we have
  // branch on a phi of an icmp (branch on icmp is much better).

  // We don't want to do this tranformation for switches, because we don't
  // really want to duplicate a switch.
  if (isa<SwitchInst>(BB->getTerminator()))
    return false;
  
  // Look for unconditional branch predecessors.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *PredBB = PN->getIncomingBlock(i);
    if (BranchInst *PredBr = dyn_cast<BranchInst>(PredBB->getTerminator()))
      if (PredBr->isUnconditional() &&
          // Try to duplicate BB into PredBB.
          DuplicateCondBranchOnPHIIntoPred(BB, PredBB))
        return true;
  }

  return false;
}


/// ProcessJumpOnLogicalPHI - PN's basic block contains a conditional branch
/// whose condition is an AND/OR where one side is PN.  If PN has constant
/// operands that permit us to evaluate the condition for some operand, thread
/// through the block.  For example with:
///   br (and X, phi(Y, Z, false))
/// the predecessor corresponding to the 'false' will always jump to the false
/// destination of the branch.
///
bool JumpThreading::ProcessBranchOnLogical(Value *V, BasicBlock *BB,
                                           bool isAnd) {
  // If this is a binary operator tree of the same AND/OR opcode, check the
  // LHS/RHS.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V))
    if ((isAnd && BO->getOpcode() == Instruction::And) ||
        (!isAnd && BO->getOpcode() == Instruction::Or)) {
      if (ProcessBranchOnLogical(BO->getOperand(0), BB, isAnd))
        return true;
      if (ProcessBranchOnLogical(BO->getOperand(1), BB, isAnd))
        return true;
    }
      
  // If this isn't a PHI node, we can't handle it.
  PHINode *PN = dyn_cast<PHINode>(V);
  if (!PN || PN->getParent() != BB) return false;
                                             
  // We can only do the simplification for phi nodes of 'false' with AND or
  // 'true' with OR.  See if we have any entries in the phi for this.
  unsigned PredNo = ~0U;
  ConstantInt *PredCst = ConstantInt::get(Type::getInt1Ty(BB->getContext()),
                                          !isAnd);
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (PN->getIncomingValue(i) == PredCst) {
      PredNo = i;
      break;
    }
  }
  
  // If no match, bail out.
  if (PredNo == ~0U)
    return false;
  
  // If so, we can actually do this threading.  Merge any common predecessors
  // that will act the same.
  BasicBlock *PredBB = FactorCommonPHIPreds(PN, PredCst);
  
  // Next, figure out which successor we are threading to.  If this was an AND,
  // the constant must be FALSE, and we must be targeting the 'false' block.
  // If this is an OR, the constant must be TRUE, and we must be targeting the
  // 'true' block.
  BasicBlock *SuccBB = BB->getTerminator()->getSuccessor(isAnd);
  
  // Ok, try to thread it!
  return ThreadEdge(BB, PredBB, SuccBB);
}

/// GetResultOfComparison - Given an icmp/fcmp predicate and the left and right
/// hand sides of the compare instruction, try to determine the result. If the
/// result can not be determined, a null pointer is returned.
static Constant *GetResultOfComparison(CmpInst::Predicate pred,
                                       Value *LHS, Value *RHS,
                                       LLVMContext &Context) {
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantExpr::getCompare(pred, CLHS, CRHS);

  if (LHS == RHS)
    if (isa<IntegerType>(LHS->getType()) || isa<PointerType>(LHS->getType()))
      return ICmpInst::isTrueWhenEqual(pred) ? 
                 ConstantInt::getTrue(Context) : ConstantInt::getFalse(Context);

  return 0;
}

/// ProcessBranchOnCompare - We found a branch on a comparison between a phi
/// node and a value.  If we can identify when the comparison is true between
/// the phi inputs and the value, we can fold the compare for that edge and
/// thread through it.
bool JumpThreading::ProcessBranchOnCompare(CmpInst *Cmp, BasicBlock *BB) {
  PHINode *PN = cast<PHINode>(Cmp->getOperand(0));
  Value *RHS = Cmp->getOperand(1);
  
  // If the phi isn't in the current block, an incoming edge to this block
  // doesn't control the destination.
  if (PN->getParent() != BB)
    return false;
  
  // We can do this simplification if any comparisons fold to true or false.
  // See if any do.
  Value *PredVal = 0;
  bool TrueDirection = false;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    PredVal = PN->getIncomingValue(i);
    
    Constant *Res = GetResultOfComparison(Cmp->getPredicate(), PredVal,
                                          RHS, Cmp->getContext());
    if (!Res) {
      PredVal = 0;
      continue;
    }
    
    // If this folded to a constant expr, we can't do anything.
    if (ConstantInt *ResC = dyn_cast<ConstantInt>(Res)) {
      TrueDirection = ResC->getZExtValue();
      break;
    }
    // If this folded to undef, just go the false way.
    if (isa<UndefValue>(Res)) {
      TrueDirection = false;
      break;
    }
    
    // Otherwise, we can't fold this input.
    PredVal = 0;
  }
  
  // If no match, bail out.
  if (PredVal == 0)
    return false;
  
  // If so, we can actually do this threading.  Merge any common predecessors
  // that will act the same.
  BasicBlock *PredBB = FactorCommonPHIPreds(PN, PredVal);
  
  // Next, get our successor.
  BasicBlock *SuccBB = BB->getTerminator()->getSuccessor(!TrueDirection);
  
  // Ok, try to thread it!
  return ThreadEdge(BB, PredBB, SuccBB);
}


/// AddPHINodeEntriesForMappedBlock - We're adding 'NewPred' as a new
/// predecessor to the PHIBB block.  If it has PHI nodes, add entries for
/// NewPred using the entries from OldPred (suitably mapped).
static void AddPHINodeEntriesForMappedBlock(BasicBlock *PHIBB,
                                            BasicBlock *OldPred,
                                            BasicBlock *NewPred,
                                     DenseMap<Instruction*, Value*> &ValueMap) {
  for (BasicBlock::iterator PNI = PHIBB->begin();
       PHINode *PN = dyn_cast<PHINode>(PNI); ++PNI) {
    // Ok, we have a PHI node.  Figure out what the incoming value was for the
    // DestBlock.
    Value *IV = PN->getIncomingValueForBlock(OldPred);
    
    // Remap the value if necessary.
    if (Instruction *Inst = dyn_cast<Instruction>(IV)) {
      DenseMap<Instruction*, Value*>::iterator I = ValueMap.find(Inst);
      if (I != ValueMap.end())
        IV = I->second;
    }
    
    PN->addIncoming(IV, NewPred);
  }
}

/// ThreadEdge - We have decided that it is safe and profitable to thread an
/// edge from PredBB to SuccBB across BB.  Transform the IR to reflect this
/// change.
bool JumpThreading::ThreadEdge(BasicBlock *BB, BasicBlock *PredBB, 
                               BasicBlock *SuccBB) {
  // If threading to the same block as we come from, we would infinite loop.
  if (SuccBB == BB) {
    DEBUG(errs() << "  Not threading across BB '" << BB->getName()
          << "' - would thread to self!\n");
    return false;
  }
  
  // If threading this would thread across a loop header, don't thread the edge.
  // See the comments above FindLoopHeaders for justifications and caveats.
  if (LoopHeaders.count(BB)) {
    DEBUG(errs() << "  Not threading from '" << PredBB->getName()
          << "' across loop header BB '" << BB->getName()
          << "' to dest BB '" << SuccBB->getName()
          << "' - it might create an irreducible loop!\n");
    return false;
  }

  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB);
  if (JumpThreadCost > Threshold) {
    DEBUG(errs() << "  Not threading BB '" << BB->getName()
          << "' - Cost is too high: " << JumpThreadCost << "\n");
    return false;
  }
  
  // And finally, do it!
  DEBUG(errs() << "  Threading edge from '" << PredBB->getName() << "' to '"
        << SuccBB->getName() << "' with cost: " << JumpThreadCost
        << ", across block:\n    "
        << *BB << "\n");
  
  // We are going to have to map operands from the original BB block to the new
  // copy of the block 'NewBB'.  If there are PHI nodes in BB, evaluate them to
  // account for entry from PredBB.
  DenseMap<Instruction*, Value*> ValueMapping;
  
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), 
                                         BB->getName()+".thread", 
                                         BB->getParent(), BB);
  NewBB->moveAfter(PredBB);
  
  BasicBlock::iterator BI = BB->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(PredBB);
  
  // Clone the non-phi instructions of BB into NewBB, keeping track of the
  // mapping and using it to remap operands in the cloned instructions.
  for (; !isa<TerminatorInst>(BI); ++BI) {
    Instruction *New = BI->clone();
    New->setName(BI->getName());
    NewBB->getInstList().push_back(New);
    ValueMapping[BI] = New;
   
    // Remap operands to patch up intra-block references.
    for (unsigned i = 0, e = New->getNumOperands(); i != e; ++i)
      if (Instruction *Inst = dyn_cast<Instruction>(New->getOperand(i))) {
        DenseMap<Instruction*, Value*>::iterator I = ValueMapping.find(Inst);
        if (I != ValueMapping.end())
          New->setOperand(i, I->second);
      }
  }
  
  // We didn't copy the terminator from BB over to NewBB, because there is now
  // an unconditional jump to SuccBB.  Insert the unconditional jump.
  BranchInst::Create(SuccBB, NewBB);
  
  // Check to see if SuccBB has PHI nodes. If so, we need to add entries to the
  // PHI nodes for NewBB now.
  AddPHINodeEntriesForMappedBlock(SuccBB, BB, NewBB, ValueMapping);
  
  // If there were values defined in BB that are used outside the block, then we
  // now have to update all uses of the value to use either the original value,
  // the cloned value, or some PHI derived value.  This can require arbitrary
  // PHI insertion, of which we are prepared to do, clean these up now.
  SSAUpdater SSAUpdate;
  SmallVector<Use*, 16> UsesToRename;
  for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    // Scan all uses of this instruction to see if it is used outside of its
    // block, and if so, record them in UsesToRename.
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
         ++UI) {
      Instruction *User = cast<Instruction>(*UI);
      if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
        if (UserPN->getIncomingBlock(UI) == BB)
          continue;
      } else if (User->getParent() == BB)
        continue;
      
      UsesToRename.push_back(&UI.getUse());
    }
    
    // If there are no uses outside the block, we're done with this instruction.
    if (UsesToRename.empty())
      continue;
    
    DEBUG(errs() << "JT: Renaming non-local uses of: " << *I << "\n");

    // We found a use of I outside of BB.  Rename all uses of I that are outside
    // its block to be uses of the appropriate PHI node etc.  See ValuesInBlocks
    // with the two values we know.
    SSAUpdate.Initialize(I);
    SSAUpdate.AddAvailableValue(BB, I);
    SSAUpdate.AddAvailableValue(NewBB, ValueMapping[I]);
    
    while (!UsesToRename.empty())
      SSAUpdate.RewriteUse(*UsesToRename.pop_back_val());
    DEBUG(errs() << "\n");
  }
  
  
  // Ok, NewBB is good to go.  Update the terminator of PredBB to jump to
  // NewBB instead of BB.  This eliminates predecessors from BB, which requires
  // us to simplify any PHI nodes in BB.
  TerminatorInst *PredTerm = PredBB->getTerminator();
  for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i)
    if (PredTerm->getSuccessor(i) == BB) {
      BB->removePredecessor(PredBB);
      PredTerm->setSuccessor(i, NewBB);
    }
  
  // At this point, the IR is fully up to date and consistent.  Do a quick scan
  // over the new instructions and zap any that are constants or dead.  This
  // frequently happens because of phi translation.
  BI = NewBB->begin();
  for (BasicBlock::iterator E = NewBB->end(); BI != E; ) {
    Instruction *Inst = BI++;
    if (Constant *C = ConstantFoldInstruction(Inst, BB->getContext(), TD)) {
      Inst->replaceAllUsesWith(C);
      Inst->eraseFromParent();
      continue;
    }
    
    RecursivelyDeleteTriviallyDeadInstructions(Inst);
  }
  
  // Threaded an edge!
  ++NumThreads;
  return true;
}

/// DuplicateCondBranchOnPHIIntoPred - PredBB contains an unconditional branch
/// to BB which contains an i1 PHI node and a conditional branch on that PHI.
/// If we can duplicate the contents of BB up into PredBB do so now, this
/// improves the odds that the branch will be on an analyzable instruction like
/// a compare.
bool JumpThreading::DuplicateCondBranchOnPHIIntoPred(BasicBlock *BB,
                                                     BasicBlock *PredBB) {
  // If BB is a loop header, then duplicating this block outside the loop would
  // cause us to transform this into an irreducible loop, don't do this.
  // See the comments above FindLoopHeaders for justifications and caveats.
  if (LoopHeaders.count(BB)) {
    DEBUG(errs() << "  Not duplicating loop header '" << BB->getName()
          << "' into predecessor block '" << PredBB->getName()
          << "' - it might create an irreducible loop!\n");
    return false;
  }
  
  unsigned DuplicationCost = getJumpThreadDuplicationCost(BB);
  if (DuplicationCost > Threshold) {
    DEBUG(errs() << "  Not duplicating BB '" << BB->getName()
          << "' - Cost is too high: " << DuplicationCost << "\n");
    return false;
  }
  
  // Okay, we decided to do this!  Clone all the instructions in BB onto the end
  // of PredBB.
  DEBUG(errs() << "  Duplicating block '" << BB->getName() << "' into end of '"
        << PredBB->getName() << "' to eliminate branch on phi.  Cost: "
        << DuplicationCost << " block is:" << *BB << "\n");
  
  // We are going to have to map operands from the original BB block into the
  // PredBB block.  Evaluate PHI nodes in BB.
  DenseMap<Instruction*, Value*> ValueMapping;
  
  BasicBlock::iterator BI = BB->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(PredBB);
  
  BranchInst *OldPredBranch = cast<BranchInst>(PredBB->getTerminator());
  
  // Clone the non-phi instructions of BB into PredBB, keeping track of the
  // mapping and using it to remap operands in the cloned instructions.
  for (; BI != BB->end(); ++BI) {
    Instruction *New = BI->clone();
    New->setName(BI->getName());
    PredBB->getInstList().insert(OldPredBranch, New);
    ValueMapping[BI] = New;
    
    // Remap operands to patch up intra-block references.
    for (unsigned i = 0, e = New->getNumOperands(); i != e; ++i)
      if (Instruction *Inst = dyn_cast<Instruction>(New->getOperand(i))) {
        DenseMap<Instruction*, Value*>::iterator I = ValueMapping.find(Inst);
        if (I != ValueMapping.end())
          New->setOperand(i, I->second);
      }
  }
  
  // Check to see if the targets of the branch had PHI nodes. If so, we need to
  // add entries to the PHI nodes for branch from PredBB now.
  BranchInst *BBBranch = cast<BranchInst>(BB->getTerminator());
  AddPHINodeEntriesForMappedBlock(BBBranch->getSuccessor(0), BB, PredBB,
                                  ValueMapping);
  AddPHINodeEntriesForMappedBlock(BBBranch->getSuccessor(1), BB, PredBB,
                                  ValueMapping);
  
  // If there were values defined in BB that are used outside the block, then we
  // now have to update all uses of the value to use either the original value,
  // the cloned value, or some PHI derived value.  This can require arbitrary
  // PHI insertion, of which we are prepared to do, clean these up now.
  SSAUpdater SSAUpdate;
  SmallVector<Use*, 16> UsesToRename;
  for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    // Scan all uses of this instruction to see if it is used outside of its
    // block, and if so, record them in UsesToRename.
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
         ++UI) {
      Instruction *User = cast<Instruction>(*UI);
      if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
        if (UserPN->getIncomingBlock(UI) == BB)
          continue;
      } else if (User->getParent() == BB)
        continue;
      
      UsesToRename.push_back(&UI.getUse());
    }
    
    // If there are no uses outside the block, we're done with this instruction.
    if (UsesToRename.empty())
      continue;
    
    DEBUG(errs() << "JT: Renaming non-local uses of: " << *I << "\n");
    
    // We found a use of I outside of BB.  Rename all uses of I that are outside
    // its block to be uses of the appropriate PHI node etc.  See ValuesInBlocks
    // with the two values we know.
    SSAUpdate.Initialize(I);
    SSAUpdate.AddAvailableValue(BB, I);
    SSAUpdate.AddAvailableValue(PredBB, ValueMapping[I]);
    
    while (!UsesToRename.empty())
      SSAUpdate.RewriteUse(*UsesToRename.pop_back_val());
    DEBUG(errs() << "\n");
  }
  
  // PredBB no longer jumps to BB, remove entries in the PHI node for the edge
  // that we nuked.
  BB->removePredecessor(PredBB);
  
  // Remove the unconditional branch at the end of the PredBB block.
  OldPredBranch->eraseFromParent();
  
  ++NumDupes;
  return true;
}


