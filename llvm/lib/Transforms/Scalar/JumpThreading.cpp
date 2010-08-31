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
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumThreads, "Number of jumps threaded");
STATISTIC(NumFolds,   "Number of terminators folded");
STATISTIC(NumDupes,   "Number of branch blocks duplicated to eliminate phi");

static cl::opt<unsigned>
Threshold("jump-threading-threshold", 
          cl::desc("Max block size to duplicate for jump threading"),
          cl::init(6), cl::Hidden);

// Turn on use of LazyValueInfo.
static cl::opt<bool>
EnableLVI("enable-jump-threading-lvi",
          cl::desc("Use LVI for jump threading"),
          cl::init(true),
          cl::ReallyHidden);



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
    LazyValueInfo *LVI;
#ifdef NDEBUG
    SmallPtrSet<BasicBlock*, 16> LoopHeaders;
#else
    SmallSet<AssertingVH<BasicBlock>, 16> LoopHeaders;
#endif
    DenseSet<std::pair<Value*, BasicBlock*> > RecursionSet;
    
    // RAII helper for updating the recursion stack.
    struct RecursionSetRemover {
      DenseSet<std::pair<Value*, BasicBlock*> > &TheSet;
      std::pair<Value*, BasicBlock*> ThePair;
      
      RecursionSetRemover(DenseSet<std::pair<Value*, BasicBlock*> > &S,
                          std::pair<Value*, BasicBlock*> P)
        : TheSet(S), ThePair(P) { }
      
      ~RecursionSetRemover() {
        TheSet.erase(ThePair);
      }
    };
  public:
    static char ID; // Pass identification
    JumpThreading() : FunctionPass(ID) {}

    bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      if (EnableLVI)
        AU.addRequired<LazyValueInfo>();
    }
    
    void FindLoopHeaders(Function &F);
    bool ProcessBlock(BasicBlock *BB);
    bool ThreadEdge(BasicBlock *BB, const SmallVectorImpl<BasicBlock*> &PredBBs,
                    BasicBlock *SuccBB);
    bool DuplicateCondBranchOnPHIIntoPred(BasicBlock *BB,
                                  const SmallVectorImpl<BasicBlock *> &PredBBs);
    
    typedef SmallVectorImpl<std::pair<ConstantInt*,
                                      BasicBlock*> > PredValueInfo;
    
    bool ComputeValueKnownInPredecessors(Value *V, BasicBlock *BB,
                                         PredValueInfo &Result);
    bool ProcessThreadableEdges(Value *Cond, BasicBlock *BB);
    
    
    bool ProcessBranchOnDuplicateCond(BasicBlock *PredBB, BasicBlock *DestBB);
    bool ProcessSwitchOnDuplicateCond(BasicBlock *PredBB, BasicBlock *DestBB);

    bool ProcessBranchOnPHI(PHINode *PN);
    bool ProcessBranchOnXOR(BinaryOperator *BO);
    
    bool SimplifyPartiallyRedundantLoad(LoadInst *LI);
  };
}

char JumpThreading::ID = 0;
INITIALIZE_PASS(JumpThreading, "jump-threading",
                "Jump Threading", false, false);

// Public interface to the Jump Threading pass
FunctionPass *llvm::createJumpThreadingPass() { return new JumpThreading(); }

/// runOnFunction - Top level algorithm.
///
bool JumpThreading::runOnFunction(Function &F) {
  DEBUG(dbgs() << "Jump threading on function '" << F.getName() << "'\n");
  TD = getAnalysisIfAvailable<TargetData>();
  LVI = EnableLVI ? &getAnalysis<LazyValueInfo>() : 0;
  
  FindLoopHeaders(F);
  
  bool Changed, EverChanged = false;
  do {
    Changed = false;
    for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
      BasicBlock *BB = I;
      // Thread all of the branches we can over this block. 
      while (ProcessBlock(BB))
        Changed = true;
      
      ++I;
      
      // If the block is trivially dead, zap it.  This eliminates the successor
      // edges which simplifies the CFG.
      if (pred_begin(BB) == pred_end(BB) &&
          BB != &BB->getParent()->getEntryBlock()) {
        DEBUG(dbgs() << "  JT: Deleting dead block '" << BB->getName()
              << "' with terminator: " << *BB->getTerminator() << '\n');
        LoopHeaders.erase(BB);
        if (LVI) LVI->eraseBlock(BB);
        DeleteDeadBlock(BB);
        Changed = true;
      } else if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
        // Can't thread an unconditional jump, but if the block is "almost
        // empty", we can replace uses of it with uses of the successor and make
        // this dead.
        if (BI->isUnconditional() && 
            BB != &BB->getParent()->getEntryBlock()) {
          BasicBlock::iterator BBI = BB->getFirstNonPHI();
          // Ignore dbg intrinsics.
          while (isa<DbgInfoIntrinsic>(BBI))
            ++BBI;
          // If the terminator is the only non-phi instruction, try to nuke it.
          if (BBI->isTerminator()) {
            // Since TryToSimplifyUncondBranchFromEmptyBlock may delete the
            // block, we have to make sure it isn't in the LoopHeaders set.  We
            // reinsert afterward if needed.
            bool ErasedFromLoopHeaders = LoopHeaders.erase(BB);
            BasicBlock *Succ = BI->getSuccessor(0);
            
            // FIXME: It is always conservatively correct to drop the info
            // for a block even if it doesn't get erased.  This isn't totally
            // awesome, but it allows us to use AssertingVH to prevent nasty
            // dangling pointer issues within LazyValueInfo.
            if (LVI) LVI->eraseBlock(BB);
            if (TryToSimplifyUncondBranchFromEmptyBlock(BB)) {
              Changed = true;
              // If we deleted BB and BB was the header of a loop, then the
              // successor is now the header of the loop.
              BB = Succ;
            }
            
            if (ErasedFromLoopHeaders)
              LoopHeaders.insert(BB);
          }
        }
      }
    }
    EverChanged |= Changed;
  } while (Changed);
  
  LoopHeaders.clear();
  return EverChanged;
}

/// getJumpThreadDuplicationCost - Return the cost of duplicating this block to
/// thread across it.
static unsigned getJumpThreadDuplicationCost(const BasicBlock *BB) {
  /// Ignore PHI nodes, these will be flattened when duplication happens.
  BasicBlock::const_iterator I = BB->getFirstNonPHI();
  
  // FIXME: THREADING will delete values that are just used to compute the
  // branch, so they shouldn't count against the duplication cost.
  
  
  // Sum up the cost of each instruction until we get to the terminator.  Don't
  // include the terminator because the copy won't include it.
  unsigned Size = 0;
  for (; !isa<TerminatorInst>(I); ++I) {
    // Debugger intrinsics don't incur code size.
    if (isa<DbgInfoIntrinsic>(I)) continue;
    
    // If this is a pointer->pointer bitcast, it is free.
    if (isa<BitCastInst>(I) && I->getType()->isPointerTy())
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
      else if (!CI->getType()->isVectorTy())
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

// Helper method for ComputeValueKnownInPredecessors.  If Value is a
// ConstantInt, push it.  If it's an undef, push 0.  Otherwise, do nothing.
static void PushConstantIntOrUndef(SmallVectorImpl<std::pair<ConstantInt*,
                                                        BasicBlock*> > &Result,
                              Constant *Value, BasicBlock* BB){
  if (ConstantInt *FoldedCInt = dyn_cast<ConstantInt>(Value))
    Result.push_back(std::make_pair(FoldedCInt, BB));
  else if (isa<UndefValue>(Value))
    Result.push_back(std::make_pair((ConstantInt*)0, BB));
}

/// ComputeValueKnownInPredecessors - Given a basic block BB and a value V, see
/// if we can infer that the value is a known ConstantInt in any of our
/// predecessors.  If so, return the known list of value and pred BB in the
/// result vector.  If a value is known to be undef, it is returned as null.
///
/// This returns true if there were any known values.
///
bool JumpThreading::
ComputeValueKnownInPredecessors(Value *V, BasicBlock *BB,PredValueInfo &Result){
  // This method walks up use-def chains recursively.  Because of this, we could
  // get into an infinite loop going around loops in the use-def chain.  To
  // prevent this, keep track of what (value, block) pairs we've already visited
  // and terminate the search if we loop back to them
  if (!RecursionSet.insert(std::make_pair(V, BB)).second)
    return false;
  
  // An RAII help to remove this pair from the recursion set once the recursion
  // stack pops back out again.
  RecursionSetRemover remover(RecursionSet, std::make_pair(V, BB));
  
  // If V is a constantint, then it is known in all predecessors.
  if (isa<ConstantInt>(V) || isa<UndefValue>(V)) {
    ConstantInt *CI = dyn_cast<ConstantInt>(V);
    
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
      Result.push_back(std::make_pair(CI, *PI));
    
    return true;
  }
  
  // If V is a non-instruction value, or an instruction in a different block,
  // then it can't be derived from a PHI.
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0 || I->getParent() != BB) {
    
    // Okay, if this is a live-in value, see if it has a known value at the end
    // of any of our predecessors.
    //
    // FIXME: This should be an edge property, not a block end property.
    /// TODO: Per PR2563, we could infer value range information about a
    /// predecessor based on its terminator.
    //
    if (LVI) {
      // FIXME: change this to use the more-rich 'getPredicateOnEdge' method if
      // "I" is a non-local compare-with-a-constant instruction.  This would be
      // able to handle value inequalities better, for example if the compare is
      // "X < 4" and "X < 3" is known true but "X < 4" itself is not available.
      // Perhaps getConstantOnEdge should be smart enough to do this?
      
      for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
        BasicBlock *P = *PI;
        // If the value is known by LazyValueInfo to be a constant in a
        // predecessor, use that information to try to thread this block.
        Constant *PredCst = LVI->getConstantOnEdge(V, P, BB);
        if (PredCst == 0 ||
            (!isa<ConstantInt>(PredCst) && !isa<UndefValue>(PredCst)))
          continue;
        
        Result.push_back(std::make_pair(dyn_cast<ConstantInt>(PredCst), P));
      }
      
      return !Result.empty();
    }
    
    return false;
  }
  
  /// If I is a PHI node, then we know the incoming values for any constants.
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *InVal = PN->getIncomingValue(i);
      if (isa<ConstantInt>(InVal) || isa<UndefValue>(InVal)) {
        ConstantInt *CI = dyn_cast<ConstantInt>(InVal);
        Result.push_back(std::make_pair(CI, PN->getIncomingBlock(i)));
      } else if (LVI) {
        Constant *CI = LVI->getConstantOnEdge(InVal,
                                              PN->getIncomingBlock(i), BB);
        // LVI returns null is no value could be determined.
        if (!CI) continue;
        PushConstantIntOrUndef(Result, CI, PN->getIncomingBlock(i));
      }
    }
    
    return !Result.empty();
  }
  
  SmallVector<std::pair<ConstantInt*, BasicBlock*>, 8> LHSVals, RHSVals;

  // Handle some boolean conditions.
  if (I->getType()->getPrimitiveSizeInBits() == 1) { 
    // X | true -> true
    // X & false -> false
    if (I->getOpcode() == Instruction::Or ||
        I->getOpcode() == Instruction::And) {
      ComputeValueKnownInPredecessors(I->getOperand(0), BB, LHSVals);
      ComputeValueKnownInPredecessors(I->getOperand(1), BB, RHSVals);
      
      if (LHSVals.empty() && RHSVals.empty())
        return false;
      
      ConstantInt *InterestingVal;
      if (I->getOpcode() == Instruction::Or)
        InterestingVal = ConstantInt::getTrue(I->getContext());
      else
        InterestingVal = ConstantInt::getFalse(I->getContext());
      
      SmallPtrSet<BasicBlock*, 4> LHSKnownBBs;
      
      // Scan for the sentinel.  If we find an undef, force it to the
      // interesting value: x|undef -> true and x&undef -> false.
      for (unsigned i = 0, e = LHSVals.size(); i != e; ++i)
        if (LHSVals[i].first == InterestingVal || LHSVals[i].first == 0) {
          Result.push_back(LHSVals[i]);
          Result.back().first = InterestingVal;
          LHSKnownBBs.insert(LHSVals[i].second);
        }
      for (unsigned i = 0, e = RHSVals.size(); i != e; ++i)
        if (RHSVals[i].first == InterestingVal || RHSVals[i].first == 0) {
          // If we already inferred a value for this block on the LHS, don't
          // re-add it.
          if (!LHSKnownBBs.count(RHSVals[i].second)) {
            Result.push_back(RHSVals[i]);
            Result.back().first = InterestingVal;
          }
        }
      
      return !Result.empty();
    }
    
    // Handle the NOT form of XOR.
    if (I->getOpcode() == Instruction::Xor &&
        isa<ConstantInt>(I->getOperand(1)) &&
        cast<ConstantInt>(I->getOperand(1))->isOne()) {
      ComputeValueKnownInPredecessors(I->getOperand(0), BB, Result);
      if (Result.empty())
        return false;

      // Invert the known values.
      for (unsigned i = 0, e = Result.size(); i != e; ++i)
        if (Result[i].first)
          Result[i].first =
            cast<ConstantInt>(ConstantExpr::getNot(Result[i].first));
      
      return true;
    }
  
  // Try to simplify some other binary operator values.
  } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(BO->getOperand(1))) {
      SmallVector<std::pair<ConstantInt*, BasicBlock*>, 8> LHSVals;
      ComputeValueKnownInPredecessors(BO->getOperand(0), BB, LHSVals);
    
      // Try to use constant folding to simplify the binary operator.
      for (unsigned i = 0, e = LHSVals.size(); i != e; ++i) {
        Constant *V = LHSVals[i].first ? LHSVals[i].first :
                                 cast<Constant>(UndefValue::get(BO->getType()));
        Constant *Folded = ConstantExpr::get(BO->getOpcode(), V, CI);
        
        PushConstantIntOrUndef(Result, Folded, LHSVals[i].second);
      }
    }
      
    return !Result.empty();
  }
  
  // Handle compare with phi operand, where the PHI is defined in this block.
  if (CmpInst *Cmp = dyn_cast<CmpInst>(I)) {
    PHINode *PN = dyn_cast<PHINode>(Cmp->getOperand(0));
    if (PN && PN->getParent() == BB) {
      // We can do this simplification if any comparisons fold to true or false.
      // See if any do.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        BasicBlock *PredBB = PN->getIncomingBlock(i);
        Value *LHS = PN->getIncomingValue(i);
        Value *RHS = Cmp->getOperand(1)->DoPHITranslation(BB, PredBB);
        
        Value *Res = SimplifyCmpInst(Cmp->getPredicate(), LHS, RHS, TD);
        if (Res == 0) {
          if (!LVI || !isa<Constant>(RHS))
            continue;
          
          LazyValueInfo::Tristate 
            ResT = LVI->getPredicateOnEdge(Cmp->getPredicate(), LHS,
                                           cast<Constant>(RHS), PredBB, BB);
          if (ResT == LazyValueInfo::Unknown)
            continue;
          Res = ConstantInt::get(Type::getInt1Ty(LHS->getContext()), ResT);
        }
        
        if (Constant *ConstRes = dyn_cast<Constant>(Res))
          PushConstantIntOrUndef(Result, ConstRes, PredBB);
      }
      
      return !Result.empty();
    }
    
    
    // If comparing a live-in value against a constant, see if we know the
    // live-in value on any predecessors.
    if (LVI && isa<Constant>(Cmp->getOperand(1)) &&
        Cmp->getType()->isIntegerTy()) {
      if (!isa<Instruction>(Cmp->getOperand(0)) ||
          cast<Instruction>(Cmp->getOperand(0))->getParent() != BB) {
        Constant *RHSCst = cast<Constant>(Cmp->getOperand(1));

        for (pred_iterator PI = pred_begin(BB), E = pred_end(BB);PI != E; ++PI){
          BasicBlock *P = *PI;
          // If the value is known by LazyValueInfo to be a constant in a
          // predecessor, use that information to try to thread this block.
          LazyValueInfo::Tristate Res =
            LVI->getPredicateOnEdge(Cmp->getPredicate(), Cmp->getOperand(0),
                                    RHSCst, P, BB);
          if (Res == LazyValueInfo::Unknown)
            continue;

          Constant *ResC = ConstantInt::get(Cmp->getType(), Res);
          Result.push_back(std::make_pair(cast<ConstantInt>(ResC), P));
        }

        return !Result.empty();
      }
      
      // Try to find a constant value for the LHS of a comparison,
      // and evaluate it statically if we can.
      if (Constant *CmpConst = dyn_cast<Constant>(Cmp->getOperand(1))) {
        SmallVector<std::pair<ConstantInt*, BasicBlock*>, 8> LHSVals;
        ComputeValueKnownInPredecessors(I->getOperand(0), BB, LHSVals);
        
        for (unsigned i = 0, e = LHSVals.size(); i != e; ++i) {
          Constant *V = LHSVals[i].first ? LHSVals[i].first :
                           cast<Constant>(UndefValue::get(CmpConst->getType()));
          Constant *Folded = ConstantExpr::getCompare(Cmp->getPredicate(),
                                                      V, CmpConst);
          PushConstantIntOrUndef(Result, Folded, LHSVals[i].second);
        }
        
        return !Result.empty();
      }
    }
  }
  
  if (LVI) {
    // If all else fails, see if LVI can figure out a constant value for us.
    Constant *CI = LVI->getConstant(V, BB);
    ConstantInt *CInt = dyn_cast_or_null<ConstantInt>(CI);
    if (CInt) {
      for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
        Result.push_back(std::make_pair(CInt, *PI));
    }
    
    return !Result.empty();
  }
  
  return false;
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
  // If the block is trivially dead, just return and let the caller nuke it.
  // This simplifies other transformations.
  if (pred_begin(BB) == pred_end(BB) &&
      BB != &BB->getParent()->getEntryBlock())
    return false;
  
  // If this block has a single predecessor, and if that pred has a single
  // successor, merge the blocks.  This encourages recursive jump threading
  // because now the condition in this block can be threaded through
  // predecessors of our predecessor block.
  if (BasicBlock *SinglePred = BB->getSinglePredecessor()) {
    if (SinglePred->getTerminator()->getNumSuccessors() == 1 &&
        SinglePred != BB) {
      // If SinglePred was a loop header, BB becomes one.
      if (LoopHeaders.erase(SinglePred))
        LoopHeaders.insert(BB);
      
      // Remember if SinglePred was the entry block of the function.  If so, we
      // will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      if (LVI) LVI->eraseBlock(SinglePred);
      MergeBasicBlockIntoOnlyPred(BB);
      
      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());
      return true;
    }
  }

  // Look to see if the terminator is a branch of switch, if not we can't thread
  // it.
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
    DEBUG(dbgs() << "  In block '" << BB->getName()
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
      RemovePredecessorAndSimplify(BBTerm->getSuccessor(i), BB, TD);
    }
    
    DEBUG(dbgs() << "  In block '" << BB->getName()
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
  if (!LVI &&
      !Condition->hasOneUse() && // Multiple uses.
      (CondInst == 0 || CondInst->getParent() != BB)) { // Non-local definition.
    pred_iterator PI = pred_begin(BB), E = pred_end(BB);
    if (isa<BranchInst>(BB->getTerminator())) {
      for (; PI != E; ++PI) {
        BasicBlock *P = *PI;
        if (BranchInst *PBI = dyn_cast<BranchInst>(P->getTerminator()))
          if (PBI->isConditional() && PBI->getCondition() == Condition &&
              ProcessBranchOnDuplicateCond(P, BB))
            return true;
      }
    } else {
      assert(isa<SwitchInst>(BB->getTerminator()) && "Unknown jump terminator");
      for (; PI != E; ++PI) {
        BasicBlock *P = *PI;
        if (SwitchInst *PSI = dyn_cast<SwitchInst>(P->getTerminator()))
          if (PSI->getCondition() == Condition &&
              ProcessSwitchOnDuplicateCond(P, BB))
            return true;
      }
    }
  }

  // All the rest of our checks depend on the condition being an instruction.
  if (CondInst == 0) {
    // FIXME: Unify this with code below.
    if (LVI && ProcessThreadableEdges(Condition, BB))
      return true;
    return false;
  }  
    
  
  if (CmpInst *CondCmp = dyn_cast<CmpInst>(CondInst)) {
    if (!LVI &&
        (!isa<PHINode>(CondCmp->getOperand(0)) ||
         cast<PHINode>(CondCmp->getOperand(0))->getParent() != BB)) {
      // If we have a comparison, loop over the predecessors to see if there is
      // a condition with a lexically identical value.
      pred_iterator PI = pred_begin(BB), E = pred_end(BB);
      for (; PI != E; ++PI) {
        BasicBlock *P = *PI;
        if (BranchInst *PBI = dyn_cast<BranchInst>(P->getTerminator()))
          if (PBI->isConditional() && P != BB) {
            if (CmpInst *CI = dyn_cast<CmpInst>(PBI->getCondition())) {
              if (CI->getOperand(0) == CondCmp->getOperand(0) &&
                  CI->getOperand(1) == CondCmp->getOperand(1) &&
                  CI->getPredicate() == CondCmp->getPredicate()) {
                // TODO: Could handle things like (x != 4) --> (x == 17)
                if (ProcessBranchOnDuplicateCond(P, BB))
                  return true;
              }
            }
          }
      }
    }
    
    // For a comparison where the LHS is outside this block, it's possible
    // that we've branched on it before.  Used LVI to see if we can simplify
    // the branch based on that.
    BranchInst *CondBr = dyn_cast<BranchInst>(BB->getTerminator());
    Constant *CondConst = dyn_cast<Constant>(CondCmp->getOperand(1));
    pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
    if (LVI && CondBr && CondConst && CondBr->isConditional() && PI != PE &&
        (!isa<Instruction>(CondCmp->getOperand(0)) ||
         cast<Instruction>(CondCmp->getOperand(0))->getParent() != BB)) {
      // For predecessor edge, determine if the comparison is true or false
      // on that edge.  If they're all true or all false, we can simplify the
      // branch.
      // FIXME: We could handle mixed true/false by duplicating code.
      LazyValueInfo::Tristate Baseline =      
        LVI->getPredicateOnEdge(CondCmp->getPredicate(), CondCmp->getOperand(0),
                                CondConst, *PI, BB);
      if (Baseline != LazyValueInfo::Unknown) {
        // Check that all remaining incoming values match the first one.
        while (++PI != PE) {
          LazyValueInfo::Tristate Ret = LVI->getPredicateOnEdge(
                                          CondCmp->getPredicate(),
                                          CondCmp->getOperand(0),
                                          CondConst, *PI, BB);
          if (Ret != Baseline) break;
        }
        
        // If we terminated early, then one of the values didn't match.
        if (PI == PE) {
          unsigned ToRemove = Baseline == LazyValueInfo::True ? 1 : 0;
          unsigned ToKeep = Baseline == LazyValueInfo::True ? 0 : 1;
          RemovePredecessorAndSimplify(CondBr->getSuccessor(ToRemove), BB, TD);
          BranchInst::Create(CondBr->getSuccessor(ToKeep), CondBr);
          CondBr->eraseFromParent();
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
  Value *SimplifyValue = CondInst;
  if (CmpInst *CondCmp = dyn_cast<CmpInst>(SimplifyValue))
    if (isa<Constant>(CondCmp->getOperand(1)))
      SimplifyValue = CondCmp->getOperand(0);
  
  // TODO: There are other places where load PRE would be profitable, such as
  // more complex comparisons.
  if (LoadInst *LI = dyn_cast<LoadInst>(SimplifyValue))
    if (SimplifyPartiallyRedundantLoad(LI))
      return true;
  
  
  // Handle a variety of cases where we are branching on something derived from
  // a PHI node in the current block.  If we can prove that any predecessors
  // compute a predictable value based on a PHI node, thread those predecessors.
  //
  if (ProcessThreadableEdges(CondInst, BB))
    return true;
  
  // If this is an otherwise-unfoldable branch on a phi node in the current
  // block, see if we can simplify.
  if (PHINode *PN = dyn_cast<PHINode>(CondInst))
    if (PN->getParent() == BB && isa<BranchInst>(BB->getTerminator()))
      return ProcessBranchOnPHI(PN);
  
  
  // If this is an otherwise-unfoldable branch on a XOR, see if we can simplify.
  if (CondInst->getOpcode() == Instruction::Xor &&
      CondInst->getParent() == BB && isa<BranchInst>(BB->getTerminator()))
    return ProcessBranchOnXOR(cast<BinaryOperator>(CondInst));
  
  
  // TODO: If we have: "br (X > 0)"  and we have a predecessor where we know
  // "(X == 4)", thread through this block.
  
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
    DEBUG(dbgs() << "  In block '" << PredBB->getName()
          << "' folding terminator: " << *PredBB->getTerminator() << '\n');
    ++NumFolds;
    ConstantFoldTerminator(PredBB);
    return true;
  }
   
  BranchInst *DestBI = cast<BranchInst>(BB->getTerminator());

  // If the dest block has one predecessor, just fix the branch condition to a
  // constant and fold it.
  if (BB->getSinglePredecessor()) {
    DEBUG(dbgs() << "  In block '" << BB->getName()
          << "' folding condition to '" << BranchDir << "': "
          << *BB->getTerminator() << '\n');
    ++NumFolds;
    Value *OldCond = DestBI->getCondition();
    DestBI->setCondition(ConstantInt::get(Type::getInt1Ty(BB->getContext()),
                                          BranchDir));
    // Delete dead instructions before we fold the branch.  Folding the branch
    // can eliminate edges from the CFG which can end up deleting OldCond.
    RecursivelyDeleteTriviallyDeadInstructions(OldCond);
    ConstantFoldTerminator(BB);
    return true;
  }
 
  
  // Next, figure out which successor we are threading to.
  BasicBlock *SuccBB = DestBI->getSuccessor(!BranchDir);
  
  SmallVector<BasicBlock*, 2> Preds;
  Preds.push_back(PredBB);
  
  // Ok, try to thread it!
  return ThreadEdge(BB, Preds, SuccBB);
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
      
      // Do not forward this if it already goes to this destination, this would
      // be an infinite loop.
      if (PredSI->getSuccessor(PredCase) == DestSucc)
        continue;

      // Otherwise, we're safe to make the change.  Make sure that the edge from
      // DestSI to DestSucc is not critical and has no PHI nodes.
      DEBUG(dbgs() << "FORWARDING EDGE " << *DestVal << "   FROM: " << *PredSI);
      DEBUG(dbgs() << "THROUGH: " << *DestSI);

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
  // TODO: Could do simple PHI translation, that would be fun :)
  if (Instruction *PtrOp = dyn_cast<Instruction>(LoadedPtr))
    if (PtrOp->getParent() == LoadBB)
      return false;
  
  // Scan a few instructions up from the load, to see if it is obviously live at
  // the entry to its block.
  BasicBlock::iterator BBIt = LI;

  if (Value *AvailableVal = 
        FindAvailableLoadedValue(LoadedPtr, LoadBB, BBIt, 6)) {
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
         PI != PE; ++PI) {
      BasicBlock *P = *PI;
      // If the predecessor is an indirect goto, we can't split the edge.
      if (isa<IndirectBrInst>(P->getTerminator()))
        return false;
      
      if (!AvailablePredSet.count(P))
        PredsToSplit.push_back(P);
    }
    
    // Split them out to their own block.
    UnavailablePred =
      SplitBlockPredecessors(LoadBB, &PredsToSplit[0], PredsToSplit.size(),
                             "thread-pre-split", this);
  }
  
  // If the value isn't available in all predecessors, then there will be
  // exactly one where it isn't available.  Insert a load on that edge and add
  // it to the AvailablePreds list.
  if (UnavailablePred) {
    assert(UnavailablePred->getTerminator()->getNumSuccessors() == 1 &&
           "Can't handle critical edge here!");
    Value *NewVal = new LoadInst(LoadedPtr, LI->getName()+".pr", false,
                                 LI->getAlignment(),
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
    BasicBlock *P = *PI;
    AvailablePredsTy::iterator I = 
      std::lower_bound(AvailablePreds.begin(), AvailablePreds.end(),
                       std::make_pair(P, (Value*)0));
    
    assert(I != AvailablePreds.end() && I->first == P &&
           "Didn't find entry for predecessor!");
    
    PN->addIncoming(I->second, I->first);
  }
  
  //cerr << "PRE: " << *LI << *PN << "\n";
  
  LI->replaceAllUsesWith(PN);
  LI->eraseFromParent();
  
  return true;
}

/// FindMostPopularDest - The specified list contains multiple possible
/// threadable destinations.  Pick the one that occurs the most frequently in
/// the list.
static BasicBlock *
FindMostPopularDest(BasicBlock *BB,
                    const SmallVectorImpl<std::pair<BasicBlock*,
                                  BasicBlock*> > &PredToDestList) {
  assert(!PredToDestList.empty());
  
  // Determine popularity.  If there are multiple possible destinations, we
  // explicitly choose to ignore 'undef' destinations.  We prefer to thread
  // blocks with known and real destinations to threading undef.  We'll handle
  // them later if interesting.
  DenseMap<BasicBlock*, unsigned> DestPopularity;
  for (unsigned i = 0, e = PredToDestList.size(); i != e; ++i)
    if (PredToDestList[i].second)
      DestPopularity[PredToDestList[i].second]++;
  
  // Find the most popular dest.
  DenseMap<BasicBlock*, unsigned>::iterator DPI = DestPopularity.begin();
  BasicBlock *MostPopularDest = DPI->first;
  unsigned Popularity = DPI->second;
  SmallVector<BasicBlock*, 4> SamePopularity;
  
  for (++DPI; DPI != DestPopularity.end(); ++DPI) {
    // If the popularity of this entry isn't higher than the popularity we've
    // seen so far, ignore it.
    if (DPI->second < Popularity)
      ; // ignore.
    else if (DPI->second == Popularity) {
      // If it is the same as what we've seen so far, keep track of it.
      SamePopularity.push_back(DPI->first);
    } else {
      // If it is more popular, remember it.
      SamePopularity.clear();
      MostPopularDest = DPI->first;
      Popularity = DPI->second;
    }      
  }
  
  // Okay, now we know the most popular destination.  If there is more than
  // destination, we need to determine one.  This is arbitrary, but we need
  // to make a deterministic decision.  Pick the first one that appears in the
  // successor list.
  if (!SamePopularity.empty()) {
    SamePopularity.push_back(MostPopularDest);
    TerminatorInst *TI = BB->getTerminator();
    for (unsigned i = 0; ; ++i) {
      assert(i != TI->getNumSuccessors() && "Didn't find any successor!");
      
      if (std::find(SamePopularity.begin(), SamePopularity.end(),
                    TI->getSuccessor(i)) == SamePopularity.end())
        continue;
      
      MostPopularDest = TI->getSuccessor(i);
      break;
    }
  }
  
  // Okay, we have finally picked the most popular destination.
  return MostPopularDest;
}

bool JumpThreading::ProcessThreadableEdges(Value *Cond, BasicBlock *BB) {
  // If threading this would thread across a loop header, don't even try to
  // thread the edge.
  if (LoopHeaders.count(BB))
    return false;
  
  SmallVector<std::pair<ConstantInt*, BasicBlock*>, 8> PredValues;
  if (!ComputeValueKnownInPredecessors(Cond, BB, PredValues))
    return false;
  
  assert(!PredValues.empty() &&
         "ComputeValueKnownInPredecessors returned true with no values");

  DEBUG(dbgs() << "IN BB: " << *BB;
        for (unsigned i = 0, e = PredValues.size(); i != e; ++i) {
          dbgs() << "  BB '" << BB->getName() << "': FOUND condition = ";
          if (PredValues[i].first)
            dbgs() << *PredValues[i].first;
          else
            dbgs() << "UNDEF";
          dbgs() << " for pred '" << PredValues[i].second->getName()
          << "'.\n";
        });
  
  // Decide what we want to thread through.  Convert our list of known values to
  // a list of known destinations for each pred.  This also discards duplicate
  // predecessors and keeps track of the undefined inputs (which are represented
  // as a null dest in the PredToDestList).
  SmallPtrSet<BasicBlock*, 16> SeenPreds;
  SmallVector<std::pair<BasicBlock*, BasicBlock*>, 16> PredToDestList;
  
  BasicBlock *OnlyDest = 0;
  BasicBlock *MultipleDestSentinel = (BasicBlock*)(intptr_t)~0ULL;
  
  for (unsigned i = 0, e = PredValues.size(); i != e; ++i) {
    BasicBlock *Pred = PredValues[i].second;
    if (!SeenPreds.insert(Pred))
      continue;  // Duplicate predecessor entry.
    
    // If the predecessor ends with an indirect goto, we can't change its
    // destination.
    if (isa<IndirectBrInst>(Pred->getTerminator()))
      continue;
    
    ConstantInt *Val = PredValues[i].first;
    
    BasicBlock *DestBB;
    if (Val == 0)      // Undef.
      DestBB = 0;
    else if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
      DestBB = BI->getSuccessor(Val->isZero());
    else {
      SwitchInst *SI = cast<SwitchInst>(BB->getTerminator());
      DestBB = SI->getSuccessor(SI->findCaseValue(Val));
    }

    // If we have exactly one destination, remember it for efficiency below.
    if (i == 0)
      OnlyDest = DestBB;
    else if (OnlyDest != DestBB)
      OnlyDest = MultipleDestSentinel;
    
    PredToDestList.push_back(std::make_pair(Pred, DestBB));
  }
  
  // If all edges were unthreadable, we fail.
  if (PredToDestList.empty())
    return false;
  
  // Determine which is the most common successor.  If we have many inputs and
  // this block is a switch, we want to start by threading the batch that goes
  // to the most popular destination first.  If we only know about one
  // threadable destination (the common case) we can avoid this.
  BasicBlock *MostPopularDest = OnlyDest;
  
  if (MostPopularDest == MultipleDestSentinel)
    MostPopularDest = FindMostPopularDest(BB, PredToDestList);
  
  // Now that we know what the most popular destination is, factor all
  // predecessors that will jump to it into a single predecessor.
  SmallVector<BasicBlock*, 16> PredsToFactor;
  for (unsigned i = 0, e = PredToDestList.size(); i != e; ++i)
    if (PredToDestList[i].second == MostPopularDest) {
      BasicBlock *Pred = PredToDestList[i].first;
      
      // This predecessor may be a switch or something else that has multiple
      // edges to the block.  Factor each of these edges by listing them
      // according to # occurrences in PredsToFactor.
      TerminatorInst *PredTI = Pred->getTerminator();
      for (unsigned i = 0, e = PredTI->getNumSuccessors(); i != e; ++i)
        if (PredTI->getSuccessor(i) == BB)
          PredsToFactor.push_back(Pred);
    }

  // If the threadable edges are branching on an undefined value, we get to pick
  // the destination that these predecessors should get to.
  if (MostPopularDest == 0)
    MostPopularDest = BB->getTerminator()->
                            getSuccessor(GetBestDestForJumpOnUndef(BB));
        
  // Ok, try to thread it!
  return ThreadEdge(BB, PredsToFactor, MostPopularDest);
}

/// ProcessBranchOnPHI - We have an otherwise unthreadable conditional branch on
/// a PHI node in the current block.  See if there are any simplifications we
/// can do based on inputs to the phi node.
/// 
bool JumpThreading::ProcessBranchOnPHI(PHINode *PN) {
  BasicBlock *BB = PN->getParent();
  
  // TODO: We could make use of this to do it once for blocks with common PHI
  // values.
  SmallVector<BasicBlock*, 1> PredBBs;
  PredBBs.resize(1);
  
  // If any of the predecessor blocks end in an unconditional branch, we can
  // *duplicate* the conditional branch into that block in order to further
  // encourage jump threading and to eliminate cases where we have branch on a
  // phi of an icmp (branch on icmp is much better).
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *PredBB = PN->getIncomingBlock(i);
    if (BranchInst *PredBr = dyn_cast<BranchInst>(PredBB->getTerminator()))
      if (PredBr->isUnconditional()) {
        PredBBs[0] = PredBB;
        // Try to duplicate BB into PredBB.
        if (DuplicateCondBranchOnPHIIntoPred(BB, PredBBs))
          return true;
      }
  }

  return false;
}

/// ProcessBranchOnXOR - We have an otherwise unthreadable conditional branch on
/// a xor instruction in the current block.  See if there are any
/// simplifications we can do based on inputs to the xor.
/// 
bool JumpThreading::ProcessBranchOnXOR(BinaryOperator *BO) {
  BasicBlock *BB = BO->getParent();
  
  // If either the LHS or RHS of the xor is a constant, don't do this
  // optimization.
  if (isa<ConstantInt>(BO->getOperand(0)) ||
      isa<ConstantInt>(BO->getOperand(1)))
    return false;
  
  // If the first instruction in BB isn't a phi, we won't be able to infer
  // anything special about any particular predecessor.
  if (!isa<PHINode>(BB->front()))
    return false;
  
  // If we have a xor as the branch input to this block, and we know that the
  // LHS or RHS of the xor in any predecessor is true/false, then we can clone
  // the condition into the predecessor and fix that value to true, saving some
  // logical ops on that path and encouraging other paths to simplify.
  //
  // This copies something like this:
  //
  //  BB:
  //    %X = phi i1 [1],  [%X']
  //    %Y = icmp eq i32 %A, %B
  //    %Z = xor i1 %X, %Y
  //    br i1 %Z, ...
  //
  // Into:
  //  BB':
  //    %Y = icmp ne i32 %A, %B
  //    br i1 %Z, ...

  SmallVector<std::pair<ConstantInt*, BasicBlock*>, 8> XorOpValues;
  bool isLHS = true;
  if (!ComputeValueKnownInPredecessors(BO->getOperand(0), BB, XorOpValues)) {
    assert(XorOpValues.empty());
    if (!ComputeValueKnownInPredecessors(BO->getOperand(1), BB, XorOpValues))
      return false;
    isLHS = false;
  }
  
  assert(!XorOpValues.empty() &&
         "ComputeValueKnownInPredecessors returned true with no values");

  // Scan the information to see which is most popular: true or false.  The
  // predecessors can be of the set true, false, or undef.
  unsigned NumTrue = 0, NumFalse = 0;
  for (unsigned i = 0, e = XorOpValues.size(); i != e; ++i) {
    if (!XorOpValues[i].first) continue;  // Ignore undefs for the count.
    if (XorOpValues[i].first->isZero())
      ++NumFalse;
    else
      ++NumTrue;
  }
  
  // Determine which value to split on, true, false, or undef if neither.
  ConstantInt *SplitVal = 0;
  if (NumTrue > NumFalse)
    SplitVal = ConstantInt::getTrue(BB->getContext());
  else if (NumTrue != 0 || NumFalse != 0)
    SplitVal = ConstantInt::getFalse(BB->getContext());
  
  // Collect all of the blocks that this can be folded into so that we can
  // factor this once and clone it once.
  SmallVector<BasicBlock*, 8> BlocksToFoldInto;
  for (unsigned i = 0, e = XorOpValues.size(); i != e; ++i) {
    if (XorOpValues[i].first != SplitVal && XorOpValues[i].first != 0) continue;

    BlocksToFoldInto.push_back(XorOpValues[i].second);
  }
  
  // If we inferred a value for all of the predecessors, then duplication won't
  // help us.  However, we can just replace the LHS or RHS with the constant.
  if (BlocksToFoldInto.size() ==
      cast<PHINode>(BB->front()).getNumIncomingValues()) {
    if (SplitVal == 0) {
      // If all preds provide undef, just nuke the xor, because it is undef too.
      BO->replaceAllUsesWith(UndefValue::get(BO->getType()));
      BO->eraseFromParent();
    } else if (SplitVal->isZero()) {
      // If all preds provide 0, replace the xor with the other input.
      BO->replaceAllUsesWith(BO->getOperand(isLHS));
      BO->eraseFromParent();
    } else {
      // If all preds provide 1, set the computed value to 1.
      BO->setOperand(!isLHS, SplitVal);
    }
    
    return true;
  }
  
  // Try to duplicate BB into PredBB.
  return DuplicateCondBranchOnPHIIntoPred(BB, BlocksToFoldInto);
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

/// ThreadEdge - We have decided that it is safe and profitable to factor the
/// blocks in PredBBs to one predecessor, then thread an edge from it to SuccBB
/// across BB.  Transform the IR to reflect this change.
bool JumpThreading::ThreadEdge(BasicBlock *BB, 
                               const SmallVectorImpl<BasicBlock*> &PredBBs, 
                               BasicBlock *SuccBB) {
  // If threading to the same block as we come from, we would infinite loop.
  if (SuccBB == BB) {
    DEBUG(dbgs() << "  Not threading across BB '" << BB->getName()
          << "' - would thread to self!\n");
    return false;
  }
  
  // If threading this would thread across a loop header, don't thread the edge.
  // See the comments above FindLoopHeaders for justifications and caveats.
  if (LoopHeaders.count(BB)) {
    DEBUG(dbgs() << "  Not threading across loop header BB '" << BB->getName()
          << "' to dest BB '" << SuccBB->getName()
          << "' - it might create an irreducible loop!\n");
    return false;
  }

  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB);
  if (JumpThreadCost > Threshold) {
    DEBUG(dbgs() << "  Not threading BB '" << BB->getName()
          << "' - Cost is too high: " << JumpThreadCost << "\n");
    return false;
  }
  
  // And finally, do it!  Start by factoring the predecessors is needed.
  BasicBlock *PredBB;
  if (PredBBs.size() == 1)
    PredBB = PredBBs[0];
  else {
    DEBUG(dbgs() << "  Factoring out " << PredBBs.size()
          << " common predecessors.\n");
    PredBB = SplitBlockPredecessors(BB, &PredBBs[0], PredBBs.size(),
                                    ".thr_comm", this);
  }
  
  // And finally, do it!
  DEBUG(dbgs() << "  Threading edge from '" << PredBB->getName() << "' to '"
        << SuccBB->getName() << "' with cost: " << JumpThreadCost
        << ", across block:\n    "
        << *BB << "\n");
  
  if (LVI)
    LVI->threadEdge(PredBB, BB, SuccBB);
  
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
    
    DEBUG(dbgs() << "JT: Renaming non-local uses of: " << *I << "\n");

    // We found a use of I outside of BB.  Rename all uses of I that are outside
    // its block to be uses of the appropriate PHI node etc.  See ValuesInBlocks
    // with the two values we know.
    SSAUpdate.Initialize(I);
    SSAUpdate.AddAvailableValue(BB, I);
    SSAUpdate.AddAvailableValue(NewBB, ValueMapping[I]);
    
    while (!UsesToRename.empty())
      SSAUpdate.RewriteUse(*UsesToRename.pop_back_val());
    DEBUG(dbgs() << "\n");
  }
  
  
  // Ok, NewBB is good to go.  Update the terminator of PredBB to jump to
  // NewBB instead of BB.  This eliminates predecessors from BB, which requires
  // us to simplify any PHI nodes in BB.
  TerminatorInst *PredTerm = PredBB->getTerminator();
  for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i)
    if (PredTerm->getSuccessor(i) == BB) {
      RemovePredecessorAndSimplify(BB, PredBB, TD);
      PredTerm->setSuccessor(i, NewBB);
    }
  
  // At this point, the IR is fully up to date and consistent.  Do a quick scan
  // over the new instructions and zap any that are constants or dead.  This
  // frequently happens because of phi translation.
  SimplifyInstructionsInBlock(NewBB, TD);
  
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
                                 const SmallVectorImpl<BasicBlock *> &PredBBs) {
  assert(!PredBBs.empty() && "Can't handle an empty set");

  // If BB is a loop header, then duplicating this block outside the loop would
  // cause us to transform this into an irreducible loop, don't do this.
  // See the comments above FindLoopHeaders for justifications and caveats.
  if (LoopHeaders.count(BB)) {
    DEBUG(dbgs() << "  Not duplicating loop header '" << BB->getName()
          << "' into predecessor block '" << PredBBs[0]->getName()
          << "' - it might create an irreducible loop!\n");
    return false;
  }
  
  unsigned DuplicationCost = getJumpThreadDuplicationCost(BB);
  if (DuplicationCost > Threshold) {
    DEBUG(dbgs() << "  Not duplicating BB '" << BB->getName()
          << "' - Cost is too high: " << DuplicationCost << "\n");
    return false;
  }
  
  // And finally, do it!  Start by factoring the predecessors is needed.
  BasicBlock *PredBB;
  if (PredBBs.size() == 1)
    PredBB = PredBBs[0];
  else {
    DEBUG(dbgs() << "  Factoring out " << PredBBs.size()
          << " common predecessors.\n");
    PredBB = SplitBlockPredecessors(BB, &PredBBs[0], PredBBs.size(),
                                    ".thr_comm", this);
  }
  
  // Okay, we decided to do this!  Clone all the instructions in BB onto the end
  // of PredBB.
  DEBUG(dbgs() << "  Duplicating block '" << BB->getName() << "' into end of '"
        << PredBB->getName() << "' to eliminate branch on phi.  Cost: "
        << DuplicationCost << " block is:" << *BB << "\n");
  
  // Unless PredBB ends with an unconditional branch, split the edge so that we
  // can just clone the bits from BB into the end of the new PredBB.
  BranchInst *OldPredBranch = dyn_cast<BranchInst>(PredBB->getTerminator());
  
  if (OldPredBranch == 0 || !OldPredBranch->isUnconditional()) {
    PredBB = SplitEdge(PredBB, BB, this);
    OldPredBranch = cast<BranchInst>(PredBB->getTerminator());
  }
  
  // We are going to have to map operands from the original BB block into the
  // PredBB block.  Evaluate PHI nodes in BB.
  DenseMap<Instruction*, Value*> ValueMapping;
  
  BasicBlock::iterator BI = BB->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(PredBB);
  
  // Clone the non-phi instructions of BB into PredBB, keeping track of the
  // mapping and using it to remap operands in the cloned instructions.
  for (; BI != BB->end(); ++BI) {
    Instruction *New = BI->clone();
    
    // Remap operands to patch up intra-block references.
    for (unsigned i = 0, e = New->getNumOperands(); i != e; ++i)
      if (Instruction *Inst = dyn_cast<Instruction>(New->getOperand(i))) {
        DenseMap<Instruction*, Value*>::iterator I = ValueMapping.find(Inst);
        if (I != ValueMapping.end())
          New->setOperand(i, I->second);
      }

    // If this instruction can be simplified after the operands are updated,
    // just use the simplified value instead.  This frequently happens due to
    // phi translation.
    if (Value *IV = SimplifyInstruction(New, TD)) {
      delete New;
      ValueMapping[BI] = IV;
    } else {
      // Otherwise, insert the new instruction into the block.
      New->setName(BI->getName());
      PredBB->getInstList().insert(OldPredBranch, New);
      ValueMapping[BI] = New;
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
    
    DEBUG(dbgs() << "JT: Renaming non-local uses of: " << *I << "\n");
    
    // We found a use of I outside of BB.  Rename all uses of I that are outside
    // its block to be uses of the appropriate PHI node etc.  See ValuesInBlocks
    // with the two values we know.
    SSAUpdate.Initialize(I);
    SSAUpdate.AddAvailableValue(BB, I);
    SSAUpdate.AddAvailableValue(PredBB, ValueMapping[I]);
    
    while (!UsesToRename.empty())
      SSAUpdate.RewriteUse(*UsesToRename.pop_back_val());
    DEBUG(dbgs() << "\n");
  }
  
  // PredBB no longer jumps to BB, remove entries in the PHI node for the edge
  // that we nuked.
  RemovePredecessorAndSimplify(BB, PredBB, TD);
  
  // Remove the unconditional branch at the end of the PredBB block.
  OldPredBranch->eraseFromParent();
  
  ++NumDupes;
  return true;
}


