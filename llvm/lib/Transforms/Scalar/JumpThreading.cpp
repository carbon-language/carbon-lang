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
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

STATISTIC(NumThreads, "Number of jumps threaded");
STATISTIC(NumFolds,   "Number of terminators folded");

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
  class VISIBILITY_HIDDEN JumpThreading : public FunctionPass {
  public:
    static char ID; // Pass identification
    JumpThreading() : FunctionPass((intptr_t)&ID) {}

    bool runOnFunction(Function &F);
    bool ThreadBlock(BasicBlock *BB);
    void ThreadEdge(BasicBlock *BB, BasicBlock *PredBB, BasicBlock *SuccBB);
    BasicBlock *FactorCommonPHIPreds(PHINode *PN, Constant *CstVal);

    bool ProcessJumpOnPHI(PHINode *PN);
    bool ProcessBranchOnLogical(Value *V, BasicBlock *BB, bool isAnd);
    bool ProcessBranchOnCompare(CmpInst *Cmp, BasicBlock *BB);
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
  DOUT << "Jump threading on function '" << F.getNameStart() << "'\n";
  
  bool AnotherIteration = true, EverChanged = false;
  while (AnotherIteration) {
    AnotherIteration = false;
    bool Changed = false;
    for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
      while (ThreadBlock(I))
        Changed = true;
    AnotherIteration = Changed;
    EverChanged |= Changed;
  }
  return EverChanged;
}

/// FactorCommonPHIPreds - If there are multiple preds with the same incoming
/// value for the PHI, factor them together so we get one block to thread for
/// the whole group.
/// This is important for things like "phi i1 [true, true, false, true, x]"
/// where we only need to clone the block for the true blocks once.
///
BasicBlock *JumpThreading::FactorCommonPHIPreds(PHINode *PN, Constant *CstVal) {
  SmallVector<BasicBlock*, 16> CommonPreds;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingValue(i) == CstVal)
      CommonPreds.push_back(PN->getIncomingBlock(i));
  
  if (CommonPreds.size() == 1)
    return CommonPreds[0];
    
  DOUT << "  Factoring out " << CommonPreds.size()
       << " common predecessors.\n";
  return SplitBlockPredecessors(PN->getParent(),
                                &CommonPreds[0], CommonPreds.size(),
                                ".thr_comm", this);
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
      else if (isa<VectorType>(CI->getType()))
        Size += 1;
    }
  }
  
  // Threading through a switch statement is particularly profitable.  If this
  // block ends in a switch, decrease its cost to make it more likely to happen.
  if (isa<SwitchInst>(I))
    Size = Size > 6 ? Size-6 : 0;
  
  return Size;
}


/// ThreadBlock - If there are any predecessors whose control can be threaded
/// through to a successor, transform them now.
bool JumpThreading::ThreadBlock(BasicBlock *BB) {
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
    DOUT << "  In block '" << BB->getNameStart()
         << "' folding terminator: " << *BB->getTerminator();
    ++NumFolds;
    ConstantFoldTerminator(BB);
    return true;
  }
  
  // If there is only a single predecessor of this block, nothing to fold.
  if (BB->getSinglePredecessor())
    return false;

  // See if this is a phi node in the current block.
  PHINode *PN = dyn_cast<PHINode>(Condition);
  if (PN && PN->getParent() == BB)
    return ProcessJumpOnPHI(PN);
  
  // If this is a conditional branch whose condition is and/or of a phi, try to
  // simplify it.
  if (BinaryOperator *CondI = dyn_cast<BinaryOperator>(Condition)) {
    if ((CondI->getOpcode() == Instruction::And || 
         CondI->getOpcode() == Instruction::Or) &&
        isa<BranchInst>(BB->getTerminator()) &&
        ProcessBranchOnLogical(CondI, BB,
                               CondI->getOpcode() == Instruction::And))
      return true;
  }
  
  // If we have "br (phi != 42)" and the phi node has any constant values as 
  // operands, we can thread through this block.
  if (CmpInst *CondCmp = dyn_cast<CmpInst>(Condition))
    if (isa<PHINode>(CondCmp->getOperand(0)) &&
        isa<Constant>(CondCmp->getOperand(1)) &&
        ProcessBranchOnCompare(CondCmp, BB))
      return true;
  
  return false;
}

/// ProcessJumpOnPHI - We have a conditional branch of switch on a PHI node in
/// the current block.  See if there are any simplifications we can do based on
/// inputs to the phi node.
/// 
bool JumpThreading::ProcessJumpOnPHI(PHINode *PN) {
  // See if the phi node has any constant values.  If so, we can determine where
  // the corresponding predecessor will branch.
  ConstantInt *PredCst = 0;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if ((PredCst = dyn_cast<ConstantInt>(PN->getIncomingValue(i))))
      break;
  
  // If no incoming value has a constant, we don't know the destination of any
  // predecessors.
  if (PredCst == 0)
    return false;
  
  // See if the cost of duplicating this block is low enough.
  BasicBlock *BB = PN->getParent();
  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB);
  if (JumpThreadCost > Threshold) {
    DOUT << "  Not threading BB '" << BB->getNameStart()
         << "' - Cost is too high: " << JumpThreadCost << "\n";
    return false;
  }
  
  // If so, we can actually do this threading.  Merge any common predecessors
  // that will act the same.
  BasicBlock *PredBB = FactorCommonPHIPreds(PN, PredCst);
  
  // Next, figure out which successor we are threading to.
  BasicBlock *SuccBB;
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
    SuccBB = BI->getSuccessor(PredCst == ConstantInt::getFalse());
  else {
    SwitchInst *SI = cast<SwitchInst>(BB->getTerminator());
    SuccBB = SI->getSuccessor(SI->findCaseValue(PredCst));
  }
  
  // If threading to the same block as we come from, we would infinite loop.
  if (SuccBB == BB) {
    DOUT << "  Not threading BB '" << BB->getNameStart()
         << "' - would thread to self!\n";
    return false;
  }
  
  // And finally, do it!
  DOUT << "  Threading edge from '" << PredBB->getNameStart() << "' to '"
       << SuccBB->getNameStart() << "' with cost: " << JumpThreadCost
       << ", across block:\n    "
       << *BB << "\n";
       
  ThreadEdge(BB, PredBB, SuccBB);
  ++NumThreads;
  return true;
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
    if (isAnd && BO->getOpcode() == Instruction::And ||
        !isAnd && BO->getOpcode() == Instruction::Or) {
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
  ConstantInt *PredCst = ConstantInt::get(Type::Int1Ty, !isAnd);
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (PN->getIncomingValue(i) == PredCst) {
      PredNo = i;
      break;
    }
  }
  
  // If no match, bail out.
  if (PredNo == ~0U)
    return false;
  
  // See if the cost of duplicating this block is low enough.
  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB);
  if (JumpThreadCost > Threshold) {
    DOUT << "  Not threading BB '" << BB->getNameStart()
         << "' - Cost is too high: " << JumpThreadCost << "\n";
    return false;
  }

  // If so, we can actually do this threading.  Merge any common predecessors
  // that will act the same.
  BasicBlock *PredBB = FactorCommonPHIPreds(PN, PredCst);
  
  // Next, figure out which successor we are threading to.  If this was an AND,
  // the constant must be FALSE, and we must be targeting the 'false' block.
  // If this is an OR, the constant must be TRUE, and we must be targeting the
  // 'true' block.
  BasicBlock *SuccBB = BB->getTerminator()->getSuccessor(isAnd);
  
  // If threading to the same block as we come from, we would infinite loop.
  if (SuccBB == BB) {
    DOUT << "  Not threading BB '" << BB->getNameStart()
    << "' - would thread to self!\n";
    return false;
  }
  
  // And finally, do it!
  DOUT << "  Threading edge through bool from '" << PredBB->getNameStart()
       << "' to '" << SuccBB->getNameStart() << "' with cost: "
       << JumpThreadCost << ", across block:\n    "
       << *BB << "\n";
  
  ThreadEdge(BB, PredBB, SuccBB);
  ++NumThreads;
  return true;
}

/// ProcessBranchOnCompare - We found a branch on a comparison between a phi
/// node and a constant.  If the PHI node contains any constants as inputs, we
/// can fold the compare for that edge and thread through it.
bool JumpThreading::ProcessBranchOnCompare(CmpInst *Cmp, BasicBlock *BB) {
  PHINode *PN = cast<PHINode>(Cmp->getOperand(0));
  Constant *RHS = cast<Constant>(Cmp->getOperand(1));
  
  // If the phi isn't in the current block, an incoming edge to this block
  // doesn't control the destination.
  if (PN->getParent() != BB)
    return false;
  
  // We can do this simplification if any comparisons fold to true or false.
  // See if any do.
  Constant *PredCst = 0;
  bool TrueDirection = false;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    PredCst = dyn_cast<Constant>(PN->getIncomingValue(i));
    if (PredCst == 0) continue;
    
    Constant *Res;
    if (ICmpInst *ICI = dyn_cast<ICmpInst>(Cmp))
      Res = ConstantExpr::getICmp(ICI->getPredicate(), PredCst, RHS);
    else
      Res = ConstantExpr::getFCmp(cast<FCmpInst>(Cmp)->getPredicate(),
                                  PredCst, RHS);
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
    PredCst = 0;
  }
  
  // If no match, bail out.
  if (PredCst == 0)
    return false;
  
  // See if the cost of duplicating this block is low enough.
  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB);
  if (JumpThreadCost > Threshold) {
    DOUT << "  Not threading BB '" << BB->getNameStart()
         << "' - Cost is too high: " << JumpThreadCost << "\n";
    return false;
  }
  
  // If so, we can actually do this threading.  Merge any common predecessors
  // that will act the same.
  BasicBlock *PredBB = FactorCommonPHIPreds(PN, PredCst);
  
  // Next, get our successor.
  BasicBlock *SuccBB = BB->getTerminator()->getSuccessor(!TrueDirection);
  
  // If threading to the same block as we come from, we would infinite loop.
  if (SuccBB == BB) {
    DOUT << "  Not threading BB '" << BB->getNameStart()
    << "' - would thread to self!\n";
    return false;
  }
  
  
  // And finally, do it!
  DOUT << "  Threading edge through bool from '" << PredBB->getNameStart()
       << "' to '" << SuccBB->getNameStart() << "' with cost: "
       << JumpThreadCost << ", across block:\n    "
       << *BB << "\n";
  
  ThreadEdge(BB, PredBB, SuccBB);
  ++NumThreads;
  return true;
}


/// ThreadEdge - We have decided that it is safe and profitable to thread an
/// edge from PredBB to SuccBB across BB.  Transform the IR to reflect this
/// change.
void JumpThreading::ThreadEdge(BasicBlock *BB, BasicBlock *PredBB, 
                               BasicBlock *SuccBB) {

  // Jump Threading can not update SSA properties correctly if the values
  // defined in the duplicated block are used outside of the block itself.  For
  // this reason, we spill all values that are used outside of BB to the stack.
  for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    if (!I->isUsedOutsideOfBlock(BB))
      continue;
    
    // We found a use of I outside of BB.  Create a new stack slot to
    // break this inter-block usage pattern.
    if (!isa<StructType>(I->getType())) {
      DemoteRegToStack(*I);
      continue;
    }
    
    // Alternatively, I must be a call or invoke that returns multiple retvals.
    // We can't use 'DemoteRegToStack' because that will create loads and
    // stores of aggregates which is not valid yet.  If I is a call, we can just
    // pull all the getresult instructions up to this block.  If I is an invoke,
    // we are out of luck.
    BasicBlock::iterator IP = I; ++IP;
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
         UI != E; ++UI)
      cast<GetResultInst>(UI)->moveBefore(IP);
  }
 
  // We are going to have to map operands from the original BB block to the new
  // copy of the block 'NewBB'.  If there are PHI nodes in BB, evaluate them to
  // account for entry from PredBB.
  DenseMap<Instruction*, Value*> ValueMapping;
  
  BasicBlock *NewBB =
    BasicBlock::Create(BB->getName()+".thread", BB->getParent(), BB);
  NewBB->moveAfter(PredBB);
  
  BasicBlock::iterator BI = BB->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(PredBB);
  
  // Clone the non-phi instructions of BB into NewBB, keeping track of the
  // mapping and using it to remap operands in the cloned instructions.
  for (; !isa<TerminatorInst>(BI); ++BI) {
    Instruction *New = BI->clone();
    New->setName(BI->getNameStart());
    NewBB->getInstList().push_back(New);
    ValueMapping[BI] = New;
   
    // Remap operands to patch up intra-block references.
    for (unsigned i = 0, e = New->getNumOperands(); i != e; ++i)
      if (Instruction *Inst = dyn_cast<Instruction>(New->getOperand(i)))
        if (Value *Remapped = ValueMapping[Inst])
          New->setOperand(i, Remapped);
  }
  
  // We didn't copy the terminator from BB over to NewBB, because there is now
  // an unconditional jump to SuccBB.  Insert the unconditional jump.
  BranchInst::Create(SuccBB, NewBB);
  
  // Check to see if SuccBB has PHI nodes. If so, we need to add entries to the
  // PHI nodes for NewBB now.
  for (BasicBlock::iterator PNI = SuccBB->begin(); isa<PHINode>(PNI); ++PNI) {
    PHINode *PN = cast<PHINode>(PNI);
    // Ok, we have a PHI node.  Figure out what the incoming value was for the
    // DestBlock.
    Value *IV = PN->getIncomingValueForBlock(BB);
    
    // Remap the value if necessary.
    if (Instruction *Inst = dyn_cast<Instruction>(IV))
      if (Value *MappedIV = ValueMapping[Inst])
        IV = MappedIV;
    PN->addIncoming(IV, NewBB);
  }
  
  // Finally, NewBB is good to go.  Update the terminator of PredBB to jump to
  // NewBB instead of BB.  This eliminates predecessors from BB, which requires
  // us to simplify any PHI nodes in BB.
  TerminatorInst *PredTerm = PredBB->getTerminator();
  for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i)
    if (PredTerm->getSuccessor(i) == BB) {
      BB->removePredecessor(PredBB);
      PredTerm->setSuccessor(i, NewBB);
    }
}
