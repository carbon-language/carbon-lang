//===- SimplifyCFG.cpp - Code to perform CFG simplification ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Peephole optimize the CFG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplifycfg"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
#include <map>
using namespace llvm;

static cl::opt<bool>
DupRet("simplifycfg-dup-ret", cl::Hidden, cl::init(false),
       cl::desc("Duplicate return instructions into unconditional branches"));

STATISTIC(NumSpeculations, "Number of speculative executed instructions");

namespace {
class SimplifyCFGOpt {
  const TargetData *const TD;

  Value *isValueEqualityComparison(TerminatorInst *TI);
  BasicBlock *GetValueEqualityComparisonCases(TerminatorInst *TI,
    std::vector<std::pair<ConstantInt*, BasicBlock*> > &Cases);
  bool SimplifyEqualityComparisonWithOnlyPredecessor(TerminatorInst *TI,
                                                     BasicBlock *Pred);
  bool FoldValueComparisonIntoPredecessors(TerminatorInst *TI);

  bool SimplifyReturn(ReturnInst *RI);
  bool SimplifyUnwind(UnwindInst *UI);
  bool SimplifyUnreachable(UnreachableInst *UI);
  bool SimplifySwitch(SwitchInst *SI);
  bool SimplifyIndirectBr(IndirectBrInst *IBI);
  bool SimplifyUncondBranch(BranchInst *BI);
  bool SimplifyCondBranch(BranchInst *BI);

public:
  explicit SimplifyCFGOpt(const TargetData *td) : TD(td) {}
  bool run(BasicBlock *BB);
};
}

/// SafeToMergeTerminators - Return true if it is safe to merge these two
/// terminator instructions together.
///
static bool SafeToMergeTerminators(TerminatorInst *SI1, TerminatorInst *SI2) {
  if (SI1 == SI2) return false;  // Can't merge with self!
  
  // It is not safe to merge these two switch instructions if they have a common
  // successor, and if that successor has a PHI node, and if *that* PHI node has
  // conflicting incoming values from the two switch blocks.
  BasicBlock *SI1BB = SI1->getParent();
  BasicBlock *SI2BB = SI2->getParent();
  SmallPtrSet<BasicBlock*, 16> SI1Succs(succ_begin(SI1BB), succ_end(SI1BB));
  
  for (succ_iterator I = succ_begin(SI2BB), E = succ_end(SI2BB); I != E; ++I)
    if (SI1Succs.count(*I))
      for (BasicBlock::iterator BBI = (*I)->begin();
           isa<PHINode>(BBI); ++BBI) {
        PHINode *PN = cast<PHINode>(BBI);
        if (PN->getIncomingValueForBlock(SI1BB) !=
            PN->getIncomingValueForBlock(SI2BB))
          return false;
      }
        
  return true;
}

/// AddPredecessorToBlock - Update PHI nodes in Succ to indicate that there will
/// now be entries in it from the 'NewPred' block.  The values that will be
/// flowing into the PHI nodes will be the same as those coming in from
/// ExistPred, an existing predecessor of Succ.
static void AddPredecessorToBlock(BasicBlock *Succ, BasicBlock *NewPred,
                                  BasicBlock *ExistPred) {
  if (!isa<PHINode>(Succ->begin())) return; // Quick exit if nothing to do
  
  PHINode *PN;
  for (BasicBlock::iterator I = Succ->begin();
       (PN = dyn_cast<PHINode>(I)); ++I)
    PN->addIncoming(PN->getIncomingValueForBlock(ExistPred), NewPred);
}


/// GetIfCondition - Given a basic block (BB) with two predecessors (and at
/// least one PHI node in it), check to see if the merge at this block is due
/// to an "if condition".  If so, return the boolean condition that determines
/// which entry into BB will be taken.  Also, return by references the block
/// that will be entered from if the condition is true, and the block that will
/// be entered if the condition is false.
///
/// This does no checking to see if the true/false blocks have large or unsavory
/// instructions in them.
static Value *GetIfCondition(BasicBlock *BB, BasicBlock *&IfTrue,
                             BasicBlock *&IfFalse) {
  PHINode *SomePHI = cast<PHINode>(BB->begin());
  assert(SomePHI->getNumIncomingValues() == 2 &&
         "Function can only handle blocks with 2 predecessors!");
  BasicBlock *Pred1 = SomePHI->getIncomingBlock(0);
  BasicBlock *Pred2 = SomePHI->getIncomingBlock(1);

  // We can only handle branches.  Other control flow will be lowered to
  // branches if possible anyway.
  BranchInst *Pred1Br = dyn_cast<BranchInst>(Pred1->getTerminator());
  BranchInst *Pred2Br = dyn_cast<BranchInst>(Pred2->getTerminator());
  if (Pred1Br == 0 || Pred2Br == 0)
    return 0;

  // Eliminate code duplication by ensuring that Pred1Br is conditional if
  // either are.
  if (Pred2Br->isConditional()) {
    // If both branches are conditional, we don't have an "if statement".  In
    // reality, we could transform this case, but since the condition will be
    // required anyway, we stand no chance of eliminating it, so the xform is
    // probably not profitable.
    if (Pred1Br->isConditional())
      return 0;

    std::swap(Pred1, Pred2);
    std::swap(Pred1Br, Pred2Br);
  }

  if (Pred1Br->isConditional()) {
    // The only thing we have to watch out for here is to make sure that Pred2
    // doesn't have incoming edges from other blocks.  If it does, the condition
    // doesn't dominate BB.
    if (Pred2->getSinglePredecessor() == 0)
      return 0;
    
    // If we found a conditional branch predecessor, make sure that it branches
    // to BB and Pred2Br.  If it doesn't, this isn't an "if statement".
    if (Pred1Br->getSuccessor(0) == BB &&
        Pred1Br->getSuccessor(1) == Pred2) {
      IfTrue = Pred1;
      IfFalse = Pred2;
    } else if (Pred1Br->getSuccessor(0) == Pred2 &&
               Pred1Br->getSuccessor(1) == BB) {
      IfTrue = Pred2;
      IfFalse = Pred1;
    } else {
      // We know that one arm of the conditional goes to BB, so the other must
      // go somewhere unrelated, and this must not be an "if statement".
      return 0;
    }

    return Pred1Br->getCondition();
  }

  // Ok, if we got here, both predecessors end with an unconditional branch to
  // BB.  Don't panic!  If both blocks only have a single (identical)
  // predecessor, and THAT is a conditional branch, then we're all ok!
  BasicBlock *CommonPred = Pred1->getSinglePredecessor();
  if (CommonPred == 0 || CommonPred != Pred2->getSinglePredecessor())
    return 0;

  // Otherwise, if this is a conditional branch, then we can use it!
  BranchInst *BI = dyn_cast<BranchInst>(CommonPred->getTerminator());
  if (BI == 0) return 0;
  
  assert(BI->isConditional() && "Two successors but not conditional?");
  if (BI->getSuccessor(0) == Pred1) {
    IfTrue = Pred1;
    IfFalse = Pred2;
  } else {
    IfTrue = Pred2;
    IfFalse = Pred1;
  }
  return BI->getCondition();
}

/// DominatesMergePoint - If we have a merge point of an "if condition" as
/// accepted above, return true if the specified value dominates the block.  We
/// don't handle the true generality of domination here, just a special case
/// which works well enough for us.
///
/// If AggressiveInsts is non-null, and if V does not dominate BB, we check to
/// see if V (which must be an instruction) is cheap to compute and is
/// non-trapping.  If both are true, the instruction is inserted into the set
/// and true is returned.
static bool DominatesMergePoint(Value *V, BasicBlock *BB,
                                SmallPtrSet<Instruction*, 4> *AggressiveInsts) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    // Non-instructions all dominate instructions, but not all constantexprs
    // can be executed unconditionally.
    if (ConstantExpr *C = dyn_cast<ConstantExpr>(V))
      if (C->canTrap())
        return false;
    return true;
  }
  BasicBlock *PBB = I->getParent();

  // We don't want to allow weird loops that might have the "if condition" in
  // the bottom of this block.
  if (PBB == BB) return false;

  // If this instruction is defined in a block that contains an unconditional
  // branch to BB, then it must be in the 'conditional' part of the "if
  // statement".  If not, it definitely dominates the region.
  BranchInst *BI = dyn_cast<BranchInst>(PBB->getTerminator());
  if (BI == 0 || BI->isConditional() || BI->getSuccessor(0) != BB)
    return true;

  // If we aren't allowing aggressive promotion anymore, then don't consider
  // instructions in the 'if region'.
  if (AggressiveInsts == 0) return false;
  
  // Okay, it looks like the instruction IS in the "condition".  Check to
  // see if it's a cheap instruction to unconditionally compute, and if it
  // only uses stuff defined outside of the condition.  If so, hoist it out.
  if (!I->isSafeToSpeculativelyExecute())
    return false;

  switch (I->getOpcode()) {
  default: return false;  // Cannot hoist this out safely.
  case Instruction::Load:
    // We have to check to make sure there are no instructions before the
    // load in its basic block, as we are going to hoist the load out to its
    // predecessor.
    if (PBB->getFirstNonPHIOrDbg() != I)
      return false;
    break;
  case Instruction::GetElementPtr:
    // GEPs are cheap if all indices are constant.
    if (!cast<GetElementPtrInst>(I)->hasAllConstantIndices())
      return false;
    break;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::ICmp:
    break;   // These are all cheap and non-trapping instructions.
  }

  // Okay, we can only really hoist these out if their operands are not
  // defined in the conditional region.
  for (User::op_iterator i = I->op_begin(), e = I->op_end(); i != e; ++i)
    if (!DominatesMergePoint(*i, BB, 0))
      return false;
  // Okay, it's safe to do this!  Remember this instruction.
  AggressiveInsts->insert(I);
  return true;
}

/// GetConstantInt - Extract ConstantInt from value, looking through IntToPtr
/// and PointerNullValue. Return NULL if value is not a constant int.
static ConstantInt *GetConstantInt(Value *V, const TargetData *TD) {
  // Normal constant int.
  ConstantInt *CI = dyn_cast<ConstantInt>(V);
  if (CI || !TD || !isa<Constant>(V) || !V->getType()->isPointerTy())
    return CI;

  // This is some kind of pointer constant. Turn it into a pointer-sized
  // ConstantInt if possible.
  const IntegerType *PtrTy = TD->getIntPtrType(V->getContext());

  // Null pointer means 0, see SelectionDAGBuilder::getValue(const Value*).
  if (isa<ConstantPointerNull>(V))
    return ConstantInt::get(PtrTy, 0);

  // IntToPtr const int.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::IntToPtr)
      if (ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(0))) {
        // The constant is very likely to have the right type already.
        if (CI->getType() == PtrTy)
          return CI;
        else
          return cast<ConstantInt>
            (ConstantExpr::getIntegerCast(CI, PtrTy, /*isSigned=*/false));
      }
  return 0;
}

/// GatherConstantCompares - Given a potentially 'or'd or 'and'd together
/// collection of icmp eq/ne instructions that compare a value against a
/// constant, return the value being compared, and stick the constant into the
/// Values vector.
static Value *
GatherConstantCompares(Value *V, std::vector<ConstantInt*> &Vals, Value *&Extra,
                       const TargetData *TD, bool isEQ, unsigned &UsedICmps) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return 0;
  
  // If this is an icmp against a constant, handle this as one of the cases.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(I)) {
    if (ConstantInt *C = GetConstantInt(I->getOperand(1), TD)) {
      if (ICI->getPredicate() == (isEQ ? ICmpInst::ICMP_EQ:ICmpInst::ICMP_NE)) {
        UsedICmps++;
        Vals.push_back(C);
        return I->getOperand(0);
      }
      
      // If we have "x ult 3" comparison, for example, then we can add 0,1,2 to
      // the set.
      ConstantRange Span =
        ConstantRange::makeICmpRegion(ICI->getPredicate(), C->getValue());
      
      // If this is an and/!= check then we want to optimize "x ugt 2" into
      // x != 0 && x != 1.
      if (!isEQ)
        Span = Span.inverse();
      
      // If there are a ton of values, we don't want to make a ginormous switch.
      if (Span.getSetSize().ugt(8) || Span.isEmptySet() ||
          // We don't handle wrapped sets yet.
          Span.isWrappedSet())
        return 0;
      
      for (APInt Tmp = Span.getLower(); Tmp != Span.getUpper(); ++Tmp)
        Vals.push_back(ConstantInt::get(V->getContext(), Tmp));
      UsedICmps++;
      return I->getOperand(0);
    }
    return 0;
  }
  
  // Otherwise, we can only handle an | or &, depending on isEQ.
  if (I->getOpcode() != (isEQ ? Instruction::Or : Instruction::And))
    return 0;
  
  unsigned NumValsBeforeLHS = Vals.size();
  unsigned UsedICmpsBeforeLHS = UsedICmps;
  if (Value *LHS = GatherConstantCompares(I->getOperand(0), Vals, Extra, TD,
                                          isEQ, UsedICmps)) {
    unsigned NumVals = Vals.size();
    unsigned UsedICmpsBeforeRHS = UsedICmps;
    if (Value *RHS = GatherConstantCompares(I->getOperand(1), Vals, Extra, TD,
                                            isEQ, UsedICmps)) {
      if (LHS == RHS)
        return LHS;
      Vals.resize(NumVals);
      UsedICmps = UsedICmpsBeforeRHS;
    }

    // The RHS of the or/and can't be folded in and we haven't used "Extra" yet,
    // set it and return success.
    if (Extra == 0 || Extra == I->getOperand(1)) {
      Extra = I->getOperand(1);
      return LHS;
    }
    
    Vals.resize(NumValsBeforeLHS);
    UsedICmps = UsedICmpsBeforeLHS;
    return 0;
  }
  
  // If the LHS can't be folded in, but Extra is available and RHS can, try to
  // use LHS as Extra.
  if (Extra == 0 || Extra == I->getOperand(0)) {
    Value *OldExtra = Extra;
    Extra = I->getOperand(0);
    if (Value *RHS = GatherConstantCompares(I->getOperand(1), Vals, Extra, TD,
                                            isEQ, UsedICmps))
      return RHS;
    assert(Vals.size() == NumValsBeforeLHS);
    Extra = OldExtra;
  }
  
  return 0;
}
      
static void EraseTerminatorInstAndDCECond(TerminatorInst *TI) {
  Instruction* Cond = 0;
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    Cond = dyn_cast<Instruction>(SI->getCondition());
  } else if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isConditional())
      Cond = dyn_cast<Instruction>(BI->getCondition());
  } else if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(TI)) {
    Cond = dyn_cast<Instruction>(IBI->getAddress());
  }

  TI->eraseFromParent();
  if (Cond) RecursivelyDeleteTriviallyDeadInstructions(Cond);
}

/// isValueEqualityComparison - Return true if the specified terminator checks
/// to see if a value is equal to constant integer value.
Value *SimplifyCFGOpt::isValueEqualityComparison(TerminatorInst *TI) {
  Value *CV = 0;
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    // Do not permit merging of large switch instructions into their
    // predecessors unless there is only one predecessor.
    if (SI->getNumSuccessors()*std::distance(pred_begin(SI->getParent()),
                                             pred_end(SI->getParent())) <= 128)
      CV = SI->getCondition();
  } else if (BranchInst *BI = dyn_cast<BranchInst>(TI))
    if (BI->isConditional() && BI->getCondition()->hasOneUse())
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition()))
        if ((ICI->getPredicate() == ICmpInst::ICMP_EQ ||
             ICI->getPredicate() == ICmpInst::ICMP_NE) &&
            GetConstantInt(ICI->getOperand(1), TD))
          CV = ICI->getOperand(0);

  // Unwrap any lossless ptrtoint cast.
  if (TD && CV && CV->getType() == TD->getIntPtrType(CV->getContext()))
    if (PtrToIntInst *PTII = dyn_cast<PtrToIntInst>(CV))
      CV = PTII->getOperand(0);
  return CV;
}

/// GetValueEqualityComparisonCases - Given a value comparison instruction,
/// decode all of the 'cases' that it represents and return the 'default' block.
BasicBlock *SimplifyCFGOpt::
GetValueEqualityComparisonCases(TerminatorInst *TI,
                                std::vector<std::pair<ConstantInt*,
                                                      BasicBlock*> > &Cases) {
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    Cases.reserve(SI->getNumCases());
    for (unsigned i = 1, e = SI->getNumCases(); i != e; ++i)
      Cases.push_back(std::make_pair(SI->getCaseValue(i), SI->getSuccessor(i)));
    return SI->getDefaultDest();
  }

  BranchInst *BI = cast<BranchInst>(TI);
  ICmpInst *ICI = cast<ICmpInst>(BI->getCondition());
  Cases.push_back(std::make_pair(GetConstantInt(ICI->getOperand(1), TD),
                                 BI->getSuccessor(ICI->getPredicate() ==
                                                  ICmpInst::ICMP_NE)));
  return BI->getSuccessor(ICI->getPredicate() == ICmpInst::ICMP_EQ);
}


/// EliminateBlockCases - Given a vector of bb/value pairs, remove any entries
/// in the list that match the specified block.
static void EliminateBlockCases(BasicBlock *BB,
               std::vector<std::pair<ConstantInt*, BasicBlock*> > &Cases) {
  for (unsigned i = 0, e = Cases.size(); i != e; ++i)
    if (Cases[i].second == BB) {
      Cases.erase(Cases.begin()+i);
      --i; --e;
    }
}

/// ValuesOverlap - Return true if there are any keys in C1 that exist in C2 as
/// well.
static bool
ValuesOverlap(std::vector<std::pair<ConstantInt*, BasicBlock*> > &C1,
              std::vector<std::pair<ConstantInt*, BasicBlock*> > &C2) {
  std::vector<std::pair<ConstantInt*, BasicBlock*> > *V1 = &C1, *V2 = &C2;

  // Make V1 be smaller than V2.
  if (V1->size() > V2->size())
    std::swap(V1, V2);

  if (V1->size() == 0) return false;
  if (V1->size() == 1) {
    // Just scan V2.
    ConstantInt *TheVal = (*V1)[0].first;
    for (unsigned i = 0, e = V2->size(); i != e; ++i)
      if (TheVal == (*V2)[i].first)
        return true;
  }

  // Otherwise, just sort both lists and compare element by element.
  array_pod_sort(V1->begin(), V1->end());
  array_pod_sort(V2->begin(), V2->end());
  unsigned i1 = 0, i2 = 0, e1 = V1->size(), e2 = V2->size();
  while (i1 != e1 && i2 != e2) {
    if ((*V1)[i1].first == (*V2)[i2].first)
      return true;
    if ((*V1)[i1].first < (*V2)[i2].first)
      ++i1;
    else
      ++i2;
  }
  return false;
}

/// SimplifyEqualityComparisonWithOnlyPredecessor - If TI is known to be a
/// terminator instruction and its block is known to only have a single
/// predecessor block, check to see if that predecessor is also a value
/// comparison with the same value, and if that comparison determines the
/// outcome of this comparison.  If so, simplify TI.  This does a very limited
/// form of jump threading.
bool SimplifyCFGOpt::
SimplifyEqualityComparisonWithOnlyPredecessor(TerminatorInst *TI,
                                              BasicBlock *Pred) {
  Value *PredVal = isValueEqualityComparison(Pred->getTerminator());
  if (!PredVal) return false;  // Not a value comparison in predecessor.

  Value *ThisVal = isValueEqualityComparison(TI);
  assert(ThisVal && "This isn't a value comparison!!");
  if (ThisVal != PredVal) return false;  // Different predicates.

  // Find out information about when control will move from Pred to TI's block.
  std::vector<std::pair<ConstantInt*, BasicBlock*> > PredCases;
  BasicBlock *PredDef = GetValueEqualityComparisonCases(Pred->getTerminator(),
                                                        PredCases);
  EliminateBlockCases(PredDef, PredCases);  // Remove default from cases.

  // Find information about how control leaves this block.
  std::vector<std::pair<ConstantInt*, BasicBlock*> > ThisCases;
  BasicBlock *ThisDef = GetValueEqualityComparisonCases(TI, ThisCases);
  EliminateBlockCases(ThisDef, ThisCases);  // Remove default from cases.

  // If TI's block is the default block from Pred's comparison, potentially
  // simplify TI based on this knowledge.
  if (PredDef == TI->getParent()) {
    // If we are here, we know that the value is none of those cases listed in
    // PredCases.  If there are any cases in ThisCases that are in PredCases, we
    // can simplify TI.
    if (!ValuesOverlap(PredCases, ThisCases))
      return false;
    
    if (isa<BranchInst>(TI)) {
      // Okay, one of the successors of this condbr is dead.  Convert it to a
      // uncond br.
      assert(ThisCases.size() == 1 && "Branch can only have one case!");
      // Insert the new branch.
      Instruction *NI = BranchInst::Create(ThisDef, TI);
      (void) NI;

      // Remove PHI node entries for the dead edge.
      ThisCases[0].second->removePredecessor(TI->getParent());

      DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
           << "Through successor TI: " << *TI << "Leaving: " << *NI << "\n");

      EraseTerminatorInstAndDCECond(TI);
      return true;
    }
      
    SwitchInst *SI = cast<SwitchInst>(TI);
    // Okay, TI has cases that are statically dead, prune them away.
    SmallPtrSet<Constant*, 16> DeadCases;
    for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
      DeadCases.insert(PredCases[i].first);

    DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
                 << "Through successor TI: " << *TI);

    for (unsigned i = SI->getNumCases()-1; i != 0; --i)
      if (DeadCases.count(SI->getCaseValue(i))) {
        SI->getSuccessor(i)->removePredecessor(TI->getParent());
        SI->removeCase(i);
      }

    DEBUG(dbgs() << "Leaving: " << *TI << "\n");
    return true;
  }
  
  // Otherwise, TI's block must correspond to some matched value.  Find out
  // which value (or set of values) this is.
  ConstantInt *TIV = 0;
  BasicBlock *TIBB = TI->getParent();
  for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
    if (PredCases[i].second == TIBB) {
      if (TIV != 0)
        return false;  // Cannot handle multiple values coming to this block.
      TIV = PredCases[i].first;
    }
  assert(TIV && "No edge from pred to succ?");

  // Okay, we found the one constant that our value can be if we get into TI's
  // BB.  Find out which successor will unconditionally be branched to.
  BasicBlock *TheRealDest = 0;
  for (unsigned i = 0, e = ThisCases.size(); i != e; ++i)
    if (ThisCases[i].first == TIV) {
      TheRealDest = ThisCases[i].second;
      break;
    }

  // If not handled by any explicit cases, it is handled by the default case.
  if (TheRealDest == 0) TheRealDest = ThisDef;

  // Remove PHI node entries for dead edges.
  BasicBlock *CheckEdge = TheRealDest;
  for (succ_iterator SI = succ_begin(TIBB), e = succ_end(TIBB); SI != e; ++SI)
    if (*SI != CheckEdge)
      (*SI)->removePredecessor(TIBB);
    else
      CheckEdge = 0;

  // Insert the new branch.
  Instruction *NI = BranchInst::Create(TheRealDest, TI);
  (void) NI;

  DEBUG(dbgs() << "Threading pred instr: " << *Pred->getTerminator()
            << "Through successor TI: " << *TI << "Leaving: " << *NI << "\n");

  EraseTerminatorInstAndDCECond(TI);
  return true;
}

namespace {
  /// ConstantIntOrdering - This class implements a stable ordering of constant
  /// integers that does not depend on their address.  This is important for
  /// applications that sort ConstantInt's to ensure uniqueness.
  struct ConstantIntOrdering {
    bool operator()(const ConstantInt *LHS, const ConstantInt *RHS) const {
      return LHS->getValue().ult(RHS->getValue());
    }
  };
}

static int ConstantIntSortPredicate(const void *P1, const void *P2) {
  const ConstantInt *LHS = *(const ConstantInt**)P1;
  const ConstantInt *RHS = *(const ConstantInt**)P2;
  if (LHS->getValue().ult(RHS->getValue()))
    return 1;
  if (LHS->getValue() == RHS->getValue())
    return 0;
  return -1;
}

/// FoldValueComparisonIntoPredecessors - The specified terminator is a value
/// equality comparison instruction (either a switch or a branch on "X == c").
/// See if any of the predecessors of the terminator block are value comparisons
/// on the same value.  If so, and if safe to do so, fold them together.
bool SimplifyCFGOpt::FoldValueComparisonIntoPredecessors(TerminatorInst *TI) {
  BasicBlock *BB = TI->getParent();
  Value *CV = isValueEqualityComparison(TI);  // CondVal
  assert(CV && "Not a comparison?");
  bool Changed = false;

  SmallVector<BasicBlock*, 16> Preds(pred_begin(BB), pred_end(BB));
  while (!Preds.empty()) {
    BasicBlock *Pred = Preds.pop_back_val();

    // See if the predecessor is a comparison with the same value.
    TerminatorInst *PTI = Pred->getTerminator();
    Value *PCV = isValueEqualityComparison(PTI);  // PredCondVal

    if (PCV == CV && SafeToMergeTerminators(TI, PTI)) {
      // Figure out which 'cases' to copy from SI to PSI.
      std::vector<std::pair<ConstantInt*, BasicBlock*> > BBCases;
      BasicBlock *BBDefault = GetValueEqualityComparisonCases(TI, BBCases);

      std::vector<std::pair<ConstantInt*, BasicBlock*> > PredCases;
      BasicBlock *PredDefault = GetValueEqualityComparisonCases(PTI, PredCases);

      // Based on whether the default edge from PTI goes to BB or not, fill in
      // PredCases and PredDefault with the new switch cases we would like to
      // build.
      SmallVector<BasicBlock*, 8> NewSuccessors;

      if (PredDefault == BB) {
        // If this is the default destination from PTI, only the edges in TI
        // that don't occur in PTI, or that branch to BB will be activated.
        std::set<ConstantInt*, ConstantIntOrdering> PTIHandled;
        for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
          if (PredCases[i].second != BB)
            PTIHandled.insert(PredCases[i].first);
          else {
            // The default destination is BB, we don't need explicit targets.
            std::swap(PredCases[i], PredCases.back());
            PredCases.pop_back();
            --i; --e;
          }

        // Reconstruct the new switch statement we will be building.
        if (PredDefault != BBDefault) {
          PredDefault->removePredecessor(Pred);
          PredDefault = BBDefault;
          NewSuccessors.push_back(BBDefault);
        }
        for (unsigned i = 0, e = BBCases.size(); i != e; ++i)
          if (!PTIHandled.count(BBCases[i].first) &&
              BBCases[i].second != BBDefault) {
            PredCases.push_back(BBCases[i]);
            NewSuccessors.push_back(BBCases[i].second);
          }

      } else {
        // If this is not the default destination from PSI, only the edges
        // in SI that occur in PSI with a destination of BB will be
        // activated.
        std::set<ConstantInt*, ConstantIntOrdering> PTIHandled;
        for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
          if (PredCases[i].second == BB) {
            PTIHandled.insert(PredCases[i].first);
            std::swap(PredCases[i], PredCases.back());
            PredCases.pop_back();
            --i; --e;
          }

        // Okay, now we know which constants were sent to BB from the
        // predecessor.  Figure out where they will all go now.
        for (unsigned i = 0, e = BBCases.size(); i != e; ++i)
          if (PTIHandled.count(BBCases[i].first)) {
            // If this is one we are capable of getting...
            PredCases.push_back(BBCases[i]);
            NewSuccessors.push_back(BBCases[i].second);
            PTIHandled.erase(BBCases[i].first);// This constant is taken care of
          }

        // If there are any constants vectored to BB that TI doesn't handle,
        // they must go to the default destination of TI.
        for (std::set<ConstantInt*, ConstantIntOrdering>::iterator I = 
                                    PTIHandled.begin(),
               E = PTIHandled.end(); I != E; ++I) {
          PredCases.push_back(std::make_pair(*I, BBDefault));
          NewSuccessors.push_back(BBDefault);
        }
      }

      // Okay, at this point, we know which new successor Pred will get.  Make
      // sure we update the number of entries in the PHI nodes for these
      // successors.
      for (unsigned i = 0, e = NewSuccessors.size(); i != e; ++i)
        AddPredecessorToBlock(NewSuccessors[i], Pred, BB);

      // Convert pointer to int before we switch.
      if (CV->getType()->isPointerTy()) {
        assert(TD && "Cannot switch on pointer without TargetData");
        CV = new PtrToIntInst(CV, TD->getIntPtrType(CV->getContext()),
                              "magicptr", PTI);
      }

      // Now that the successors are updated, create the new Switch instruction.
      SwitchInst *NewSI = SwitchInst::Create(CV, PredDefault,
                                             PredCases.size(), PTI);
      for (unsigned i = 0, e = PredCases.size(); i != e; ++i)
        NewSI->addCase(PredCases[i].first, PredCases[i].second);

      EraseTerminatorInstAndDCECond(PTI);

      // Okay, last check.  If BB is still a successor of PSI, then we must
      // have an infinite loop case.  If so, add an infinitely looping block
      // to handle the case to preserve the behavior of the code.
      BasicBlock *InfLoopBlock = 0;
      for (unsigned i = 0, e = NewSI->getNumSuccessors(); i != e; ++i)
        if (NewSI->getSuccessor(i) == BB) {
          if (InfLoopBlock == 0) {
            // Insert it at the end of the function, because it's either code,
            // or it won't matter if it's hot. :)
            InfLoopBlock = BasicBlock::Create(BB->getContext(),
                                              "infloop", BB->getParent());
            BranchInst::Create(InfLoopBlock, InfLoopBlock);
          }
          NewSI->setSuccessor(i, InfLoopBlock);
        }

      Changed = true;
    }
  }
  return Changed;
}

// isSafeToHoistInvoke - If we would need to insert a select that uses the
// value of this invoke (comments in HoistThenElseCodeToIf explain why we
// would need to do this), we can't hoist the invoke, as there is nowhere
// to put the select in this case.
static bool isSafeToHoistInvoke(BasicBlock *BB1, BasicBlock *BB2,
                                Instruction *I1, Instruction *I2) {
  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI) {
    PHINode *PN;
    for (BasicBlock::iterator BBI = SI->begin();
         (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
      Value *BB1V = PN->getIncomingValueForBlock(BB1);
      Value *BB2V = PN->getIncomingValueForBlock(BB2);
      if (BB1V != BB2V && (BB1V==I1 || BB2V==I2)) {
        return false;
      }
    }
  }
  return true;
}

/// HoistThenElseCodeToIf - Given a conditional branch that goes to BB1 and
/// BB2, hoist any common code in the two blocks up into the branch block.  The
/// caller of this function guarantees that BI's block dominates BB1 and BB2.
static bool HoistThenElseCodeToIf(BranchInst *BI) {
  // This does very trivial matching, with limited scanning, to find identical
  // instructions in the two blocks.  In particular, we don't want to get into
  // O(M*N) situations here where M and N are the sizes of BB1 and BB2.  As
  // such, we currently just scan for obviously identical instructions in an
  // identical order.
  BasicBlock *BB1 = BI->getSuccessor(0);  // The true destination.
  BasicBlock *BB2 = BI->getSuccessor(1);  // The false destination

  BasicBlock::iterator BB1_Itr = BB1->begin();
  BasicBlock::iterator BB2_Itr = BB2->begin();

  Instruction *I1 = BB1_Itr++, *I2 = BB2_Itr++;
  // Skip debug info if it is not identical.
  DbgInfoIntrinsic *DBI1 = dyn_cast<DbgInfoIntrinsic>(I1);
  DbgInfoIntrinsic *DBI2 = dyn_cast<DbgInfoIntrinsic>(I2);
  if (!DBI1 || !DBI2 || !DBI1->isIdenticalToWhenDefined(DBI2)) {
    while (isa<DbgInfoIntrinsic>(I1))
      I1 = BB1_Itr++;
    while (isa<DbgInfoIntrinsic>(I2))
      I2 = BB2_Itr++;
  }
  if (isa<PHINode>(I1) || !I1->isIdenticalToWhenDefined(I2) ||
      (isa<InvokeInst>(I1) && !isSafeToHoistInvoke(BB1, BB2, I1, I2)))
    return false;

  // If we get here, we can hoist at least one instruction.
  BasicBlock *BIParent = BI->getParent();

  do {
    // If we are hoisting the terminator instruction, don't move one (making a
    // broken BB), instead clone it, and remove BI.
    if (isa<TerminatorInst>(I1))
      goto HoistTerminator;

    // For a normal instruction, we just move one to right before the branch,
    // then replace all uses of the other with the first.  Finally, we remove
    // the now redundant second instruction.
    BIParent->getInstList().splice(BI, BB1->getInstList(), I1);
    if (!I2->use_empty())
      I2->replaceAllUsesWith(I1);
    I1->intersectOptionalDataWith(I2);
    I2->eraseFromParent();

    I1 = BB1_Itr++;
    I2 = BB2_Itr++;
    // Skip debug info if it is not identical.
    DbgInfoIntrinsic *DBI1 = dyn_cast<DbgInfoIntrinsic>(I1);
    DbgInfoIntrinsic *DBI2 = dyn_cast<DbgInfoIntrinsic>(I2);
    if (!DBI1 || !DBI2 || !DBI1->isIdenticalToWhenDefined(DBI2)) {
      while (isa<DbgInfoIntrinsic>(I1))
        I1 = BB1_Itr++;
      while (isa<DbgInfoIntrinsic>(I2))
        I2 = BB2_Itr++;
    }
  } while (I1->isIdenticalToWhenDefined(I2));

  return true;

HoistTerminator:
  // It may not be possible to hoist an invoke.
  if (isa<InvokeInst>(I1) && !isSafeToHoistInvoke(BB1, BB2, I1, I2))
    return true;

  // Okay, it is safe to hoist the terminator.
  Instruction *NT = I1->clone();
  BIParent->getInstList().insert(BI, NT);
  if (!NT->getType()->isVoidTy()) {
    I1->replaceAllUsesWith(NT);
    I2->replaceAllUsesWith(NT);
    NT->takeName(I1);
  }

  // Hoisting one of the terminators from our successor is a great thing.
  // Unfortunately, the successors of the if/else blocks may have PHI nodes in
  // them.  If they do, all PHI entries for BB1/BB2 must agree for all PHI
  // nodes, so we insert select instruction to compute the final result.
  std::map<std::pair<Value*,Value*>, SelectInst*> InsertedSelects;
  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI) {
    PHINode *PN;
    for (BasicBlock::iterator BBI = SI->begin();
         (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
      Value *BB1V = PN->getIncomingValueForBlock(BB1);
      Value *BB2V = PN->getIncomingValueForBlock(BB2);
      if (BB1V == BB2V) continue;
      
      // These values do not agree.  Insert a select instruction before NT
      // that determines the right value.
      SelectInst *&SI = InsertedSelects[std::make_pair(BB1V, BB2V)];
      if (SI == 0)
        SI = SelectInst::Create(BI->getCondition(), BB1V, BB2V,
                                BB1V->getName()+"."+BB2V->getName(), NT);
      // Make the PHI node use the select for all incoming values for BB1/BB2
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingBlock(i) == BB1 || PN->getIncomingBlock(i) == BB2)
          PN->setIncomingValue(i, SI);
    }
  }

  // Update any PHI nodes in our new successors.
  for (succ_iterator SI = succ_begin(BB1), E = succ_end(BB1); SI != E; ++SI)
    AddPredecessorToBlock(*SI, BIParent, BB1);

  EraseTerminatorInstAndDCECond(BI);
  return true;
}

/// SpeculativelyExecuteBB - Given a conditional branch that goes to BB1
/// and an BB2 and the only successor of BB1 is BB2, hoist simple code
/// (for now, restricted to a single instruction that's side effect free) from
/// the BB1 into the branch block to speculatively execute it.
static bool SpeculativelyExecuteBB(BranchInst *BI, BasicBlock *BB1) {
  // Only speculatively execution a single instruction (not counting the
  // terminator) for now.
  Instruction *HInst = NULL;
  Instruction *Term = BB1->getTerminator();
  for (BasicBlock::iterator BBI = BB1->begin(), BBE = BB1->end();
       BBI != BBE; ++BBI) {
    Instruction *I = BBI;
    // Skip debug info.
    if (isa<DbgInfoIntrinsic>(I)) continue;
    if (I == Term) break;

    if (HInst)
      return false;
    HInst = I;
  }
  if (!HInst)
    return false;

  // Be conservative for now. FP select instruction can often be expensive.
  Value *BrCond = BI->getCondition();
  if (isa<FCmpInst>(BrCond))
    return false;

  // If BB1 is actually on the false edge of the conditional branch, remember
  // to swap the select operands later.
  bool Invert = false;
  if (BB1 != BI->getSuccessor(0)) {
    assert(BB1 == BI->getSuccessor(1) && "No edge from 'if' block?");
    Invert = true;
  }

  // Turn
  // BB:
  //     %t1 = icmp
  //     br i1 %t1, label %BB1, label %BB2
  // BB1:
  //     %t3 = add %t2, c
  //     br label BB2
  // BB2:
  // =>
  // BB:
  //     %t1 = icmp
  //     %t4 = add %t2, c
  //     %t3 = select i1 %t1, %t2, %t3
  switch (HInst->getOpcode()) {
  default: return false;  // Not safe / profitable to hoist.
  case Instruction::Add:
  case Instruction::Sub:
    // Not worth doing for vector ops.
    if (HInst->getType()->isVectorTy())
      return false;
    break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    // Don't mess with vector operations.
    if (HInst->getType()->isVectorTy())
      return false;
    break;   // These are all cheap and non-trapping instructions.
  }
  
  // If the instruction is obviously dead, don't try to predicate it.
  if (HInst->use_empty()) {
    HInst->eraseFromParent();
    return true;
  }

  // Can we speculatively execute the instruction? And what is the value 
  // if the condition is false? Consider the phi uses, if the incoming value
  // from the "if" block are all the same V, then V is the value of the
  // select if the condition is false.
  BasicBlock *BIParent = BI->getParent();
  SmallVector<PHINode*, 4> PHIUses;
  Value *FalseV = NULL;
  
  BasicBlock *BB2 = BB1->getTerminator()->getSuccessor(0);
  for (Value::use_iterator UI = HInst->use_begin(), E = HInst->use_end();
       UI != E; ++UI) {
    // Ignore any user that is not a PHI node in BB2.  These can only occur in
    // unreachable blocks, because they would not be dominated by the instr.
    PHINode *PN = dyn_cast<PHINode>(*UI);
    if (!PN || PN->getParent() != BB2)
      return false;
    PHIUses.push_back(PN);
    
    Value *PHIV = PN->getIncomingValueForBlock(BIParent);
    if (!FalseV)
      FalseV = PHIV;
    else if (FalseV != PHIV)
      return false;  // Inconsistent value when condition is false.
  }
  
  assert(FalseV && "Must have at least one user, and it must be a PHI");

  // Do not hoist the instruction if any of its operands are defined but not
  // used in this BB. The transformation will prevent the operand from
  // being sunk into the use block.
  for (User::op_iterator i = HInst->op_begin(), e = HInst->op_end(); 
       i != e; ++i) {
    Instruction *OpI = dyn_cast<Instruction>(*i);
    if (OpI && OpI->getParent() == BIParent &&
        !OpI->isUsedInBasicBlock(BIParent))
      return false;
  }

  // If we get here, we can hoist the instruction. Try to place it
  // before the icmp instruction preceding the conditional branch.
  BasicBlock::iterator InsertPos = BI;
  if (InsertPos != BIParent->begin())
    --InsertPos;
  // Skip debug info between condition and branch.
  while (InsertPos != BIParent->begin() && isa<DbgInfoIntrinsic>(InsertPos))
    --InsertPos;
  if (InsertPos == BrCond && !isa<PHINode>(BrCond)) {
    SmallPtrSet<Instruction *, 4> BB1Insns;
    for(BasicBlock::iterator BB1I = BB1->begin(), BB1E = BB1->end(); 
        BB1I != BB1E; ++BB1I) 
      BB1Insns.insert(BB1I);
    for(Value::use_iterator UI = BrCond->use_begin(), UE = BrCond->use_end();
        UI != UE; ++UI) {
      Instruction *Use = cast<Instruction>(*UI);
      if (!BB1Insns.count(Use)) continue;
      
      // If BrCond uses the instruction that place it just before
      // branch instruction.
      InsertPos = BI;
      break;
    }
  } else
    InsertPos = BI;
  BIParent->getInstList().splice(InsertPos, BB1->getInstList(), HInst);

  // Create a select whose true value is the speculatively executed value and
  // false value is the previously determined FalseV.
  SelectInst *SI;
  if (Invert)
    SI = SelectInst::Create(BrCond, FalseV, HInst,
                            FalseV->getName() + "." + HInst->getName(), BI);
  else
    SI = SelectInst::Create(BrCond, HInst, FalseV,
                            HInst->getName() + "." + FalseV->getName(), BI);

  // Make the PHI node use the select for all incoming values for "then" and
  // "if" blocks.
  for (unsigned i = 0, e = PHIUses.size(); i != e; ++i) {
    PHINode *PN = PHIUses[i];
    for (unsigned j = 0, ee = PN->getNumIncomingValues(); j != ee; ++j)
      if (PN->getIncomingBlock(j) == BB1 || PN->getIncomingBlock(j) == BIParent)
        PN->setIncomingValue(j, SI);
  }

  ++NumSpeculations;
  return true;
}

/// BlockIsSimpleEnoughToThreadThrough - Return true if we can thread a branch
/// across this block.
static bool BlockIsSimpleEnoughToThreadThrough(BasicBlock *BB) {
  BranchInst *BI = cast<BranchInst>(BB->getTerminator());
  unsigned Size = 0;
  
  for (BasicBlock::iterator BBI = BB->begin(); &*BBI != BI; ++BBI) {
    if (isa<DbgInfoIntrinsic>(BBI))
      continue;
    if (Size > 10) return false;  // Don't clone large BB's.
    ++Size;
    
    // We can only support instructions that do not define values that are
    // live outside of the current basic block.
    for (Value::use_iterator UI = BBI->use_begin(), E = BBI->use_end();
         UI != E; ++UI) {
      Instruction *U = cast<Instruction>(*UI);
      if (U->getParent() != BB || isa<PHINode>(U)) return false;
    }
    
    // Looks ok, continue checking.
  }

  return true;
}

/// FoldCondBranchOnPHI - If we have a conditional branch on a PHI node value
/// that is defined in the same block as the branch and if any PHI entries are
/// constants, thread edges corresponding to that entry to be branches to their
/// ultimate destination.
static bool FoldCondBranchOnPHI(BranchInst *BI, const TargetData *TD) {
  BasicBlock *BB = BI->getParent();
  PHINode *PN = dyn_cast<PHINode>(BI->getCondition());
  // NOTE: we currently cannot transform this case if the PHI node is used
  // outside of the block.
  if (!PN || PN->getParent() != BB || !PN->hasOneUse())
    return false;
  
  // Degenerate case of a single entry PHI.
  if (PN->getNumIncomingValues() == 1) {
    FoldSingleEntryPHINodes(PN->getParent());
    return true;    
  }

  // Now we know that this block has multiple preds and two succs.
  if (!BlockIsSimpleEnoughToThreadThrough(BB)) return false;
  
  // Okay, this is a simple enough basic block.  See if any phi values are
  // constants.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    ConstantInt *CB = dyn_cast<ConstantInt>(PN->getIncomingValue(i));
    if (CB == 0 || !CB->getType()->isIntegerTy(1)) continue;
    
    // Okay, we now know that all edges from PredBB should be revectored to
    // branch to RealDest.
    BasicBlock *PredBB = PN->getIncomingBlock(i);
    BasicBlock *RealDest = BI->getSuccessor(!CB->getZExtValue());
    
    if (RealDest == BB) continue;  // Skip self loops.
    
    // The dest block might have PHI nodes, other predecessors and other
    // difficult cases.  Instead of being smart about this, just insert a new
    // block that jumps to the destination block, effectively splitting
    // the edge we are about to create.
    BasicBlock *EdgeBB = BasicBlock::Create(BB->getContext(),
                                            RealDest->getName()+".critedge",
                                            RealDest->getParent(), RealDest);
    BranchInst::Create(RealDest, EdgeBB);
    
    // Update PHI nodes.
    AddPredecessorToBlock(RealDest, EdgeBB, BB);

    // BB may have instructions that are being threaded over.  Clone these
    // instructions into EdgeBB.  We know that there will be no uses of the
    // cloned instructions outside of EdgeBB.
    BasicBlock::iterator InsertPt = EdgeBB->begin();
    DenseMap<Value*, Value*> TranslateMap;  // Track translated values.
    for (BasicBlock::iterator BBI = BB->begin(); &*BBI != BI; ++BBI) {
      if (PHINode *PN = dyn_cast<PHINode>(BBI)) {
        TranslateMap[PN] = PN->getIncomingValueForBlock(PredBB);
        continue;
      }
      // Clone the instruction.
      Instruction *N = BBI->clone();
      if (BBI->hasName()) N->setName(BBI->getName()+".c");
      
      // Update operands due to translation.
      for (User::op_iterator i = N->op_begin(), e = N->op_end();
           i != e; ++i) {
        DenseMap<Value*, Value*>::iterator PI = TranslateMap.find(*i);
        if (PI != TranslateMap.end())
          *i = PI->second;
      }
      
      // Check for trivial simplification.
      if (Value *V = SimplifyInstruction(N, TD)) {
        TranslateMap[BBI] = V;
        delete N;   // Instruction folded away, don't need actual inst
      } else {
        // Insert the new instruction into its new home.
        EdgeBB->getInstList().insert(InsertPt, N);
        if (!BBI->use_empty())
          TranslateMap[BBI] = N;
      }
    }

    // Loop over all of the edges from PredBB to BB, changing them to branch
    // to EdgeBB instead.
    TerminatorInst *PredBBTI = PredBB->getTerminator();
    for (unsigned i = 0, e = PredBBTI->getNumSuccessors(); i != e; ++i)
      if (PredBBTI->getSuccessor(i) == BB) {
        BB->removePredecessor(PredBB);
        PredBBTI->setSuccessor(i, EdgeBB);
      }
    
    // Recurse, simplifying any other constants.
    return FoldCondBranchOnPHI(BI, TD) | true;
  }

  return false;
}

/// FoldTwoEntryPHINode - Given a BB that starts with the specified two-entry
/// PHI node, see if we can eliminate it.
static bool FoldTwoEntryPHINode(PHINode *PN, const TargetData *TD) {
  // Ok, this is a two entry PHI node.  Check to see if this is a simple "if
  // statement", which has a very simple dominance structure.  Basically, we
  // are trying to find the condition that is being branched on, which
  // subsequently causes this merge to happen.  We really want control
  // dependence information for this check, but simplifycfg can't keep it up
  // to date, and this catches most of the cases we care about anyway.
  BasicBlock *BB = PN->getParent();
  BasicBlock *IfTrue, *IfFalse;
  Value *IfCond = GetIfCondition(BB, IfTrue, IfFalse);
  if (!IfCond ||
      // Don't bother if the branch will be constant folded trivially.
      isa<ConstantInt>(IfCond))
    return false;
  
  // Okay, we found that we can merge this two-entry phi node into a select.
  // Doing so would require us to fold *all* two entry phi nodes in this block.
  // At some point this becomes non-profitable (particularly if the target
  // doesn't support cmov's).  Only do this transformation if there are two or
  // fewer PHI nodes in this block.
  unsigned NumPhis = 0;
  for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ++NumPhis, ++I)
    if (NumPhis > 2)
      return false;
  
  // Loop over the PHI's seeing if we can promote them all to select
  // instructions.  While we are at it, keep track of the instructions
  // that need to be moved to the dominating block.
  SmallPtrSet<Instruction*, 4> AggressiveInsts;
  
  for (BasicBlock::iterator II = BB->begin(); isa<PHINode>(II);) {
    PHINode *PN = cast<PHINode>(II++);
    if (Value *V = SimplifyInstruction(PN, TD)) {
      PN->replaceAllUsesWith(V);
      PN->eraseFromParent();
      continue;
    }
    
    if (!DominatesMergePoint(PN->getIncomingValue(0), BB, &AggressiveInsts) ||
        !DominatesMergePoint(PN->getIncomingValue(1), BB, &AggressiveInsts))
      return false;
  }
  
  // If we folded the the first phi, PN dangles at this point.  Refresh it.  If
  // we ran out of PHIs then we simplified them all.
  PN = dyn_cast<PHINode>(BB->begin());
  if (PN == 0) return true;
  
  // Don't fold i1 branches on PHIs which contain binary operators.  These can
  // often be turned into switches and other things.
  if (PN->getType()->isIntegerTy(1) &&
      (isa<BinaryOperator>(PN->getIncomingValue(0)) ||
       isa<BinaryOperator>(PN->getIncomingValue(1)) ||
       isa<BinaryOperator>(IfCond)))
    return false;
  
  // If we all PHI nodes are promotable, check to make sure that all
  // instructions in the predecessor blocks can be promoted as well.  If
  // not, we won't be able to get rid of the control flow, so it's not
  // worth promoting to select instructions.
  BasicBlock *DomBlock = 0;
  BasicBlock *IfBlock1 = PN->getIncomingBlock(0);
  BasicBlock *IfBlock2 = PN->getIncomingBlock(1);
  if (cast<BranchInst>(IfBlock1->getTerminator())->isConditional()) {
    IfBlock1 = 0;
  } else {
    DomBlock = *pred_begin(IfBlock1);
    for (BasicBlock::iterator I = IfBlock1->begin();!isa<TerminatorInst>(I);++I)
      if (!AggressiveInsts.count(I) && !isa<DbgInfoIntrinsic>(I)) {
        // This is not an aggressive instruction that we can promote.
        // Because of this, we won't be able to get rid of the control
        // flow, so the xform is not worth it.
        return false;
      }
  }
    
  if (cast<BranchInst>(IfBlock2->getTerminator())->isConditional()) {
    IfBlock2 = 0;
  } else {
    DomBlock = *pred_begin(IfBlock2);
    for (BasicBlock::iterator I = IfBlock2->begin();!isa<TerminatorInst>(I);++I)
      if (!AggressiveInsts.count(I) && !isa<DbgInfoIntrinsic>(I)) {
        // This is not an aggressive instruction that we can promote.
        // Because of this, we won't be able to get rid of the control
        // flow, so the xform is not worth it.
        return false;
      }
  }
  
  DEBUG(dbgs() << "FOUND IF CONDITION!  " << *IfCond << "  T: "
               << IfTrue->getName() << "  F: " << IfFalse->getName() << "\n");
      
  // If we can still promote the PHI nodes after this gauntlet of tests,
  // do all of the PHI's now.
  Instruction *InsertPt = DomBlock->getTerminator();
  
  // Move all 'aggressive' instructions, which are defined in the
  // conditional parts of the if's up to the dominating block.
  if (IfBlock1)
    DomBlock->getInstList().splice(InsertPt,
                                   IfBlock1->getInstList(), IfBlock1->begin(),
                                   IfBlock1->getTerminator());
  if (IfBlock2)
    DomBlock->getInstList().splice(InsertPt,
                                   IfBlock2->getInstList(), IfBlock2->begin(),
                                   IfBlock2->getTerminator());
  
  while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
    // Change the PHI node into a select instruction.
    Value *TrueVal  = PN->getIncomingValue(PN->getIncomingBlock(0) == IfFalse);
    Value *FalseVal = PN->getIncomingValue(PN->getIncomingBlock(0) == IfTrue);
    
    Value *NV = SelectInst::Create(IfCond, TrueVal, FalseVal, "", InsertPt);
    PN->replaceAllUsesWith(NV);
    NV->takeName(PN);
    PN->eraseFromParent();
  }
  
  // At this point, IfBlock1 and IfBlock2 are both empty, so our if statement
  // has been flattened.  Change DomBlock to jump directly to our new block to
  // avoid other simplifycfg's kicking in on the diamond.
  TerminatorInst *OldTI = DomBlock->getTerminator();
  BranchInst::Create(BB, OldTI);
  OldTI->eraseFromParent();
  return true;
}

/// SimplifyCondBranchToTwoReturns - If we found a conditional branch that goes
/// to two returning blocks, try to merge them together into one return,
/// introducing a select if the return values disagree.
static bool SimplifyCondBranchToTwoReturns(BranchInst *BI) {
  assert(BI->isConditional() && "Must be a conditional branch");
  BasicBlock *TrueSucc = BI->getSuccessor(0);
  BasicBlock *FalseSucc = BI->getSuccessor(1);
  ReturnInst *TrueRet = cast<ReturnInst>(TrueSucc->getTerminator());
  ReturnInst *FalseRet = cast<ReturnInst>(FalseSucc->getTerminator());
  
  // Check to ensure both blocks are empty (just a return) or optionally empty
  // with PHI nodes.  If there are other instructions, merging would cause extra
  // computation on one path or the other.
  if (!TrueSucc->getFirstNonPHIOrDbg()->isTerminator())
    return false;
  if (!FalseSucc->getFirstNonPHIOrDbg()->isTerminator())
    return false;

  // Okay, we found a branch that is going to two return nodes.  If
  // there is no return value for this function, just change the
  // branch into a return.
  if (FalseRet->getNumOperands() == 0) {
    TrueSucc->removePredecessor(BI->getParent());
    FalseSucc->removePredecessor(BI->getParent());
    ReturnInst::Create(BI->getContext(), 0, BI);
    EraseTerminatorInstAndDCECond(BI);
    return true;
  }
    
  // Otherwise, figure out what the true and false return values are
  // so we can insert a new select instruction.
  Value *TrueValue = TrueRet->getReturnValue();
  Value *FalseValue = FalseRet->getReturnValue();
  
  // Unwrap any PHI nodes in the return blocks.
  if (PHINode *TVPN = dyn_cast_or_null<PHINode>(TrueValue))
    if (TVPN->getParent() == TrueSucc)
      TrueValue = TVPN->getIncomingValueForBlock(BI->getParent());
  if (PHINode *FVPN = dyn_cast_or_null<PHINode>(FalseValue))
    if (FVPN->getParent() == FalseSucc)
      FalseValue = FVPN->getIncomingValueForBlock(BI->getParent());
  
  // In order for this transformation to be safe, we must be able to
  // unconditionally execute both operands to the return.  This is
  // normally the case, but we could have a potentially-trapping
  // constant expression that prevents this transformation from being
  // safe.
  if (ConstantExpr *TCV = dyn_cast_or_null<ConstantExpr>(TrueValue))
    if (TCV->canTrap())
      return false;
  if (ConstantExpr *FCV = dyn_cast_or_null<ConstantExpr>(FalseValue))
    if (FCV->canTrap())
      return false;
  
  // Okay, we collected all the mapped values and checked them for sanity, and
  // defined to really do this transformation.  First, update the CFG.
  TrueSucc->removePredecessor(BI->getParent());
  FalseSucc->removePredecessor(BI->getParent());
  
  // Insert select instructions where needed.
  Value *BrCond = BI->getCondition();
  if (TrueValue) {
    // Insert a select if the results differ.
    if (TrueValue == FalseValue || isa<UndefValue>(FalseValue)) {
    } else if (isa<UndefValue>(TrueValue)) {
      TrueValue = FalseValue;
    } else {
      TrueValue = SelectInst::Create(BrCond, TrueValue,
                                     FalseValue, "retval", BI);
    }
  }

  Value *RI = !TrueValue ?
              ReturnInst::Create(BI->getContext(), BI) :
              ReturnInst::Create(BI->getContext(), TrueValue, BI);
  (void) RI;
      
  DEBUG(dbgs() << "\nCHANGING BRANCH TO TWO RETURNS INTO SELECT:"
               << "\n  " << *BI << "NewRet = " << *RI
               << "TRUEBLOCK: " << *TrueSucc << "FALSEBLOCK: "<< *FalseSucc);
      
  EraseTerminatorInstAndDCECond(BI);

  return true;
}

/// FoldBranchToCommonDest - If this basic block is ONLY a setcc and a branch,
/// and if a predecessor branches to us and one of our successors, fold the
/// setcc into the predecessor and use logical operations to pick the right
/// destination.
bool llvm::FoldBranchToCommonDest(BranchInst *BI) {
  BasicBlock *BB = BI->getParent();
  Instruction *Cond = dyn_cast<Instruction>(BI->getCondition());
  if (Cond == 0 || (!isa<CmpInst>(Cond) && !isa<BinaryOperator>(Cond)) ||
    Cond->getParent() != BB || !Cond->hasOneUse())
  return false;

  SmallVector<DbgInfoIntrinsic *, 8> DbgValues;
  // Only allow this if the condition is a simple instruction that can be
  // executed unconditionally.  It must be in the same block as the branch, and
  // must be at the front of the block.
  BasicBlock::iterator FrontIt = BB->front();
  // Ignore dbg intrinsics.
  while (DbgInfoIntrinsic *DBI = dyn_cast<DbgInfoIntrinsic>(FrontIt)) {
    DbgValues.push_back(DBI);
    ++FrontIt;
  }
    
  // Allow a single instruction to be hoisted in addition to the compare
  // that feeds the branch.  We later ensure that any values that _it_ uses
  // were also live in the predecessor, so that we don't unnecessarily create
  // register pressure or inhibit out-of-order execution.
  Instruction *BonusInst = 0;
  if (&*FrontIt != Cond &&
      FrontIt->hasOneUse() && *FrontIt->use_begin() == Cond &&
      FrontIt->isSafeToSpeculativelyExecute()) {
    BonusInst = &*FrontIt;
    ++FrontIt;
  }
  
  // Ignore dbg intrinsics.
  while (DbgInfoIntrinsic *DBI = dyn_cast<DbgInfoIntrinsic>(FrontIt)) {
    DbgValues.push_back(DBI);
    ++FrontIt;
  }

  // Only a single bonus inst is allowed.
  if (&*FrontIt != Cond)
    return false;
  
  // Make sure the instruction after the condition is the cond branch.
  BasicBlock::iterator CondIt = Cond; ++CondIt;
  // Ingore dbg intrinsics.
  while(DbgInfoIntrinsic *DBI = dyn_cast<DbgInfoIntrinsic>(CondIt)) {
    DbgValues.push_back(DBI);
    ++CondIt;
  }
  if (&*CondIt != BI) {
    assert (!isa<DbgInfoIntrinsic>(CondIt) && "Hey do not forget debug info!");
    return false;
  }

  // Cond is known to be a compare or binary operator.  Check to make sure that
  // neither operand is a potentially-trapping constant expression.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Cond->getOperand(0)))
    if (CE->canTrap())
      return false;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Cond->getOperand(1)))
    if (CE->canTrap())
      return false;
  
  
  // Finally, don't infinitely unroll conditional loops.
  BasicBlock *TrueDest  = BI->getSuccessor(0);
  BasicBlock *FalseDest = BI->getSuccessor(1);
  if (TrueDest == BB || FalseDest == BB)
    return false;

  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *PredBlock = *PI;
    BranchInst *PBI = dyn_cast<BranchInst>(PredBlock->getTerminator());
    
    // Check that we have two conditional branches.  If there is a PHI node in
    // the common successor, verify that the same value flows in from both
    // blocks.
    if (PBI == 0 || PBI->isUnconditional() ||
        !SafeToMergeTerminators(BI, PBI))
      continue;
    
    // Ensure that any values used in the bonus instruction are also used
    // by the terminator of the predecessor.  This means that those values
    // must already have been resolved, so we won't be inhibiting the 
    // out-of-order core by speculating them earlier.
    if (BonusInst) {
      // Collect the values used by the bonus inst
      SmallPtrSet<Value*, 4> UsedValues;
      for (Instruction::op_iterator OI = BonusInst->op_begin(),
           OE = BonusInst->op_end(); OI != OE; ++OI) {
        Value* V = *OI;
        if (!isa<Constant>(V))
          UsedValues.insert(V);
      }

      SmallVector<std::pair<Value*, unsigned>, 4> Worklist;
      Worklist.push_back(std::make_pair(PBI->getOperand(0), 0));
      
      // Walk up to four levels back up the use-def chain of the predecessor's
      // terminator to see if all those values were used.  The choice of four
      // levels is arbitrary, to provide a compile-time-cost bound.
      while (!Worklist.empty()) {
        std::pair<Value*, unsigned> Pair = Worklist.back();
        Worklist.pop_back();
        
        if (Pair.second >= 4) continue;
        UsedValues.erase(Pair.first);
        if (UsedValues.empty()) break;
        
        if (Instruction *I = dyn_cast<Instruction>(Pair.first)) {
          for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
               OI != OE; ++OI)
            Worklist.push_back(std::make_pair(OI->get(), Pair.second+1));
        }       
      }
      
      if (!UsedValues.empty()) return false;
    }
    
    Instruction::BinaryOps Opc;
    bool InvertPredCond = false;

    if (PBI->getSuccessor(0) == TrueDest)
      Opc = Instruction::Or;
    else if (PBI->getSuccessor(1) == FalseDest)
      Opc = Instruction::And;
    else if (PBI->getSuccessor(0) == FalseDest)
      Opc = Instruction::And, InvertPredCond = true;
    else if (PBI->getSuccessor(1) == TrueDest)
      Opc = Instruction::Or, InvertPredCond = true;
    else
      continue;

    DEBUG(dbgs() << "FOLDING BRANCH TO COMMON DEST:\n" << *PBI << *BB);
    
    // If we need to invert the condition in the pred block to match, do so now.
    if (InvertPredCond) {
      Value *NewCond = PBI->getCondition();
      
      if (NewCond->hasOneUse() && isa<CmpInst>(NewCond)) {
        CmpInst *CI = cast<CmpInst>(NewCond);
        CI->setPredicate(CI->getInversePredicate());
      } else {
        NewCond = BinaryOperator::CreateNot(NewCond,
                                  PBI->getCondition()->getName()+".not", PBI);
      }
      
      PBI->setCondition(NewCond);
      BasicBlock *OldTrue = PBI->getSuccessor(0);
      BasicBlock *OldFalse = PBI->getSuccessor(1);
      PBI->setSuccessor(0, OldFalse);
      PBI->setSuccessor(1, OldTrue);
    }
    
    // If we have a bonus inst, clone it into the predecessor block.
    Instruction *NewBonus = 0;
    if (BonusInst) {
      NewBonus = BonusInst->clone();
      PredBlock->getInstList().insert(PBI, NewBonus);
      NewBonus->takeName(BonusInst);
      BonusInst->setName(BonusInst->getName()+".old");
    }
    
    // Clone Cond into the predecessor basic block, and or/and the
    // two conditions together.
    Instruction *New = Cond->clone();
    if (BonusInst) New->replaceUsesOfWith(BonusInst, NewBonus);
    PredBlock->getInstList().insert(PBI, New);
    New->takeName(Cond);
    Cond->setName(New->getName()+".old");
    
    Value *NewCond = BinaryOperator::Create(Opc, PBI->getCondition(),
                                            New, "or.cond", PBI);
    PBI->setCondition(NewCond);
    if (PBI->getSuccessor(0) == BB) {
      AddPredecessorToBlock(TrueDest, PredBlock, BB);
      PBI->setSuccessor(0, TrueDest);
    }
    if (PBI->getSuccessor(1) == BB) {
      AddPredecessorToBlock(FalseDest, PredBlock, BB);
      PBI->setSuccessor(1, FalseDest);
    }

    // Move dbg value intrinsics in PredBlock.
    for (SmallVector<DbgInfoIntrinsic *, 8>::iterator DBI = DbgValues.begin(),
           DBE = DbgValues.end(); DBI != DBE; ++DBI) {
      DbgInfoIntrinsic *DB = *DBI;
      DB->removeFromParent();
      DB->insertBefore(PBI);
    }
    return true;
  }
  return false;
}

/// SimplifyCondBranchToCondBranch - If we have a conditional branch as a
/// predecessor of another block, this function tries to simplify it.  We know
/// that PBI and BI are both conditional branches, and BI is in one of the
/// successor blocks of PBI - PBI branches to BI.
static bool SimplifyCondBranchToCondBranch(BranchInst *PBI, BranchInst *BI) {
  assert(PBI->isConditional() && BI->isConditional());
  BasicBlock *BB = BI->getParent();

  // If this block ends with a branch instruction, and if there is a
  // predecessor that ends on a branch of the same condition, make 
  // this conditional branch redundant.
  if (PBI->getCondition() == BI->getCondition() &&
      PBI->getSuccessor(0) != PBI->getSuccessor(1)) {
    // Okay, the outcome of this conditional branch is statically
    // knowable.  If this block had a single pred, handle specially.
    if (BB->getSinglePredecessor()) {
      // Turn this into a branch on constant.
      bool CondIsTrue = PBI->getSuccessor(0) == BB;
      BI->setCondition(ConstantInt::get(Type::getInt1Ty(BB->getContext()), 
                                        CondIsTrue));
      return true;  // Nuke the branch on constant.
    }
    
    // Otherwise, if there are multiple predecessors, insert a PHI that merges
    // in the constant and simplify the block result.  Subsequent passes of
    // simplifycfg will thread the block.
    if (BlockIsSimpleEnoughToThreadThrough(BB)) {
      pred_iterator PB = pred_begin(BB), PE = pred_end(BB);
      PHINode *NewPN = PHINode::Create(Type::getInt1Ty(BB->getContext()),
                                       std::distance(PB, PE),
                                       BI->getCondition()->getName() + ".pr",
                                       BB->begin());
      // Okay, we're going to insert the PHI node.  Since PBI is not the only
      // predecessor, compute the PHI'd conditional value for all of the preds.
      // Any predecessor where the condition is not computable we keep symbolic.
      for (pred_iterator PI = PB; PI != PE; ++PI) {
        BasicBlock *P = *PI;
        if ((PBI = dyn_cast<BranchInst>(P->getTerminator())) &&
            PBI != BI && PBI->isConditional() &&
            PBI->getCondition() == BI->getCondition() &&
            PBI->getSuccessor(0) != PBI->getSuccessor(1)) {
          bool CondIsTrue = PBI->getSuccessor(0) == BB;
          NewPN->addIncoming(ConstantInt::get(Type::getInt1Ty(BB->getContext()), 
                                              CondIsTrue), P);
        } else {
          NewPN->addIncoming(BI->getCondition(), P);
        }
      }
      
      BI->setCondition(NewPN);
      return true;
    }
  }
  
  // If this is a conditional branch in an empty block, and if any
  // predecessors is a conditional branch to one of our destinations,
  // fold the conditions into logical ops and one cond br.
  BasicBlock::iterator BBI = BB->begin();
  // Ignore dbg intrinsics.
  while (isa<DbgInfoIntrinsic>(BBI))
    ++BBI;
  if (&*BBI != BI)
    return false;

  
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(BI->getCondition()))
    if (CE->canTrap())
      return false;
  
  int PBIOp, BIOp;
  if (PBI->getSuccessor(0) == BI->getSuccessor(0))
    PBIOp = BIOp = 0;
  else if (PBI->getSuccessor(0) == BI->getSuccessor(1))
    PBIOp = 0, BIOp = 1;
  else if (PBI->getSuccessor(1) == BI->getSuccessor(0))
    PBIOp = 1, BIOp = 0;
  else if (PBI->getSuccessor(1) == BI->getSuccessor(1))
    PBIOp = BIOp = 1;
  else
    return false;
    
  // Check to make sure that the other destination of this branch
  // isn't BB itself.  If so, this is an infinite loop that will
  // keep getting unwound.
  if (PBI->getSuccessor(PBIOp) == BB)
    return false;
    
  // Do not perform this transformation if it would require 
  // insertion of a large number of select instructions. For targets
  // without predication/cmovs, this is a big pessimization.
  BasicBlock *CommonDest = PBI->getSuccessor(PBIOp);
      
  unsigned NumPhis = 0;
  for (BasicBlock::iterator II = CommonDest->begin();
       isa<PHINode>(II); ++II, ++NumPhis)
    if (NumPhis > 2) // Disable this xform.
      return false;
    
  // Finally, if everything is ok, fold the branches to logical ops.
  BasicBlock *OtherDest  = BI->getSuccessor(BIOp ^ 1);
  
  DEBUG(dbgs() << "FOLDING BRs:" << *PBI->getParent()
               << "AND: " << *BI->getParent());
  
  
  // If OtherDest *is* BB, then BB is a basic block with a single conditional
  // branch in it, where one edge (OtherDest) goes back to itself but the other
  // exits.  We don't *know* that the program avoids the infinite loop
  // (even though that seems likely).  If we do this xform naively, we'll end up
  // recursively unpeeling the loop.  Since we know that (after the xform is
  // done) that the block *is* infinite if reached, we just make it an obviously
  // infinite loop with no cond branch.
  if (OtherDest == BB) {
    // Insert it at the end of the function, because it's either code,
    // or it won't matter if it's hot. :)
    BasicBlock *InfLoopBlock = BasicBlock::Create(BB->getContext(),
                                                  "infloop", BB->getParent());
    BranchInst::Create(InfLoopBlock, InfLoopBlock);
    OtherDest = InfLoopBlock;
  }  
  
  DEBUG(dbgs() << *PBI->getParent()->getParent());
  
  // BI may have other predecessors.  Because of this, we leave
  // it alone, but modify PBI.
  
  // Make sure we get to CommonDest on True&True directions.
  Value *PBICond = PBI->getCondition();
  if (PBIOp)
    PBICond = BinaryOperator::CreateNot(PBICond,
                                        PBICond->getName()+".not",
                                        PBI);
  Value *BICond = BI->getCondition();
  if (BIOp)
    BICond = BinaryOperator::CreateNot(BICond,
                                       BICond->getName()+".not",
                                       PBI);
  // Merge the conditions.
  Value *Cond = BinaryOperator::CreateOr(PBICond, BICond, "brmerge", PBI);
  
  // Modify PBI to branch on the new condition to the new dests.
  PBI->setCondition(Cond);
  PBI->setSuccessor(0, CommonDest);
  PBI->setSuccessor(1, OtherDest);
  
  // OtherDest may have phi nodes.  If so, add an entry from PBI's
  // block that are identical to the entries for BI's block.
  AddPredecessorToBlock(OtherDest, PBI->getParent(), BB);
  
  // We know that the CommonDest already had an edge from PBI to
  // it.  If it has PHIs though, the PHIs may have different
  // entries for BB and PBI's BB.  If so, insert a select to make
  // them agree.
  PHINode *PN;
  for (BasicBlock::iterator II = CommonDest->begin();
       (PN = dyn_cast<PHINode>(II)); ++II) {
    Value *BIV = PN->getIncomingValueForBlock(BB);
    unsigned PBBIdx = PN->getBasicBlockIndex(PBI->getParent());
    Value *PBIV = PN->getIncomingValue(PBBIdx);
    if (BIV != PBIV) {
      // Insert a select in PBI to pick the right value.
      Value *NV = SelectInst::Create(PBICond, PBIV, BIV,
                                     PBIV->getName()+".mux", PBI);
      PN->setIncomingValue(PBBIdx, NV);
    }
  }
  
  DEBUG(dbgs() << "INTO: " << *PBI->getParent());
  DEBUG(dbgs() << *PBI->getParent()->getParent());
  
  // This basic block is probably dead.  We know it has at least
  // one fewer predecessor.
  return true;
}

// SimplifyTerminatorOnSelect - Simplifies a terminator by replacing it with a
// branch to TrueBB if Cond is true or to FalseBB if Cond is false.
// Takes care of updating the successors and removing the old terminator.
// Also makes sure not to introduce new successors by assuming that edges to
// non-successor TrueBBs and FalseBBs aren't reachable.
static bool SimplifyTerminatorOnSelect(TerminatorInst *OldTerm, Value *Cond,
                                       BasicBlock *TrueBB, BasicBlock *FalseBB){
  // Remove any superfluous successor edges from the CFG.
  // First, figure out which successors to preserve.
  // If TrueBB and FalseBB are equal, only try to preserve one copy of that
  // successor.
  BasicBlock *KeepEdge1 = TrueBB;
  BasicBlock *KeepEdge2 = TrueBB != FalseBB ? FalseBB : 0;

  // Then remove the rest.
  for (unsigned I = 0, E = OldTerm->getNumSuccessors(); I != E; ++I) {
    BasicBlock *Succ = OldTerm->getSuccessor(I);
    // Make sure only to keep exactly one copy of each edge.
    if (Succ == KeepEdge1)
      KeepEdge1 = 0;
    else if (Succ == KeepEdge2)
      KeepEdge2 = 0;
    else
      Succ->removePredecessor(OldTerm->getParent());
  }

  // Insert an appropriate new terminator.
  if ((KeepEdge1 == 0) && (KeepEdge2 == 0)) {
    if (TrueBB == FalseBB)
      // We were only looking for one successor, and it was present.
      // Create an unconditional branch to it.
      BranchInst::Create(TrueBB, OldTerm);
    else
      // We found both of the successors we were looking for.
      // Create a conditional branch sharing the condition of the select.
      BranchInst::Create(TrueBB, FalseBB, Cond, OldTerm);
  } else if (KeepEdge1 && (KeepEdge2 || TrueBB == FalseBB)) {
    // Neither of the selected blocks were successors, so this
    // terminator must be unreachable.
    new UnreachableInst(OldTerm->getContext(), OldTerm);
  } else {
    // One of the selected values was a successor, but the other wasn't.
    // Insert an unconditional branch to the one that was found;
    // the edge to the one that wasn't must be unreachable.
    if (KeepEdge1 == 0)
      // Only TrueBB was found.
      BranchInst::Create(TrueBB, OldTerm);
    else
      // Only FalseBB was found.
      BranchInst::Create(FalseBB, OldTerm);
  }

  EraseTerminatorInstAndDCECond(OldTerm);
  return true;
}

// SimplifySwitchOnSelect - Replaces
//   (switch (select cond, X, Y)) on constant X, Y
// with a branch - conditional if X and Y lead to distinct BBs,
// unconditional otherwise.
static bool SimplifySwitchOnSelect(SwitchInst *SI, SelectInst *Select) {
  // Check for constant integer values in the select.
  ConstantInt *TrueVal = dyn_cast<ConstantInt>(Select->getTrueValue());
  ConstantInt *FalseVal = dyn_cast<ConstantInt>(Select->getFalseValue());
  if (!TrueVal || !FalseVal)
    return false;

  // Find the relevant condition and destinations.
  Value *Condition = Select->getCondition();
  BasicBlock *TrueBB = SI->getSuccessor(SI->findCaseValue(TrueVal));
  BasicBlock *FalseBB = SI->getSuccessor(SI->findCaseValue(FalseVal));

  // Perform the actual simplification.
  return SimplifyTerminatorOnSelect(SI, Condition, TrueBB, FalseBB);
}

// SimplifyIndirectBrOnSelect - Replaces
//   (indirectbr (select cond, blockaddress(@fn, BlockA),
//                             blockaddress(@fn, BlockB)))
// with
//   (br cond, BlockA, BlockB).
static bool SimplifyIndirectBrOnSelect(IndirectBrInst *IBI, SelectInst *SI) {
  // Check that both operands of the select are block addresses.
  BlockAddress *TBA = dyn_cast<BlockAddress>(SI->getTrueValue());
  BlockAddress *FBA = dyn_cast<BlockAddress>(SI->getFalseValue());
  if (!TBA || !FBA)
    return false;

  // Extract the actual blocks.
  BasicBlock *TrueBB = TBA->getBasicBlock();
  BasicBlock *FalseBB = FBA->getBasicBlock();

  // Perform the actual simplification.
  return SimplifyTerminatorOnSelect(IBI, SI->getCondition(), TrueBB, FalseBB);
}

/// TryToSimplifyUncondBranchWithICmpInIt - This is called when we find an icmp
/// instruction (a seteq/setne with a constant) as the only instruction in a
/// block that ends with an uncond branch.  We are looking for a very specific
/// pattern that occurs when "A == 1 || A == 2 || A == 3" gets simplified.  In
/// this case, we merge the first two "or's of icmp" into a switch, but then the
/// default value goes to an uncond block with a seteq in it, we get something
/// like:
///
///   switch i8 %A, label %DEFAULT [ i8 1, label %end    i8 2, label %end ]
/// DEFAULT:
///   %tmp = icmp eq i8 %A, 92
///   br label %end
/// end:
///   ... = phi i1 [ true, %entry ], [ %tmp, %DEFAULT ], [ true, %entry ]
/// 
/// We prefer to split the edge to 'end' so that there is a true/false entry to
/// the PHI, merging the third icmp into the switch.
static bool TryToSimplifyUncondBranchWithICmpInIt(ICmpInst *ICI,
                                                  const TargetData *TD) {
  BasicBlock *BB = ICI->getParent();
  // If the block has any PHIs in it or the icmp has multiple uses, it is too
  // complex.
  if (isa<PHINode>(BB->begin()) || !ICI->hasOneUse()) return false;

  Value *V = ICI->getOperand(0);
  ConstantInt *Cst = cast<ConstantInt>(ICI->getOperand(1));
  
  // The pattern we're looking for is where our only predecessor is a switch on
  // 'V' and this block is the default case for the switch.  In this case we can
  // fold the compared value into the switch to simplify things.
  BasicBlock *Pred = BB->getSinglePredecessor();
  if (Pred == 0 || !isa<SwitchInst>(Pred->getTerminator())) return false;
  
  SwitchInst *SI = cast<SwitchInst>(Pred->getTerminator());
  if (SI->getCondition() != V)
    return false;
  
  // If BB is reachable on a non-default case, then we simply know the value of
  // V in this block.  Substitute it and constant fold the icmp instruction
  // away.
  if (SI->getDefaultDest() != BB) {
    ConstantInt *VVal = SI->findCaseDest(BB);
    assert(VVal && "Should have a unique destination value");
    ICI->setOperand(0, VVal);
    
    if (Value *V = SimplifyInstruction(ICI, TD)) {
      ICI->replaceAllUsesWith(V);
      ICI->eraseFromParent();
    }
    // BB is now empty, so it is likely to simplify away.
    return SimplifyCFG(BB) | true;
  }
  
  // Ok, the block is reachable from the default dest.  If the constant we're
  // comparing exists in one of the other edges, then we can constant fold ICI
  // and zap it.
  if (SI->findCaseValue(Cst) != 0) {
    Value *V;
    if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
      V = ConstantInt::getFalse(BB->getContext());
    else
      V = ConstantInt::getTrue(BB->getContext());
    
    ICI->replaceAllUsesWith(V);
    ICI->eraseFromParent();
    // BB is now empty, so it is likely to simplify away.
    return SimplifyCFG(BB) | true;
  }
  
  // The use of the icmp has to be in the 'end' block, by the only PHI node in
  // the block.
  BasicBlock *SuccBlock = BB->getTerminator()->getSuccessor(0);
  PHINode *PHIUse = dyn_cast<PHINode>(ICI->use_back());
  if (PHIUse == 0 || PHIUse != &SuccBlock->front() ||
      isa<PHINode>(++BasicBlock::iterator(PHIUse)))
    return false;

  // If the icmp is a SETEQ, then the default dest gets false, the new edge gets
  // true in the PHI.
  Constant *DefaultCst = ConstantInt::getTrue(BB->getContext());
  Constant *NewCst     = ConstantInt::getFalse(BB->getContext());

  if (ICI->getPredicate() == ICmpInst::ICMP_EQ)
    std::swap(DefaultCst, NewCst);

  // Replace ICI (which is used by the PHI for the default value) with true or
  // false depending on if it is EQ or NE.
  ICI->replaceAllUsesWith(DefaultCst);
  ICI->eraseFromParent();

  // Okay, the switch goes to this block on a default value.  Add an edge from
  // the switch to the merge point on the compared value.
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), "switch.edge",
                                         BB->getParent(), BB);
  SI->addCase(Cst, NewBB);
  
  // NewBB branches to the phi block, add the uncond branch and the phi entry.
  BranchInst::Create(SuccBlock, NewBB);
  PHIUse->addIncoming(NewCst, NewBB);
  return true;
}

/// SimplifyBranchOnICmpChain - The specified branch is a conditional branch.
/// Check to see if it is branching on an or/and chain of icmp instructions, and
/// fold it into a switch instruction if so.
static bool SimplifyBranchOnICmpChain(BranchInst *BI, const TargetData *TD) {
  Instruction *Cond = dyn_cast<Instruction>(BI->getCondition());
  if (Cond == 0) return false;
  
  
  // Change br (X == 0 | X == 1), T, F into a switch instruction.
  // If this is a bunch of seteq's or'd together, or if it's a bunch of
  // 'setne's and'ed together, collect them.
  Value *CompVal = 0;
  std::vector<ConstantInt*> Values;
  bool TrueWhenEqual = true;
  Value *ExtraCase = 0;
  unsigned UsedICmps = 0;
  
  if (Cond->getOpcode() == Instruction::Or) {
    CompVal = GatherConstantCompares(Cond, Values, ExtraCase, TD, true,
                                     UsedICmps);
  } else if (Cond->getOpcode() == Instruction::And) {
    CompVal = GatherConstantCompares(Cond, Values, ExtraCase, TD, false,
                                     UsedICmps);
    TrueWhenEqual = false;
  }
  
  // If we didn't have a multiply compared value, fail.
  if (CompVal == 0) return false;

  // Avoid turning single icmps into a switch.
  if (UsedICmps <= 1)
    return false;

  // There might be duplicate constants in the list, which the switch
  // instruction can't handle, remove them now.
  array_pod_sort(Values.begin(), Values.end(), ConstantIntSortPredicate);
  Values.erase(std::unique(Values.begin(), Values.end()), Values.end());
  
  // If Extra was used, we require at least two switch values to do the
  // transformation.  A switch with one value is just an cond branch.
  if (ExtraCase && Values.size() < 2) return false;
  
  // Figure out which block is which destination.
  BasicBlock *DefaultBB = BI->getSuccessor(1);
  BasicBlock *EdgeBB    = BI->getSuccessor(0);
  if (!TrueWhenEqual) std::swap(DefaultBB, EdgeBB);
  
  BasicBlock *BB = BI->getParent();
  
  DEBUG(dbgs() << "Converting 'icmp' chain with " << Values.size()
               << " cases into SWITCH.  BB is:\n" << *BB);
  
  // If there are any extra values that couldn't be folded into the switch
  // then we evaluate them with an explicit branch first.  Split the block
  // right before the condbr to handle it.
  if (ExtraCase) {
    BasicBlock *NewBB = BB->splitBasicBlock(BI, "switch.early.test");
    // Remove the uncond branch added to the old block.
    TerminatorInst *OldTI = BB->getTerminator();
    
    if (TrueWhenEqual)
      BranchInst::Create(EdgeBB, NewBB, ExtraCase, OldTI);
    else
      BranchInst::Create(NewBB, EdgeBB, ExtraCase, OldTI);
      
    OldTI->eraseFromParent();
    
    // If there are PHI nodes in EdgeBB, then we need to add a new entry to them
    // for the edge we just added.
    AddPredecessorToBlock(EdgeBB, BB, NewBB);
    
    DEBUG(dbgs() << "  ** 'icmp' chain unhandled condition: " << *ExtraCase
          << "\nEXTRABB = " << *BB);
    BB = NewBB;
  }
  
  // Convert pointer to int before we switch.
  if (CompVal->getType()->isPointerTy()) {
    assert(TD && "Cannot switch on pointer without TargetData");
    CompVal = new PtrToIntInst(CompVal,
                               TD->getIntPtrType(CompVal->getContext()),
                               "magicptr", BI);
  }
  
  // Create the new switch instruction now.
  SwitchInst *New = SwitchInst::Create(CompVal, DefaultBB, Values.size(), BI);
  
  // Add all of the 'cases' to the switch instruction.
  for (unsigned i = 0, e = Values.size(); i != e; ++i)
    New->addCase(Values[i], EdgeBB);
  
  // We added edges from PI to the EdgeBB.  As such, if there were any
  // PHI nodes in EdgeBB, they need entries to be added corresponding to
  // the number of edges added.
  for (BasicBlock::iterator BBI = EdgeBB->begin();
       isa<PHINode>(BBI); ++BBI) {
    PHINode *PN = cast<PHINode>(BBI);
    Value *InVal = PN->getIncomingValueForBlock(BB);
    for (unsigned i = 0, e = Values.size()-1; i != e; ++i)
      PN->addIncoming(InVal, BB);
  }
  
  // Erase the old branch instruction.
  EraseTerminatorInstAndDCECond(BI);
  
  DEBUG(dbgs() << "  ** 'icmp' chain result is:\n" << *BB << '\n');
  return true;
}

bool SimplifyCFGOpt::SimplifyReturn(ReturnInst *RI) {
  BasicBlock *BB = RI->getParent();
  if (!BB->getFirstNonPHIOrDbg()->isTerminator()) return false;
  
  // Find predecessors that end with branches.
  SmallVector<BasicBlock*, 8> UncondBranchPreds;
  SmallVector<BranchInst*, 8> CondBranchPreds;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    BasicBlock *P = *PI;
    TerminatorInst *PTI = P->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(PTI)) {
      if (BI->isUnconditional())
        UncondBranchPreds.push_back(P);
      else
        CondBranchPreds.push_back(BI);
    }
  }
  
  // If we found some, do the transformation!
  if (!UncondBranchPreds.empty() && DupRet) {
    while (!UncondBranchPreds.empty()) {
      BasicBlock *Pred = UncondBranchPreds.pop_back_val();
      DEBUG(dbgs() << "FOLDING: " << *BB
            << "INTO UNCOND BRANCH PRED: " << *Pred);
      (void)FoldReturnIntoUncondBranch(RI, BB, Pred);
    }
    
    // If we eliminated all predecessors of the block, delete the block now.
    if (pred_begin(BB) == pred_end(BB))
      // We know there are no successors, so just nuke the block.
      BB->eraseFromParent();
    
    return true;
  }
  
  // Check out all of the conditional branches going to this return
  // instruction.  If any of them just select between returns, change the
  // branch itself into a select/return pair.
  while (!CondBranchPreds.empty()) {
    BranchInst *BI = CondBranchPreds.pop_back_val();
    
    // Check to see if the non-BB successor is also a return block.
    if (isa<ReturnInst>(BI->getSuccessor(0)->getTerminator()) &&
        isa<ReturnInst>(BI->getSuccessor(1)->getTerminator()) &&
        SimplifyCondBranchToTwoReturns(BI))
      return true;
  }
  return false;
}

bool SimplifyCFGOpt::SimplifyUnwind(UnwindInst *UI) {
  // Check to see if the first instruction in this block is just an unwind.
  // If so, replace any invoke instructions which use this as an exception
  // destination with call instructions.
  BasicBlock *BB = UI->getParent();
  if (!BB->getFirstNonPHIOrDbg()->isTerminator()) return false;

  bool Changed = false;
  SmallVector<BasicBlock*, 8> Preds(pred_begin(BB), pred_end(BB));
  while (!Preds.empty()) {
    BasicBlock *Pred = Preds.back();
    InvokeInst *II = dyn_cast<InvokeInst>(Pred->getTerminator());
    if (II && II->getUnwindDest() == BB) {
      // Insert a new branch instruction before the invoke, because this
      // is now a fall through.
      BranchInst *BI = BranchInst::Create(II->getNormalDest(), II);
      Pred->getInstList().remove(II);   // Take out of symbol table
      
      // Insert the call now.
      SmallVector<Value*,8> Args(II->op_begin(), II->op_end()-3);
      CallInst *CI = CallInst::Create(II->getCalledValue(),
                                      Args.begin(), Args.end(),
                                      II->getName(), BI);
      CI->setCallingConv(II->getCallingConv());
      CI->setAttributes(II->getAttributes());
      // If the invoke produced a value, the Call now does instead.
      II->replaceAllUsesWith(CI);
      delete II;
      Changed = true;
    }
    
    Preds.pop_back();
  }
  
  // If this block is now dead (and isn't the entry block), remove it.
  if (pred_begin(BB) == pred_end(BB) &&
      BB != &BB->getParent()->getEntryBlock()) {
    // We know there are no successors, so just nuke the block.
    BB->eraseFromParent();
    return true;
  }
  
  return Changed;  
}

bool SimplifyCFGOpt::SimplifyUnreachable(UnreachableInst *UI) {
  BasicBlock *BB = UI->getParent();
  
  bool Changed = false;
  
  // If there are any instructions immediately before the unreachable that can
  // be removed, do so.
  while (UI != BB->begin()) {
    BasicBlock::iterator BBI = UI;
    --BBI;
    // Do not delete instructions that can have side effects, like calls
    // (which may never return) and volatile loads and stores.
    if (isa<CallInst>(BBI) && !isa<DbgInfoIntrinsic>(BBI)) break;
    
    if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
      if (SI->isVolatile())
        break;
    
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI))
      if (LI->isVolatile())
        break;
    
    // Delete this instruction (any uses are guaranteed to be dead)
    if (!BBI->use_empty())
      BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
    BBI->eraseFromParent();
    Changed = true;
  }
  
  // If the unreachable instruction is the first in the block, take a gander
  // at all of the predecessors of this instruction, and simplify them.
  if (&BB->front() != UI) return Changed;
  
  SmallVector<BasicBlock*, 8> Preds(pred_begin(BB), pred_end(BB));
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    TerminatorInst *TI = Preds[i]->getTerminator();
    
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (BI->isUnconditional()) {
        if (BI->getSuccessor(0) == BB) {
          new UnreachableInst(TI->getContext(), TI);
          TI->eraseFromParent();
          Changed = true;
        }
      } else {
        if (BI->getSuccessor(0) == BB) {
          BranchInst::Create(BI->getSuccessor(1), BI);
          EraseTerminatorInstAndDCECond(BI);
        } else if (BI->getSuccessor(1) == BB) {
          BranchInst::Create(BI->getSuccessor(0), BI);
          EraseTerminatorInstAndDCECond(BI);
          Changed = true;
        }
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      for (unsigned i = 1, e = SI->getNumCases(); i != e; ++i)
        if (SI->getSuccessor(i) == BB) {
          BB->removePredecessor(SI->getParent());
          SI->removeCase(i);
          --i; --e;
          Changed = true;
        }
      // If the default value is unreachable, figure out the most popular
      // destination and make it the default.
      if (SI->getSuccessor(0) == BB) {
        std::map<BasicBlock*, std::pair<unsigned, unsigned> > Popularity;
        for (unsigned i = 1, e = SI->getNumCases(); i != e; ++i) {
          std::pair<unsigned, unsigned>& entry =
              Popularity[SI->getSuccessor(i)];
          if (entry.first == 0) {
            entry.first = 1;
            entry.second = i;
          } else {
            entry.first++;
          }
        }

        // Find the most popular block.
        unsigned MaxPop = 0;
        unsigned MaxIndex = 0;
        BasicBlock *MaxBlock = 0;
        for (std::map<BasicBlock*, std::pair<unsigned, unsigned> >::iterator
             I = Popularity.begin(), E = Popularity.end(); I != E; ++I) {
          if (I->second.first > MaxPop || 
              (I->second.first == MaxPop && MaxIndex > I->second.second)) {
            MaxPop = I->second.first;
            MaxIndex = I->second.second;
            MaxBlock = I->first;
          }
        }
        if (MaxBlock) {
          // Make this the new default, allowing us to delete any explicit
          // edges to it.
          SI->setSuccessor(0, MaxBlock);
          Changed = true;
          
          // If MaxBlock has phinodes in it, remove MaxPop-1 entries from
          // it.
          if (isa<PHINode>(MaxBlock->begin()))
            for (unsigned i = 0; i != MaxPop-1; ++i)
              MaxBlock->removePredecessor(SI->getParent());
          
          for (unsigned i = 1, e = SI->getNumCases(); i != e; ++i)
            if (SI->getSuccessor(i) == MaxBlock) {
              SI->removeCase(i);
              --i; --e;
            }
        }
      }
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(TI)) {
      if (II->getUnwindDest() == BB) {
        // Convert the invoke to a call instruction.  This would be a good
        // place to note that the call does not throw though.
        BranchInst *BI = BranchInst::Create(II->getNormalDest(), II);
        II->removeFromParent();   // Take out of symbol table
        
        // Insert the call now...
        SmallVector<Value*, 8> Args(II->op_begin(), II->op_end()-3);
        CallInst *CI = CallInst::Create(II->getCalledValue(),
                                        Args.begin(), Args.end(),
                                        II->getName(), BI);
        CI->setCallingConv(II->getCallingConv());
        CI->setAttributes(II->getAttributes());
        // If the invoke produced a value, the call does now instead.
        II->replaceAllUsesWith(CI);
        delete II;
        Changed = true;
      }
    }
  }
  
  // If this block is now dead, remove it.
  if (pred_begin(BB) == pred_end(BB) &&
      BB != &BB->getParent()->getEntryBlock()) {
    // We know there are no successors, so just nuke the block.
    BB->eraseFromParent();
    return true;
  }

  return Changed;
}

/// TurnSwitchRangeIntoICmp - Turns a switch with that contains only a
/// integer range comparison into a sub, an icmp and a branch.
static bool TurnSwitchRangeIntoICmp(SwitchInst *SI) {
  assert(SI->getNumCases() > 2 && "Degenerate switch?");

  // Make sure all cases point to the same destination and gather the values.
  SmallVector<ConstantInt *, 16> Cases;
  Cases.push_back(SI->getCaseValue(1));
  for (unsigned I = 2, E = SI->getNumCases(); I != E; ++I) {
    if (SI->getSuccessor(I-1) != SI->getSuccessor(I))
      return false;
    Cases.push_back(SI->getCaseValue(I));
  }
  assert(Cases.size() == SI->getNumCases()-1 && "Not all cases gathered");

  // Sort the case values, then check if they form a range we can transform.
  array_pod_sort(Cases.begin(), Cases.end(), ConstantIntSortPredicate);
  for (unsigned I = 1, E = Cases.size(); I != E; ++I) {
    if (Cases[I-1]->getValue() != Cases[I]->getValue()+1)
      return false;
  }

  Constant *Offset = ConstantExpr::getNeg(Cases.back());
  Constant *NumCases = ConstantInt::get(Offset->getType(), SI->getNumCases()-1);

  Value *Sub = SI->getCondition();
  if (!Offset->isNullValue())
    Sub = BinaryOperator::CreateAdd(Sub, Offset, Sub->getName()+".off", SI);
  Value *Cmp = new ICmpInst(SI, ICmpInst::ICMP_ULT, Sub, NumCases, "switch");
  BranchInst::Create(SI->getSuccessor(1), SI->getDefaultDest(), Cmp, SI);

  // Prune obsolete incoming values off the successor's PHI nodes.
  for (BasicBlock::iterator BBI = SI->getSuccessor(1)->begin();
       isa<PHINode>(BBI); ++BBI) {
    for (unsigned I = 0, E = SI->getNumCases()-2; I != E; ++I)
      cast<PHINode>(BBI)->removeIncomingValue(SI->getParent());
  }
  SI->eraseFromParent();

  return true;
}

bool SimplifyCFGOpt::SimplifySwitch(SwitchInst *SI) {
  // If this switch is too complex to want to look at, ignore it.
  if (!isValueEqualityComparison(SI))
    return false;

  BasicBlock *BB = SI->getParent();

  // If we only have one predecessor, and if it is a branch on this value,
  // see if that predecessor totally determines the outcome of this switch.
  if (BasicBlock *OnlyPred = BB->getSinglePredecessor())
    if (SimplifyEqualityComparisonWithOnlyPredecessor(SI, OnlyPred))
      return SimplifyCFG(BB) | true;

  Value *Cond = SI->getCondition();
  if (SelectInst *Select = dyn_cast<SelectInst>(Cond))
    if (SimplifySwitchOnSelect(SI, Select))
      return SimplifyCFG(BB) | true;

  // If the block only contains the switch, see if we can fold the block
  // away into any preds.
  BasicBlock::iterator BBI = BB->begin();
  // Ignore dbg intrinsics.
  while (isa<DbgInfoIntrinsic>(BBI))
    ++BBI;
  if (SI == &*BBI)
    if (FoldValueComparisonIntoPredecessors(SI))
      return SimplifyCFG(BB) | true;

  // Try to transform the switch into an icmp and a branch.
  if (TurnSwitchRangeIntoICmp(SI))
    return SimplifyCFG(BB) | true;
  
  return false;
}

bool SimplifyCFGOpt::SimplifyIndirectBr(IndirectBrInst *IBI) {
  BasicBlock *BB = IBI->getParent();
  bool Changed = false;
  
  // Eliminate redundant destinations.
  SmallPtrSet<Value *, 8> Succs;
  for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
    BasicBlock *Dest = IBI->getDestination(i);
    if (!Dest->hasAddressTaken() || !Succs.insert(Dest)) {
      Dest->removePredecessor(BB);
      IBI->removeDestination(i);
      --i; --e;
      Changed = true;
    }
  } 

  if (IBI->getNumDestinations() == 0) {
    // If the indirectbr has no successors, change it to unreachable.
    new UnreachableInst(IBI->getContext(), IBI);
    EraseTerminatorInstAndDCECond(IBI);
    return true;
  }
  
  if (IBI->getNumDestinations() == 1) {
    // If the indirectbr has one successor, change it to a direct branch.
    BranchInst::Create(IBI->getDestination(0), IBI);
    EraseTerminatorInstAndDCECond(IBI);
    return true;
  }
  
  if (SelectInst *SI = dyn_cast<SelectInst>(IBI->getAddress())) {
    if (SimplifyIndirectBrOnSelect(IBI, SI))
      return SimplifyCFG(BB) | true;
  }
  return Changed;
}

bool SimplifyCFGOpt::SimplifyUncondBranch(BranchInst *BI) {
  BasicBlock *BB = BI->getParent();
  
  // If the Terminator is the only non-phi instruction, simplify the block.
  BasicBlock::iterator I = BB->getFirstNonPHIOrDbg();
  if (I->isTerminator() && BB != &BB->getParent()->getEntryBlock() &&
      TryToSimplifyUncondBranchFromEmptyBlock(BB))
    return true;
  
  // If the only instruction in the block is a seteq/setne comparison
  // against a constant, try to simplify the block.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(I))
    if (ICI->isEquality() && isa<ConstantInt>(ICI->getOperand(1))) {
      for (++I; isa<DbgInfoIntrinsic>(I); ++I)
        ;
      if (I->isTerminator() && TryToSimplifyUncondBranchWithICmpInIt(ICI, TD))
        return true;
    }
  
  return false;
}


bool SimplifyCFGOpt::SimplifyCondBranch(BranchInst *BI) {
  BasicBlock *BB = BI->getParent();
  
  // Conditional branch
  if (isValueEqualityComparison(BI)) {
    // If we only have one predecessor, and if it is a branch on this value,
    // see if that predecessor totally determines the outcome of this
    // switch.
    if (BasicBlock *OnlyPred = BB->getSinglePredecessor())
      if (SimplifyEqualityComparisonWithOnlyPredecessor(BI, OnlyPred))
        return SimplifyCFG(BB) | true;
    
    // This block must be empty, except for the setcond inst, if it exists.
    // Ignore dbg intrinsics.
    BasicBlock::iterator I = BB->begin();
    // Ignore dbg intrinsics.
    while (isa<DbgInfoIntrinsic>(I))
      ++I;
    if (&*I == BI) {
      if (FoldValueComparisonIntoPredecessors(BI))
        return SimplifyCFG(BB) | true;
    } else if (&*I == cast<Instruction>(BI->getCondition())){
      ++I;
      // Ignore dbg intrinsics.
      while (isa<DbgInfoIntrinsic>(I))
        ++I;
      if (&*I == BI && FoldValueComparisonIntoPredecessors(BI))
        return SimplifyCFG(BB) | true;
    }
  }
  
  // Try to turn "br (X == 0 | X == 1), T, F" into a switch instruction.
  if (SimplifyBranchOnICmpChain(BI, TD))
    return true;
  
  // We have a conditional branch to two blocks that are only reachable
  // from BI.  We know that the condbr dominates the two blocks, so see if
  // there is any identical code in the "then" and "else" blocks.  If so, we
  // can hoist it up to the branching block.
  if (BI->getSuccessor(0)->getSinglePredecessor() != 0) {
    if (BI->getSuccessor(1)->getSinglePredecessor() != 0) {
      if (HoistThenElseCodeToIf(BI))
        return SimplifyCFG(BB) | true;
    } else {
      // If Successor #1 has multiple preds, we may be able to conditionally
      // execute Successor #0 if it branches to successor #1.
      TerminatorInst *Succ0TI = BI->getSuccessor(0)->getTerminator();
      if (Succ0TI->getNumSuccessors() == 1 &&
          Succ0TI->getSuccessor(0) == BI->getSuccessor(1))
        if (SpeculativelyExecuteBB(BI, BI->getSuccessor(0)))
          return SimplifyCFG(BB) | true;
    }
  } else if (BI->getSuccessor(1)->getSinglePredecessor() != 0) {
    // If Successor #0 has multiple preds, we may be able to conditionally
    // execute Successor #1 if it branches to successor #0.
    TerminatorInst *Succ1TI = BI->getSuccessor(1)->getTerminator();
    if (Succ1TI->getNumSuccessors() == 1 &&
        Succ1TI->getSuccessor(0) == BI->getSuccessor(0))
      if (SpeculativelyExecuteBB(BI, BI->getSuccessor(1)))
        return SimplifyCFG(BB) | true;
  }
  
  // If this is a branch on a phi node in the current block, thread control
  // through this block if any PHI node entries are constants.
  if (PHINode *PN = dyn_cast<PHINode>(BI->getCondition()))
    if (PN->getParent() == BI->getParent())
      if (FoldCondBranchOnPHI(BI, TD))
        return SimplifyCFG(BB) | true;
  
  // If this basic block is ONLY a setcc and a branch, and if a predecessor
  // branches to us and one of our successors, fold the setcc into the
  // predecessor and use logical operations to pick the right destination.
  if (FoldBranchToCommonDest(BI))
    return SimplifyCFG(BB) | true;
  
  // Scan predecessor blocks for conditional branches.
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    if (BranchInst *PBI = dyn_cast<BranchInst>((*PI)->getTerminator()))
      if (PBI != BI && PBI->isConditional())
        if (SimplifyCondBranchToCondBranch(PBI, BI))
          return SimplifyCFG(BB) | true;

  return false;
}

bool SimplifyCFGOpt::run(BasicBlock *BB) {
  bool Changed = false;

  assert(BB && BB->getParent() && "Block not embedded in function!");
  assert(BB->getTerminator() && "Degenerate basic block encountered!");

  // Remove basic blocks that have no predecessors (except the entry block)...
  // or that just have themself as a predecessor.  These are unreachable.
  if ((pred_begin(BB) == pred_end(BB) &&
       BB != &BB->getParent()->getEntryBlock()) ||
      BB->getSinglePredecessor() == BB) {
    DEBUG(dbgs() << "Removing BB: \n" << *BB);
    DeleteDeadBlock(BB);
    return true;
  }

  // Check to see if we can constant propagate this terminator instruction
  // away...
  Changed |= ConstantFoldTerminator(BB);

  // Check for and eliminate duplicate PHI nodes in this block.
  Changed |= EliminateDuplicatePHINodes(BB);

  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  //
  if (MergeBlockIntoPredecessor(BB))
    return true;
  
  // If there is a trivial two-entry PHI node in this basic block, and we can
  // eliminate it, do so now.
  if (PHINode *PN = dyn_cast<PHINode>(BB->begin()))
    if (PN->getNumIncomingValues() == 2)
      Changed |= FoldTwoEntryPHINode(PN, TD);

  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
    if (BI->isUnconditional()) {
      if (SimplifyUncondBranch(BI)) return true;
    } else {
      if (SimplifyCondBranch(BI)) return true;
    }
  } else if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
    if (SimplifyReturn(RI)) return true;
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
    if (SimplifySwitch(SI)) return true;
  } else if (UnreachableInst *UI =
               dyn_cast<UnreachableInst>(BB->getTerminator())) {
    if (SimplifyUnreachable(UI)) return true;
  } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
    if (SimplifyUnwind(UI)) return true;
  } else if (IndirectBrInst *IBI =
               dyn_cast<IndirectBrInst>(BB->getTerminator())) {
    if (SimplifyIndirectBr(IBI)) return true;
  }

  return Changed;
}

/// SimplifyCFG - This function is used to do simplification of a CFG.  For
/// example, it adjusts branches to branches to eliminate the extra hop, it
/// eliminates unreachable basic blocks, and does other "peephole" optimization
/// of the CFG.  It returns true if a modification was made.
///
bool llvm::SimplifyCFG(BasicBlock *BB, const TargetData *TD) {
  return SimplifyCFGOpt(TD).run(BB);
}
