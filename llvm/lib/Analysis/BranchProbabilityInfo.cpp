//===-- BranchProbabilityInfo.cpp - Branch Probability Analysis -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include <climits>

using namespace llvm;

INITIALIZE_PASS_BEGIN(BranchProbabilityInfo, "branch-prob",
                      "Branch Probability Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(BranchProbabilityInfo, "branch-prob",
                    "Branch Probability Analysis", false, true)

char BranchProbabilityInfo::ID = 0;


// Please note that BranchProbabilityAnalysis is not a FunctionPass.
// It is created by BranchProbabilityInfo (which is a FunctionPass), which
// provides a clear interface. Thanks to that, all heuristics and other
// private methods are hidden in the .cpp file.
class BranchProbabilityAnalysis {

  typedef std::pair<BasicBlock *, BasicBlock *> Edge;

  DenseMap<Edge, unsigned> *Weights;

  BranchProbabilityInfo *BP;

  LoopInfo *LI;


  // Weights are for internal use only. They are used by heuristics to help to
  // estimate edges' probability. Example:
  //
  // Using "Loop Branch Heuristics" we predict weights of edges for the
  // block BB2.
  //         ...
  //          |
  //          V
  //         BB1<-+
  //          |   |
  //          |   | (Weight = 128)
  //          V   |
  //         BB2--+
  //          |
  //          | (Weight = 4)
  //          V
  //         BB3
  //
  // Probability of the edge BB2->BB1 = 128 / (128 + 4) = 0.9696..
  // Probability of the edge BB2->BB3 = 4 / (128 + 4) = 0.0303..

  static const unsigned int LBH_TAKEN_WEIGHT = 128;
  static const unsigned int LBH_NONTAKEN_WEIGHT = 4;

  // Standard weight value. Used when none of the heuristics set weight for
  // the edge.
  static const unsigned int NORMAL_WEIGHT = 16;

  // Minimum weight of an edge. Please note, that weight is NEVER 0.
  static const unsigned int MIN_WEIGHT = 1;

  // Return TRUE if BB leads directly to a Return Instruction.
  static bool isReturningBlock(BasicBlock *BB) {
    SmallPtrSet<BasicBlock *, 8> Visited;

    while (true) {
      TerminatorInst *TI = BB->getTerminator();
      if (isa<ReturnInst>(TI))
        return true;

      if (TI->getNumSuccessors() > 1)
        break;

      // It is unreachable block which we can consider as a return instruction.
      if (TI->getNumSuccessors() == 0)
        return true;

      Visited.insert(BB);
      BB = TI->getSuccessor(0);

      // Stop if cycle is detected.
      if (Visited.count(BB))
        return false;
    }

    return false;
  }

  // Multiply Edge Weight by two.
  void incEdgeWeight(BasicBlock *Src, BasicBlock *Dst) {
    unsigned Weight = BP->getEdgeWeight(Src, Dst);
    unsigned MaxWeight = getMaxWeightFor(Src);

    if (Weight * 2 > MaxWeight)
      BP->setEdgeWeight(Src, Dst, MaxWeight);
    else
      BP->setEdgeWeight(Src, Dst, Weight * 2);
  }

  // Divide Edge Weight by two.
  void decEdgeWeight(BasicBlock *Src, BasicBlock *Dst) {
    unsigned Weight = BP->getEdgeWeight(Src, Dst);

    assert(Weight > 0);
    if (Weight / 2 < MIN_WEIGHT)
      BP->setEdgeWeight(Src, Dst, MIN_WEIGHT);
    else
      BP->setEdgeWeight(Src, Dst, Weight / 2);
  }


  unsigned getMaxWeightFor(BasicBlock *BB) const {
    return UINT_MAX / BB->getTerminator()->getNumSuccessors();
  }

public:
  BranchProbabilityAnalysis(DenseMap<Edge, unsigned> *W,
                            BranchProbabilityInfo *BP, LoopInfo *LI)
    : Weights(W), BP(BP), LI(LI) {
  }

  // Return Heuristics
  void calcReturnHeuristics(BasicBlock *BB);

  // Pointer Heuristics
  void calcPointerHeuristics(BasicBlock *BB);

  // Loop Branch Heuristics
  void calcLoopBranchHeuristics(BasicBlock *BB);

  bool runOnFunction(Function &F);
};

// Calculate Edge Weights using "Return Heuristics". Predict a successor which
// leads directly to Return Instruction will not be taken.
void BranchProbabilityAnalysis::calcReturnHeuristics(BasicBlock *BB){
  if (BB->getTerminator()->getNumSuccessors() == 1)
    return;

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    BasicBlock *Succ = *I;
    if (isReturningBlock(Succ)) {
      decEdgeWeight(BB, Succ);
    }
  }
}

// Calculate Edge Weights using "Pointer Heuristics". Predict a comparsion
// between two pointer or pointer and NULL will fail.
void BranchProbabilityAnalysis::calcPointerHeuristics(BasicBlock *BB) {
  BranchInst * BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isConditional())
    return;

  Value *Cond = BI->getCondition();
  ICmpInst *CI = dyn_cast<ICmpInst>(Cond);
  if (!CI)
    return;

  Value *LHS = CI->getOperand(0);
  Value *RHS = CI->getOperand(1);

  if (!LHS->getType()->isPointerTy())
    return;

  assert(RHS->getType()->isPointerTy());

  BasicBlock *Taken = BI->getSuccessor(0);
  BasicBlock *NonTaken = BI->getSuccessor(1);

  // p != 0   ->   isProb = true
  // p == 0   ->   isProb = false
  // p != q   ->   isProb = true
  // p == q   ->   isProb = false;
  bool isProb = !CI->isEquality();
  if (!isProb)
    std::swap(Taken, NonTaken);

  incEdgeWeight(BB, Taken);
  decEdgeWeight(BB, NonTaken);
}

// Calculate Edge Weights using "Loop Branch Heuristics". Predict backedges
// as taken, exiting edges as not-taken.
void BranchProbabilityAnalysis::calcLoopBranchHeuristics(BasicBlock *BB) {
  unsigned numSuccs = BB->getTerminator()->getNumSuccessors();

  Loop *L = LI->getLoopFor(BB);
  if (!L)
    return;

  SmallVector<BasicBlock *, 8> BackEdges;
  SmallVector<BasicBlock *, 8> ExitingEdges;

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    BasicBlock *Succ = *I;
    Loop *SuccL = LI->getLoopFor(Succ);
    if (SuccL != L)
      ExitingEdges.push_back(Succ);
    else if (Succ == L->getHeader())
      BackEdges.push_back(Succ);
  }

  if (unsigned numBackEdges = BackEdges.size()) {
    unsigned backWeight = LBH_TAKEN_WEIGHT / numBackEdges;
    if (backWeight < NORMAL_WEIGHT)
      backWeight = NORMAL_WEIGHT;

    for (SmallVector<BasicBlock *, 8>::iterator EI = BackEdges.begin(),
         EE = BackEdges.end(); EI != EE; ++EI) {
      BasicBlock *Back = *EI;
      BP->setEdgeWeight(BB, Back, backWeight);
    }
  }

  unsigned numExitingEdges = ExitingEdges.size();
  if (unsigned numNonExitingEdges = numSuccs - numExitingEdges) {
    unsigned exitWeight = LBH_NONTAKEN_WEIGHT / numNonExitingEdges;
    if (exitWeight < MIN_WEIGHT)
      exitWeight = MIN_WEIGHT;

    for (SmallVector<BasicBlock *, 8>::iterator EI = ExitingEdges.begin(),
         EE = ExitingEdges.end(); EI != EE; ++EI) {
      BasicBlock *Exiting = *EI;
      BP->setEdgeWeight(BB, Exiting, exitWeight);
    }
  }
}

bool BranchProbabilityAnalysis::runOnFunction(Function &F) {

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ) {
    BasicBlock *BB = I++;

    // Only LBH uses setEdgeWeight method.
    calcLoopBranchHeuristics(BB);

    // PH and RH use only incEdgeWeight and decEwdgeWeight methods to
    // not efface LBH results.
    calcPointerHeuristics(BB);
    calcReturnHeuristics(BB);
  }

  return false;
}


bool BranchProbabilityInfo::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();
  BranchProbabilityAnalysis BPA(&Weights, this, &LI);
  bool ret = BPA.runOnFunction(F);
  return ret;
}

// TODO: This currently hardcodes 80% as a fraction 4/5. We will soon add a
// BranchProbability class to encapsulate the fractional probability and
// define a few static instances of the class for use as predefined thresholds.
bool BranchProbabilityInfo::isEdgeHot(BasicBlock *Src, BasicBlock *Dst) const {
  unsigned Sum = 0;
  for (succ_iterator I = succ_begin(Src), E = succ_end(Src); I != E; ++I) {
    BasicBlock *Succ = *I;
    unsigned Weight = getEdgeWeight(Src, Succ);
    unsigned PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;
  }

  return getEdgeWeight(Src, Dst) * 5 > Sum * 4;
}

BasicBlock *BranchProbabilityInfo::getHotSucc(BasicBlock *BB) const {
  unsigned Sum = 0;
  unsigned MaxWeight = 0;
  BasicBlock *MaxSucc = 0;

  for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
    BasicBlock *Succ = *I;
    unsigned Weight = getEdgeWeight(BB, Succ);
    unsigned PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;

    if (Weight > MaxWeight) {
      MaxWeight = Weight;
      MaxSucc = Succ;
    }
  }

  if (MaxWeight * 5 > Sum * 4)
    return MaxSucc;

  return 0;
}

// Return edge's weight. If can't find it, return DEFAULT_WEIGHT value.
unsigned
BranchProbabilityInfo::getEdgeWeight(BasicBlock *Src, BasicBlock *Dst) const {
  Edge E(Src, Dst);
  DenseMap<Edge, unsigned>::const_iterator I = Weights.find(E);

  if (I != Weights.end())
    return I->second;

  return DEFAULT_WEIGHT;
}

void BranchProbabilityInfo::setEdgeWeight(BasicBlock *Src, BasicBlock *Dst,
                                     unsigned Weight) {
  Weights[std::make_pair(Src, Dst)] = Weight;
  DEBUG(dbgs() << "setEdgeWeight: " << Src->getNameStr() << " -> "
        << Dst->getNameStr() << " to " << Weight
        << (isEdgeHot(Src, Dst) ? " [is HOT now]\n" : "\n"));
}

raw_ostream &
BranchProbabilityInfo::printEdgeProbability(raw_ostream &OS, BasicBlock *Src,
                                        BasicBlock *Dst) const {

  unsigned Sum = 0;
  for (succ_iterator I = succ_begin(Src), E = succ_end(Src); I != E; ++I) {
    BasicBlock *Succ = *I;
    unsigned Weight = getEdgeWeight(Src, Succ);
    unsigned PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;
  }

  double Prob = (double)getEdgeWeight(Src, Dst) / Sum;
  OS << "probability (" << Src->getNameStr() << " --> " << Dst->getNameStr()
     << ") = " << Prob << "\n";

  return OS;
}
