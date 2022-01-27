//===- LoopInterchange.cpp - Loop interchange pass-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Pass handles loop interchange transform.
// This pass interchanges loops to provide a more cache-friendly memory access
// patterns.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopInterchange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "loop-interchange"

STATISTIC(LoopsInterchanged, "Number of loops interchanged");

static cl::opt<int> LoopInterchangeCostThreshold(
    "loop-interchange-threshold", cl::init(0), cl::Hidden,
    cl::desc("Interchange if you gain more than this number"));

namespace {

using LoopVector = SmallVector<Loop *, 8>;

// TODO: Check if we can use a sparse matrix here.
using CharMatrix = std::vector<std::vector<char>>;

} // end anonymous namespace

// Maximum number of dependencies that can be handled in the dependency matrix.
static const unsigned MaxMemInstrCount = 100;

// Maximum loop depth supported.
static const unsigned MaxLoopNestDepth = 10;

#ifdef DUMP_DEP_MATRICIES
static void printDepMatrix(CharMatrix &DepMatrix) {
  for (auto &Row : DepMatrix) {
    for (auto D : Row)
      LLVM_DEBUG(dbgs() << D << " ");
    LLVM_DEBUG(dbgs() << "\n");
  }
}
#endif

static bool populateDependencyMatrix(CharMatrix &DepMatrix, unsigned Level,
                                     Loop *L, DependenceInfo *DI) {
  using ValueVector = SmallVector<Value *, 16>;

  ValueVector MemInstr;

  // For each block.
  for (BasicBlock *BB : L->blocks()) {
    // Scan the BB and collect legal loads and stores.
    for (Instruction &I : *BB) {
      if (!isa<Instruction>(I))
        return false;
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        if (!Ld->isSimple())
          return false;
        MemInstr.push_back(&I);
      } else if (auto *St = dyn_cast<StoreInst>(&I)) {
        if (!St->isSimple())
          return false;
        MemInstr.push_back(&I);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Found " << MemInstr.size()
                    << " Loads and Stores to analyze\n");

  ValueVector::iterator I, IE, J, JE;

  for (I = MemInstr.begin(), IE = MemInstr.end(); I != IE; ++I) {
    for (J = I, JE = MemInstr.end(); J != JE; ++J) {
      std::vector<char> Dep;
      Instruction *Src = cast<Instruction>(*I);
      Instruction *Dst = cast<Instruction>(*J);
      if (Src == Dst)
        continue;
      // Ignore Input dependencies.
      if (isa<LoadInst>(Src) && isa<LoadInst>(Dst))
        continue;
      // Track Output, Flow, and Anti dependencies.
      if (auto D = DI->depends(Src, Dst, true)) {
        assert(D->isOrdered() && "Expected an output, flow or anti dep.");
        LLVM_DEBUG(StringRef DepType =
                       D->isFlow() ? "flow" : D->isAnti() ? "anti" : "output";
                   dbgs() << "Found " << DepType
                          << " dependency between Src and Dst\n"
                          << " Src:" << *Src << "\n Dst:" << *Dst << '\n');
        unsigned Levels = D->getLevels();
        char Direction;
        for (unsigned II = 1; II <= Levels; ++II) {
          const SCEV *Distance = D->getDistance(II);
          const SCEVConstant *SCEVConst =
              dyn_cast_or_null<SCEVConstant>(Distance);
          if (SCEVConst) {
            const ConstantInt *CI = SCEVConst->getValue();
            if (CI->isNegative())
              Direction = '<';
            else if (CI->isZero())
              Direction = '=';
            else
              Direction = '>';
            Dep.push_back(Direction);
          } else if (D->isScalar(II)) {
            Direction = 'S';
            Dep.push_back(Direction);
          } else {
            unsigned Dir = D->getDirection(II);
            if (Dir == Dependence::DVEntry::LT ||
                Dir == Dependence::DVEntry::LE)
              Direction = '<';
            else if (Dir == Dependence::DVEntry::GT ||
                     Dir == Dependence::DVEntry::GE)
              Direction = '>';
            else if (Dir == Dependence::DVEntry::EQ)
              Direction = '=';
            else
              Direction = '*';
            Dep.push_back(Direction);
          }
        }
        while (Dep.size() != Level) {
          Dep.push_back('I');
        }

        DepMatrix.push_back(Dep);
        if (DepMatrix.size() > MaxMemInstrCount) {
          LLVM_DEBUG(dbgs() << "Cannot handle more than " << MaxMemInstrCount
                            << " dependencies inside loop\n");
          return false;
        }
      }
    }
  }

  return true;
}

// A loop is moved from index 'from' to an index 'to'. Update the Dependence
// matrix by exchanging the two columns.
static void interChangeDependencies(CharMatrix &DepMatrix, unsigned FromIndx,
                                    unsigned ToIndx) {
  for (unsigned I = 0, E = DepMatrix.size(); I < E; ++I)
    std::swap(DepMatrix[I][ToIndx], DepMatrix[I][FromIndx]);
}

// Checks if outermost non '=','S'or'I' dependence in the dependence matrix is
// '>'
static bool isOuterMostDepPositive(CharMatrix &DepMatrix, unsigned Row,
                                   unsigned Column) {
  for (unsigned i = 0; i <= Column; ++i) {
    if (DepMatrix[Row][i] == '<')
      return false;
    if (DepMatrix[Row][i] == '>')
      return true;
  }
  // All dependencies were '=','S' or 'I'
  return false;
}

// Checks if no dependence exist in the dependency matrix in Row before Column.
static bool containsNoDependence(CharMatrix &DepMatrix, unsigned Row,
                                 unsigned Column) {
  for (unsigned i = 0; i < Column; ++i) {
    if (DepMatrix[Row][i] != '=' && DepMatrix[Row][i] != 'S' &&
        DepMatrix[Row][i] != 'I')
      return false;
  }
  return true;
}

static bool validDepInterchange(CharMatrix &DepMatrix, unsigned Row,
                                unsigned OuterLoopId, char InnerDep,
                                char OuterDep) {
  if (isOuterMostDepPositive(DepMatrix, Row, OuterLoopId))
    return false;

  if (InnerDep == OuterDep)
    return true;

  // It is legal to interchange if and only if after interchange no row has a
  // '>' direction as the leftmost non-'='.

  if (InnerDep == '=' || InnerDep == 'S' || InnerDep == 'I')
    return true;

  if (InnerDep == '<')
    return true;

  if (InnerDep == '>') {
    // If OuterLoopId represents outermost loop then interchanging will make the
    // 1st dependency as '>'
    if (OuterLoopId == 0)
      return false;

    // If all dependencies before OuterloopId are '=','S'or 'I'. Then
    // interchanging will result in this row having an outermost non '='
    // dependency of '>'
    if (!containsNoDependence(DepMatrix, Row, OuterLoopId))
      return true;
  }

  return false;
}

// Checks if it is legal to interchange 2 loops.
// [Theorem] A permutation of the loops in a perfect nest is legal if and only
// if the direction matrix, after the same permutation is applied to its
// columns, has no ">" direction as the leftmost non-"=" direction in any row.
static bool isLegalToInterChangeLoops(CharMatrix &DepMatrix,
                                      unsigned InnerLoopId,
                                      unsigned OuterLoopId) {
  unsigned NumRows = DepMatrix.size();
  // For each row check if it is valid to interchange.
  for (unsigned Row = 0; Row < NumRows; ++Row) {
    char InnerDep = DepMatrix[Row][InnerLoopId];
    char OuterDep = DepMatrix[Row][OuterLoopId];
    if (InnerDep == '*' || OuterDep == '*')
      return false;
    if (!validDepInterchange(DepMatrix, Row, OuterLoopId, InnerDep, OuterDep))
      return false;
  }
  return true;
}

static LoopVector populateWorklist(Loop &L) {
  LLVM_DEBUG(dbgs() << "Calling populateWorklist on Func: "
                    << L.getHeader()->getParent()->getName() << " Loop: %"
                    << L.getHeader()->getName() << '\n');
  LoopVector LoopList;
  Loop *CurrentLoop = &L;
  const std::vector<Loop *> *Vec = &CurrentLoop->getSubLoops();
  while (!Vec->empty()) {
    // The current loop has multiple subloops in it hence it is not tightly
    // nested.
    // Discard all loops above it added into Worklist.
    if (Vec->size() != 1)
      return {};

    LoopList.push_back(CurrentLoop);
    CurrentLoop = Vec->front();
    Vec = &CurrentLoop->getSubLoops();
  }
  LoopList.push_back(CurrentLoop);
  return LoopList;
}

static PHINode *getInductionVariable(Loop *L, ScalarEvolution *SE) {
  PHINode *InnerIndexVar = L->getCanonicalInductionVariable();
  if (InnerIndexVar)
    return InnerIndexVar;
  if (L->getLoopLatch() == nullptr || L->getLoopPredecessor() == nullptr)
    return nullptr;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    PHINode *PhiVar = cast<PHINode>(I);
    Type *PhiTy = PhiVar->getType();
    if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
        !PhiTy->isPointerTy())
      return nullptr;
    const SCEVAddRecExpr *AddRec =
        dyn_cast<SCEVAddRecExpr>(SE->getSCEV(PhiVar));
    if (!AddRec || !AddRec->isAffine())
      continue;
    const SCEV *Step = AddRec->getStepRecurrence(*SE);
    if (!isa<SCEVConstant>(Step))
      continue;
    // Found the induction variable.
    // FIXME: Handle loops with more than one induction variable. Note that,
    // currently, legality makes sure we have only one induction variable.
    return PhiVar;
  }
  return nullptr;
}

namespace {

/// LoopInterchangeLegality checks if it is legal to interchange the loop.
class LoopInterchangeLegality {
public:
  LoopInterchangeLegality(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                          OptimizationRemarkEmitter *ORE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), ORE(ORE) {}

  /// Check if the loops can be interchanged.
  bool canInterchangeLoops(unsigned InnerLoopId, unsigned OuterLoopId,
                           CharMatrix &DepMatrix);

  /// Check if the loop structure is understood. We do not handle triangular
  /// loops for now.
  bool isLoopStructureUnderstood(PHINode *InnerInductionVar);

  bool currentLimitations();

  const SmallPtrSetImpl<PHINode *> &getOuterInnerReductions() const {
    return OuterInnerReductions;
  }

private:
  bool tightlyNested(Loop *Outer, Loop *Inner);
  bool containsUnsafeInstructions(BasicBlock *BB);

  /// Discover induction and reduction PHIs in the header of \p L. Induction
  /// PHIs are added to \p Inductions, reductions are added to
  /// OuterInnerReductions. When the outer loop is passed, the inner loop needs
  /// to be passed as \p InnerLoop.
  bool findInductionAndReductions(Loop *L,
                                  SmallVector<PHINode *, 8> &Inductions,
                                  Loop *InnerLoop);

  Loop *OuterLoop;
  Loop *InnerLoop;

  ScalarEvolution *SE;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// Set of reduction PHIs taking part of a reduction across the inner and
  /// outer loop.
  SmallPtrSet<PHINode *, 4> OuterInnerReductions;
};

/// LoopInterchangeProfitability checks if it is profitable to interchange the
/// loop.
class LoopInterchangeProfitability {
public:
  LoopInterchangeProfitability(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                               OptimizationRemarkEmitter *ORE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), ORE(ORE) {}

  /// Check if the loop interchange is profitable.
  bool isProfitable(unsigned InnerLoopId, unsigned OuterLoopId,
                    CharMatrix &DepMatrix);

private:
  int getInstrOrderCost();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;
};

/// LoopInterchangeTransform interchanges the loop.
class LoopInterchangeTransform {
public:
  LoopInterchangeTransform(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                           LoopInfo *LI, DominatorTree *DT,
                           const LoopInterchangeLegality &LIL)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), LI(LI), DT(DT), LIL(LIL) {}

  /// Interchange OuterLoop and InnerLoop.
  bool transform();
  void restructureLoops(Loop *NewInner, Loop *NewOuter,
                        BasicBlock *OrigInnerPreHeader,
                        BasicBlock *OrigOuterPreHeader);
  void removeChildLoop(Loop *OuterLoop, Loop *InnerLoop);

private:
  bool adjustLoopLinks();
  bool adjustLoopBranches();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;

  LoopInfo *LI;
  DominatorTree *DT;

  const LoopInterchangeLegality &LIL;
};

struct LoopInterchange {
  ScalarEvolution *SE = nullptr;
  LoopInfo *LI = nullptr;
  DependenceInfo *DI = nullptr;
  DominatorTree *DT = nullptr;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  LoopInterchange(ScalarEvolution *SE, LoopInfo *LI, DependenceInfo *DI,
                  DominatorTree *DT, OptimizationRemarkEmitter *ORE)
      : SE(SE), LI(LI), DI(DI), DT(DT), ORE(ORE) {}

  bool run(Loop *L) {
    if (L->getParentLoop())
      return false;

    return processLoopList(populateWorklist(*L));
  }

  bool run(LoopNest &LN) {
    const auto &LoopList = LN.getLoops();
    for (unsigned I = 1; I < LoopList.size(); ++I)
      if (LoopList[I]->getParentLoop() != LoopList[I - 1])
        return false;
    return processLoopList(LoopList);
  }

  bool isComputableLoopNest(ArrayRef<Loop *> LoopList) {
    for (Loop *L : LoopList) {
      const SCEV *ExitCountOuter = SE->getBackedgeTakenCount(L);
      if (isa<SCEVCouldNotCompute>(ExitCountOuter)) {
        LLVM_DEBUG(dbgs() << "Couldn't compute backedge count\n");
        return false;
      }
      if (L->getNumBackEdges() != 1) {
        LLVM_DEBUG(dbgs() << "NumBackEdges is not equal to 1\n");
        return false;
      }
      if (!L->getExitingBlock()) {
        LLVM_DEBUG(dbgs() << "Loop doesn't have unique exit block\n");
        return false;
      }
    }
    return true;
  }

  unsigned selectLoopForInterchange(ArrayRef<Loop *> LoopList) {
    // TODO: Add a better heuristic to select the loop to be interchanged based
    // on the dependence matrix. Currently we select the innermost loop.
    return LoopList.size() - 1;
  }

  bool processLoopList(ArrayRef<Loop *> LoopList) {
    bool Changed = false;
    unsigned LoopNestDepth = LoopList.size();
    if (LoopNestDepth < 2) {
      LLVM_DEBUG(dbgs() << "Loop doesn't contain minimum nesting level.\n");
      return false;
    }
    if (LoopNestDepth > MaxLoopNestDepth) {
      LLVM_DEBUG(dbgs() << "Cannot handle loops of depth greater than "
                        << MaxLoopNestDepth << "\n");
      return false;
    }
    if (!isComputableLoopNest(LoopList)) {
      LLVM_DEBUG(dbgs() << "Not valid loop candidate for interchange\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "Processing LoopList of size = " << LoopNestDepth
                      << "\n");

    CharMatrix DependencyMatrix;
    Loop *OuterMostLoop = *(LoopList.begin());
    if (!populateDependencyMatrix(DependencyMatrix, LoopNestDepth,
                                  OuterMostLoop, DI)) {
      LLVM_DEBUG(dbgs() << "Populating dependency matrix failed\n");
      return false;
    }
#ifdef DUMP_DEP_MATRICIES
    LLVM_DEBUG(dbgs() << "Dependence before interchange\n");
    printDepMatrix(DependencyMatrix);
#endif

    // Get the Outermost loop exit.
    BasicBlock *LoopNestExit = OuterMostLoop->getExitBlock();
    if (!LoopNestExit) {
      LLVM_DEBUG(dbgs() << "OuterMostLoop needs an unique exit block");
      return false;
    }

    unsigned SelecLoopId = selectLoopForInterchange(LoopList);
    // Move the selected loop outwards to the best possible position.
    Loop *LoopToBeInterchanged = LoopList[SelecLoopId];
    for (unsigned i = SelecLoopId; i > 0; i--) {
      bool Interchanged = processLoop(LoopToBeInterchanged, LoopList[i - 1], i,
                                      i - 1, DependencyMatrix);
      if (!Interchanged)
        return Changed;
      // Update the DependencyMatrix
      interChangeDependencies(DependencyMatrix, i, i - 1);
#ifdef DUMP_DEP_MATRICIES
      LLVM_DEBUG(dbgs() << "Dependence after interchange\n");
      printDepMatrix(DependencyMatrix);
#endif
      Changed |= Interchanged;
    }
    return Changed;
  }

  bool processLoop(Loop *InnerLoop, Loop *OuterLoop, unsigned InnerLoopId,
                   unsigned OuterLoopId,
                   std::vector<std::vector<char>> &DependencyMatrix) {
    LLVM_DEBUG(dbgs() << "Processing InnerLoopId = " << InnerLoopId
                      << " and OuterLoopId = " << OuterLoopId << "\n");
    LoopInterchangeLegality LIL(OuterLoop, InnerLoop, SE, ORE);
    if (!LIL.canInterchangeLoops(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      LLVM_DEBUG(dbgs() << "Not interchanging loops. Cannot prove legality.\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "Loops are legal to interchange\n");
    LoopInterchangeProfitability LIP(OuterLoop, InnerLoop, SE, ORE);
    if (!LIP.isProfitable(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      LLVM_DEBUG(dbgs() << "Interchanging loops not profitable.\n");
      return false;
    }

    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Interchanged",
                                InnerLoop->getStartLoc(),
                                InnerLoop->getHeader())
             << "Loop interchanged with enclosing loop.";
    });

    LoopInterchangeTransform LIT(OuterLoop, InnerLoop, SE, LI, DT, LIL);
    LIT.transform();
    LLVM_DEBUG(dbgs() << "Loops interchanged.\n");
    LoopsInterchanged++;

    assert(InnerLoop->isLCSSAForm(*DT) &&
           "Inner loop not left in LCSSA form after loop interchange!");
    assert(OuterLoop->isLCSSAForm(*DT) &&
           "Outer loop not left in LCSSA form after loop interchange!");

    return true;
  }
};

} // end anonymous namespace

bool LoopInterchangeLegality::containsUnsafeInstructions(BasicBlock *BB) {
  return any_of(*BB, [](const Instruction &I) {
    return I.mayHaveSideEffects() || I.mayReadFromMemory();
  });
}

bool LoopInterchangeLegality::tightlyNested(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();

  LLVM_DEBUG(dbgs() << "Checking if loops are tightly nested\n");

  // A perfectly nested loop will not have any branch in between the outer and
  // inner block i.e. outer header will branch to either inner preheader and
  // outerloop latch.
  BranchInst *OuterLoopHeaderBI =
      dyn_cast<BranchInst>(OuterLoopHeader->getTerminator());
  if (!OuterLoopHeaderBI)
    return false;

  for (BasicBlock *Succ : successors(OuterLoopHeaderBI))
    if (Succ != InnerLoopPreHeader && Succ != InnerLoop->getHeader() &&
        Succ != OuterLoopLatch)
      return false;

  LLVM_DEBUG(dbgs() << "Checking instructions in Loop header and Loop latch\n");
  // We do not have any basic block in between now make sure the outer header
  // and outer loop latch doesn't contain any unsafe instructions.
  if (containsUnsafeInstructions(OuterLoopHeader) ||
      containsUnsafeInstructions(OuterLoopLatch))
    return false;

  // Also make sure the inner loop preheader does not contain any unsafe
  // instructions. Note that all instructions in the preheader will be moved to
  // the outer loop header when interchanging.
  if (InnerLoopPreHeader != OuterLoopHeader &&
      containsUnsafeInstructions(InnerLoopPreHeader))
    return false;

  BasicBlock *InnerLoopExit = InnerLoop->getExitBlock();
  // Ensure the inner loop exit block flows to the outer loop latch possibly
  // through empty blocks.
  const BasicBlock &SuccInner =
      LoopNest::skipEmptyBlockUntil(InnerLoopExit, OuterLoopLatch);
  if (&SuccInner != OuterLoopLatch) {
    LLVM_DEBUG(dbgs() << "Inner loop exit block " << *InnerLoopExit
                      << " does not lead to the outer loop latch.\n";);
    return false;
  }
  // The inner loop exit block does flow to the outer loop latch and not some
  // other BBs, now make sure it contains safe instructions, since it will be
  // moved into the (new) inner loop after interchange.
  if (containsUnsafeInstructions(InnerLoopExit))
    return false;

  LLVM_DEBUG(dbgs() << "Loops are perfectly nested\n");
  // We have a perfect loop nest.
  return true;
}

bool LoopInterchangeLegality::isLoopStructureUnderstood(
    PHINode *InnerInduction) {
  unsigned Num = InnerInduction->getNumOperands();
  BasicBlock *InnerLoopPreheader = InnerLoop->getLoopPreheader();
  for (unsigned i = 0; i < Num; ++i) {
    Value *Val = InnerInduction->getOperand(i);
    if (isa<Constant>(Val))
      continue;
    Instruction *I = dyn_cast<Instruction>(Val);
    if (!I)
      return false;
    // TODO: Handle triangular loops.
    // e.g. for(int i=0;i<N;i++)
    //        for(int j=i;j<N;j++)
    unsigned IncomBlockIndx = PHINode::getIncomingValueNumForOperand(i);
    if (InnerInduction->getIncomingBlock(IncomBlockIndx) ==
            InnerLoopPreheader &&
        !OuterLoop->isLoopInvariant(I)) {
      return false;
    }
  }

  // TODO: Handle triangular loops of another form.
  // e.g. for(int i=0;i<N;i++)
  //        for(int j=0;j<i;j++)
  // or,
  //      for(int i=0;i<N;i++)
  //        for(int j=0;j*i<N;j++)
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BranchInst *InnerLoopLatchBI =
      dyn_cast<BranchInst>(InnerLoopLatch->getTerminator());
  if (!InnerLoopLatchBI->isConditional())
    return false;
  if (CmpInst *InnerLoopCmp =
          dyn_cast<CmpInst>(InnerLoopLatchBI->getCondition())) {
    Value *Op0 = InnerLoopCmp->getOperand(0);
    Value *Op1 = InnerLoopCmp->getOperand(1);

    // LHS and RHS of the inner loop exit condition, e.g.,
    // in "for(int j=0;j<i;j++)", LHS is j and RHS is i.
    Value *Left = nullptr;
    Value *Right = nullptr;

    // Check if V only involves inner loop induction variable.
    // Return true if V is InnerInduction, or a cast from
    // InnerInduction, or a binary operator that involves
    // InnerInduction and a constant.
    std::function<bool(Value *)> IsPathToIndVar;
    IsPathToIndVar = [&InnerInduction, &IsPathToIndVar](Value *V) -> bool {
      if (V == InnerInduction)
        return true;
      if (isa<Constant>(V))
        return true;
      Instruction *I = dyn_cast<Instruction>(V);
      if (!I)
        return false;
      if (isa<CastInst>(I))
        return IsPathToIndVar(I->getOperand(0));
      if (isa<BinaryOperator>(I))
        return IsPathToIndVar(I->getOperand(0)) &&
               IsPathToIndVar(I->getOperand(1));
      return false;
    };

    if (IsPathToIndVar(Op0) && !isa<Constant>(Op0)) {
      Left = Op0;
      Right = Op1;
    } else if (IsPathToIndVar(Op1) && !isa<Constant>(Op1)) {
      Left = Op1;
      Right = Op0;
    }

    if (Left == nullptr)
      return false;

    const SCEV *S = SE->getSCEV(Right);
    if (!SE->isLoopInvariant(S, OuterLoop))
      return false;
  }

  return true;
}

// If SV is a LCSSA PHI node with a single incoming value, return the incoming
// value.
static Value *followLCSSA(Value *SV) {
  PHINode *PHI = dyn_cast<PHINode>(SV);
  if (!PHI)
    return SV;

  if (PHI->getNumIncomingValues() != 1)
    return SV;
  return followLCSSA(PHI->getIncomingValue(0));
}

// Check V's users to see if it is involved in a reduction in L.
static PHINode *findInnerReductionPhi(Loop *L, Value *V) {
  // Reduction variables cannot be constants.
  if (isa<Constant>(V))
    return nullptr;

  for (Value *User : V->users()) {
    if (PHINode *PHI = dyn_cast<PHINode>(User)) {
      if (PHI->getNumIncomingValues() == 1)
        continue;
      RecurrenceDescriptor RD;
      if (RecurrenceDescriptor::isReductionPHI(PHI, L, RD))
        return PHI;
      return nullptr;
    }
  }

  return nullptr;
}

bool LoopInterchangeLegality::findInductionAndReductions(
    Loop *L, SmallVector<PHINode *, 8> &Inductions, Loop *InnerLoop) {
  if (!L->getLoopLatch() || !L->getLoopPredecessor())
    return false;
  for (PHINode &PHI : L->getHeader()->phis()) {
    RecurrenceDescriptor RD;
    InductionDescriptor ID;
    if (InductionDescriptor::isInductionPHI(&PHI, L, SE, ID))
      Inductions.push_back(&PHI);
    else {
      // PHIs in inner loops need to be part of a reduction in the outer loop,
      // discovered when checking the PHIs of the outer loop earlier.
      if (!InnerLoop) {
        if (!OuterInnerReductions.count(&PHI)) {
          LLVM_DEBUG(dbgs() << "Inner loop PHI is not part of reductions "
                               "across the outer loop.\n");
          return false;
        }
      } else {
        assert(PHI.getNumIncomingValues() == 2 &&
               "Phis in loop header should have exactly 2 incoming values");
        // Check if we have a PHI node in the outer loop that has a reduction
        // result from the inner loop as an incoming value.
        Value *V = followLCSSA(PHI.getIncomingValueForBlock(L->getLoopLatch()));
        PHINode *InnerRedPhi = findInnerReductionPhi(InnerLoop, V);
        if (!InnerRedPhi ||
            !llvm::is_contained(InnerRedPhi->incoming_values(), &PHI)) {
          LLVM_DEBUG(
              dbgs()
              << "Failed to recognize PHI as an induction or reduction.\n");
          return false;
        }
        OuterInnerReductions.insert(&PHI);
        OuterInnerReductions.insert(InnerRedPhi);
      }
    }
  }
  return true;
}

// This function indicates the current limitations in the transform as a result
// of which we do not proceed.
bool LoopInterchangeLegality::currentLimitations() {
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();

  // transform currently expects the loop latches to also be the exiting
  // blocks.
  if (InnerLoop->getExitingBlock() != InnerLoopLatch ||
      OuterLoop->getExitingBlock() != OuterLoop->getLoopLatch() ||
      !isa<BranchInst>(InnerLoopLatch->getTerminator()) ||
      !isa<BranchInst>(OuterLoop->getLoopLatch()->getTerminator())) {
    LLVM_DEBUG(
        dbgs() << "Loops where the latch is not the exiting block are not"
               << " supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "ExitingNotLatch",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Loops where the latch is not the exiting block cannot be"
                " interchange currently.";
    });
    return true;
  }

  PHINode *InnerInductionVar;
  SmallVector<PHINode *, 8> Inductions;
  if (!findInductionAndReductions(OuterLoop, Inductions, InnerLoop)) {
    LLVM_DEBUG(
        dbgs() << "Only outer loops with induction or reduction PHI nodes "
               << "are supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedPHIOuter",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Only outer loops with induction or reduction PHI nodes can be"
                " interchanged currently.";
    });
    return true;
  }

  // TODO: Currently we handle only loops with 1 induction variable.
  if (Inductions.size() != 1) {
    LLVM_DEBUG(dbgs() << "Loops with more than 1 induction variables are not "
                      << "supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "MultiIndutionOuter",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Only outer loops with 1 induction variable can be "
                "interchanged currently.";
    });
    return true;
  }

  Inductions.clear();
  if (!findInductionAndReductions(InnerLoop, Inductions, nullptr)) {
    LLVM_DEBUG(
        dbgs() << "Only inner loops with induction or reduction PHI nodes "
               << "are supported currently.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedPHIInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Only inner loops with induction or reduction PHI nodes can be"
                " interchange currently.";
    });
    return true;
  }

  // TODO: Currently we handle only loops with 1 induction variable.
  if (Inductions.size() != 1) {
    LLVM_DEBUG(
        dbgs() << "We currently only support loops with 1 induction variable."
               << "Failed to interchange due to current limitation\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "MultiInductionInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Only inner loops with 1 induction variable can be "
                "interchanged currently.";
    });
    return true;
  }
  InnerInductionVar = Inductions.pop_back_val();

  // TODO: Triangular loops are not handled for now.
  if (!isLoopStructureUnderstood(InnerInductionVar)) {
    LLVM_DEBUG(dbgs() << "Loop structure not understood by pass\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedStructureInner",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Inner loop structure not understood currently.";
    });
    return true;
  }

  return false;
}

// We currently only support LCSSA PHI nodes in the inner loop exit, if their
// users are either reduction PHIs or PHIs outside the outer loop (which means
// the we are only interested in the final value after the loop).
static bool
areInnerLoopExitPHIsSupported(Loop *InnerL, Loop *OuterL,
                              SmallPtrSetImpl<PHINode *> &Reductions) {
  BasicBlock *InnerExit = OuterL->getUniqueExitBlock();
  for (PHINode &PHI : InnerExit->phis()) {
    // Reduction lcssa phi will have only 1 incoming block that from loop latch.
    if (PHI.getNumIncomingValues() > 1)
      return false;
    if (any_of(PHI.users(), [&Reductions, OuterL](User *U) {
          PHINode *PN = dyn_cast<PHINode>(U);
          return !PN ||
                 (!Reductions.count(PN) && OuterL->contains(PN->getParent()));
        })) {
      return false;
    }
  }
  return true;
}

// We currently support LCSSA PHI nodes in the outer loop exit, if their
// incoming values do not come from the outer loop latch or if the
// outer loop latch has a single predecessor. In that case, the value will
// be available if both the inner and outer loop conditions are true, which
// will still be true after interchanging. If we have multiple predecessor,
// that may not be the case, e.g. because the outer loop latch may be executed
// if the inner loop is not executed.
static bool areOuterLoopExitPHIsSupported(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *LoopNestExit = OuterLoop->getUniqueExitBlock();
  for (PHINode &PHI : LoopNestExit->phis()) {
    //  FIXME: We currently are not able to detect floating point reductions
    //         and have to use floating point PHIs as a proxy to prevent
    //         interchanging in the presence of floating point reductions.
    if (PHI.getType()->isFloatingPointTy())
      return false;
    for (unsigned i = 0; i < PHI.getNumIncomingValues(); i++) {
     Instruction *IncomingI = dyn_cast<Instruction>(PHI.getIncomingValue(i));
     if (!IncomingI || IncomingI->getParent() != OuterLoop->getLoopLatch())
       continue;

     // The incoming value is defined in the outer loop latch. Currently we
     // only support that in case the outer loop latch has a single predecessor.
     // This guarantees that the outer loop latch is executed if and only if
     // the inner loop is executed (because tightlyNested() guarantees that the
     // outer loop header only branches to the inner loop or the outer loop
     // latch).
     // FIXME: We could weaken this logic and allow multiple predecessors,
     //        if the values are produced outside the loop latch. We would need
     //        additional logic to update the PHI nodes in the exit block as
     //        well.
     if (OuterLoop->getLoopLatch()->getUniquePredecessor() == nullptr)
       return false;
    }
  }
  return true;
}

// In case of multi-level nested loops, it may occur that lcssa phis exist in
// the latch of InnerLoop, i.e., when defs of the incoming values are further
// inside the loopnest. Sometimes those incoming values are not available
// after interchange, since the original inner latch will become the new outer
// latch which may have predecessor paths that do not include those incoming
// values.
// TODO: Handle transformation of lcssa phis in the InnerLoop latch in case of
// multi-level loop nests.
static bool areInnerLoopLatchPHIsSupported(Loop *OuterLoop, Loop *InnerLoop) {
  if (InnerLoop->getSubLoops().empty())
    return true;
  // If the original outer latch has only one predecessor, then values defined
  // further inside the looploop, e.g., in the innermost loop, will be available
  // at the new outer latch after interchange.
  if (OuterLoop->getLoopLatch()->getUniquePredecessor() != nullptr)
    return true;

  // The outer latch has more than one predecessors, i.e., the inner
  // exit and the inner header.
  // PHI nodes in the inner latch are lcssa phis where the incoming values
  // are defined further inside the loopnest. Check if those phis are used
  // in the original inner latch. If that is the case then bail out since
  // those incoming values may not be available at the new outer latch.
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  for (PHINode &PHI : InnerLoopLatch->phis()) {
    for (auto *U : PHI.users()) {
      Instruction *UI = cast<Instruction>(U);
      if (InnerLoopLatch == UI->getParent())
        return false;
    }
  }
  return true;
}

bool LoopInterchangeLegality::canInterchangeLoops(unsigned InnerLoopId,
                                                  unsigned OuterLoopId,
                                                  CharMatrix &DepMatrix) {
  if (!isLegalToInterChangeLoops(DepMatrix, InnerLoopId, OuterLoopId)) {
    LLVM_DEBUG(dbgs() << "Failed interchange InnerLoopId = " << InnerLoopId
                      << " and OuterLoopId = " << OuterLoopId
                      << " due to dependence\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "Dependence",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Cannot interchange loops due to dependences.";
    });
    return false;
  }
  // Check if outer and inner loop contain legal instructions only.
  for (auto *BB : OuterLoop->blocks())
    for (Instruction &I : BB->instructionsWithoutDebug())
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        // readnone functions do not prevent interchanging.
        if (CI->onlyWritesMemory())
          continue;
        LLVM_DEBUG(
            dbgs() << "Loops with call instructions cannot be interchanged "
                   << "safely.");
        ORE->emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "CallInst",
                                          CI->getDebugLoc(),
                                          CI->getParent())
                 << "Cannot interchange loops due to call instruction.";
        });

        return false;
      }

  if (!areInnerLoopLatchPHIsSupported(OuterLoop, InnerLoop)) {
    LLVM_DEBUG(dbgs() << "Found unsupported PHI nodes in inner loop latch.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedInnerLatchPHI",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Cannot interchange loops because unsupported PHI nodes found "
                "in inner loop latch.";
    });
    return false;
  }

  // TODO: The loops could not be interchanged due to current limitations in the
  // transform module.
  if (currentLimitations()) {
    LLVM_DEBUG(dbgs() << "Not legal because of current transform limitation\n");
    return false;
  }

  // Check if the loops are tightly nested.
  if (!tightlyNested(OuterLoop, InnerLoop)) {
    LLVM_DEBUG(dbgs() << "Loops not tightly nested\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "NotTightlyNested",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Cannot interchange loops because they are not tightly "
                "nested.";
    });
    return false;
  }

  if (!areInnerLoopExitPHIsSupported(OuterLoop, InnerLoop,
                                     OuterInnerReductions)) {
    LLVM_DEBUG(dbgs() << "Found unsupported PHI nodes in inner loop exit.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedExitPHI",
                                      InnerLoop->getStartLoc(),
                                      InnerLoop->getHeader())
             << "Found unsupported PHI node in loop exit.";
    });
    return false;
  }

  if (!areOuterLoopExitPHIsSupported(OuterLoop, InnerLoop)) {
    LLVM_DEBUG(dbgs() << "Found unsupported PHI nodes in outer loop exit.\n");
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedExitPHI",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Found unsupported PHI node in loop exit.";
    });
    return false;
  }

  return true;
}

int LoopInterchangeProfitability::getInstrOrderCost() {
  unsigned GoodOrder, BadOrder;
  BadOrder = GoodOrder = 0;
  for (BasicBlock *BB : InnerLoop->blocks()) {
    for (Instruction &Ins : *BB) {
      if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&Ins)) {
        unsigned NumOp = GEP->getNumOperands();
        bool FoundInnerInduction = false;
        bool FoundOuterInduction = false;
        for (unsigned i = 0; i < NumOp; ++i) {
          // Skip operands that are not SCEV-able.
          if (!SE->isSCEVable(GEP->getOperand(i)->getType()))
            continue;

          const SCEV *OperandVal = SE->getSCEV(GEP->getOperand(i));
          const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(OperandVal);
          if (!AR)
            continue;

          // If we find the inner induction after an outer induction e.g.
          // for(int i=0;i<N;i++)
          //   for(int j=0;j<N;j++)
          //     A[i][j] = A[i-1][j-1]+k;
          // then it is a good order.
          if (AR->getLoop() == InnerLoop) {
            // We found an InnerLoop induction after OuterLoop induction. It is
            // a good order.
            FoundInnerInduction = true;
            if (FoundOuterInduction) {
              GoodOrder++;
              break;
            }
          }
          // If we find the outer induction after an inner induction e.g.
          // for(int i=0;i<N;i++)
          //   for(int j=0;j<N;j++)
          //     A[j][i] = A[j-1][i-1]+k;
          // then it is a bad order.
          if (AR->getLoop() == OuterLoop) {
            // We found an OuterLoop induction after InnerLoop induction. It is
            // a bad order.
            FoundOuterInduction = true;
            if (FoundInnerInduction) {
              BadOrder++;
              break;
            }
          }
        }
      }
    }
  }
  return GoodOrder - BadOrder;
}

static bool isProfitableForVectorization(unsigned InnerLoopId,
                                         unsigned OuterLoopId,
                                         CharMatrix &DepMatrix) {
  // TODO: Improve this heuristic to catch more cases.
  // If the inner loop is loop independent or doesn't carry any dependency it is
  // profitable to move this to outer position.
  for (auto &Row : DepMatrix) {
    if (Row[InnerLoopId] != 'S' && Row[InnerLoopId] != 'I')
      return false;
    // TODO: We need to improve this heuristic.
    if (Row[OuterLoopId] != '=')
      return false;
  }
  // If outer loop has dependence and inner loop is loop independent then it is
  // profitable to interchange to enable parallelism.
  // If there are no dependences, interchanging will not improve anything.
  return !DepMatrix.empty();
}

bool LoopInterchangeProfitability::isProfitable(unsigned InnerLoopId,
                                                unsigned OuterLoopId,
                                                CharMatrix &DepMatrix) {
  // TODO: Add better profitability checks.
  // e.g
  // 1) Construct dependency matrix and move the one with no loop carried dep
  //    inside to enable vectorization.

  // This is rough cost estimation algorithm. It counts the good and bad order
  // of induction variables in the instruction and allows reordering if number
  // of bad orders is more than good.
  int Cost = getInstrOrderCost();
  LLVM_DEBUG(dbgs() << "Cost = " << Cost << "\n");
  if (Cost < -LoopInterchangeCostThreshold)
    return true;

  // It is not profitable as per current cache profitability model. But check if
  // we can move this loop outside to improve parallelism.
  if (isProfitableForVectorization(InnerLoopId, OuterLoopId, DepMatrix))
    return true;

  ORE->emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, "InterchangeNotProfitable",
                                    InnerLoop->getStartLoc(),
                                    InnerLoop->getHeader())
           << "Interchanging loops is too costly (cost="
           << ore::NV("Cost", Cost) << ", threshold="
           << ore::NV("Threshold", LoopInterchangeCostThreshold)
           << ") and it does not improve parallelism.";
  });
  return false;
}

void LoopInterchangeTransform::removeChildLoop(Loop *OuterLoop,
                                               Loop *InnerLoop) {
  for (Loop *L : *OuterLoop)
    if (L == InnerLoop) {
      OuterLoop->removeChildLoop(L);
      return;
    }
  llvm_unreachable("Couldn't find loop");
}

/// Update LoopInfo, after interchanging. NewInner and NewOuter refer to the
/// new inner and outer loop after interchanging: NewInner is the original
/// outer loop and NewOuter is the original inner loop.
///
/// Before interchanging, we have the following structure
/// Outer preheader
//  Outer header
//    Inner preheader
//    Inner header
//      Inner body
//      Inner latch
//   outer bbs
//   Outer latch
//
// After interchanging:
// Inner preheader
// Inner header
//   Outer preheader
//   Outer header
//     Inner body
//     outer bbs
//     Outer latch
//   Inner latch
void LoopInterchangeTransform::restructureLoops(
    Loop *NewInner, Loop *NewOuter, BasicBlock *OrigInnerPreHeader,
    BasicBlock *OrigOuterPreHeader) {
  Loop *OuterLoopParent = OuterLoop->getParentLoop();
  // The original inner loop preheader moves from the new inner loop to
  // the parent loop, if there is one.
  NewInner->removeBlockFromLoop(OrigInnerPreHeader);
  LI->changeLoopFor(OrigInnerPreHeader, OuterLoopParent);

  // Switch the loop levels.
  if (OuterLoopParent) {
    // Remove the loop from its parent loop.
    removeChildLoop(OuterLoopParent, NewInner);
    removeChildLoop(NewInner, NewOuter);
    OuterLoopParent->addChildLoop(NewOuter);
  } else {
    removeChildLoop(NewInner, NewOuter);
    LI->changeTopLevelLoop(NewInner, NewOuter);
  }
  while (!NewOuter->isInnermost())
    NewInner->addChildLoop(NewOuter->removeChildLoop(NewOuter->begin()));
  NewOuter->addChildLoop(NewInner);

  // BBs from the original inner loop.
  SmallVector<BasicBlock *, 8> OrigInnerBBs(NewOuter->blocks());

  // Add BBs from the original outer loop to the original inner loop (excluding
  // BBs already in inner loop)
  for (BasicBlock *BB : NewInner->blocks())
    if (LI->getLoopFor(BB) == NewInner)
      NewOuter->addBlockEntry(BB);

  // Now remove inner loop header and latch from the new inner loop and move
  // other BBs (the loop body) to the new inner loop.
  BasicBlock *OuterHeader = NewOuter->getHeader();
  BasicBlock *OuterLatch = NewOuter->getLoopLatch();
  for (BasicBlock *BB : OrigInnerBBs) {
    // Nothing will change for BBs in child loops.
    if (LI->getLoopFor(BB) != NewOuter)
      continue;
    // Remove the new outer loop header and latch from the new inner loop.
    if (BB == OuterHeader || BB == OuterLatch)
      NewInner->removeBlockFromLoop(BB);
    else
      LI->changeLoopFor(BB, NewInner);
  }

  // The preheader of the original outer loop becomes part of the new
  // outer loop.
  NewOuter->addBlockEntry(OrigOuterPreHeader);
  LI->changeLoopFor(OrigOuterPreHeader, NewOuter);

  // Tell SE that we move the loops around.
  SE->forgetLoop(NewOuter);
  SE->forgetLoop(NewInner);
}

bool LoopInterchangeTransform::transform() {
  bool Transformed = false;
  Instruction *InnerIndexVar;

  if (InnerLoop->getSubLoops().empty()) {
    BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
    LLVM_DEBUG(dbgs() << "Splitting the inner loop latch\n");
    PHINode *InductionPHI = getInductionVariable(InnerLoop, SE);
    if (!InductionPHI) {
      LLVM_DEBUG(dbgs() << "Failed to find the point to split loop latch \n");
      return false;
    }

    if (InductionPHI->getIncomingBlock(0) == InnerLoopPreHeader)
      InnerIndexVar = dyn_cast<Instruction>(InductionPHI->getIncomingValue(1));
    else
      InnerIndexVar = dyn_cast<Instruction>(InductionPHI->getIncomingValue(0));

    // Ensure that InductionPHI is the first Phi node.
    if (&InductionPHI->getParent()->front() != InductionPHI)
      InductionPHI->moveBefore(&InductionPHI->getParent()->front());

    // Create a new latch block for the inner loop. We split at the
    // current latch's terminator and then move the condition and all
    // operands that are not either loop-invariant or the induction PHI into the
    // new latch block.
    BasicBlock *NewLatch =
        SplitBlock(InnerLoop->getLoopLatch(),
                   InnerLoop->getLoopLatch()->getTerminator(), DT, LI);

    SmallSetVector<Instruction *, 4> WorkList;
    unsigned i = 0;
    auto MoveInstructions = [&i, &WorkList, this, InductionPHI, NewLatch]() {
      for (; i < WorkList.size(); i++) {
        // Duplicate instruction and move it the new latch. Update uses that
        // have been moved.
        Instruction *NewI = WorkList[i]->clone();
        NewI->insertBefore(NewLatch->getFirstNonPHI());
        assert(!NewI->mayHaveSideEffects() &&
               "Moving instructions with side-effects may change behavior of "
               "the loop nest!");
        for (Use &U : llvm::make_early_inc_range(WorkList[i]->uses())) {
          Instruction *UserI = cast<Instruction>(U.getUser());
          if (!InnerLoop->contains(UserI->getParent()) ||
              UserI->getParent() == NewLatch || UserI == InductionPHI)
            U.set(NewI);
        }
        // Add operands of moved instruction to the worklist, except if they are
        // outside the inner loop or are the induction PHI.
        for (Value *Op : WorkList[i]->operands()) {
          Instruction *OpI = dyn_cast<Instruction>(Op);
          if (!OpI ||
              this->LI->getLoopFor(OpI->getParent()) != this->InnerLoop ||
              OpI == InductionPHI)
            continue;
          WorkList.insert(OpI);
        }
      }
    };

    // FIXME: Should we interchange when we have a constant condition?
    Instruction *CondI = dyn_cast<Instruction>(
        cast<BranchInst>(InnerLoop->getLoopLatch()->getTerminator())
            ->getCondition());
    if (CondI)
      WorkList.insert(CondI);
    MoveInstructions();
    WorkList.insert(cast<Instruction>(InnerIndexVar));
    MoveInstructions();

    // Splits the inner loops phi nodes out into a separate basic block.
    BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
    SplitBlock(InnerLoopHeader, InnerLoopHeader->getFirstNonPHI(), DT, LI);
    LLVM_DEBUG(dbgs() << "splitting InnerLoopHeader done\n");
  }

  // Instructions in the original inner loop preheader may depend on values
  // defined in the outer loop header. Move them there, because the original
  // inner loop preheader will become the entry into the interchanged loop nest.
  // Currently we move all instructions and rely on LICM to move invariant
  // instructions outside the loop nest.
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  if (InnerLoopPreHeader != OuterLoopHeader) {
    SmallPtrSet<Instruction *, 4> NeedsMoving;
    for (Instruction &I :
         make_early_inc_range(make_range(InnerLoopPreHeader->begin(),
                                         std::prev(InnerLoopPreHeader->end()))))
      I.moveBefore(OuterLoopHeader->getTerminator());
  }

  Transformed |= adjustLoopLinks();
  if (!Transformed) {
    LLVM_DEBUG(dbgs() << "adjustLoopLinks failed\n");
    return false;
  }

  return true;
}

/// \brief Move all instructions except the terminator from FromBB right before
/// InsertBefore
static void moveBBContents(BasicBlock *FromBB, Instruction *InsertBefore) {
  auto &ToList = InsertBefore->getParent()->getInstList();
  auto &FromList = FromBB->getInstList();

  ToList.splice(InsertBefore->getIterator(), FromList, FromList.begin(),
                FromBB->getTerminator()->getIterator());
}

/// Swap instructions between \p BB1 and \p BB2 but keep terminators intact.
static void swapBBContents(BasicBlock *BB1, BasicBlock *BB2) {
  // Save all non-terminator instructions of BB1 into TempInstrs and unlink them
  // from BB1 afterwards.
  auto Iter = map_range(*BB1, [](Instruction &I) { return &I; });
  SmallVector<Instruction *, 4> TempInstrs(Iter.begin(), std::prev(Iter.end()));
  for (Instruction *I : TempInstrs)
    I->removeFromParent();

  // Move instructions from BB2 to BB1.
  moveBBContents(BB2, BB1->getTerminator());

  // Move instructions from TempInstrs to BB2.
  for (Instruction *I : TempInstrs)
    I->insertBefore(BB2->getTerminator());
}

// Update BI to jump to NewBB instead of OldBB. Records updates to the
// dominator tree in DTUpdates. If \p MustUpdateOnce is true, assert that
// \p OldBB  is exactly once in BI's successor list.
static void updateSuccessor(BranchInst *BI, BasicBlock *OldBB,
                            BasicBlock *NewBB,
                            std::vector<DominatorTree::UpdateType> &DTUpdates,
                            bool MustUpdateOnce = true) {
  assert((!MustUpdateOnce ||
          llvm::count_if(successors(BI),
                         [OldBB](BasicBlock *BB) {
                           return BB == OldBB;
                         }) == 1) && "BI must jump to OldBB exactly once.");
  bool Changed = false;
  for (Use &Op : BI->operands())
    if (Op == OldBB) {
      Op.set(NewBB);
      Changed = true;
    }

  if (Changed) {
    DTUpdates.push_back(
        {DominatorTree::UpdateKind::Insert, BI->getParent(), NewBB});
    DTUpdates.push_back(
        {DominatorTree::UpdateKind::Delete, BI->getParent(), OldBB});
  }
  assert(Changed && "Expected a successor to be updated");
}

// Move Lcssa PHIs to the right place.
static void moveLCSSAPhis(BasicBlock *InnerExit, BasicBlock *InnerHeader,
                          BasicBlock *InnerLatch, BasicBlock *OuterHeader,
                          BasicBlock *OuterLatch, BasicBlock *OuterExit,
                          Loop *InnerLoop, LoopInfo *LI) {

  // Deal with LCSSA PHI nodes in the exit block of the inner loop, that are
  // defined either in the header or latch. Those blocks will become header and
  // latch of the new outer loop, and the only possible users can PHI nodes
  // in the exit block of the loop nest or the outer loop header (reduction
  // PHIs, in that case, the incoming value must be defined in the inner loop
  // header). We can just substitute the user with the incoming value and remove
  // the PHI.
  for (PHINode &P : make_early_inc_range(InnerExit->phis())) {
    assert(P.getNumIncomingValues() == 1 &&
           "Only loops with a single exit are supported!");

    // Incoming values are guaranteed be instructions currently.
    auto IncI = cast<Instruction>(P.getIncomingValueForBlock(InnerLatch));
    // Skip phis with incoming values from the inner loop body, excluding the
    // header and latch.
    if (IncI->getParent() != InnerLatch && IncI->getParent() != InnerHeader)
      continue;

    assert(all_of(P.users(),
                  [OuterHeader, OuterExit, IncI, InnerHeader](User *U) {
                    return (cast<PHINode>(U)->getParent() == OuterHeader &&
                            IncI->getParent() == InnerHeader) ||
                           cast<PHINode>(U)->getParent() == OuterExit;
                  }) &&
           "Can only replace phis iff the uses are in the loop nest exit or "
           "the incoming value is defined in the inner header (it will "
           "dominate all loop blocks after interchanging)");
    P.replaceAllUsesWith(IncI);
    P.eraseFromParent();
  }

  SmallVector<PHINode *, 8> LcssaInnerExit;
  for (PHINode &P : InnerExit->phis())
    LcssaInnerExit.push_back(&P);

  SmallVector<PHINode *, 8> LcssaInnerLatch;
  for (PHINode &P : InnerLatch->phis())
    LcssaInnerLatch.push_back(&P);

  // Lcssa PHIs for values used outside the inner loop are in InnerExit.
  // If a PHI node has users outside of InnerExit, it has a use outside the
  // interchanged loop and we have to preserve it. We move these to
  // InnerLatch, which will become the new exit block for the innermost
  // loop after interchanging.
  for (PHINode *P : LcssaInnerExit)
    P->moveBefore(InnerLatch->getFirstNonPHI());

  // If the inner loop latch contains LCSSA PHIs, those come from a child loop
  // and we have to move them to the new inner latch.
  for (PHINode *P : LcssaInnerLatch)
    P->moveBefore(InnerExit->getFirstNonPHI());

  // Deal with LCSSA PHI nodes in the loop nest exit block. For PHIs that have
  // incoming values defined in the outer loop, we have to add a new PHI
  // in the inner loop latch, which became the exit block of the outer loop,
  // after interchanging.
  if (OuterExit) {
    for (PHINode &P : OuterExit->phis()) {
      if (P.getNumIncomingValues() != 1)
        continue;
      // Skip Phis with incoming values defined in the inner loop. Those should
      // already have been updated.
      auto I = dyn_cast<Instruction>(P.getIncomingValue(0));
      if (!I || LI->getLoopFor(I->getParent()) == InnerLoop)
        continue;

      PHINode *NewPhi = dyn_cast<PHINode>(P.clone());
      NewPhi->setIncomingValue(0, P.getIncomingValue(0));
      NewPhi->setIncomingBlock(0, OuterLatch);
      // We might have incoming edges from other BBs, i.e., the original outer
      // header.
      for (auto *Pred : predecessors(InnerLatch)) {
        if (Pred == OuterLatch)
          continue;
        NewPhi->addIncoming(P.getIncomingValue(0), Pred);
      }
      NewPhi->insertBefore(InnerLatch->getFirstNonPHI());
      P.setIncomingValue(0, NewPhi);
    }
  }

  // Now adjust the incoming blocks for the LCSSA PHIs.
  // For PHIs moved from Inner's exit block, we need to replace Inner's latch
  // with the new latch.
  InnerLatch->replacePhiUsesWith(InnerLatch, OuterLatch);
}

bool LoopInterchangeTransform::adjustLoopBranches() {
  LLVM_DEBUG(dbgs() << "adjustLoopBranches called\n");
  std::vector<DominatorTree::UpdateType> DTUpdates;

  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();

  assert(OuterLoopPreHeader != OuterLoop->getHeader() &&
         InnerLoopPreHeader != InnerLoop->getHeader() && OuterLoopPreHeader &&
         InnerLoopPreHeader && "Guaranteed by loop-simplify form");
  // Ensure that both preheaders do not contain PHI nodes and have single
  // predecessors. This allows us to move them easily. We use
  // InsertPreHeaderForLoop to create an 'extra' preheader, if the existing
  // preheaders do not satisfy those conditions.
  if (isa<PHINode>(OuterLoopPreHeader->begin()) ||
      !OuterLoopPreHeader->getUniquePredecessor())
    OuterLoopPreHeader =
        InsertPreheaderForLoop(OuterLoop, DT, LI, nullptr, true);
  if (InnerLoopPreHeader == OuterLoop->getHeader())
    InnerLoopPreHeader =
        InsertPreheaderForLoop(InnerLoop, DT, LI, nullptr, true);

  // Adjust the loop preheader
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();
  BasicBlock *OuterLoopPredecessor = OuterLoopPreHeader->getUniquePredecessor();
  BasicBlock *InnerLoopLatchPredecessor =
      InnerLoopLatch->getUniquePredecessor();
  BasicBlock *InnerLoopLatchSuccessor;
  BasicBlock *OuterLoopLatchSuccessor;

  BranchInst *OuterLoopLatchBI =
      dyn_cast<BranchInst>(OuterLoopLatch->getTerminator());
  BranchInst *InnerLoopLatchBI =
      dyn_cast<BranchInst>(InnerLoopLatch->getTerminator());
  BranchInst *OuterLoopHeaderBI =
      dyn_cast<BranchInst>(OuterLoopHeader->getTerminator());
  BranchInst *InnerLoopHeaderBI =
      dyn_cast<BranchInst>(InnerLoopHeader->getTerminator());

  if (!OuterLoopPredecessor || !InnerLoopLatchPredecessor ||
      !OuterLoopLatchBI || !InnerLoopLatchBI || !OuterLoopHeaderBI ||
      !InnerLoopHeaderBI)
    return false;

  BranchInst *InnerLoopLatchPredecessorBI =
      dyn_cast<BranchInst>(InnerLoopLatchPredecessor->getTerminator());
  BranchInst *OuterLoopPredecessorBI =
      dyn_cast<BranchInst>(OuterLoopPredecessor->getTerminator());

  if (!OuterLoopPredecessorBI || !InnerLoopLatchPredecessorBI)
    return false;
  BasicBlock *InnerLoopHeaderSuccessor = InnerLoopHeader->getUniqueSuccessor();
  if (!InnerLoopHeaderSuccessor)
    return false;

  // Adjust Loop Preheader and headers.
  // The branches in the outer loop predecessor and the outer loop header can
  // be unconditional branches or conditional branches with duplicates. Consider
  // this when updating the successors.
  updateSuccessor(OuterLoopPredecessorBI, OuterLoopPreHeader,
                  InnerLoopPreHeader, DTUpdates, /*MustUpdateOnce=*/false);
  // The outer loop header might or might not branch to the outer latch.
  // We are guaranteed to branch to the inner loop preheader.
  if (llvm::is_contained(OuterLoopHeaderBI->successors(), OuterLoopLatch)) {
    // In this case the outerLoopHeader should branch to the InnerLoopLatch.
    updateSuccessor(OuterLoopHeaderBI, OuterLoopLatch, InnerLoopLatch,
                    DTUpdates,
                    /*MustUpdateOnce=*/false);
  }
  updateSuccessor(OuterLoopHeaderBI, InnerLoopPreHeader,
                  InnerLoopHeaderSuccessor, DTUpdates,
                  /*MustUpdateOnce=*/false);

  // Adjust reduction PHI's now that the incoming block has changed.
  InnerLoopHeaderSuccessor->replacePhiUsesWith(InnerLoopHeader,
                                               OuterLoopHeader);

  updateSuccessor(InnerLoopHeaderBI, InnerLoopHeaderSuccessor,
                  OuterLoopPreHeader, DTUpdates);

  // -------------Adjust loop latches-----------
  if (InnerLoopLatchBI->getSuccessor(0) == InnerLoopHeader)
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(1);
  else
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(0);

  updateSuccessor(InnerLoopLatchPredecessorBI, InnerLoopLatch,
                  InnerLoopLatchSuccessor, DTUpdates);


  if (OuterLoopLatchBI->getSuccessor(0) == OuterLoopHeader)
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(1);
  else
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(0);

  updateSuccessor(InnerLoopLatchBI, InnerLoopLatchSuccessor,
                  OuterLoopLatchSuccessor, DTUpdates);
  updateSuccessor(OuterLoopLatchBI, OuterLoopLatchSuccessor, InnerLoopLatch,
                  DTUpdates);

  DT->applyUpdates(DTUpdates);
  restructureLoops(OuterLoop, InnerLoop, InnerLoopPreHeader,
                   OuterLoopPreHeader);

  moveLCSSAPhis(InnerLoopLatchSuccessor, InnerLoopHeader, InnerLoopLatch,
                OuterLoopHeader, OuterLoopLatch, InnerLoop->getExitBlock(),
                InnerLoop, LI);
  // For PHIs in the exit block of the outer loop, outer's latch has been
  // replaced by Inners'.
  OuterLoopLatchSuccessor->replacePhiUsesWith(OuterLoopLatch, InnerLoopLatch);

  auto &OuterInnerReductions = LIL.getOuterInnerReductions();
  // Now update the reduction PHIs in the inner and outer loop headers.
  SmallVector<PHINode *, 4> InnerLoopPHIs, OuterLoopPHIs;
  for (PHINode &PHI : InnerLoopHeader->phis())
    if (OuterInnerReductions.contains(&PHI))
      InnerLoopPHIs.push_back(cast<PHINode>(&PHI));
  for (PHINode &PHI : OuterLoopHeader->phis())
    if (OuterInnerReductions.contains(&PHI))
      OuterLoopPHIs.push_back(cast<PHINode>(&PHI));

  // Now move the remaining reduction PHIs from outer to inner loop header and
  // vice versa. The PHI nodes must be part of a reduction across the inner and
  // outer loop and all the remains to do is and updating the incoming blocks.
  for (PHINode *PHI : OuterLoopPHIs) {
    PHI->moveBefore(InnerLoopHeader->getFirstNonPHI());
    assert(OuterInnerReductions.count(PHI) && "Expected a reduction PHI node");
  }
  for (PHINode *PHI : InnerLoopPHIs) {
    PHI->moveBefore(OuterLoopHeader->getFirstNonPHI());
    assert(OuterInnerReductions.count(PHI) && "Expected a reduction PHI node");
  }

  // Update the incoming blocks for moved PHI nodes.
  OuterLoopHeader->replacePhiUsesWith(InnerLoopPreHeader, OuterLoopPreHeader);
  OuterLoopHeader->replacePhiUsesWith(InnerLoopLatch, OuterLoopLatch);
  InnerLoopHeader->replacePhiUsesWith(OuterLoopPreHeader, InnerLoopPreHeader);
  InnerLoopHeader->replacePhiUsesWith(OuterLoopLatch, InnerLoopLatch);

  // Values defined in the outer loop header could be used in the inner loop
  // latch. In that case, we need to create LCSSA phis for them, because after
  // interchanging they will be defined in the new inner loop and used in the
  // new outer loop.
  IRBuilder<> Builder(OuterLoopHeader->getContext());
  SmallVector<Instruction *, 4> MayNeedLCSSAPhis;
  for (Instruction &I :
       make_range(OuterLoopHeader->begin(), std::prev(OuterLoopHeader->end())))
    MayNeedLCSSAPhis.push_back(&I);
  formLCSSAForInstructions(MayNeedLCSSAPhis, *DT, *LI, SE, Builder);

  return true;
}

bool LoopInterchangeTransform::adjustLoopLinks() {
  // Adjust all branches in the inner and outer loop.
  bool Changed = adjustLoopBranches();
  if (Changed) {
    // We have interchanged the preheaders so we need to interchange the data in
    // the preheaders as well. This is because the content of the inner
    // preheader was previously executed inside the outer loop.
    BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
    BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
    swapBBContents(OuterLoopPreHeader, InnerLoopPreHeader);
  }
  return Changed;
}

namespace {
/// Main LoopInterchange Pass.
struct LoopInterchangeLegacyPass : public LoopPass {
  static char ID;

  LoopInterchangeLegacyPass() : LoopPass(ID) {
    initializeLoopInterchangeLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DependenceAnalysisWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();

    getLoopAnalysisUsage(AU);
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    if (skipLoop(L))
      return false;

    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *DI = &getAnalysis<DependenceAnalysisWrapperPass>().getDI();
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    return LoopInterchange(SE, LI, DI, DT, ORE).run(L);
  }
};
} // namespace

char LoopInterchangeLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(LoopInterchangeLegacyPass, "loop-interchange",
                      "Interchanges loops for cache reuse", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)

INITIALIZE_PASS_END(LoopInterchangeLegacyPass, "loop-interchange",
                    "Interchanges loops for cache reuse", false, false)

Pass *llvm::createLoopInterchangePass() {
  return new LoopInterchangeLegacyPass();
}

PreservedAnalyses LoopInterchangePass::run(LoopNest &LN,
                                           LoopAnalysisManager &AM,
                                           LoopStandardAnalysisResults &AR,
                                           LPMUpdater &U) {
  Function &F = *LN.getParent();

  DependenceInfo DI(&F, &AR.AA, &AR.SE, &AR.LI);
  OptimizationRemarkEmitter ORE(&F);
  if (!LoopInterchange(&AR.SE, &AR.LI, &DI, &AR.DT, &ORE).run(LN))
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}
