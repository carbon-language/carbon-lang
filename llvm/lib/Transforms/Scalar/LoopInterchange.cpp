//===- LoopInterchange.cpp - Loop interchange pass------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This Pass handles loop interchange transform.
// This pass interchanges loops to provide a more cache-friendly memory access
// patterns.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;

#define DEBUG_TYPE "loop-interchange"

namespace {

typedef SmallVector<Loop *, 8> LoopVector;

// TODO: Check if we can use a sparse matrix here.
typedef std::vector<std::vector<char>> CharMatrix;

// Maximum number of dependencies that can be handled in the dependency matrix.
static const unsigned MaxMemInstrCount = 100;

// Maximum loop depth supported.
static const unsigned MaxLoopNestDepth = 10;

struct LoopInterchange;

#ifdef DUMP_DEP_MATRICIES
void printDepMatrix(CharMatrix &DepMatrix) {
  for (auto I = DepMatrix.begin(), E = DepMatrix.end(); I != E; ++I) {
    std::vector<char> Vec = *I;
    for (auto II = Vec.begin(), EE = Vec.end(); II != EE; ++II)
      DEBUG(dbgs() << *II << " ");
    DEBUG(dbgs() << "\n");
  }
}
#endif

bool populateDependencyMatrix(CharMatrix &DepMatrix, unsigned Level, Loop *L,
                              DependenceAnalysis *DA) {
  typedef SmallVector<Value *, 16> ValueVector;
  ValueVector MemInstr;

  if (Level > MaxLoopNestDepth) {
    DEBUG(dbgs() << "Cannot handle loops of depth greater than "
                 << MaxLoopNestDepth << "\n");
    return false;
  }

  // For each block.
  for (Loop::block_iterator BB = L->block_begin(), BE = L->block_end();
       BB != BE; ++BB) {
    // Scan the BB and collect legal loads and stores.
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E;
         ++I) {
      Instruction *Ins = dyn_cast<Instruction>(I);
      if (!Ins)
        return false;
      LoadInst *Ld = dyn_cast<LoadInst>(I);
      StoreInst *St = dyn_cast<StoreInst>(I);
      if (!St && !Ld)
        continue;
      if (Ld && !Ld->isSimple())
        return false;
      if (St && !St->isSimple())
        return false;
      MemInstr.push_back(I);
    }
  }

  DEBUG(dbgs() << "Found " << MemInstr.size()
               << " Loads and Stores to analyze\n");

  ValueVector::iterator I, IE, J, JE;

  for (I = MemInstr.begin(), IE = MemInstr.end(); I != IE; ++I) {
    for (J = I, JE = MemInstr.end(); J != JE; ++J) {
      std::vector<char> Dep;
      Instruction *Src = dyn_cast<Instruction>(*I);
      Instruction *Des = dyn_cast<Instruction>(*J);
      if (Src == Des)
        continue;
      if (isa<LoadInst>(Src) && isa<LoadInst>(Des))
        continue;
      if (auto D = DA->depends(Src, Des, true)) {
        DEBUG(dbgs() << "Found Dependency between Src=" << Src << " Des=" << Des
                     << "\n");
        if (D->isFlow()) {
          // TODO: Handle Flow dependence.Check if it is sufficient to populate
          // the Dependence Matrix with the direction reversed.
          DEBUG(dbgs() << "Flow dependence not handled");
          return false;
        }
        if (D->isAnti()) {
          DEBUG(dbgs() << "Found Anti dependence \n");
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
            DEBUG(dbgs() << "Cannot handle more than " << MaxMemInstrCount
                         << " dependencies inside loop\n");
            return false;
          }
        }
      }
    }
  }

  // We don't have a DepMatrix to check legality return false
  if (DepMatrix.size() == 0)
    return false;
  return true;
}

// A loop is moved from index 'from' to an index 'to'. Update the Dependence
// matrix by exchanging the two columns.
void interChangeDepedencies(CharMatrix &DepMatrix, unsigned FromIndx,
                            unsigned ToIndx) {
  unsigned numRows = DepMatrix.size();
  for (unsigned i = 0; i < numRows; ++i) {
    char TmpVal = DepMatrix[i][ToIndx];
    DepMatrix[i][ToIndx] = DepMatrix[i][FromIndx];
    DepMatrix[i][FromIndx] = TmpVal;
  }
}

// Checks if outermost non '=','S'or'I' dependence in the dependence matrix is
// '>'
bool isOuterMostDepPositive(CharMatrix &DepMatrix, unsigned Row,
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
bool containsNoDependence(CharMatrix &DepMatrix, unsigned Row,
                          unsigned Column) {
  for (unsigned i = 0; i < Column; ++i) {
    if (DepMatrix[Row][i] != '=' || DepMatrix[Row][i] != 'S' ||
        DepMatrix[Row][i] != 'I')
      return false;
  }
  return true;
}

bool validDepInterchange(CharMatrix &DepMatrix, unsigned Row,
                         unsigned OuterLoopId, char InnerDep, char OuterDep) {

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
// [Theorm] A permutation of the loops in a perfect nest is legal if and only if
// the direction matrix, after the same permutation is applied to its columns,
// has no ">" direction as the leftmost non-"=" direction in any row.
bool isLegalToInterChangeLoops(CharMatrix &DepMatrix, unsigned InnerLoopId,
                               unsigned OuterLoopId) {

  unsigned NumRows = DepMatrix.size();
  // For each row check if it is valid to interchange.
  for (unsigned Row = 0; Row < NumRows; ++Row) {
    char InnerDep = DepMatrix[Row][InnerLoopId];
    char OuterDep = DepMatrix[Row][OuterLoopId];
    if (InnerDep == '*' || OuterDep == '*')
      return false;
    else if (!validDepInterchange(DepMatrix, Row, OuterLoopId, InnerDep,
                                  OuterDep))
      return false;
  }
  return true;
}

static void populateWorklist(Loop &L, SmallVector<LoopVector, 8> &V) {

  DEBUG(dbgs() << "Calling populateWorklist called\n");
  LoopVector LoopList;
  Loop *CurrentLoop = &L;
  std::vector<Loop *> vec = CurrentLoop->getSubLoopsVector();
  while (vec.size() != 0) {
    // The current loop has multiple subloops in it hence it is not tightly
    // nested.
    // Discard all loops above it added into Worklist.
    if (vec.size() != 1) {
      LoopList.clear();
      return;
    }
    LoopList.push_back(CurrentLoop);
    CurrentLoop = *(vec.begin());
    vec = CurrentLoop->getSubLoopsVector();
  }
  LoopList.push_back(CurrentLoop);
  V.push_back(LoopList);
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
    const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
    if (!C)
      continue;
    // Found the induction variable.
    // FIXME: Handle loops with more than one induction variable. Note that,
    // currently, legality makes sure we have only one induction variable.
    return PhiVar;
  }
  return nullptr;
}

/// LoopInterchangeLegality checks if it is legal to interchange the loop.
class LoopInterchangeLegality {
public:
  LoopInterchangeLegality(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                          LoopInterchange *Pass)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), CurrentPass(Pass) {}

  /// Check if the loops can be interchanged.
  bool canInterchangeLoops(unsigned InnerLoopId, unsigned OuterLoopId,
                           CharMatrix &DepMatrix);
  /// Check if the loop structure is understood. We do not handle triangular
  /// loops for now.
  bool isLoopStructureUnderstood(PHINode *InnerInductionVar);

  bool currentLimitations();

private:
  bool tightlyNested(Loop *Outer, Loop *Inner);

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;
  LoopInterchange *CurrentPass;
};

/// LoopInterchangeProfitability checks if it is profitable to interchange the
/// loop.
class LoopInterchangeProfitability {
public:
  LoopInterchangeProfitability(Loop *Outer, Loop *Inner, ScalarEvolution *SE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE) {}

  /// Check if the loop interchange is profitable
  bool isProfitable(unsigned InnerLoopId, unsigned OuterLoopId,
                    CharMatrix &DepMatrix);

private:
  int getInstrOrderCost();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;
};

/// LoopInterchangeTransform interchanges the loop
class LoopInterchangeTransform {
public:
  LoopInterchangeTransform(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                           LoopInfo *LI, DominatorTree *DT,
                           LoopInterchange *Pass, BasicBlock *LoopNestExit)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), LI(LI), DT(DT),
        LoopExit(LoopNestExit) {
    initialize();
  }

  /// Interchange OuterLoop and InnerLoop.
  bool transform();
  void restructureLoops(Loop *InnerLoop, Loop *OuterLoop);
  void removeChildLoop(Loop *OuterLoop, Loop *InnerLoop);
  void initialize();

private:
  void splitInnerLoopLatch(Instruction *);
  void splitOuterLoopLatch();
  void splitInnerLoopHeader();
  bool adjustLoopLinks();
  void adjustLoopPreheaders();
  void adjustOuterLoopPreheader();
  void adjustInnerLoopPreheader();
  bool adjustLoopBranches();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;
  LoopInfo *LI;
  DominatorTree *DT;
  BasicBlock *LoopExit;
};

// Main LoopInterchange Pass
struct LoopInterchange : public FunctionPass {
  static char ID;
  ScalarEvolution *SE;
  LoopInfo *LI;
  DependenceAnalysis *DA;
  DominatorTree *DT;
  LoopInterchange()
      : FunctionPass(ID), SE(nullptr), LI(nullptr), DA(nullptr), DT(nullptr) {
    initializeLoopInterchangePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DependenceAnalysis>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
  }

  bool runOnFunction(Function &F) override {
    SE = &getAnalysis<ScalarEvolution>();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DA = &getAnalysis<DependenceAnalysis>();
    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    DT = DTWP ? &DTWP->getDomTree() : nullptr;
    // Build up a worklist of loop pairs to analyze.
    SmallVector<LoopVector, 8> Worklist;

    for (Loop *L : *LI)
      populateWorklist(*L, Worklist);

    DEBUG(dbgs() << "Worklist size = " << Worklist.size() << "\n");
    bool Changed = true;
    while (!Worklist.empty()) {
      LoopVector LoopList = Worklist.pop_back_val();
      Changed = processLoopList(LoopList);
    }
    return Changed;
  }

  bool isComputableLoopNest(LoopVector LoopList) {
    for (auto I = LoopList.begin(), E = LoopList.end(); I != E; ++I) {
      Loop *L = *I;
      const SCEV *ExitCountOuter = SE->getBackedgeTakenCount(L);
      if (ExitCountOuter == SE->getCouldNotCompute()) {
        DEBUG(dbgs() << "Couldn't compute Backedge count\n");
        return false;
      }
      if (L->getNumBackEdges() != 1) {
        DEBUG(dbgs() << "NumBackEdges is not equal to 1\n");
        return false;
      }
      if (!L->getExitingBlock()) {
        DEBUG(dbgs() << "Loop Doesn't have unique exit block\n");
        return false;
      }
    }
    return true;
  }

  unsigned selectLoopForInterchange(LoopVector LoopList) {
    // TODO: Add a better heuristic to select the loop to be interchanged based
    // on the dependece matrix. Currently we select the innermost loop.
    return LoopList.size() - 1;
  }

  bool processLoopList(LoopVector LoopList) {
    bool Changed = false;
    bool containsLCSSAPHI = false;
    CharMatrix DependencyMatrix;
    if (LoopList.size() < 2) {
      DEBUG(dbgs() << "Loop doesn't contain minimum nesting level.\n");
      return false;
    }
    if (!isComputableLoopNest(LoopList)) {
      DEBUG(dbgs() << "Not vaild loop candidate for interchange\n");
      return false;
    }
    Loop *OuterMostLoop = *(LoopList.begin());

    DEBUG(dbgs() << "Processing LoopList of size = " << LoopList.size()
                 << "\n");

    if (!populateDependencyMatrix(DependencyMatrix, LoopList.size(),
                                  OuterMostLoop, DA)) {
      DEBUG(dbgs() << "Populating Dependency matrix failed\n");
      return false;
    }
#ifdef DUMP_DEP_MATRICIES
    DEBUG(dbgs() << "Dependence before inter change \n");
    printDepMatrix(DependencyMatrix);
#endif

    BasicBlock *OuterMostLoopLatch = OuterMostLoop->getLoopLatch();
    BranchInst *OuterMostLoopLatchBI =
        dyn_cast<BranchInst>(OuterMostLoopLatch->getTerminator());
    if (!OuterMostLoopLatchBI)
      return false;

    // Since we currently do not handle LCSSA PHI's any failure in loop
    // condition will now branch to LoopNestExit.
    // TODO: This should be removed once we handle LCSSA PHI nodes.

    // Get the Outermost loop exit.
    BasicBlock *LoopNestExit;
    if (OuterMostLoopLatchBI->getSuccessor(0) == OuterMostLoop->getHeader())
      LoopNestExit = OuterMostLoopLatchBI->getSuccessor(1);
    else
      LoopNestExit = OuterMostLoopLatchBI->getSuccessor(0);

    for (auto I = LoopList.begin(), E = LoopList.end(); I != E; ++I) {
      Loop *L = *I;
      BasicBlock *Latch = L->getLoopLatch();
      BasicBlock *Header = L->getHeader();
      if (Latch && Latch != Header && isa<PHINode>(Latch->begin())) {
        containsLCSSAPHI = true;
        break;
      }
    }

    // TODO: Handle lcssa PHI's. Currently LCSSA PHI's are not handled. Handle
    // the same by splitting the loop latch and adjusting loop links
    // accordingly.
    if (containsLCSSAPHI)
      return false;

    unsigned SelecLoopId = selectLoopForInterchange(LoopList);
    // Move the selected loop outwards to the best posible position.
    for (unsigned i = SelecLoopId; i > 0; i--) {
      bool Interchanged =
          processLoop(LoopList, i, i - 1, LoopNestExit, DependencyMatrix);
      if (!Interchanged)
        return Changed;
      // Loops interchanged reflect the same in LoopList
      Loop *OldOuterLoop = LoopList[i - 1];
      LoopList[i - 1] = LoopList[i];
      LoopList[i] = OldOuterLoop;

      // Update the DependencyMatrix
      interChangeDepedencies(DependencyMatrix, i, i - 1);

#ifdef DUMP_DEP_MATRICIES
      DEBUG(dbgs() << "Dependence after inter change \n");
      printDepMatrix(DependencyMatrix);
#endif
      Changed |= Interchanged;
    }
    return Changed;
  }

  bool processLoop(LoopVector LoopList, unsigned InnerLoopId,
                   unsigned OuterLoopId, BasicBlock *LoopNestExit,
                   std::vector<std::vector<char>> &DependencyMatrix) {

    DEBUG(dbgs() << "Processing Innder Loop Id = " << InnerLoopId
                 << " and OuterLoopId = " << OuterLoopId << "\n");
    Loop *InnerLoop = LoopList[InnerLoopId];
    Loop *OuterLoop = LoopList[OuterLoopId];

    LoopInterchangeLegality LIL(OuterLoop, InnerLoop, SE, this);
    if (!LIL.canInterchangeLoops(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      DEBUG(dbgs() << "Not interchanging Loops. Cannot prove legality\n");
      return false;
    }
    DEBUG(dbgs() << "Loops are legal to interchange\n");
    LoopInterchangeProfitability LIP(OuterLoop, InnerLoop, SE);
    if (!LIP.isProfitable(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      DEBUG(dbgs() << "Interchanging Loops not profitable\n");
      return false;
    }

    LoopInterchangeTransform LIT(OuterLoop, InnerLoop, SE, LI, DT, this,
                                 LoopNestExit);
    LIT.transform();
    DEBUG(dbgs() << "Loops interchanged\n");
    return true;
  }
};

} // end of namespace

static bool containsUnsafeInstructions(BasicBlock *BB) {
  for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
    if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      return true;
  }
  return false;
}

bool LoopInterchangeLegality::tightlyNested(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();

  DEBUG(dbgs() << "Checking if Loops are Tightly Nested\n");

  // A perfectly nested loop will not have any branch in between the outer and
  // inner block i.e. outer header will branch to either inner preheader and
  // outerloop latch.
  BranchInst *outerLoopHeaderBI =
      dyn_cast<BranchInst>(OuterLoopHeader->getTerminator());
  if (!outerLoopHeaderBI)
    return false;
  unsigned num = outerLoopHeaderBI->getNumSuccessors();
  for (unsigned i = 0; i < num; i++) {
    if (outerLoopHeaderBI->getSuccessor(i) != InnerLoopPreHeader &&
        outerLoopHeaderBI->getSuccessor(i) != OuterLoopLatch)
      return false;
  }

  DEBUG(dbgs() << "Checking instructions in Loop header and Loop latch \n");
  // We do not have any basic block in between now make sure the outer header
  // and outer loop latch doesnt contain any unsafe instructions.
  if (containsUnsafeInstructions(OuterLoopHeader) ||
      containsUnsafeInstructions(OuterLoopLatch))
    return false;

  DEBUG(dbgs() << "Loops are perfectly nested \n");
  // We have a perfect loop nest.
  return true;
}

static unsigned getPHICount(BasicBlock *BB) {
  unsigned PhiCount = 0;
  for (auto I = BB->begin(); isa<PHINode>(I); ++I)
    PhiCount++;
  return PhiCount;
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
  return true;
}

// This function indicates the current limitations in the transform as a result
// of which we do not proceed.
bool LoopInterchangeLegality::currentLimitations() {

  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();

  PHINode *InnerInductionVar;
  PHINode *OuterInductionVar;

  // We currently handle only 1 induction variable inside the loop. We also do
  // not handle reductions as of now.
  if (getPHICount(InnerLoopHeader) > 1)
    return true;

  if (getPHICount(OuterLoopHeader) > 1)
    return true;

  InnerInductionVar = getInductionVariable(InnerLoop, SE);
  OuterInductionVar = getInductionVariable(OuterLoop, SE);

  if (!OuterInductionVar || !InnerInductionVar) {
    DEBUG(dbgs() << "Induction variable not found\n");
    return true;
  }

  // TODO: Triangular loops are not handled for now.
  if (!isLoopStructureUnderstood(InnerInductionVar)) {
    DEBUG(dbgs() << "Loop structure not understood by pass\n");
    return true;
  }

  // TODO: Loops with LCSSA PHI's are currently not handled.
  if (isa<PHINode>(OuterLoopLatch->begin())) {
    DEBUG(dbgs() << "Found and LCSSA PHI in outer loop latch\n");
    return true;
  }
  if (InnerLoopLatch != InnerLoopHeader &&
      isa<PHINode>(InnerLoopLatch->begin())) {
    DEBUG(dbgs() << "Found and LCSSA PHI in inner loop latch\n");
    return true;
  }

  // TODO: Current limitation: Since we split the inner loop latch at the point
  // were induction variable is incremented (induction.next); We cannot have
  // more than 1 user of induction.next since it would result in broken code
  // after split.
  // e.g.
  // for(i=0;i<N;i++) {
  //    for(j = 0;j<M;j++) {
  //      A[j+1][i+2] = A[j][i]+k;
  //  }
  // }
  bool FoundInduction = false;
  Instruction *InnerIndexVarInc = nullptr;
  if (InnerInductionVar->getIncomingBlock(0) == InnerLoopPreHeader)
    InnerIndexVarInc =
        dyn_cast<Instruction>(InnerInductionVar->getIncomingValue(1));
  else
    InnerIndexVarInc =
        dyn_cast<Instruction>(InnerInductionVar->getIncomingValue(0));

  if (!InnerIndexVarInc)
    return true;

  // Since we split the inner loop latch on this induction variable. Make sure
  // we do not have any instruction between the induction variable and branch
  // instruction.

  for (auto I = InnerLoopLatch->rbegin(), E = InnerLoopLatch->rend();
       I != E && !FoundInduction; ++I) {
    if (isa<BranchInst>(*I) || isa<CmpInst>(*I) || isa<TruncInst>(*I))
      continue;
    const Instruction &Ins = *I;
    // We found an instruction. If this is not induction variable then it is not
    // safe to split this loop latch.
    if (!Ins.isIdenticalTo(InnerIndexVarInc))
      return true;
    else
      FoundInduction = true;
  }
  // The loop latch ended and we didnt find the induction variable return as
  // current limitation.
  if (!FoundInduction)
    return true;

  return false;
}

bool LoopInterchangeLegality::canInterchangeLoops(unsigned InnerLoopId,
                                                  unsigned OuterLoopId,
                                                  CharMatrix &DepMatrix) {

  if (!isLegalToInterChangeLoops(DepMatrix, InnerLoopId, OuterLoopId)) {
    DEBUG(dbgs() << "Failed interchange InnerLoopId = " << InnerLoopId
                 << "and OuterLoopId = " << OuterLoopId
                 << "due to dependence\n");
    return false;
  }

  // Create unique Preheaders if we already do not have one.
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();

  // Create  a unique outer preheader -
  // 1) If OuterLoop preheader is not present.
  // 2) If OuterLoop Preheader is same as OuterLoop Header
  // 3) If OuterLoop Preheader is same as Header of the previous loop.
  // 4) If OuterLoop Preheader is Entry node.
  if (!OuterLoopPreHeader || OuterLoopPreHeader == OuterLoop->getHeader() ||
      isa<PHINode>(OuterLoopPreHeader->begin()) ||
      !OuterLoopPreHeader->getUniquePredecessor()) {
    OuterLoopPreHeader = InsertPreheaderForLoop(OuterLoop, CurrentPass);
  }

  if (!InnerLoopPreHeader || InnerLoopPreHeader == InnerLoop->getHeader() ||
      InnerLoopPreHeader == OuterLoop->getHeader()) {
    InnerLoopPreHeader = InsertPreheaderForLoop(InnerLoop, CurrentPass);
  }

  // Check if the loops are tightly nested.
  if (!tightlyNested(OuterLoop, InnerLoop)) {
    DEBUG(dbgs() << "Loops not tightly nested\n");
    return false;
  }

  // TODO: The loops could not be interchanged due to current limitations in the
  // transform module.
  if (currentLimitations()) {
    DEBUG(dbgs() << "Not legal because of current transform limitation\n");
    return false;
  }

  return true;
}

int LoopInterchangeProfitability::getInstrOrderCost() {
  unsigned GoodOrder, BadOrder;
  BadOrder = GoodOrder = 0;
  for (auto BI = InnerLoop->block_begin(), BE = InnerLoop->block_end();
       BI != BE; ++BI) {
    for (auto I = (*BI)->begin(), E = (*BI)->end(); I != E; ++I) {
      const Instruction &Ins = *I;
      if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&Ins)) {
        unsigned NumOp = GEP->getNumOperands();
        bool FoundInnerInduction = false;
        bool FoundOuterInduction = false;
        for (unsigned i = 0; i < NumOp; ++i) {
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

bool isProfitabileForVectorization(unsigned InnerLoopId, unsigned OuterLoopId,
                                   CharMatrix &DepMatrix) {
  // TODO: Improve this heuristic to catch more cases.
  // If the inner loop is loop independent or doesn't carry any dependency it is
  // profitable to move this to outer position.
  unsigned Row = DepMatrix.size();
  for (unsigned i = 0; i < Row; ++i) {
    if (DepMatrix[i][InnerLoopId] != 'S' && DepMatrix[i][InnerLoopId] != 'I')
      return false;
    // TODO: We need to improve this heuristic.
    if (DepMatrix[i][OuterLoopId] != '=')
      return false;
  }
  // If outer loop has dependence and inner loop is loop independent then it is
  // profitable to interchange to enable parallelism.
  return true;
}

bool LoopInterchangeProfitability::isProfitable(unsigned InnerLoopId,
                                                unsigned OuterLoopId,
                                                CharMatrix &DepMatrix) {

  // TODO: Add Better Profitibility checks.
  // e.g
  // 1) Construct dependency matrix and move the one with no loop carried dep
  //    inside to enable vectorization.

  // This is rough cost estimation algorithm. It counts the good and bad order
  // of induction variables in the instruction and allows reordering if number
  // of bad orders is more than good.
  int Cost = 0;
  Cost += getInstrOrderCost();
  DEBUG(dbgs() << "Cost = " << Cost << "\n");
  if (Cost < 0)
    return true;

  // It is not profitable as per current cache profitibility model. But check if
  // we can move this loop outside to improve parallelism.
  bool ImprovesPar =
      isProfitabileForVectorization(InnerLoopId, OuterLoopId, DepMatrix);
  return ImprovesPar;
}

void LoopInterchangeTransform::removeChildLoop(Loop *OuterLoop,
                                               Loop *InnerLoop) {
  for (Loop::iterator I = OuterLoop->begin(), E = OuterLoop->end();; ++I) {
    assert(I != E && "Couldn't find loop");
    if (*I == InnerLoop) {
      OuterLoop->removeChildLoop(I);
      return;
    }
  }
}
void LoopInterchangeTransform::restructureLoops(Loop *InnerLoop,
                                                Loop *OuterLoop) {
  Loop *OuterLoopParent = OuterLoop->getParentLoop();
  if (OuterLoopParent) {
    // Remove the loop from its parent loop.
    removeChildLoop(OuterLoopParent, OuterLoop);
    removeChildLoop(OuterLoop, InnerLoop);
    OuterLoopParent->addChildLoop(InnerLoop);
  } else {
    removeChildLoop(OuterLoop, InnerLoop);
    LI->changeTopLevelLoop(OuterLoop, InnerLoop);
  }

  for (Loop::iterator I = InnerLoop->begin(), E = InnerLoop->end(); I != E; ++I)
    OuterLoop->addChildLoop(InnerLoop->removeChildLoop(I));

  InnerLoop->addChildLoop(OuterLoop);
}

bool LoopInterchangeTransform::transform() {

  DEBUG(dbgs() << "transform\n");
  bool Transformed = false;
  Instruction *InnerIndexVar;

  if (InnerLoop->getSubLoops().size() == 0) {
    BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
    DEBUG(dbgs() << "Calling Split Inner Loop\n");
    PHINode *InductionPHI = getInductionVariable(InnerLoop, SE);
    if (!InductionPHI) {
      DEBUG(dbgs() << "Failed to find the point to split loop latch \n");
      return false;
    }

    if (InductionPHI->getIncomingBlock(0) == InnerLoopPreHeader)
      InnerIndexVar = dyn_cast<Instruction>(InductionPHI->getIncomingValue(1));
    else
      InnerIndexVar = dyn_cast<Instruction>(InductionPHI->getIncomingValue(0));

    //
    // Split at the place were the induction variable is
    // incremented/decremented.
    // TODO: This splitting logic may not work always. Fix this.
    splitInnerLoopLatch(InnerIndexVar);
    DEBUG(dbgs() << "splitInnerLoopLatch Done\n");

    // Splits the inner loops phi nodes out into a seperate basic block.
    splitInnerLoopHeader();
    DEBUG(dbgs() << "splitInnerLoopHeader Done\n");
  }

  Transformed |= adjustLoopLinks();
  if (!Transformed) {
    DEBUG(dbgs() << "adjustLoopLinks Failed\n");
    return false;
  }

  restructureLoops(InnerLoop, OuterLoop);
  return true;
}

void LoopInterchangeTransform::initialize() {}

void LoopInterchangeTransform::splitInnerLoopLatch(Instruction *inc) {

  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock::iterator I = InnerLoopLatch->begin();
  BasicBlock::iterator E = InnerLoopLatch->end();
  for (; I != E; ++I) {
    if (inc == I)
      break;
  }

  BasicBlock *InnerLoopLatchPred = InnerLoopLatch;
  InnerLoopLatch = SplitBlock(InnerLoopLatchPred, I, DT, LI);
}

void LoopInterchangeTransform::splitOuterLoopLatch() {
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();
  BasicBlock *OuterLatchLcssaPhiBlock = OuterLoopLatch;
  OuterLoopLatch = SplitBlock(OuterLatchLcssaPhiBlock,
                              OuterLoopLatch->getFirstNonPHI(), DT, LI);
}

void LoopInterchangeTransform::splitInnerLoopHeader() {

  // Split the inner loop header out.
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  SplitBlock(InnerLoopHeader, InnerLoopHeader->getFirstNonPHI(), DT, LI);

  DEBUG(dbgs() << "Output of splitInnerLoopHeader InnerLoopHeaderSucc & "
                  "InnerLoopHeader \n");
}

void LoopInterchangeTransform::adjustOuterLoopPreheader() {
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  SmallVector<Instruction *, 8> Inst;
  for (auto I = OuterLoopPreHeader->begin(), E = OuterLoopPreHeader->end();
       I != E; ++I) {
    if (isa<BranchInst>(*I))
      break;
    Inst.push_back(I);
  }

  BasicBlock *InnerPreHeader = InnerLoop->getLoopPreheader();
  for (auto I = Inst.begin(), E = Inst.end(); I != E; ++I) {
    Instruction *Ins = cast<Instruction>(*I);
    Ins->moveBefore(InnerPreHeader->getTerminator());
  }
}

void LoopInterchangeTransform::adjustInnerLoopPreheader() {

  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  SmallVector<Instruction *, 8> Inst;
  for (auto I = InnerLoopPreHeader->begin(), E = InnerLoopPreHeader->end();
       I != E; ++I) {
    if (isa<BranchInst>(*I))
      break;
    Inst.push_back(I);
  }
  BasicBlock *OuterHeader = OuterLoop->getHeader();
  for (auto I = Inst.begin(), E = Inst.end(); I != E; ++I) {
    Instruction *Ins = cast<Instruction>(*I);
    Ins->moveBefore(OuterHeader->getTerminator());
  }
}

bool LoopInterchangeTransform::adjustLoopBranches() {

  DEBUG(dbgs() << "adjustLoopBranches called\n");
  // Adjust the loop preheader
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
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
  BasicBlock *InnerLoopHeaderSucessor = InnerLoopHeader->getUniqueSuccessor();
  if (!InnerLoopHeaderSucessor)
    return false;

  // Adjust Loop Preheader and headers

  unsigned NumSucc = OuterLoopPredecessorBI->getNumSuccessors();
  for (unsigned i = 0; i < NumSucc; ++i) {
    if (OuterLoopPredecessorBI->getSuccessor(i) == OuterLoopPreHeader)
      OuterLoopPredecessorBI->setSuccessor(i, InnerLoopPreHeader);
  }

  NumSucc = OuterLoopHeaderBI->getNumSuccessors();
  for (unsigned i = 0; i < NumSucc; ++i) {
    if (OuterLoopHeaderBI->getSuccessor(i) == OuterLoopLatch)
      OuterLoopHeaderBI->setSuccessor(i, LoopExit);
    else if (OuterLoopHeaderBI->getSuccessor(i) == InnerLoopPreHeader)
      OuterLoopHeaderBI->setSuccessor(i, InnerLoopHeaderSucessor);
  }

  BranchInst::Create(OuterLoopPreHeader, InnerLoopHeaderBI);
  InnerLoopHeaderBI->eraseFromParent();

  // -------------Adjust loop latches-----------
  if (InnerLoopLatchBI->getSuccessor(0) == InnerLoopHeader)
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(1);
  else
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(0);

  NumSucc = InnerLoopLatchPredecessorBI->getNumSuccessors();
  for (unsigned i = 0; i < NumSucc; ++i) {
    if (InnerLoopLatchPredecessorBI->getSuccessor(i) == InnerLoopLatch)
      InnerLoopLatchPredecessorBI->setSuccessor(i, InnerLoopLatchSuccessor);
  }

  if (OuterLoopLatchBI->getSuccessor(0) == OuterLoopHeader)
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(1);
  else
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(0);

  if (InnerLoopLatchBI->getSuccessor(1) == InnerLoopLatchSuccessor)
    InnerLoopLatchBI->setSuccessor(1, OuterLoopLatchSuccessor);
  else
    InnerLoopLatchBI->setSuccessor(0, OuterLoopLatchSuccessor);

  if (OuterLoopLatchBI->getSuccessor(0) == OuterLoopLatchSuccessor) {
    OuterLoopLatchBI->setSuccessor(0, InnerLoopLatch);
  } else {
    OuterLoopLatchBI->setSuccessor(1, InnerLoopLatch);
  }

  return true;
}
void LoopInterchangeTransform::adjustLoopPreheaders() {

  // We have interchanged the preheaders so we need to interchange the data in
  // the preheader as well.
  // This is because the content of inner preheader was previously executed
  // inside the outer loop.
  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BranchInst *InnerTermBI =
      cast<BranchInst>(InnerLoopPreHeader->getTerminator());

  SmallVector<Value *, 16> OuterPreheaderInstr;
  SmallVector<Value *, 16> InnerPreheaderInstr;

  for (auto I = OuterLoopPreHeader->begin(); !isa<BranchInst>(I); ++I)
    OuterPreheaderInstr.push_back(I);

  for (auto I = InnerLoopPreHeader->begin(); !isa<BranchInst>(I); ++I)
    InnerPreheaderInstr.push_back(I);

  BasicBlock *HeaderSplit =
      SplitBlock(OuterLoopHeader, OuterLoopHeader->getTerminator(), DT, LI);
  Instruction *InsPoint = HeaderSplit->getFirstNonPHI();
  // These instructions should now be executed inside the loop.
  // Move instruction into a new block after outer header.
  for (auto I = InnerPreheaderInstr.begin(), E = InnerPreheaderInstr.end();
       I != E; ++I) {
    Instruction *Ins = cast<Instruction>(*I);
    Ins->moveBefore(InsPoint);
  }
  // These instructions were not executed previously in the loop so move them to
  // the older inner loop preheader.
  for (auto I = OuterPreheaderInstr.begin(), E = OuterPreheaderInstr.end();
       I != E; ++I) {
    Instruction *Ins = cast<Instruction>(*I);
    Ins->moveBefore(InnerTermBI);
  }
}

bool LoopInterchangeTransform::adjustLoopLinks() {

  // Adjust all branches in the inner and outer loop.
  bool Changed = adjustLoopBranches();
  if (Changed)
    adjustLoopPreheaders();
  return Changed;
}

char LoopInterchange::ID = 0;
INITIALIZE_PASS_BEGIN(LoopInterchange, "loop-interchange",
                      "Interchanges loops for cache reuse", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)

INITIALIZE_PASS_END(LoopInterchange, "loop-interchange",
                    "Interchanges loops for cache reuse", false, false)

Pass *llvm::createLoopInterchangePass() { return new LoopInterchange(); }
