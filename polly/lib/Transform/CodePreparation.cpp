//===---- CodePreparation.cpp - Code preparation for Scop Detection -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The Polly code preparation pass is executed before SCoP detection. Its only
// use is to translate all PHI nodes that can not be expressed by the code
// generator into explicit memory dependences.
//
// Polly's code generation can code generate all PHI nodes that do not
// reference parameters within the scop. As the code preparation pass is run
// before scop detection, we can not check this condition, because without
// a detected scop, we do not know SCEVUnknowns that appear in the SCEV of
// a PHI node may later be within or outside of the SCoP. Hence, we follow a
// heuristic and translate all PHI nodes that are either directly SCEVUnknown
// or SCEVCouldNotCompute. This will hopefully get most of the PHI nodes that
// are introduced due to conditional control flow, but not the ones that are
// referencing loop counters.
//
// XXX: In the future, we should remove the need for this pass entirely and
// instead add support for scalar dependences to ScopInfo and code generation.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace polly;

namespace {

// Helper function which (for a given PHI node):
//
// 1) Remembers all incoming values and the associated basic blocks
// 2) Demotes the phi node to the stack
// 3) Remembers the correlation between the PHI node and the new alloca
//
// When we combine the information from 1) and 3) we know the values stored
// in this alloca at the end of the predecessor basic blocks of the PHI.
static void DemotePHI(
    PHINode *PN, DenseMap<PHINode *, AllocaInst *> &PNallocMap,
    DenseMap<std::pair<Value *, BasicBlock *>, PHINode *> &ValueLocToPhiMap) {

  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    auto *InVal = PN->getIncomingValue(i);
    auto *InBB = PN->getIncomingBlock(i);
    ValueLocToPhiMap[std::make_pair(InVal, InBB)] = PN;
  }

  PNallocMap[PN] = DemotePHIToStack(PN);
}

/// @brief Prepare the IR for the scop detection.
///
class CodePreparation : public FunctionPass {
  CodePreparation(const CodePreparation &) LLVM_DELETED_FUNCTION;
  const CodePreparation &
  operator=(const CodePreparation &) LLVM_DELETED_FUNCTION;

  LoopInfo *LI;
  ScalarEvolution *SE;

  void clear();

  bool eliminatePHINodes(Function &F);

public:
  static char ID;

  explicit CodePreparation() : FunctionPass(ID) {}
  ~CodePreparation();

  /// @name FunctionPass interface.
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}
};
}

void CodePreparation::clear() {}

CodePreparation::~CodePreparation() { clear(); }

bool CodePreparation::eliminatePHINodes(Function &F) {
  // The PHINodes that will be demoted.
  std::vector<PHINode *> PNtoDemote;
  // The PHINodes that will be deleted (stack slot sharing).
  std::vector<PHINode *> PNtoDelete;
  // The PHINodes that will be preserved.
  std::vector<PHINode *> PNtoPreserve;
  // Map to remember values stored in PHINodes at the end of basic blocks.
  DenseMap<std::pair<Value *, BasicBlock *>, PHINode *> ValueLocToPhiMap;
  // Map from PHINodes to their alloca (after demotion) counterpart.
  DenseMap<PHINode *, AllocaInst *> PNallocMap;

  // Scan the PHINodes in this function and categorize them to be either:
  // o Preserved, if they are (canonical) induction variables or can be
  //              synthesized during code generation ('SCEVable')
  // o Deleted, if they are trivial PHI nodes (one incoming value) and the
  //            incoming value is a PHI node we will demote
  // o Demoted, if they do not fit any of the previous categories
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI)
    for (BasicBlock::iterator II = BI->begin(), IE = BI->getFirstNonPHI();
         II != IE; ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (SE->isSCEVable(PN->getType())) {
        const SCEV *S = SE->getSCEV(PN);
        if (!isa<SCEVUnknown>(S) && !isa<SCEVCouldNotCompute>(S)) {
          PNtoPreserve.push_back(PN);
          continue;
        }
      }

      // As DemotePHIToStack does not support invoke edges, we preserve
      // PHINodes that have invoke edges.
      if (hasInvokeEdge(PN)) {
        PNtoPreserve.push_back(PN);
      } else {
        if (PN->getNumIncomingValues() == 1)
          PNtoDelete.push_back(PN);
        else
          PNtoDemote.push_back(PN);
      }
    }

  if (PNtoDemote.empty() && PNtoDelete.empty())
    return false;

  while (!PNtoDemote.empty()) {
    PHINode *PN = PNtoDemote.back();
    PNtoDemote.pop_back();
    DemotePHI(PN, PNallocMap, ValueLocToPhiMap);
  }

  // For each trivial PHI we encountered (and we want to delete) we try to find
  // the value it will hold in a alloca we already created by PHI demotion. If
  // we succeed (the incoming value is stored in an alloca at the predecessor
  // block), we can replace the trivial PHI by the value stored in the alloca.
  // If not, we will demote this trivial PHI as any other one.
  for (auto PNIt = PNtoDelete.rbegin(), PNEnd = PNtoDelete.rend();
       PNIt != PNEnd; ++PNIt) {
    PHINode *TrivPN = *PNIt;
    assert(TrivPN->getNumIncomingValues() == 1 && "Assumed trivial PHI");

    auto *InVal = TrivPN->getIncomingValue(0);
    auto *InBB = TrivPN->getIncomingBlock(0);
    const auto &ValLocIt = ValueLocToPhiMap.find(std::make_pair(InVal, InBB));
    if (ValLocIt != ValueLocToPhiMap.end()) {
      PHINode *InPHI = ValLocIt->second;
      assert(PNallocMap.count(InPHI) &&
             "Inconsitent state, PN was not demoted!");
      auto *InPHIAlloca = PNallocMap[InPHI];
      PNallocMap[TrivPN] = InPHIAlloca;
      LoadInst *LI = new LoadInst(InPHIAlloca, "",
                                  TrivPN->getParent()->getFirstInsertionPt());
      TrivPN->replaceAllUsesWith(LI);
      TrivPN->eraseFromParent();
      continue;
    }

    DemotePHI(TrivPN, PNallocMap, ValueLocToPhiMap);
  }

  // Move preserved PHINodes to the beginning of the BasicBlock.
  while (!PNtoPreserve.empty()) {
    PHINode *PN = PNtoPreserve.back();
    PNtoPreserve.pop_back();

    BasicBlock *BB = PN->getParent();
    if (PN == BB->begin())
      continue;

    PN->moveBefore(BB->begin());
  }

  return true;
}

void CodePreparation::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();

  AU.addPreserved<LoopInfo>();
  AU.addPreserved<RegionInfoPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<DominanceFrontier>();
}

bool CodePreparation::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();

  splitEntryBlockForAlloca(&F.getEntryBlock(), this);

  eliminatePHINodes(F);

  return false;
}

void CodePreparation::releaseMemory() { clear(); }

void CodePreparation::print(raw_ostream &OS, const Module *) const {}

char CodePreparation::ID = 0;
char &polly::CodePreparationID = CodePreparation::ID;

Pass *polly::createCodePreparationPass() { return new CodePreparation(); }

INITIALIZE_PASS_BEGIN(CodePreparation, "polly-prepare",
                      "Polly - Prepare code for polly", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(CodePreparation, "polly-prepare",
                    "Polly - Prepare code for polly", false, false)
