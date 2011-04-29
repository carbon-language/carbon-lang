//===---- CodePreparation.cpp - Code preparation for Scop Detection -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implement the code preparation for Scop detect, which will:
//    1. Translate all PHINodes that not induction variable to memory access,
//       this will easier parameter and scalar dependencies checking.
//
//===----------------------------------------------------------------------===//
#include "polly/LinkAllPasses.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "polly-code-prep"
#include "llvm/Support/Debug.h"


using namespace llvm;
using namespace polly;

namespace {
//===----------------------------------------------------------------------===//
/// @brief Scop Code Preparation - Perform some transforms to make scop detect
/// easier.
///
class CodePreperation : public FunctionPass {
  // DO NOT IMPLEMENT.
  CodePreperation(const CodePreperation &);
  // DO NOT IMPLEMENT.
  const CodePreperation &operator=(const CodePreperation &);

  // LoopInfo to compute canonical induction variable.
  LoopInfo *LI;

  // Clear the context.
  void clear();

  bool eliminatePHINodes(Function &F);

public:
  static char ID;

  explicit CodePreperation() : FunctionPass(ID) {}
  ~CodePreperation();

  /// @name FunctionPass interface.
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}

};
}

//===----------------------------------------------------------------------===//
/// CodePreperation implement.

void CodePreperation::clear() {
}

CodePreperation::~CodePreperation() {
  clear();
}

bool CodePreperation::eliminatePHINodes(Function &F) {
  // The PHINodes that will be deleted.
  std::vector<PHINode*> PNtoDel;
  // The PHINodes that will be preserved.
  std::vector<PHINode*> PreservedPNs;

  // Scan the PHINodes in this function.
  for (Function::iterator ibb = F.begin(), ibe = F.end();
      ibb != ibe; ++ibb)
    for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->getFirstNonPHI();
        iib != iie; ++iib)
      if (PHINode *PN = cast<PHINode>(iib)) {
        if (Loop *L = LI->getLoopFor(ibb)) {
          // Induction variable will be preserved.
          if (L->getCanonicalInductionVariable() == PN) {
            PreservedPNs.push_back(PN);
            continue;
          }
        }

        // As DemotePHIToStack does not support invoke edges, we preserve
        // PHINodes that have invoke edges.
        if (hasInvokeEdge(PN))
          PreservedPNs.push_back(PN);
        else
          PNtoDel.push_back(PN);
      }

  if (PNtoDel.empty())
    return false;

  // Eliminate the PHINodes that not an Induction variable.
  while (!PNtoDel.empty()) {
    PHINode *PN = PNtoDel.back();
    PNtoDel.pop_back();

    DemotePHIToStack(PN);
  }

  // Move all preserved PHINodes to the beginning of the BasicBlock.
  while (!PreservedPNs.empty()) {
    PHINode *PN = PreservedPNs.back();
    PreservedPNs.pop_back();

    BasicBlock *BB = PN->getParent();
    if (PN == BB->begin())
      continue;

    PN->moveBefore(BB->begin());
  }

  return true;
}

void CodePreperation::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();

  AU.addPreserved<LoopInfo>();
  AU.addPreserved<RegionInfo>();
  AU.addPreserved<DominatorTree>();
  AU.addPreserved<DominanceFrontier>();
}

bool CodePreperation::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfo>();

  splitEntryBlockForAlloca(&F.getEntryBlock(), this);

  eliminatePHINodes(F);

  return false;
}

void CodePreperation::releaseMemory() {
  clear();
}

void CodePreperation::print(raw_ostream &OS, const Module *) const {
}

char CodePreperation::ID = 0;

RegisterPass<CodePreperation> X("polly-prepare",
                              "Polly - Prepare code for polly.",
                              false, true);

char &polly::CodePreperationID = CodePreperation::ID;

Pass *polly::createCodePreperationPass() {
  return new CodePreperation();
}
