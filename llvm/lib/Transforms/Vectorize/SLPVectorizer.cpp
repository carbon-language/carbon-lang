//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//
#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE SV_NAME

#include "VecUtils.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace llvm;

static cl::opt<int>
SLPCostThreshold("slp-threshold", cl::init(1), cl::Hidden,
                 cl::desc("Only vectorize trees if the gain is above this "
                          "number. (gain = -cost of vectorization)"));
namespace {

/// The SLPVectorizer Pass.
struct SLPVectorizer : public BasicBlockPass {
  typedef std::map<Value*, BoUpSLP::StoreList> StoreListMap;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : BasicBlockPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;

  /// \brief Collect memory references and sort them according to their base
  /// object. We sort the stores to their base objects to reduce the cost of the
  /// quadratic search on the stores. TODO: We can further reduce this cost
  /// if we flush the chain creation every time we run into a memory barrier.
  bool CollectStores(BasicBlock *BB, BoUpSLP &R) {
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      StoreInst *SI = dyn_cast<StoreInst>(it);
      if (!SI)
        continue;

      // Check that the pointer points to scalars.
      if (SI->getValueOperand()->getType()->isAggregateType())
        return false;

      // Find the base of the GEP.
      Value *Ptr = SI->getPointerOperand();
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr))
        Ptr = GEP->getPointerOperand();

      // Save the store locations.
      StoreRefs[Ptr].push_back(SI);
    }
    return true;
  }

  bool RollStoreChains(BoUpSLP &R) {
    bool Changed = false;
    // Attempt to sort and vectorize each of the store-groups.
    for (StoreListMap::iterator it = StoreRefs.begin(), e = StoreRefs.end();
         it != e; ++it) {
      if (it->second.size() < 2)
        continue;

      DEBUG(dbgs()<<"SLP: Analyzing a store chain of length " <<
            it->second.size() << ".\n");

      Changed |= R.vectorizeStores(it->second, -SLPCostThreshold);
    }
    return Changed;
  }

  virtual bool runOnBasicBlock(BasicBlock &BB) {
    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    TTI = &getAnalysis<TargetTransformInfo>();
    AA = &getAnalysis<AliasAnalysis>();
    StoreRefs.clear();

    // Must have DataLayout. We can't require it because some tests run w/o
    // triple.
    if (!DL)
      return false;

    // Use the bollom up slp vectorizer to construct chains that start with
    // he store instructions.
    BoUpSLP R(&BB, SE, DL, TTI, AA);

    if (!CollectStores(&BB, R))
      return false;

    bool Changed = RollStoreChains(R);
    if (Changed) {
      DEBUG(dbgs()<<"SLP: vectorized in \""<<BB.getParent()->getName()<<"\"\n");
      DEBUG(verifyFunction(*BB.getParent()));
    }

    return Changed;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    BasicBlockPass::getAnalysisUsage(AU);
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<TargetTransformInfo>();
  }

private:
  StoreListMap StoreRefs;
};

} // end anonymous namespace

char SLPVectorizer::ID = 0;
static const char lv_name[] = "SLP Vectorizer";
INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

namespace llvm {
  Pass *createSLPVectorizerPass() {
    return new SLPVectorizer();
  }
}

