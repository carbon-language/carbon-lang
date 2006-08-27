//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched. This is a pass most useful for debugging
// via bugpoint.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumExtracted("loop-extract", "Number of loops extracted");

  // FIXME: This is not a function pass, but the PassManager doesn't allow
  // Module passes to require FunctionPasses, so we can't get loop info if we're
  // not a function pass.
  struct LoopExtractor : public FunctionPass {
    unsigned NumLoops;

    LoopExtractor(unsigned numLoops = ~0) : NumLoops(numLoops) {}

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<DominatorSet>();
      AU.addRequired<LoopInfo>();
    }
  };

  RegisterPass<LoopExtractor>
  X("loop-extract", "Extract loops into new functions");

  /// SingleLoopExtractor - For bugpoint.
  struct SingleLoopExtractor : public LoopExtractor {
    SingleLoopExtractor() : LoopExtractor(1) {}
  };

  RegisterPass<SingleLoopExtractor>
  Y("loop-extract-single", "Extract at most one loop into a new function");
} // End anonymous namespace

// createLoopExtractorPass - This pass extracts all natural loops from the
// program into a function if it can.
//
FunctionPass *llvm::createLoopExtractorPass() { return new LoopExtractor(); }

bool LoopExtractor::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();

  // If this function has no loops, there is nothing to do.
  if (LI.begin() == LI.end())
    return false;

  DominatorSet &DS = getAnalysis<DominatorSet>();

  // If there is more than one top-level loop in this function, extract all of
  // the loops.
  bool Changed = false;
  if (LI.end()-LI.begin() > 1) {
    for (LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i) {
      if (NumLoops == 0) return Changed;
      --NumLoops;
      Changed |= ExtractLoop(DS, *i) != 0;
      ++NumExtracted;
    }
  } else {
    // Otherwise there is exactly one top-level loop.  If this function is more
    // than a minimal wrapper around the loop, extract the loop.
    Loop *TLL = *LI.begin();
    bool ShouldExtractLoop = false;

    // Extract the loop if the entry block doesn't branch to the loop header.
    TerminatorInst *EntryTI = F.getEntryBlock().getTerminator();
    if (!isa<BranchInst>(EntryTI) ||
        !cast<BranchInst>(EntryTI)->isUnconditional() ||
        EntryTI->getSuccessor(0) != TLL->getHeader())
      ShouldExtractLoop = true;
    else {
      // Check to see if any exits from the loop are more than just return
      // blocks.
      std::vector<BasicBlock*> ExitBlocks;
      TLL->getExitBlocks(ExitBlocks);
      for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
        if (!isa<ReturnInst>(ExitBlocks[i]->getTerminator())) {
          ShouldExtractLoop = true;
          break;
        }
    }

    if (ShouldExtractLoop) {
      if (NumLoops == 0) return Changed;
      --NumLoops;
      Changed |= ExtractLoop(DS, TLL) != 0;
      ++NumExtracted;
    } else {
      // Okay, this function is a minimal container around the specified loop.
      // If we extract the loop, we will continue to just keep extracting it
      // infinitely... so don't extract it.  However, if the loop contains any
      // subloops, extract them.
      for (Loop::iterator i = TLL->begin(), e = TLL->end(); i != e; ++i) {
        if (NumLoops == 0) return Changed;
        --NumLoops;
        Changed |= ExtractLoop(DS, *i) != 0;
        ++NumExtracted;
      }
    }
  }

  return Changed;
}

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
FunctionPass *llvm::createSingleLoopExtractorPass() {
  return new SingleLoopExtractor();
}


namespace {
  /// BlockExtractorPass - This pass is used by bugpoint to extract all blocks
  /// from the module into their own functions except for those specified by the
  /// BlocksToNotExtract list.
  class BlockExtractorPass : public ModulePass {
    std::vector<BasicBlock*> BlocksToNotExtract;
  public:
    BlockExtractorPass(std::vector<BasicBlock*> &B) : BlocksToNotExtract(B) {}
    BlockExtractorPass() {}

    bool runOnModule(Module &M);
  };
  RegisterPass<BlockExtractorPass>
  XX("extract-blocks", "Extract Basic Blocks From Module (for bugpoint use)");
}

// createBlockExtractorPass - This pass extracts all blocks (except those
// specified in the argument list) from the functions in the module.
//
ModulePass *llvm::createBlockExtractorPass(std::vector<BasicBlock*> &BTNE) {
  return new BlockExtractorPass(BTNE);
}

bool BlockExtractorPass::runOnModule(Module &M) {
  std::set<BasicBlock*> TranslatedBlocksToNotExtract;
  for (unsigned i = 0, e = BlocksToNotExtract.size(); i != e; ++i) {
    BasicBlock *BB = BlocksToNotExtract[i];
    Function *F = BB->getParent();

    // Map the corresponding function in this module.
    Function *MF = M.getFunction(F->getName(), F->getFunctionType());

    // Figure out which index the basic block is in its function.
    Function::iterator BBI = MF->begin();
    std::advance(BBI, std::distance(F->begin(), Function::iterator(BB)));
    TranslatedBlocksToNotExtract.insert(BBI);
  }

  // Now that we know which blocks to not extract, figure out which ones we WANT
  // to extract.
  std::vector<BasicBlock*> BlocksToExtract;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      if (!TranslatedBlocksToNotExtract.count(BB))
        BlocksToExtract.push_back(BB);

  for (unsigned i = 0, e = BlocksToExtract.size(); i != e; ++i)
    ExtractBasicBlock(BlocksToExtract[i]);

  return !BlocksToExtract.empty();
}
