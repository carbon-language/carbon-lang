//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched. This is a pass most useful for debugging
// via bugpoint.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-extract"
#include "llvm/Transforms/IPO.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/ADT/Statistic.h"
#include <fstream>
#include <set>
using namespace llvm;

STATISTIC(NumExtracted, "Number of loops extracted");

namespace {
  struct LoopExtractor : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    unsigned NumLoops;

    explicit LoopExtractor(unsigned numLoops = ~0) 
      : LoopPass(&ID), NumLoops(numLoops) {}

    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<DominatorTree>();
    }
  };
}

char LoopExtractor::ID = 0;
static RegisterPass<LoopExtractor>
X("loop-extract", "Extract loops into new functions");

namespace {
  /// SingleLoopExtractor - For bugpoint.
  struct SingleLoopExtractor : public LoopExtractor {
    static char ID; // Pass identification, replacement for typeid
    SingleLoopExtractor() : LoopExtractor(1) {}
  };
} // End anonymous namespace

char SingleLoopExtractor::ID = 0;
static RegisterPass<SingleLoopExtractor>
Y("loop-extract-single", "Extract at most one loop into a new function");

// createLoopExtractorPass - This pass extracts all natural loops from the
// program into a function if it can.
//
Pass *llvm::createLoopExtractorPass() { return new LoopExtractor(); }

bool LoopExtractor::runOnLoop(Loop *L, LPPassManager &LPM) {
  // Only visit top-level loops.
  if (L->getParentLoop())
    return false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm())
    return false;

  DominatorTree &DT = getAnalysis<DominatorTree>();
  bool Changed = false;

  // If there is more than one top-level loop in this function, extract all of
  // the loops. Otherwise there is exactly one top-level loop; in this case if
  // this function is more than a minimal wrapper around the loop, extract
  // the loop.
  bool ShouldExtractLoop = false;

  // Extract the loop if the entry block doesn't branch to the loop header.
  TerminatorInst *EntryTI =
    L->getHeader()->getParent()->getEntryBlock().getTerminator();
  if (!isa<BranchInst>(EntryTI) ||
      !cast<BranchInst>(EntryTI)->isUnconditional() ||
      EntryTI->getSuccessor(0) != L->getHeader())
    ShouldExtractLoop = true;
  else {
    // Check to see if any exits from the loop are more than just return
    // blocks.
    SmallVector<BasicBlock*, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      if (!isa<ReturnInst>(ExitBlocks[i]->getTerminator())) {
        ShouldExtractLoop = true;
        break;
      }
  }
  if (ShouldExtractLoop) {
    if (NumLoops == 0) return Changed;
    --NumLoops;
    if (ExtractLoop(DT, L) != 0) {
      Changed = true;
      // After extraction, the loop is replaced by a function call, so
      // we shouldn't try to run any more loop passes on it.
      LPM.deleteLoopFromQueue(L);
    }
    ++NumExtracted;
  }

  return Changed;
}

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
Pass *llvm::createSingleLoopExtractorPass() {
  return new SingleLoopExtractor();
}


// BlockFile - A file which contains a list of blocks that should not be
// extracted.
static cl::opt<std::string>
BlockFile("extract-blocks-file", cl::value_desc("filename"),
          cl::desc("A file containing list of basic blocks to not extract"),
          cl::Hidden);

namespace {
  /// BlockExtractorPass - This pass is used by bugpoint to extract all blocks
  /// from the module into their own functions except for those specified by the
  /// BlocksToNotExtract list.
  class BlockExtractorPass : public ModulePass {
    void LoadFile(const char *Filename);

    std::vector<BasicBlock*> BlocksToNotExtract;
    std::vector<std::pair<std::string, std::string> > BlocksToNotExtractByName;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit BlockExtractorPass(const std::vector<BasicBlock*> &B) 
      : ModulePass(&ID), BlocksToNotExtract(B) {
      if (!BlockFile.empty())
        LoadFile(BlockFile.c_str());
    }
    BlockExtractorPass() : ModulePass(&ID) {}

    bool runOnModule(Module &M);
  };
}

char BlockExtractorPass::ID = 0;
static RegisterPass<BlockExtractorPass>
XX("extract-blocks", "Extract Basic Blocks From Module (for bugpoint use)");

// createBlockExtractorPass - This pass extracts all blocks (except those
// specified in the argument list) from the functions in the module.
//
ModulePass *llvm::createBlockExtractorPass(const std::vector<BasicBlock*> &BTNE)
{
  return new BlockExtractorPass(BTNE);
}

void BlockExtractorPass::LoadFile(const char *Filename) {
  // Load the BlockFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: BlockExtractor couldn't load file '" << Filename
           << "'!\n";
    return;
  }
  while (In) {
    std::string FunctionName, BlockName;
    In >> FunctionName;
    In >> BlockName;
    if (!BlockName.empty())
      BlocksToNotExtractByName.push_back(
          std::make_pair(FunctionName, BlockName));
  }
}

bool BlockExtractorPass::runOnModule(Module &M) {
  std::set<BasicBlock*> TranslatedBlocksToNotExtract;
  for (unsigned i = 0, e = BlocksToNotExtract.size(); i != e; ++i) {
    BasicBlock *BB = BlocksToNotExtract[i];
    Function *F = BB->getParent();

    // Map the corresponding function in this module.
    Function *MF = M.getFunction(F->getName());
    assert(MF->getFunctionType() == F->getFunctionType() && "Wrong function?");

    // Figure out which index the basic block is in its function.
    Function::iterator BBI = MF->begin();
    std::advance(BBI, std::distance(F->begin(), Function::iterator(BB)));
    TranslatedBlocksToNotExtract.insert(BBI);
  }

  while (!BlocksToNotExtractByName.empty()) {
    // There's no way to find BBs by name without looking at every BB inside
    // every Function. Fortunately, this is always empty except when used by
    // bugpoint in which case correctness is more important than performance.

    std::string &FuncName  = BlocksToNotExtractByName.back().first;
    std::string &BlockName = BlocksToNotExtractByName.back().second;

    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
      Function &F = *FI;
      if (F.getName() != FuncName) continue;

      for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
        BasicBlock &BB = *BI;
        if (BB.getName() != BlockName) continue;

        TranslatedBlocksToNotExtract.insert(BI);
      }
    }

    BlocksToNotExtractByName.pop_back();
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
