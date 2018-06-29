//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a '-dot-cfg' analysis pass, which emits the
// cfg.<fnname>.dot file for each function in the program, with a graph of the
// CFG for that function.
//
// The other main feature of this file is that it implements the
// Function::viewCFG method, which is useful for debugging passes which operate
// on the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
using namespace llvm;

static cl::opt<bool> CFGHeatPerFunction("cfg-heat-per-function",
                                        cl::init(false), cl::Hidden,
                                        cl::desc("Heat CFG per function"));

static cl::opt<bool> ShowHeatColors("cfg-heat-colors", cl::init(true),
                                    cl::Hidden,
                                    cl::desc("Show heat colors in CFG"));

static cl::opt<bool> UseRawEdgeWeight("cfg-raw-weights", cl::init(false),
                                      cl::Hidden,
                                      cl::desc("Use raw weights for labels. "
                                               "Use percentages as default."));

static cl::opt<bool> ShowEdgeWeight("cfg-weights", cl::init(true), cl::Hidden,
                                    cl::desc("Show edges labeled with weights"));

static void writeHeatCFGToDotFile(Function &F, BlockFrequencyInfo *BFI,
                                 BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                                 bool UseHeuristic, bool isSimple) {
  std::string Filename = ("cfg." + F.getName() + ".dot").str();
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::F_Text);

  CFGDOTInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeuristic(UseHeuristic);
  CFGInfo.setHeatColors(ShowHeatColors);
  CFGInfo.setEdgeWeights(ShowEdgeWeight);
  CFGInfo.setRawEdgeWeights(UseRawEdgeWeight);

  if (!EC)
    WriteGraph(File, &CFGInfo, isSimple);
  else
    errs() << "  error opening file for writing!";
  errs() << "\n";
}

static void writeAllCFGsToDotFile(Module &M,
                       function_ref<BlockFrequencyInfo *(Function &)> LookupBFI,
                       function_ref<BranchProbabilityInfo *(Function &)> LookupBPI,
                       bool isSimple) {
  bool UseHeuristic = true;
  uint64_t MaxFreq = 0;
  if (!CFGHeatPerFunction)
    MaxFreq = getMaxFreq(M, LookupBFI, UseHeuristic);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    auto *BFI = LookupBFI(F);
    auto *BPI = LookupBPI(F);
    if (CFGHeatPerFunction)
      MaxFreq = getMaxFreq(F, BFI, UseHeuristic);
    writeHeatCFGToDotFile(F, BFI, BPI, MaxFreq, UseHeuristic, isSimple);
  }

}

static void viewHeatCFG(Function &F, BlockFrequencyInfo *BFI,
                                 BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                                 bool UseHeuristic, bool isSimple) {
  CFGDOTInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeuristic(UseHeuristic);
  CFGInfo.setHeatColors(ShowHeatColors);
  CFGInfo.setEdgeWeights(ShowEdgeWeight);
  CFGInfo.setRawEdgeWeights(UseRawEdgeWeight);

  ViewGraph(&CFGInfo, "cfg." + F.getName(), isSimple);
}

static void viewAllCFGs(Module &M,
                       function_ref<BlockFrequencyInfo *(Function &)> LookupBFI,
                       function_ref<BranchProbabilityInfo *(Function &)> LookupBPI,
                       bool isSimple) {
  bool UseHeuristic = true;
  uint64_t MaxFreq = 0;
  if (!CFGHeatPerFunction)
    MaxFreq = getMaxFreq(M, LookupBFI, UseHeuristic);

  for (auto &F : M) {
    if (F.isDeclaration()) continue;
    auto *BFI = LookupBFI(F);
    auto *BPI = LookupBPI(F);
    if (CFGHeatPerFunction)
      MaxFreq = getMaxFreq(F, BFI, UseHeuristic);
    viewHeatCFG(F, BFI, BPI, MaxFreq, UseHeuristic, isSimple);
  }

}

namespace {
  struct CFGViewerLegacyPass : public ModulePass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGViewerLegacyPass() : ModulePass(ID) {
      initializeCFGViewerLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override {
      auto LookupBFI = [this](Function &F) {
        return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
      };
      auto LookupBPI = [this](Function &F) {
        return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
      };
      viewAllCFGs(M, LookupBFI, LookupBPI, /*isSimple=*/false);
      return false;
    }

    void print(raw_ostream &OS, const Module * = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
      AU.addRequired<BlockFrequencyInfoWrapperPass>();
      AU.addRequired<BranchProbabilityInfoWrapperPass>();
      AU.setPreservesAll();
    }

  };
}

char CFGViewerLegacyPass::ID = 0;
INITIALIZE_PASS(CFGViewerLegacyPass, "view-cfg", "View CFG of function", false, true)

PreservedAnalyses CFGViewerPass::run(Module &M,
                                     ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  auto LookupBPI = [&FAM](Function &F) {
    return &FAM.getResult<BranchProbabilityAnalysis>(F);
  };
  viewAllCFGs(M, LookupBFI, LookupBPI, /*isSimple=*/false);
  return PreservedAnalyses::all();
}


namespace {
  struct CFGOnlyViewerLegacyPass : public ModulePass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGOnlyViewerLegacyPass() : ModulePass(ID) {
      initializeCFGOnlyViewerLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override {
      auto LookupBFI = [this](Function &F) {
        return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
      };
      auto LookupBPI = [this](Function &F) {
        return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
      };
      viewAllCFGs(M, LookupBFI, LookupBPI, /*isSimple=*/true);
      return false;
    }

    void print(raw_ostream &OS, const Module * = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
      AU.addRequired<BlockFrequencyInfoWrapperPass>();
      AU.addRequired<BranchProbabilityInfoWrapperPass>();
      AU.setPreservesAll();
    }

  };
}

char CFGOnlyViewerLegacyPass::ID = 0;
INITIALIZE_PASS(CFGOnlyViewerLegacyPass, "view-cfg-only",
                "View CFG of function (with no function bodies)", false, true)

PreservedAnalyses CFGOnlyViewerPass::run(Module &M,
                                         ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  auto LookupBPI = [&FAM](Function &F) {
    return &FAM.getResult<BranchProbabilityAnalysis>(F);
  };
  viewAllCFGs(M, LookupBFI, LookupBPI, /*isSimple=*/true);
  return PreservedAnalyses::all();
}

namespace {
  struct CFGPrinterLegacyPass : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    CFGPrinterLegacyPass() : ModulePass(ID) {
      initializeCFGPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override {
      auto LookupBFI = [this](Function &F) {
        return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
      };
      auto LookupBPI = [this](Function &F) {
        return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
      };
      writeAllCFGsToDotFile(M, LookupBFI, LookupBPI, /*isSimple=*/false);
      return false;
    }

    void print(raw_ostream &OS, const Module * = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
      AU.addRequired<BlockFrequencyInfoWrapperPass>();
      AU.addRequired<BranchProbabilityInfoWrapperPass>();
      AU.setPreservesAll();
    }

  };
}

char CFGPrinterLegacyPass::ID = 0;
INITIALIZE_PASS(CFGPrinterLegacyPass, "dot-cfg", "Print CFG of function to 'dot' file", 
                false, true)

PreservedAnalyses CFGPrinterPass::run(Module &M,
                                      ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  auto LookupBPI = [&FAM](Function &F) {
    return &FAM.getResult<BranchProbabilityAnalysis>(F);
  };
  writeAllCFGsToDotFile(M, LookupBFI, LookupBPI, /*isSimple=*/false);
  return PreservedAnalyses::all();
}

namespace {
  struct CFGOnlyPrinterLegacyPass : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    CFGOnlyPrinterLegacyPass() : ModulePass(ID) {
      initializeCFGOnlyPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override {
      auto LookupBFI = [this](Function &F) {
        return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
      };
      auto LookupBPI = [this](Function &F) {
        return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
      };
      writeAllCFGsToDotFile(M, LookupBFI, LookupBPI, /*isSimple=*/true);
      return false;
    }

    void print(raw_ostream &OS, const Module * = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
      AU.addRequired<BlockFrequencyInfoWrapperPass>();
      AU.addRequired<BranchProbabilityInfoWrapperPass>();
      AU.setPreservesAll();
    }

  };
}

char CFGOnlyPrinterLegacyPass::ID = 0;
INITIALIZE_PASS(CFGOnlyPrinterLegacyPass, "dot-cfg-only",
   "Print CFG of function to 'dot' file (with no function bodies)",
   false, true)

PreservedAnalyses CFGOnlyPrinterPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  auto LookupBPI = [&FAM](Function &F) {
    return &FAM.getResult<BranchProbabilityAnalysis>(F);
  };
  writeAllCFGsToDotFile(M, LookupBFI, LookupBPI, /*isSimple=*/true);
  return PreservedAnalyses::all();
}

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const {

  CFGDOTInfo CFGInfo(this);
  ViewGraph(&CFGInfo, "cfg" + getName());
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG
/// this can make the graph smaller.
///
void Function::viewCFGOnly() const {

  CFGDOTInfo CFGInfo(this);
  ViewGraph(&CFGInfo, "cfg" + getName(), true);
}

ModulePass *llvm::createCFGPrinterLegacyPassPass() {
  return new CFGPrinterLegacyPass();
}

ModulePass *llvm::createCFGOnlyPrinterLegacyPassPass() {
  return new CFGOnlyPrinterLegacyPass();
}
