//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include <vector>
using namespace llvm;

namespace {

// FIXME: PassManager should allow Module passes to require FunctionPasses
struct LoopExtractor : public FunctionPass {

public:
  LoopExtractor() {}
  virtual bool run(Module &M);
  virtual bool runOnFunction(Function &F);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfo>();
  }

};

RegisterOpt<LoopExtractor> 
X("loop-extract", "Extract loops into new functions");

bool LoopExtractor::run(Module &M) {
  bool Changed = false;
  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i)
    Changed |= runOnFunction(*i);
  return Changed;
}

bool LoopExtractor::runOnFunction(Function &F) {
  std::cerr << F.getName() << "\n";

  LoopInfo &LI = getAnalysis<LoopInfo>();

  // We don't want to keep extracting the only loop of a function into a new one
  if (LI.begin() == LI.end() || LI.begin() + 1 == LI.end())
    return false;

  bool Changed = false;

  // Try to move each loop out of the code into separate function
  for (LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i)
    Changed |= (ExtractLoop(*i) != 0);

  return Changed;
}



} // End anonymous namespace 

/// createLoopExtractorPass 
///
FunctionPass* llvm::createLoopExtractorPass() {
  return new LoopExtractor();
}
