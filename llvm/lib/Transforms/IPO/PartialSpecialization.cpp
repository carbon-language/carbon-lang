//===-- PartialSpecialization.cpp - Specialize for common constants--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass finds function arguments that are often a common constant and 
// specializes a version of the called function for that constant.
//
// This pass simply does the cloning for functions it specializes.  It depends
// on IPSCCP and DAE to clean up the results.
//
// The initial heuristic favors constant arguments that are used in control 
// flow.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "partialspecialization"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Compiler.h"
#include <map>
using namespace llvm;

STATISTIC(numSpecialized, "Number of specialized functions created");

// Call must be used at least occasionally
static const int CallsMin = 5;

// Must have 10% of calls having the same constant to specialize on
static const double ConstValPercent = .1;

namespace {
  class VISIBILITY_HIDDEN PartSpec : public ModulePass {
    void scanForInterest(Function&, SmallVector<int, 6>&);
    void replaceUsersFor(Function&, int, Constant*, Function*);
    int scanDistribution(Function&, int, std::map<Constant*, int>&);
  public :
    static char ID; // Pass identification, replacement for typeid
    PartSpec() : ModulePass(&ID) {}
    bool runOnModule(Module &M);
  };
}

char PartSpec::ID = 0;
static RegisterPass<PartSpec>
X("partialspecialization", "Partial Specialization");

bool PartSpec::runOnModule(Module &M) {
  bool Changed = false;
  for (Module::iterator I = M.begin(); I != M.end(); ++I) {
    Function &F = *I;
    if (F.isDeclaration()) continue;
    SmallVector<int, 6> interestingArgs;
    scanForInterest(F, interestingArgs);

    // Find the first interesting Argument that we can specialize on
    // If there are multiple interesting Arguments, then those will be found
    // when processing the cloned function.
    bool breakOuter = false;
    for (unsigned int x = 0; !breakOuter && x < interestingArgs.size(); ++x) {
      std::map<Constant*, int> distribution;
      int total = scanDistribution(F, interestingArgs[x], distribution);
      if (total > CallsMin) 
        for (std::map<Constant*, int>::iterator ii = distribution.begin(),
               ee = distribution.end(); ii != ee; ++ii)
          if (total > ii->second && ii->first &&
               ii->second > total * ConstValPercent) {
            Function* NF = CloneFunction(&F);
            NF->setLinkage(GlobalValue::InternalLinkage);
            M.getFunctionList().push_back(NF);
            replaceUsersFor(F, interestingArgs[x], ii->first, NF);
            breakOuter = true;
            Changed = true;
          }
    }
  }
  return Changed;
}

/// scanForInterest - This function decides which arguments would be worth
/// specializing on.
void PartSpec::scanForInterest(Function& F, SmallVector<int, 6>& args) {
  for(Function::arg_iterator ii = F.arg_begin(), ee = F.arg_end();
      ii != ee; ++ii) {
    for(Value::use_iterator ui = ii->use_begin(), ue = ii->use_end();
        ui != ue; ++ui) {
      // As an initial proxy for control flow, specialize on arguments
      // that are used in comparisons.
      if (isa<CmpInst>(ui)) {
        args.push_back(std::distance(F.arg_begin(), ii));
        break;
      }
    }
  }
}

/// replaceUsersFor - Replace direct calls to F with NF if the arg argnum is
/// the constant val
void PartSpec::replaceUsersFor(Function& F , int argnum, Constant* val, 
                               Function* NF) {
  ++numSpecialized;
  for(Value::use_iterator ii = F.use_begin(), ee = F.use_end();
      ii != ee; ++ii)
    if (CallInst* CI = dyn_cast<CallInst>(ii))
      if (CI->getOperand(0) == &F && CI->getOperand(argnum + 1) == val)
        CI->setOperand(0, NF);
}

int PartSpec::scanDistribution(Function& F, int arg, 
                               std::map<Constant*, int>& dist) {
  bool hasIndirect = false;
  int total = 0;
  for(Value::use_iterator ii = F.use_begin(), ee = F.use_end();
      ii != ee; ++ii)
    if (CallInst* CI = dyn_cast<CallInst>(ii)) {
      ++dist[dyn_cast<Constant>(CI->getOperand(arg + 1))];
      ++total;
    } else
      hasIndirect = true;

  // Preserve the original address taken function even if all other uses
  // will be specialized.
  if (hasIndirect) ++total;
  return total;
}

ModulePass* llvm::createPartialSpecializationPass() { return new PartSpec(); }
