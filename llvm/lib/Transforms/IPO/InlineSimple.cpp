//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#include "Inliner.h"
#include "llvm/Function.h"
#include "llvm/iMemory.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO.h"

namespace {
  // FunctionInfo - For each function, calculate the size of it in blocks and
  // instructions.
  struct FunctionInfo {
    unsigned NumInsts, NumBlocks;

    FunctionInfo() : NumInsts(0), NumBlocks(0) {}
  };

  class SimpleInliner : public Inliner {
    std::map<const Function*, FunctionInfo> CachedFunctionInfo;
  public:
    int getInlineCost(CallSite CS);
  };
  RegisterOpt<SimpleInliner> X("inline", "Function Integration/Inlining");
}

Pass *createFunctionInliningPass() { return new SimpleInliner(); }

// getInlineCost - The heuristic used to determine if we should inline the
// function call or not.
//
int SimpleInliner::getInlineCost(CallSite CS) {
  Instruction *TheCall = CS.getInstruction();
  const Function *Callee = CS.getCalledFunction();
  const Function *Caller = TheCall->getParent()->getParent();

  // Don't inline a directly recursive call.
  if (Caller == Callee) return 2000000000;

  // InlineCost - This value measures how good of an inline candidate this call
  // site is to inline.  A lower inline cost make is more likely for the call to
  // be inlined.  This value may go negative.
  //
  int InlineCost = 0;

  // If there is only one call of the function, and it has internal linkage,
  // make it almost guaranteed to be inlined.
  //
  if (Callee->use_size() == 1 && Callee->hasInternalLinkage())
    InlineCost -= 30000;

  // Add to the inline quality for properties that make the call valueable to
  // inline.  This includes factors that indicate that the result of inlining
  // the function will be optimizable.  Currently this just looks at arguments
  // passed into the function.
  //
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
       I != E; ++I) {
    // Each argument passed in has a cost at both the caller and the callee
    // sides.  This favors functions that take many arguments over functions
    // that take few arguments.
    InlineCost -= 20;

    // If this is a function being passed in, it is very likely that we will be
    // able to turn an indirect function call into a direct function call.
    if (isa<Function>(I))
      InlineCost -= 100;

    // If a constant, global variable or alloca is passed in, inlining this
    // function is likely to allow significant future optimization possibilities
    // (constant propagation, scalar promotion, and scalarization), so encourage
    // the inlining of the function.
    //
    else if (isa<Constant>(I) || isa<GlobalVariable>(I) || isa<AllocaInst>(I))
      InlineCost -= 60;
  }

  // Now that we have considered all of the factors that make the call site more
  // likely to be inlined, look at factors that make us not want to inline it.
  FunctionInfo &CalleeFI = CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet...
  if (CalleeFI.NumBlocks == 0) {
    unsigned NumInsts = 0, NumBlocks = 0;

    // Look at the size of the callee.  Each basic block counts as 20 units, and
    // each instruction counts as 10.
    for (Function::const_iterator BB = Callee->begin(), E = Callee->end();
         BB != E; ++BB) {
      NumInsts += BB->size();
      NumBlocks++;
    }
    CalleeFI.NumBlocks = NumBlocks;
    CalleeFI.NumInsts  = NumInsts;
  }

  // Look at the size of the callee.  Each basic block counts as 21 units, and
  // each instruction counts as 10.
  InlineCost += CalleeFI.NumInsts*10 + CalleeFI.NumBlocks*20;
  return InlineCost;
}
