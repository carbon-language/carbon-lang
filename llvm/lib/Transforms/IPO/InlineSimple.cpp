//===- FunctionInlining.cpp - Code to perform function inlining -----------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "inline"
#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include <set>

namespace {
  Statistic<> NumInlined("inline", "Number of functions inlined");
  Statistic<> NumDeleted("inline", "Number of functions deleted because all callers found");
  cl::opt<unsigned>             // FIXME: 200 is VERY conservative
  InlineLimit("inline-threshold", cl::Hidden, cl::init(200),
              cl::desc("Control the amount of inlining to perform (default = 200)"));

  struct FunctionInlining : public Pass {
    virtual bool run(Module &M) {
      bool Changed = false;
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
        Changed |= doInlining(I);
      ProcessedFunctions.clear();
      return Changed;
    }

  private:
    std::set<Function*> ProcessedFunctions;  // Prevent infinite recursion
    bool doInlining(Function *F);
  };
  RegisterOpt<FunctionInlining> X("inline", "Function Integration/Inlining");
}

Pass *createFunctionInliningPass() { return new FunctionInlining(); }


// ShouldInlineFunction - The heuristic used to determine if we should inline
// the function call or not.
//
static inline bool ShouldInlineFunction(CallSite CS) {
  Instruction *TheCall = CS.getInstruction();
  assert(TheCall->getParent() && TheCall->getParent()->getParent() && 
	 "Call not embedded into a function!");

  const Function *Callee = CS.getCalledFunction();
  if (Callee == 0 || Callee->isExternal())
    return false;  // Cannot inline an indirect call... or external function.

  // Don't inline a recursive call.
  const Function *Caller = TheCall->getParent()->getParent();
  if (Caller == Callee) return false;

  // InlineQuality - This value measures how good of an inline candidate this
  // call site is to inline.  The initial value determines how aggressive the
  // inliner is.  If this value is negative after the final computation,
  // inlining is not performed.
  //
  int InlineQuality = InlineLimit;

  // If there is only one call of the function, and it has internal linkage,
  // make it almost guaranteed to be inlined.
  //
  if (Callee->use_size() == 1 && Callee->hasInternalLinkage())
    InlineQuality += 30000;

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
    InlineQuality += 20;

    // If this is a function being passed in, it is very likely that we will be
    // able to turn an indirect function call into a direct function call.
    if (isa<Function>(I))
      InlineQuality += 100;

    // If a constant, global variable or alloca is passed in, inlining this
    // function is likely to allow significant future optimization possibilities
    // (constant propagation, scalar promotion, and scalarization), so encourage
    // the inlining of the function.
    //
    else if (isa<Constant>(I) || isa<GlobalVariable>(I) || isa<AllocaInst>(I))
      InlineQuality += 60;
  }

  // Now that we have considered all of the factors that make the call site more
  // likely to be inlined, look at factors that make us not want to inline it.
  // As soon as the inline quality gets negative, bail out.

  // Look at the size of the callee.  Each basic block counts as 20 units, and
  // each instruction counts as 10.
  for (Function::const_iterator BB = Callee->begin(), E = Callee->end();
       BB != E; ++BB) {
    InlineQuality -= BB->size()*10 + 20;
    if (InlineQuality < 0) return false;
  }

  // Don't inline into something too big, which would make it bigger.  Here, we
  // count each basic block as a single unit.
  for (Function::const_iterator BB = Caller->begin(), E = Caller->end();
       BB != E; ++BB) {
    --InlineQuality;
    if (InlineQuality < 0) return false;
  }

  // If we get here, this call site is high enough "quality" to inline.
  DEBUG(std::cerr << "Inlining in '" << Caller->getName()
                  << "', quality = " << InlineQuality << ": " << *TheCall);
  return true;
}


// doInlining - Use a heuristic based approach to inline functions that seem to
// look good.
//
bool FunctionInlining::doInlining(Function *F) {
  // If we have already processed this function (ie, it is recursive) don't
  // revisit.
  std::set<Function*>::iterator PFI = ProcessedFunctions.lower_bound(F);
  if (PFI != ProcessedFunctions.end() && *PFI == F) return false;

  // Insert the function in the set so it doesn't get revisited.
  ProcessedFunctions.insert(PFI, F);

  bool Changed = false;
  for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
      bool ShouldInc = true;
      // Found a call or invoke instruction?
      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        CallSite CS = CallSite::get(I);
        if (Function *Callee = CS.getCalledFunction()) {
          doInlining(Callee);  // Inline in callees before callers!

          // Decide whether we should inline this function...
          if (ShouldInlineFunction(CS)) {
            // Save an iterator to the instruction before the call if it exists,
            // otherwise get an iterator at the end of the block... because the
            // call will be destroyed.
            //
            BasicBlock::iterator SI;
            if (I != BB->begin()) {
              SI = I; --SI;           // Instruction before the call...
            } else {
              SI = BB->end();
            }
            
            // Attempt to inline the function...
            if (InlineFunction(CS)) {
              ++NumInlined;
              Changed = true;
              // Move to instruction before the call...
              I = (SI == BB->end()) ? BB->begin() : SI;
              ShouldInc = false;  // Don't increment iterator until next time
              
              // If we inlined the last possible call site to the function,
              // delete the function body now.
              if (Callee->use_empty() &&
                  (Callee->hasInternalLinkage()||Callee->hasLinkOnceLinkage())){
                F->getParent()->getFunctionList().erase(Callee);
                ++NumDeleted;              
                if (Callee == F) return true;
              }
            }
          }
        }
      }
      if (ShouldInc) ++I;
    }

  return Changed;
}

