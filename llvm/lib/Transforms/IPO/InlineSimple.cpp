//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#include "Inliner.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO.h"
using namespace llvm;

namespace {
  // FunctionInfo - For each function, calculate the size of it in blocks and
  // instructions.
  struct FunctionInfo {
    // NumInsts, NumBlocks - Keep track of how large each function is, which is
    // used to estimate the code size cost of inlining it.
    unsigned NumInsts, NumBlocks;

    // ConstantArgumentWeights - Each formal argument of the function is
    // inspected to see if it is used in any contexts where making it a constant
    // would reduce the code size.  If so, we add some value to the argument
    // entry here.
    std::vector<unsigned> ConstantArgumentWeights;

    FunctionInfo() : NumInsts(0), NumBlocks(0) {}
  };

  class SimpleInliner : public Inliner {
    std::map<const Function*, FunctionInfo> CachedFunctionInfo;
  public:
    int getInlineCost(CallSite CS);
  };
  RegisterOpt<SimpleInliner> X("inline", "Function Integration/Inlining");
}

Pass *llvm::createFunctionInliningPass() { return new SimpleInliner(); }

// CountCodeReductionForConstant - Figure out an approximation for how many
// instructions will be constant folded if the specified value is constant.
//
static unsigned CountCodeReductionForConstant(Value *V) {
  unsigned Reduction = 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (isa<BranchInst>(*UI))
      Reduction += 40;          // Eliminating a conditional branch is a big win
    else if (SwitchInst *SI = dyn_cast<SwitchInst>(*UI))
      // Eliminating a switch is a big win, proportional to the number of edges
      // deleted.
      Reduction += (SI->getNumSuccessors()-1) * 40;
    else if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
      // Turning an indirect call into a direct call is a BIG win
      Reduction += CI->getCalledValue() == V ? 500 : 0;
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI)) {
      // Turning an indirect call into a direct call is a BIG win
      Reduction += CI->getCalledValue() == V ? 500 : 0;
    } else {
      // Figure out if this instruction will be removed due to simple constant
      // propagation.
      Instruction &Inst = cast<Instruction>(**UI);
      bool AllOperandsConstant = true;
      for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i)
        if (!isa<Constant>(Inst.getOperand(i)) &&
            !isa<GlobalValue>(Inst.getOperand(i)) && Inst.getOperand(i) != V) {
          AllOperandsConstant = false;
          break;
        }

      if (AllOperandsConstant) {
        // We will get to remove this instruction...
        Reduction += 7;
        
        // And any other instructions that use it which become constants
        // themselves.
        Reduction += CountCodeReductionForConstant(&Inst);
      }
    }

  return Reduction;
}

// getInlineCost - The heuristic used to determine if we should inline the
// function call or not.
//
int SimpleInliner::getInlineCost(CallSite CS) {
  Instruction *TheCall = CS.getInstruction();
  Function *Callee = CS.getCalledFunction();
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
  if (Callee->hasInternalLinkage() && Callee->hasOneUse())
    InlineCost -= 30000;

  // Get information about the callee...
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

    // Check out all of the arguments to the function, figuring out how much
    // code can be eliminated if one of the arguments is a constant.
    std::vector<unsigned> &ArgWeights = CalleeFI.ConstantArgumentWeights;

    for (Function::aiterator I = Callee->abegin(), E = Callee->aend();
         I != E; ++I)
      ArgWeights.push_back(CountCodeReductionForConstant(I));
  }


  // Add to the inline quality for properties that make the call valuable to
  // inline.  This includes factors that indicate that the result of inlining
  // the function will be optimizable.  Currently this just looks at arguments
  // passed into the function.
  //
  unsigned ArgNo = 0;
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
       I != E; ++I, ++ArgNo) {
    // Each argument passed in has a cost at both the caller and the callee
    // sides.  This favors functions that take many arguments over functions
    // that take few arguments.
    InlineCost -= 20;

    // If this is a function being passed in, it is very likely that we will be
    // able to turn an indirect function call into a direct function call.
    if (isa<Function>(I))
      InlineCost -= 100;

    // If an alloca is passed in, inlining this function is likely to allow
    // significant future optimization possibilities (like scalar promotion, and
    // scalarization), so encourage the inlining of the function.
    //
    else if (isa<AllocaInst>(I))
      InlineCost -= 60;

    // If this is a constant being passed into the function, use the argument
    // weights calculated for the callee to determine how much will be folded
    // away with this information.
    else if (isa<Constant>(I) || isa<GlobalVariable>(I)) {
      if (ArgNo < CalleeFI.ConstantArgumentWeights.size())
        InlineCost -= CalleeFI.ConstantArgumentWeights[ArgNo];
    }
  }

  // Now that we have considered all of the factors that make the call site more
  // likely to be inlined, look at factors that make us not want to inline it.

  // Don't inline into something too big, which would make it bigger.  Here, we
  // count each basic block as a single unit.
  InlineCost += Caller->size()*2;


  // Look at the size of the callee.  Each basic block counts as 20 units, and
  // each instruction counts as 5.
  InlineCost += CalleeFI.NumInsts*5 + CalleeFI.NumBlocks*20;
  return InlineCost;
}

