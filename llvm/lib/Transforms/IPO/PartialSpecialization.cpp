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
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
using namespace llvm;

STATISTIC(numSpecialized, "Number of specialized functions created");
STATISTIC(numReplaced, "Number of callers replaced by specialization");

// Maximum number of arguments markable interested
static const int MaxInterests = 6;

// Call must be used at least occasionally
static const int CallsMin = 5;

// Must have 10% of calls having the same constant to specialize on
static const double ConstValPercent = .1;

namespace {
  typedef SmallVector<int, MaxInterests> InterestingArgVector;
  class PartSpec : public ModulePass {
    void scanForInterest(Function&, InterestingArgVector&);
    int scanDistribution(Function&, int, std::map<Constant*, int>&);
  public :
    static char ID; // Pass identification, replacement for typeid
    PartSpec() : ModulePass(ID) {}
    bool runOnModule(Module &M);
  };
}

char PartSpec::ID = 0;
INITIALIZE_PASS(PartSpec, "partialspecialization",
                "Partial Specialization", false, false);

// Specialize F by replacing the arguments (keys) in replacements with the 
// constants (values).  Replace all calls to F with those constants with
// a call to the specialized function.  Returns the specialized function
static Function* 
SpecializeFunction(Function* F, 
                   ValueMap<const Value*, Value*>& replacements) {
  // arg numbers of deleted arguments
  DenseMap<unsigned, const Argument*> deleted;
  for (ValueMap<const Value*, Value*>::iterator 
         repb = replacements.begin(), repe = replacements.end();
       repb != repe; ++repb) {
    Argument const *arg = cast<const Argument>(repb->first);
    deleted[arg->getArgNo()] = arg;
  }

  Function* NF = CloneFunction(F, replacements,
                               /*ModuleLevelChanges=*/false);
  NF->setLinkage(GlobalValue::InternalLinkage);
  F->getParent()->getFunctionList().push_back(NF);

  for (Value::use_iterator ii = F->use_begin(), ee = F->use_end(); 
       ii != ee; ) {
    Value::use_iterator i = ii;
    ++ii;
    User *U = *i;
    CallSite CS(U);
    if (CS) {
      if (CS.getCalledFunction() == F) {
        SmallVector<Value*, 6> args;
        // Assemble the non-specialized arguments for the updated callsite.
        // In the process, make sure that the specialized arguments are
        // constant and match the specialization.  If that's not the case,
        // this callsite needs to call the original or some other
        // specialization; don't change it here.
        CallSite::arg_iterator as = CS.arg_begin(), ae = CS.arg_end();
        for (CallSite::arg_iterator ai = as; ai != ae; ++ai) {
          DenseMap<unsigned, const Argument*>::iterator delit = deleted.find(
            std::distance(as, ai));
          if (delit == deleted.end())
            args.push_back(cast<Value>(ai));
          else {
            Constant *ci = dyn_cast<Constant>(ai);
            if (!(ci && ci == replacements[delit->second]))
              goto next_use;
          }
        }
        Value* NCall;
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          NCall = CallInst::Create(NF, args.begin(), args.end(), 
                                   CI->getName(), CI);
          cast<CallInst>(NCall)->setTailCall(CI->isTailCall());
          cast<CallInst>(NCall)->setCallingConv(CI->getCallingConv());
        } else {
          InvokeInst *II = cast<InvokeInst>(U);
          NCall = InvokeInst::Create(NF, II->getNormalDest(),
                                     II->getUnwindDest(),
                                     args.begin(), args.end(), 
                                     II->getName(), II);
          cast<InvokeInst>(NCall)->setCallingConv(II->getCallingConv());
        }
        CS.getInstruction()->replaceAllUsesWith(NCall);
        CS.getInstruction()->eraseFromParent();
        ++numReplaced;
      }
    }
    next_use:;
  }
  return NF;
}


bool PartSpec::runOnModule(Module &M) {
  bool Changed = false;
  for (Module::iterator I = M.begin(); I != M.end(); ++I) {
    Function &F = *I;
    if (F.isDeclaration() || F.mayBeOverridden()) continue;
    InterestingArgVector interestingArgs;
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
            ValueMap<const Value*, Value*> m;
            Function::arg_iterator arg = F.arg_begin();
            for (int y = 0; y < interestingArgs[x]; ++y)
              ++arg;
            m[&*arg] = ii->first;
            SpecializeFunction(&F, m);
            ++numSpecialized;
            breakOuter = true;
            Changed = true;
          }
    }
  }
  return Changed;
}

/// scanForInterest - This function decides which arguments would be worth
/// specializing on.
void PartSpec::scanForInterest(Function& F, InterestingArgVector& args) {
  for(Function::arg_iterator ii = F.arg_begin(), ee = F.arg_end();
      ii != ee; ++ii) {
    for(Value::use_iterator ui = ii->use_begin(), ue = ii->use_end();
        ui != ue; ++ui) {

      bool interesting = false;
      User *U = *ui;
      if (isa<CmpInst>(U)) interesting = true;
      else if (isa<CallInst>(U))
        interesting = ui->getOperand(0) == ii;
      else if (isa<InvokeInst>(U))
        interesting = ui->getOperand(0) == ii;
      else if (isa<SwitchInst>(U)) interesting = true;
      else if (isa<BranchInst>(U)) interesting = true;

      if (interesting) {
        args.push_back(std::distance(F.arg_begin(), ii));
        break;
      }
    }
  }
}

/// scanDistribution - Construct a histogram of constants for arg of F at arg.
int PartSpec::scanDistribution(Function& F, int arg, 
                               std::map<Constant*, int>& dist) {
  bool hasIndirect = false;
  int total = 0;
  for (Value::use_iterator ii = F.use_begin(), ee = F.use_end();
      ii != ee; ++ii) {
    User *U = *ii;
    CallSite CS(U);
    if (CS && CS.getCalledFunction() == &F) {
      ++dist[dyn_cast<Constant>(CS.getArgument(arg))];
      ++total;
    } else
      hasIndirect = true;
  }

  // Preserve the original address taken function even if all other uses
  // will be specialized.
  if (hasIndirect) ++total;
  return total;
}

ModulePass* llvm::createPartialSpecializationPass() { return new PartSpec(); }
