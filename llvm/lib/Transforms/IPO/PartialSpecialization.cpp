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
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
using namespace llvm;

STATISTIC(numSpecialized, "Number of specialized functions created");
STATISTIC(numReplaced, "Number of callers replaced by specialization");

// Maximum number of arguments markable interested
static const int MaxInterests = 6;

namespace {
  typedef SmallVector<int, MaxInterests> InterestingArgVector;
  class PartSpec : public ModulePass {
    void scanForInterest(Function&, InterestingArgVector&);
    int scanDistribution(Function&, int, std::map<Constant*, int>&);
    InlineCostAnalyzer CA;
  public :
    static char ID; // Pass identification, replacement for typeid
    PartSpec() : ModulePass(ID) {
      initializePartSpecPass(*PassRegistry::getPassRegistry());
    }
    bool runOnModule(Module &M);
  };
}

char PartSpec::ID = 0;
INITIALIZE_PASS(PartSpec, "partialspecialization",
                "Partial Specialization", false, false)

// Specialize F by replacing the arguments (keys) in replacements with the 
// constants (values).  Replace all calls to F with those constants with
// a call to the specialized function.  Returns the specialized function
static Function* 
SpecializeFunction(Function* F, 
                   ValueToValueMapTy& replacements) {
  // arg numbers of deleted arguments
  DenseMap<unsigned, const Argument*> deleted;
  for (ValueToValueMapTy::iterator 
         repb = replacements.begin(), repe = replacements.end();
       repb != repe; ++repb) {
    Argument const *arg = cast<const Argument>(repb->first);
    deleted[arg->getArgNo()] = arg;
  }

  Function* NF = CloneFunction(F, replacements,
                               /*ModuleLevelChanges=*/false);
  NF->setLinkage(GlobalValue::InternalLinkage);
  F->getParent()->getFunctionList().push_back(NF);

  // FIXME: Specialized versions getting the same constants should also get
  // the same name.  That way, specializations for public functions can be
  // marked linkonce_odr and reused across modules.

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
      scanDistribution(F, interestingArgs[x], distribution);
      for (std::map<Constant*, int>::iterator ii = distribution.begin(),
             ee = distribution.end(); ii != ee; ++ii) {
        // The distribution map might have an entry for NULL (i.e., one or more
        // callsites were passing a non-constant there).  We allow that to 
        // happen so that we can see whether any callsites pass a non-constant; 
        // if none do and the function is internal, we might have an opportunity
        // to kill the original function.
        if (!ii->first) continue;
        int bonus = ii->second;
        SmallVector<unsigned, 1> argnos;
        argnos.push_back(interestingArgs[x]);
        InlineCost cost = CA.getSpecializationCost(&F, argnos);
        // FIXME: If this is the last constant entry, and no non-constant
        // entries exist, and the target function is internal, the cost should
        // be reduced by the original size of the target function, almost
        // certainly making it negative and causing a specialization that will
        // leave the original function dead and removable.
        if (cost.isAlways() || 
           (cost.isVariable() && cost.getValue() < bonus)) {
          ValueToValueMapTy m;
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
  }
  return Changed;
}

/// scanForInterest - This function decides which arguments would be worth
/// specializing on.
void PartSpec::scanForInterest(Function& F, InterestingArgVector& args) {
  for(Function::arg_iterator ii = F.arg_begin(), ee = F.arg_end();
      ii != ee; ++ii) {
    int argno = std::distance(F.arg_begin(), ii);
    SmallVector<unsigned, 1> argnos;
    argnos.push_back(argno);
    int bonus = CA.getSpecializationBonus(&F, argnos);
    if (bonus > 0) {
      args.push_back(argno);
    }
  }
}

/// scanDistribution - Construct a histogram of constants for arg of F at arg.
/// For each distinct constant, we'll compute the total of the specialization
/// bonus across all callsites passing that constant; if that total exceeds
/// the specialization cost, we will create the specialization.
int PartSpec::scanDistribution(Function& F, int arg, 
                               std::map<Constant*, int>& dist) {
  bool hasIndirect = false;
  int total = 0;
  for (Value::use_iterator ii = F.use_begin(), ee = F.use_end();
      ii != ee; ++ii) {
    User *U = *ii;
    CallSite CS(U);
    if (CS && CS.getCalledFunction() == &F) {
      SmallVector<unsigned, 1> argnos;
      argnos.push_back(arg);
      dist[dyn_cast<Constant>(CS.getArgument(arg))] += 
           CA.getSpecializationBonus(&F, argnos);
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
