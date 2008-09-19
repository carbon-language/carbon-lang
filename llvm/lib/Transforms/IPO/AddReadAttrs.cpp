//===- AddReadAttrs.cpp - Pass which marks functions readnone or readonly -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interprocedural pass which walks the
// call-graph, looking for functions which do not access or only read
// non-local memory, and marking them readnone/readonly.  It implements
// this as a bottom-up traversal of the call-graph.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "addreadattrs"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstIterator.h"
using namespace llvm;

STATISTIC(NumReadNone, "Number of functions marked readnone");
STATISTIC(NumReadOnly, "Number of functions marked readonly");

namespace {
  struct VISIBILITY_HIDDEN AddReadAttrs : public CallGraphSCCPass {
    static char ID; // Pass identification, replacement for typeid
    AddReadAttrs() : CallGraphSCCPass(&ID) {}

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(const std::vector<CallGraphNode *> &SCC);
  };
}

char AddReadAttrs::ID = 0;
static RegisterPass<AddReadAttrs>
X("addreadattrs", "Mark functions readnone/readonly");

Pass *llvm::createAddReadAttrsPass() { return new AddReadAttrs(); }


bool AddReadAttrs::runOnSCC(const std::vector<CallGraphNode *> &SCC) {
  CallGraph &CG = getAnalysis<CallGraph>();

  // Check if any of the functions in the SCC read or write memory.
  // If they write memory then just give up.
  bool ReadsMemory = false;
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();

    if (F == 0)
      // May write memory.
      return false;

    if (F->doesNotAccessMemory())
      // Already perfect!
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that writes memory, so treat them like declarations.
    if (F->isDeclaration() || F->hasWeakLinkage()) {
      if (!F->onlyReadsMemory())
        // May write memory.
        return false;

      ReadsMemory = true;
      continue;
    }

    // Scan the function body for explicit loads and stores, or calls to
    // functions that may read or write memory.
    for (inst_iterator II = inst_begin(F), E = inst_end(F); II != E; ++II) {
      Instruction *I = &*II;
      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        if (LI->isVolatile())
          // Volatile loads may have side-effects, so treat them as writing
          // memory.
          return false;
        ReadsMemory = true;
      } else if (isa<StoreInst>(I) || isa<MallocInst>(I) || isa<FreeInst>(I)) {
        // Writes memory.
        return false;
      } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        CallSite CS(I);

        if (std::find(SCC.begin(), SCC.end(), CG[CS.getCalledFunction()]) !=
            SCC.end())
          // The callee is inside our current SCC - ignore it.
          continue;

        if (!CS.onlyReadsMemory())
          // May write memory.
          return false;

        if (!CS.doesNotAccessMemory())
          ReadsMemory = true;
      }
    }
  }

  // Success!  Functions in this SCC do not access memory, or only read memory.
  // Give them the appropriate attribute.
  bool MadeChange = false;
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();

    if (F->doesNotAccessMemory())
      // Already perfect!
      continue;

    if (F->onlyReadsMemory() && ReadsMemory)
      // No change.
      continue;

    MadeChange = true;

    // Clear out any existing attributes.
    F->removeParamAttr(0, ParamAttr::ReadOnly | ParamAttr::ReadNone);

    // Add in the new attribute.
    F->addParamAttr(0, ReadsMemory ? ParamAttr::ReadOnly : ParamAttr::ReadNone);

    if (ReadsMemory)
      NumReadOnly++;
    else
      NumReadNone++;
  }

  return MadeChange;
}
