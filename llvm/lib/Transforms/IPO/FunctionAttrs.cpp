//===- FunctionAttrs.cpp - Pass which marks functions readnone or readonly ===//
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
// non-local memory, and marking them readnone/readonly.  It addition,
// it deduces which function arguments (of pointer type) do not escape,
// and marks them nocapture.  It implements this as a bottom-up traversal
// of the call-graph.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "functionattrs"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstIterator.h"
using namespace llvm;

STATISTIC(NumReadNone, "Number of functions marked readnone");
STATISTIC(NumReadOnly, "Number of functions marked readonly");
STATISTIC(NumNoCapture, "Number of arguments marked nocapture");

namespace {
  struct VISIBILITY_HIDDEN FunctionAttrs : public CallGraphSCCPass {
    static char ID; // Pass identification, replacement for typeid
    FunctionAttrs() : CallGraphSCCPass(&ID) {}

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(const std::vector<CallGraphNode *> &SCC);

    // AddReadAttrs - Deduce readonly/readnone attributes for the SCC.
    bool AddReadAttrs(const std::vector<CallGraphNode *> &SCC);

    // AddNoCaptureAttrs - Deduce nocapture attributes for the SCC.
    bool AddNoCaptureAttrs(const std::vector<CallGraphNode *> &SCC);

    // isCaptured - Returns true if this pointer value escapes.
    bool isCaptured(Function &F, Value *V);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      CallGraphSCCPass::getAnalysisUsage(AU);
    }

    bool PointsToLocalMemory(Value *V);
  };
}

char FunctionAttrs::ID = 0;
static RegisterPass<FunctionAttrs>
X("functionattrs", "Deduce function attributes");

Pass *llvm::createFunctionAttrsPass() { return new FunctionAttrs(); }


/// PointsToLocalMemory - Returns whether the given pointer value points to
/// memory that is local to the function.  Global constants are considered
/// local to all functions.
bool FunctionAttrs::PointsToLocalMemory(Value *V) {
  V = V->getUnderlyingObject();
  // An alloca instruction defines local memory.
  if (isa<AllocaInst>(V))
    return true;
  // A global constant counts as local memory for our purposes.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return GV->isConstant();
  // Could look through phi nodes and selects here, but it doesn't seem
  // to be useful in practice.
  return false;
}

/// AddReadAttrs - Deduce readonly/readnone attributes for the SCC.
bool FunctionAttrs::AddReadAttrs(const std::vector<CallGraphNode *> &SCC) {
  SmallPtrSet<CallGraphNode*, 8> SCCNodes;
  CallGraph &CG = getAnalysis<CallGraph>();

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (unsigned i = 0, e = SCC.size(); i != e; ++i)
    SCCNodes.insert(SCC[i]);

  // Check if any of the functions in the SCC read or write memory.  If they
  // write memory then they can't be marked readnone or readonly.
  bool ReadsMemory = false;
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();

    if (F == 0)
      // External node - may write memory.  Just give up.
      return false;

    if (F->doesNotAccessMemory())
      // Already perfect!
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that writes memory, so treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden()) {
      if (!F->onlyReadsMemory())
        // May write memory.  Just give up.
        return false;

      ReadsMemory = true;
      continue;
    }

    // Scan the function body for instructions that may read or write memory.
    for (inst_iterator II = inst_begin(F), E = inst_end(F); II != E; ++II) {
      Instruction *I = &*II;

      // Some instructions can be ignored even if they read or write memory.
      // Detect these now, skipping to the next instruction if one is found.
      CallSite CS = CallSite::get(I);
      if (CS.getInstruction()) {
        // Ignore calls to functions in the same SCC.
        if (SCCNodes.count(CG[CS.getCalledFunction()]))
          continue;
      } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        // Ignore loads from local memory.
        if (PointsToLocalMemory(LI->getPointerOperand()))
          continue;
      } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        // Ignore stores to local memory.
        if (PointsToLocalMemory(SI->getPointerOperand()))
          continue;
      }

      // Any remaining instructions need to be taken seriously!  Check if they
      // read or write memory.
      if (I->mayWriteToMemory())
        // Writes memory.  Just give up.
        return false;
      // If this instruction may read memory, remember that.
      ReadsMemory |= I->mayReadFromMemory();
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
    F->removeAttribute(~0, Attribute::ReadOnly | Attribute::ReadNone);

    // Add in the new attribute.
    F->addAttribute(~0, ReadsMemory? Attribute::ReadOnly : Attribute::ReadNone);

    if (ReadsMemory)
      NumReadOnly++;
    else
      NumReadNone++;
  }

  return MadeChange;
}

/// isCaptured - Returns whether this pointer value is captured.
bool FunctionAttrs::isCaptured(Function &F, Value *V) {
  SmallVector<Use*, 16> Worklist;
  SmallPtrSet<Use*, 16> Visited;

  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end(); UI != UE;
       ++UI) {
    Use *U = &UI.getUse();
    Visited.insert(U);
    Worklist.push_back(U);
  }

  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    Instruction *I = cast<Instruction>(U->getUser());
    V = U->get();

    if (isa<LoadInst>(I)) {
      // Loading a pointer does not cause it to escape.
      continue;
    }

    if (isa<StoreInst>(I)) {
      if (V == I->getOperand(0))
        // Stored the pointer - escapes.  TODO: improve this.
        return true;
      // Storing to the pointee does not cause the pointer to escape.
      continue;
    }

    CallSite CS = CallSite::get(I);
    if (CS.getInstruction()) {
      // Does not escape if only passed via 'nocapture' arguments.  Note
      // that calling a function pointer does not in itself cause that
      // function pointer to escape.  This is a subtle point considering
      // that (for example) the callee might return its own address.  It
      // is analogous to saying that loading a value from a pointer does
      // not cause the pointer to escape, even though the loaded value
      // might be the pointer itself (think of self-referential objects).
      CallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
      for (CallSite::arg_iterator A = B; A != E; ++A)
        if (A->get() == V && !CS.paramHasAttr(A-B+1, Attribute::NoCapture))
          // The parameter is not marked 'nocapture' - escapes.
          return true;
      // Only passed via 'nocapture' arguments, or is the called function.
      // Does not escape.
      continue;
    }

    if (isa<BitCastInst>(I) || isa<GetElementPtrInst>(I) ||
        isa<PHINode>(I) || isa<SelectInst>(I)) {
      // Type conversion, calculating an offset, or merging values.
      // The original value does not escape via this if the new value doesn't.
      // Note that in the case of a select instruction it is important that
      // the value not be used as the condition, since otherwise one bit of
      // information might escape.  It cannot be the condition because it has
      // the wrong type.
      for (Instruction::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {
        Use *U = &UI.getUse();
        if (Visited.insert(U))
          Worklist.push_back(U);
      }
      continue;
    }

    // Something else - be conservative and say it escapes.
    return true;
  }

  return false;
}

/// AddNoCaptureAttrs - Deduce nocapture attributes for the SCC.
bool FunctionAttrs::AddNoCaptureAttrs(const std::vector<CallGraphNode *> &SCC) {
  bool Changed = false;

  // Check each function in turn, determining which pointer arguments are not
  // captured.
  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    Function *F = SCC[i]->getFunction();

    if (F == 0)
      // External node - skip it;
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that writes memory, so treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden())
      continue;

    for (Function::arg_iterator A = F->arg_begin(), E = F->arg_end(); A!=E; ++A)
      if (isa<PointerType>(A->getType()) && !A->hasNoCaptureAttr() &&
          !isCaptured(*F, A)) {
        A->addAttr(Attribute::NoCapture);
        NumNoCapture++;
        Changed = true;
      }
  }

  return Changed;
}

bool FunctionAttrs::runOnSCC(const std::vector<CallGraphNode *> &SCC) {
  bool Changed = AddReadAttrs(SCC);
  Changed |= AddNoCaptureAttrs(SCC);
  return Changed;
}
