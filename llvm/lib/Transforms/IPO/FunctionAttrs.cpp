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
// non-local memory, and marking them readnone/readonly.  In addition,
// it marks function arguments (of pointer type) 'nocapture' if a call
// to the function does not create any copies of the pointer value that
// outlive the call.  This more or less means that the pointer is only
// dereferenced, and not returned from the function or stored in a global.
// This pass is implemented as a bottom-up traversal of the call-graph.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "functionattrs"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallSet.h"
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

    // isCaptured - Return true if this pointer value may be captured.
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
      ++NumReadOnly;
    else
      ++NumReadNone;
  }

  return MadeChange;
}

/// isCaptured - Return true if this pointer value may be captured.
bool FunctionAttrs::isCaptured(Function &F, Value *V) {
  typedef PointerIntPair<Use*, 2> UseWithDepth;
  SmallVector<UseWithDepth, 16> Worklist;
  SmallSet<UseWithDepth, 16> Visited;

  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end(); UI != UE;
       ++UI) {
    UseWithDepth UD(&UI.getUse(), 0);
    Visited.insert(UD);
    Worklist.push_back(UD);
  }

  while (!Worklist.empty()) {
    UseWithDepth UD = Worklist.pop_back_val();
    Use *U = UD.getPointer();
    Instruction *I = cast<Instruction>(U->getUser());
    // The value V may have any type if it comes from tracking a load.
    V = U->get();
    // The depth represents the number of loads that need to be performed to
    // get back the original pointer (or a bitcast etc of it).  For example,
    // if the pointer is stored to an alloca, then all uses of the alloca get
    // depth 1: if the alloca is loaded then you get the original pointer back.
    // If a load of the alloca is returned then the pointer has been captured.
    // The depth is needed in order to know which loads dereference the original
    // pointer (these do not capture), and which return a value which needs to
    // be tracked because if it is captured then so is the original pointer.
    unsigned Depth = UD.getInt();

    switch (I->getOpcode()) {
    case Instruction::Call:
    case Instruction::Invoke: {
      CallSite CS = CallSite::get(I);
      // Not captured if the callee is readonly and doesn't return a copy
      // through its return value.
      if (CS.onlyReadsMemory() && I->getType() == Type::VoidTy)
        break;

      // Not captured if only passed via 'nocapture' arguments.  Note that
      // calling a function pointer does not in itself cause the pointer to
      // be captured.  This is a subtle point considering that (for example)
      // the callee might return its own address.  It is analogous to saying
      // that loading a value from a pointer does not cause the pointer to be
      // captured, even though the loaded value might be the pointer itself
      // (think of self-referential objects).
      CallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
      for (CallSite::arg_iterator A = B; A != E; ++A)
        if (A->get() == V && !CS.paramHasAttr(A - B + 1, Attribute::NoCapture))
          // The parameter is not marked 'nocapture' - captured.
          return true;
      // Only passed via 'nocapture' arguments, or is the called function - not
      // captured.
      break;
    }
    case Instruction::Free:
      // Freeing a pointer does not cause it to be captured.
      break;
    case Instruction::Store:
      if (V == I->getOperand(0)) {
        // Stored the pointer - it may be captured.  If it is stored to a local
        // object (alloca) then track that object.  Otherwise give up.
        Value *Target = I->getOperand(1)->getUnderlyingObject();
        if (!isa<AllocaInst>(Target))
          // Didn't store to an obviously local object - captured.
          return true;
        if (Depth >= 3)
          // Alloca recursion too deep - give up.
          return true;
        // Analyze all uses of the alloca.
        for (Value::use_iterator UI = Target->use_begin(),
             UE = Target->use_end(); UI != UE; ++UI) {
          UseWithDepth NUD(&UI.getUse(), Depth + 1);
          if (Visited.insert(NUD))
            Worklist.push_back(NUD);
        }
      }
      // Storing to the pointee does not cause the pointer to be captured.
      break;
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::Load:
    case Instruction::PHI:
    case Instruction::Select:
      // Track any uses of this instruction to see if they are captured.
      // First handle any special cases.
      if (isa<GetElementPtrInst>(I)) {
        // Play safe and do not accept being used as an index.
        if (V != I->getOperand(0))
          return true;
      } else if (isa<SelectInst>(I)) {
        // Play safe and do not accept being used as the condition.
        if (V == I->getOperand(0))
          return true;
      } else if (isa<LoadInst>(I)) {
        // Usually loads can be ignored because they dereference the original
        // pointer.  However the loaded value needs to be tracked if loading
        // from an object that the original pointer was stored to.
        if (Depth == 0)
          // Loading the original pointer or a variation of it.  This does not
          // cause the pointer to be captured.  Note that the loaded value might
          // be the pointer itself (think of self-referential objects), but that
          // is fine as long as it's not this function that stored it there.
          break;
        // Loading a pointer to (a pointer to...) the original pointer or a
        // variation of it.  Track uses of the loaded value, noting that one
        // dereference was performed.  Note that the loaded value need not be
        // of pointer type.  For example, an alloca may have been bitcast to
        // a pointer to another type, which was then loaded.
        --Depth;
      }

      // The original value is not captured via this if the instruction isn't.
      for (Instruction::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {
        UseWithDepth UD(&UI.getUse(), Depth);
        if (Visited.insert(UD))
          Worklist.push_back(UD);
      }
      break;
    default:
      // Something else - be conservative and say it is captured.
      return true;
    }
  }

  // All uses examined - not captured.
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
        ++NumNoCapture;
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
