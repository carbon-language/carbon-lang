//===- StripSymbols.cpp - Strip symbols and debug info from a module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements stripping symbols out of symbol tables.
//
// Specifically, this allows you to strip all of the symbols out of:
//   * All functions in a module
//   * All non-essential symbols in a module (all function symbols + all module
//     scope symbols)
//   * Debug information.
//
// Notice that:
//   * This pass makes code much less readable, so it should only be used in
//     situations where the 'strip' utility would be used (such as reducing
//     code size, and making it harder to reverse engineer code).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/SymbolTable.h"
using namespace llvm;

namespace {
  class StripSymbols : public ModulePass {
    bool OnlyDebugInfo;
  public:
    StripSymbols(bool ODI = false) : OnlyDebugInfo(ODI) {}

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  RegisterPass<StripSymbols> X("strip", "Strip all symbols from a module");
}

ModulePass *llvm::createStripSymbolsPass(bool OnlyDebugInfo) {
  return new StripSymbols(OnlyDebugInfo);
}

static void RemoveDeadConstant(Constant *C) {
  assert(C->use_empty() && "Constant is not dead!");
  std::vector<Constant*> Operands;
  for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
    if (isa<DerivedType>(C->getOperand(i)->getType()) &&
        C->getOperand(i)->hasOneUse())
      Operands.push_back(C->getOperand(i));
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (!GV->hasInternalLinkage()) return;   // Don't delete non static globals.
    GV->eraseFromParent();
  }
  else if (!isa<Function>(C))
    C->destroyConstant();

  // If the constant referenced anything, see if we can delete it as well.
  while (!Operands.empty()) {
    RemoveDeadConstant(Operands.back());
    Operands.pop_back();
  }
}

bool StripSymbols::runOnModule(Module &M) {
  // If we're not just stripping debug info, strip all symbols from the
  // functions and the names from any internal globals.
  if (!OnlyDebugInfo) {
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I)
      if (I->hasInternalLinkage())
        I->setName("");     // Internal symbols can't participate in linkage

    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (I->hasInternalLinkage())
        I->setName("");     // Internal symbols can't participate in linkage
      I->getSymbolTable().strip();
    }
    
    // Remove all names from types.
    SymbolTable &SymTab = M.getSymbolTable();
    while (SymTab.type_begin() != SymTab.type_end())
      SymTab.remove(SymTab.type_begin());
  }

  // Strip debug info in the module if it exists.  To do this, we remove
  // llvm.dbg.func.start, llvm.dbg.stoppoint, and llvm.dbg.region.end calls, and
  // any globals they point to if now dead.
  Function *FuncStart = M.getNamedFunction("llvm.dbg.func.start");
  Function *StopPoint = M.getNamedFunction("llvm.dbg.stoppoint");
  Function *RegionStart = M.getNamedFunction("llvm.dbg.region.start");
  Function *RegionEnd = M.getNamedFunction("llvm.dbg.region.end");
  Function *Declare = M.getNamedFunction("llvm.dbg.declare");
  if (!FuncStart && !StopPoint && !RegionStart && !RegionEnd && !Declare)
    return true;

  std::vector<GlobalVariable*> DeadGlobals;

  // Remove all of the calls to the debugger intrinsics, and remove them from
  // the module.
  if (FuncStart) {
    while (!FuncStart->use_empty()) {
      CallInst *CI = cast<CallInst>(FuncStart->use_back());
      Value *Arg = CI->getOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Arg))
          DeadGlobals.push_back(GV);
    }
    FuncStart->eraseFromParent();
  }
  if (StopPoint) {
    while (!StopPoint->use_empty()) {
      CallInst *CI = cast<CallInst>(StopPoint->use_back());
      Value *Arg = CI->getOperand(3);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Arg))
          DeadGlobals.push_back(GV);
    }
    StopPoint->eraseFromParent();
  }
  if (RegionStart) {
    while (!RegionStart->use_empty()) {
      CallInst *CI = cast<CallInst>(RegionStart->use_back());
      Value *Arg = CI->getOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Arg))
          DeadGlobals.push_back(GV);
    }
    RegionStart->eraseFromParent();
  }
  if (RegionEnd) {
    while (!RegionEnd->use_empty()) {
      CallInst *CI = cast<CallInst>(RegionEnd->use_back());
      Value *Arg = CI->getOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Arg))
          DeadGlobals.push_back(GV);
    }
    RegionEnd->eraseFromParent();
  }
  if (Declare) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->use_back());
      Value *Arg = CI->getOperand(2);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Arg))
          DeadGlobals.push_back(GV);
    }
    Declare->eraseFromParent();
  }

  // Finally, delete any internal globals that were only used by the debugger
  // intrinsics.
  while (!DeadGlobals.empty()) {
    GlobalVariable *GV = DeadGlobals.back();
    DeadGlobals.pop_back();
    if (GV->hasInternalLinkage())
      RemoveDeadConstant(GV);
  }

  return true;
}
