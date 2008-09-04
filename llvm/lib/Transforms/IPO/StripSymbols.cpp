//===- StripSymbols.cpp - Strip symbols and debug info from a module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The StripSymbols transformation implements code stripping. Specifically, it
// can delete:
// 
//   * names for virtual registers
//   * symbols for internal globals and functions
//   * debug information
//
// Note that this transformation makes code much less readable, so it should
// only be used in situations where the 'strip' utility would be used, such as
// reducing code size or making it harder to reverse engineer code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN StripSymbols : public ModulePass {
    bool OnlyDebugInfo;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripSymbols(bool ODI = false) 
      : ModulePass(&ID), OnlyDebugInfo(ODI) {}

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char StripSymbols::ID = 0;
static RegisterPass<StripSymbols>
X("strip", "Strip all symbols from a module");

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

// Strip the symbol table of its names.
//
static void StripSymtab(ValueSymbolTable &ST) {
  for (ValueSymbolTable::iterator VI = ST.begin(), VE = ST.end(); VI != VE; ) {
    Value *V = VI->getValue();
    ++VI;
    if (!isa<GlobalValue>(V) || cast<GlobalValue>(V)->hasInternalLinkage()) {
      // Set name to "", removing from symbol table!
      V->setName("");
    }
  }
}

// Strip the symbol table of its names.
static void StripTypeSymtab(TypeSymbolTable &ST) {
  for (TypeSymbolTable::iterator TI = ST.begin(), E = ST.end(); TI != E; )
    ST.remove(TI++);
}



bool StripSymbols::runOnModule(Module &M) {
  // If we're not just stripping debug info, strip all symbols from the
  // functions and the names from any internal globals.
  if (!OnlyDebugInfo) {
    SmallPtrSet<const GlobalValue*, 8> llvmUsedValues;
    if (GlobalVariable *LLVMUsed = M.getGlobalVariable("llvm.used")) {
      llvmUsedValues.insert(LLVMUsed);
      // Collect values that are preserved as per explicit request.
      // llvm.used is used to list these values.
      if (ConstantArray *Inits = 
            dyn_cast<ConstantArray>(LLVMUsed->getInitializer())) {
        for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i) {
          if (GlobalValue *GV = dyn_cast<GlobalValue>(Inits->getOperand(i)))
            llvmUsedValues.insert(GV);
          else if (ConstantExpr *CE =
                       dyn_cast<ConstantExpr>(Inits->getOperand(i)))
            if (CE->getOpcode() == Instruction::BitCast)
              if (GlobalValue *GV = dyn_cast<GlobalValue>(CE->getOperand(0)))
                llvmUsedValues.insert(GV);
        }
      }
    }

    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (I->hasInternalLinkage() && llvmUsedValues.count(I) == 0)
        I->setName("");     // Internal symbols can't participate in linkage
    }

    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (I->hasInternalLinkage() && llvmUsedValues.count(I) == 0)
        I->setName("");     // Internal symbols can't participate in linkage
      StripSymtab(I->getValueSymbolTable());
    }
    
    // Remove all names from types.
    StripTypeSymtab(M.getTypeSymbolTable());
  }

  // Strip debug info in the module if it exists.  To do this, we remove
  // llvm.dbg.func.start, llvm.dbg.stoppoint, and llvm.dbg.region.end calls, and
  // any globals they point to if now dead.
  Function *FuncStart = M.getFunction("llvm.dbg.func.start");
  Function *StopPoint = M.getFunction("llvm.dbg.stoppoint");
  Function *RegionStart = M.getFunction("llvm.dbg.region.start");
  Function *RegionEnd = M.getFunction("llvm.dbg.region.end");
  Function *Declare = M.getFunction("llvm.dbg.declare");
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
