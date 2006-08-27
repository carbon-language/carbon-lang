//===- TraceBasicBlocks.cpp - Insert basic-block trace instrumentation ----===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass instruments the specified program with calls into a runtime
// library that cause it to output a trace of basic blocks as a side effect
// of normal execution.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Instructions.h"
#include "ProfilingUtils.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <iostream>
using namespace llvm;

namespace {
  class TraceBasicBlocks : public ModulePass {
    bool runOnModule(Module &M);
  };

  RegisterPass<TraceBasicBlocks> X("trace-basic-blocks",
                              "Insert instrumentation for basic block tracing");
}

ModulePass *llvm::createTraceBasicBlockPass()
{
  return new TraceBasicBlocks();
}

static void InsertInstrumentationCall (BasicBlock *BB,
                                       const std::string FnName,
                                       unsigned BBNumber) {
  DEBUG (std::cerr << "InsertInstrumentationCall (\"" << BB->getName ()
                   << "\", \"" << FnName << "\", " << BBNumber << ")\n");
  Module &M = *BB->getParent ()->getParent ();
  Function *InstrFn = M.getOrInsertFunction (FnName, Type::VoidTy,
                                             Type::UIntTy, (Type *)0);
  std::vector<Value*> Args (1);
  Args[0] = ConstantUInt::get (Type::UIntTy, BBNumber);

  // Insert the call after any alloca or PHI instructions...
  BasicBlock::iterator InsertPos = BB->begin();
  while (isa<AllocaInst>(InsertPos) || isa<PHINode>(InsertPos))
    ++InsertPos;

  Instruction *InstrCall = new CallInst (InstrFn, Args, "", InsertPos);
}

bool TraceBasicBlocks::runOnModule(Module &M) {
  Function *Main = M.getMainFunction();
  if (Main == 0) {
    std::cerr << "WARNING: cannot insert basic-block trace instrumentation"
              << " into a module with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  unsigned BBNumber = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      InsertInstrumentationCall (BB, "llvm_trace_basic_block", BBNumber);
      ++BBNumber;
    }

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_basic_block_tracing");
  return true;
}

