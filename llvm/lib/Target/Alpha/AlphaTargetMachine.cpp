//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaTargetMachine.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <iostream>
using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<AlphaTargetMachine> X("alpha", "  Alpha (incomplete)");
}

AlphaTargetMachine::AlphaTargetMachine( const Module &M, IntrinsicLowering *IL)
  : TargetMachine("alpha", IL, true), 
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) //TODO: check these
{}

/// addPassesToEmitAssembly - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool AlphaTargetMachine::addPassesToEmitAssembly(PassManager &PM,
                                                   std::ostream &Out) {
  
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  PM.add(createAlphaPatternInstructionSelector(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createPrologEpilogCodeInserter());
  
  // Must run branch selection immediately preceding the asm printer
  //PM.add(createAlphaBranchSelectionPass());
  
  PM.add(createAlphaCodePrinterPass(Out, *this));
    
  PM.add(createMachineCodeDeleter());
  return false;
}
