//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -------===//
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
#include "llvm/Module.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<AlphaTargetMachine> X("alpha", "  Alpha (incomplete)");
}

AlphaTargetMachine::AlphaTargetMachine( const Module &M, IntrinsicLowering *IL)
  : TargetMachine("alpha", IL, true), 
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) //TODO: check these
    //JITInfo(*this)
{}

bool AlphaTargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
					  MachineCodeEmitter &MCE)
{
  assert(0 && "TODO");
  return false;
}


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

  PM.add(createLowerConstantExpressionsPass());

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

//void AlphaJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
//   // FIXME: Implement efficient support for garbage collection intrinsics.
//   PM.add(createLowerGCPass());

//   // FIXME: Implement the invoke/unwind instructions!
//   PM.add(createLowerInvokePass());

//   // FIXME: Implement the switch instruction in the instruction selector!
//   PM.add(createLowerSwitchPass());

//   PM.add(createLowerConstantExpressionsPass());

//   // Make sure that no unreachable blocks are instruction selected.
//   PM.add(createUnreachableBlockEliminationPass());

//   PM.add(createPPC32ISelSimple(TM));
//   PM.add(createRegisterAllocator());
//   PM.add(createPrologEpilogCodeInserter());

//   // Must run branch selection immediately preceding the asm printer
//   PM.add(createPPCBranchSelectionPass());

//   if (PrintMachineCode)
//     PM.add(createMachineFunctionPrinterPass(&std::cerr));
//}

