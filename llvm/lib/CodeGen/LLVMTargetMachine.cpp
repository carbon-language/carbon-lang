//===-- LLVMTargetMachine.cpp - Implement the LLVMTargetMachine class -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMTargetMachine class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

bool LLVMTargetMachine::addPassesToEmitFile(FunctionPassManager &PM,
                                            std::ostream &Out,
                                            CodeGenFileType FileType,
                                            bool Fast) {
  // Standard LLVM-Level Passes.
  
  // Run loop strength reduction before anything else.
  if (!Fast) PM.add(createLoopStrengthReducePass(getTargetLowering()));
  
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());
  
  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass(getTargetLowering()));
  
  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());
  
  
  // Ask the target for an isel.
  if (addInstSelector(PM, Fast))
    return true;
  
  
  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  // Perform register allocation to convert to a concrete x86 representation
  PM.add(createRegisterAllocator());
  
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  // Run post-ra passes.
  if (addPostRegAlloc(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());
  
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (!Fast)
    PM.add(createBranchFoldingPass());
    
  // Fold redundant debug labels.
  PM.add(createDebugLabelFoldingPass());
  
  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  if (addPreEmitPass(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  switch (FileType) {
    default: return true;
    case TargetMachine::AssemblyFile:
      if (addAssemblyEmitter(PM, Fast, Out))
        return true;
      break;
    case TargetMachine::ObjectFile:
      if (addObjectWriter(PM, Fast, Out))
        return true;
      break;
  }
  
  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());
  
  return false; // success!
}

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to
/// get machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool LLVMTargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                   MachineCodeEmitter &MCE,
                                                   bool Fast) {
  // Standard LLVM-Level Passes.
  
  // Run loop strength reduction before anything else.
  if (!Fast) PM.add(createLoopStrengthReducePass(getTargetLowering()));
  
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());
  
  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass(getTargetLowering()));
  
  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());
  
  
  // Ask the target for an isel.
  if (addInstSelector(PM, Fast))
    return true;
  
  
  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  // Perform register allocation to convert to a concrete x86 representation
  PM.add(createRegisterAllocator());
  
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  // Run post-ra passes.
  if (addPostRegAlloc(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());
  
  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (!Fast)
    PM.add(createBranchFoldingPass());
  
  if (addPreEmitPass(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr.stream()));
  
  
  addCodeEmitter(PM, Fast, MCE);
  
  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());
  
  return false; // success!
}
