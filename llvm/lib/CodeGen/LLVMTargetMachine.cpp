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
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool> PrintLSR("print-lsr-output");
static cl::opt<bool> PrintISelInput("print-isel-input");
FileModel::Model
LLVMTargetMachine::addPassesToEmitFile(FunctionPassManager &PM,
                                       std::ostream &Out,
                                       CodeGenFileType FileType,
                                       bool Fast) {
  // Standard LLVM-Level Passes.
  
  // Run loop strength reduction before anything else.
  if (!Fast) {
    PM.add(createLoopStrengthReducePass(getTargetLowering()));
    if (PrintLSR)
      PM.add(new PrintFunctionPass("\n\n*** Code after LSR *** \n", &cerr));
  }
  
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());
  
  // FIXME: Implement the invoke/unwind instructions!
  if (!ExceptionHandling)
    PM.add(createLowerInvokePass(getTargetLowering()));
  
  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  if (!Fast)
    PM.add(createCodeGenPreparePass(getTargetLowering()));

  if (PrintISelInput)
    PM.add(new PrintFunctionPass("\n\n*** Final LLVM Code input to ISel *** \n",
                                 &cerr));
  
  // Ask the target for an isel.
  if (addInstSelector(PM, Fast))
    return FileModel::Error;

  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));
  
  // Perform register allocation to convert to a concrete x86 representation
  PM.add(createRegisterAllocator());
  
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Run post-ra passes.
  if (addPostRegAlloc(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());
  
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (!Fast)
    PM.add(createBranchFoldingPass());
    
  // Fold redundant debug labels.
  PM.add(createDebugLabelFoldingPass());
  
  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createMachineFunctionPrinterPass(cerr));

  if (addPreEmitPass(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  switch (FileType) {
  default:
    break;
  case TargetMachine::AssemblyFile:
    if (addAssemblyEmitter(PM, Fast, Out))
      return FileModel::Error;
    return FileModel::AsmFile;
  case TargetMachine::ObjectFile:
    if (getMachOWriterInfo())
      return FileModel::MachOFile;
    else if (getELFWriterInfo())
      return FileModel::ElfFile;
  }

  return FileModel::Error;
}
 
/// addPassesToEmitFileFinish - If the passes to emit the specified file had to
/// be split up (e.g., to add an object writer pass), this method can be used to
/// finish up adding passes to emit the file, if necessary.
bool LLVMTargetMachine::addPassesToEmitFileFinish(FunctionPassManager &PM,
                                                  MachineCodeEmitter *MCE,
                                                  bool Fast) {
  if (MCE)
    addSimpleCodeEmitter(PM, Fast, *MCE);

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
  if (!Fast) {
    PM.add(createLoopStrengthReducePass(getTargetLowering()));
    if (PrintLSR)
      PM.add(new PrintFunctionPass("\n\n*** Code after LSR *** \n", &cerr));
  }
  
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());
  
  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass(getTargetLowering()));
  
  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  if (!Fast)
    PM.add(createCodeGenPreparePass(getTargetLowering()));

  if (PrintISelInput)
    PM.add(new PrintFunctionPass("\n\n*** Final LLVM Code input to ISel *** \n",
                                 &cerr));

  // Ask the target for an isel.
  if (addInstSelector(PM, Fast))
    return true;

  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));
  
  // Perform register allocation to convert to a concrete x86 representation
  PM.add(createRegisterAllocator());
  
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Run post-ra passes.
  if (addPostRegAlloc(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());
  
  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createMachineFunctionPrinterPass(cerr));
  
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (!Fast)
    PM.add(createBranchFoldingPass());
  
  if (addPreEmitPass(PM, Fast) && PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(cerr));

  addCodeEmitter(PM, Fast, MCE);
  
  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());
  
  return false; // success!
}
