//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ARMTargetMachine.h"
#include "ARM.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <iostream>
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<ARMTargetMachine> X("arm", "  ARM");
}

/// TargetMachine ctor - Create an ILP32 architecture model
///
ARMTargetMachine::ARMTargetMachine(const Module &M, const std::string &FS)
  : TargetMachine("ARM"), DataLayout("E-p:32:32"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, -4) {
}

unsigned ARMTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 4 && std::string(TT.begin(), TT.begin()+4) == "arm-")
    return 20;

  if (M.getPointerSize() == Module::Pointer32)
    return 1;
  else
    return 0;
}

/// addPassesToEmitFile - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool ARMTargetMachine::addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                             CodeGenFileType FileType,
                                             bool Fast) {
  if (FileType != TargetMachine::AssemblyFile)
    return true;

  // Run loop strength reduction before anything else.
  if (!Fast)
    PM.add(createLoopStrengthReducePass());

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // Print LLVM code input to instruction selector:
  if (PrintMachineCode)
    PM.add(new PrintFunctionPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  PM.add(createARMISelDag(*this));

  // Print machine instructions as they were initially generated.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());

  // Print machine instructions after register allocation and prolog/epilog
  // insertion.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Output assembly language.
  PM.add(createARMCodePrinterPass(Out, *this));

  // Delete the MachineInstrs we generated, since they're no longer needed.
  PM.add(createMachineCodeDeleter());
  return false;
}

