//===-- SparcTargetMachine.cpp - Define TargetMachine for Sparc -----------===//
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

#include "SparcTargetMachine.h"
#include "Sparc.h"
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
  RegisterTarget<SparcTargetMachine> X("sparc", "  SPARC");
}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Module &M, const std::string &FS)
  : TargetMachine("Sparc"),
    DataLayout(std::string("Sparc"), std::string("e-p:32:32")),
    Subtarget(M, FS), InstrInfo(Subtarget),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
}

unsigned SparcTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "sparc-")
    return 20;

  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
#ifdef __sparc__
    return 20;   // BE/32 ==> Prefer sparc on sparc
#else
    return 5;    // BE/32 ==> Prefer ppc elsewhere
#endif
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return 0;
}

/// addPassesToEmitFile - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool SparcTargetMachine::addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                             CodeGenFileType FileType,
                                             bool Fast) {
  if (FileType != TargetMachine::AssemblyFile) return true;

  // Run loop strength reduction before anything else.
  if (!Fast) PM.add(createLoopStrengthReducePass());

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // Print LLVM code input to instruction selector:
  if (PrintMachineCode)
    PM.add(new PrintFunctionPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());
  
  PM.add(createSparcISelDag(*this));

  // Print machine instructions as they were initially generated.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());

  // Print machine instructions after register allocation and prolog/epilog
  // insertion.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createSparcFPMoverPass(*this));

  PM.add(createSparcDelaySlotFillerPass(*this));

  // Print machine instructions after filling delay slots.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Output assembly language.
  PM.add(createSparcCodePrinterPass(Out, *this));

  // Delete the MachineInstrs we generated, since they're no longer needed.
  PM.add(createMachineCodeDeleter());
  return false;
}

