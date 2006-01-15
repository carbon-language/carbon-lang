//===-- SparcV8TargetMachine.cpp - Define TargetMachine for SparcV8 -------===//
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

#include "SparcV8TargetMachine.h"
#include "SparcV8.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<SparcV8TargetMachine> X("sparcv8","  SPARC V8 (experimental)");

  cl::opt<bool> EnableV8DAGDAG("enable-v8-dag-isel", cl::Hidden,
                                cl::desc("Enable DAG-to-DAG isel for V8"),
                                cl::init(0));
}

/// SparcV8TargetMachine ctor - Create an ILP32 architecture model
///
SparcV8TargetMachine::SparcV8TargetMachine(const Module &M,
                                           IntrinsicLowering *IL,
                                           const std::string &FS)
  : TargetMachine("SparcV8", IL, false, 4, 4),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
}

unsigned SparcV8TargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "sparc-")
    return 20;

  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
#ifdef __sparc__
    return 20;   // BE/32 ==> Prefer sparcv8 on sparc
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
bool SparcV8TargetMachine::addPassesToEmitFile(PassManager &PM,
                                               std::ostream &Out,
                                               CodeGenFileType FileType,
                                               bool Fast) {
  if (FileType != TargetMachine::AssemblyFile) return true;

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: implement the switch instruction in the instruction selector.
  PM.add(createLowerSwitchPass());

  // Print LLVM code input to instruction selector:
  if (PrintMachineCode)
    PM.add(new PrintFunctionPass());

  if (!EnableV8DAGDAG) {
    // Replace malloc and free instructions with library calls.
    PM.add(createLowerAllocationsPass());
    PM.add(createLowerSelectPass());
    // Make sure that no unreachable blocks are instruction selected.
    PM.add(createUnreachableBlockEliminationPass());
    PM.add(createSparcV8SimpleInstructionSelector(*this));
  } else {
    // Make sure that no unreachable blocks are instruction selected.
    PM.add(createUnreachableBlockEliminationPass());
    PM.add(createSparcV8ISelDag(*this));
  }

  // Print machine instructions as they were initially generated.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());

  // Print machine instructions after register allocation and prolog/epilog
  // insertion.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createSparcV8FPMoverPass(*this));

  PM.add(createSparcV8DelaySlotFillerPass(*this));

  // Print machine instructions after filling delay slots.
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Output assembly language.
  PM.add(createSparcV8CodePrinterPass(Out, *this));

  // Delete the MachineInstrs we generated, since they're no longer needed.
  PM.add(createMachineCodeDeleter());
  return false;
}

