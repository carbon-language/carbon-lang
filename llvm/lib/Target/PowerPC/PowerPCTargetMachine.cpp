//===-- PowerPCTargetMachine.cpp - Define TargetMachine for PowerPC -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PowerPCTargetMachine.h"
#include "PowerPCFrameInfo.h"
#include "PPC32TargetMachine.h"
#include "PPC32JITInfo.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
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
  const char *PPC32ID = "PowerPC/32bit";

  static cl::opt<bool> DisablePPCDAGDAG("disable-ppc-dag-isel", cl::Hidden,
                             cl::desc("Disable DAG-to-DAG isel for PPC"));
  
  // Register the targets
  RegisterTarget<PPC32TargetMachine>
  X("ppc32", "  PowerPC 32-bit");
}

PowerPCTargetMachine::PowerPCTargetMachine(const std::string &name,
                                           IntrinsicLowering *IL,
                                           const Module &M,
                                           const std::string &FS,
                                           const TargetData &TD,
                                           const PowerPCFrameInfo &TFI)
: TargetMachine(name, IL, TD), FrameInfo(TFI), Subtarget(M, FS) {
  if (TargetDefault == PPCTarget) {
    if (Subtarget.isAIX()) PPCTarget = TargetAIX;
    if (Subtarget.isDarwin()) PPCTarget = TargetDarwin;
  }
}

unsigned PPC32TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  return 10;
#else
  return 0;
#endif
}

/// addPassesToEmitFile - Add passes to the specified pass manager to implement
/// a static compiler for this target.
///
bool PowerPCTargetMachine::addPassesToEmitFile(PassManager &PM,
                                               std::ostream &Out,
                                                CodeGenFileType FileType) {
  if (FileType != TargetMachine::AssemblyFile) return true;

  // Run loop strength reduction before anything else.
  PM.add(createLoopStrengthReducePass());
  PM.add(createCFGSimplificationPass());

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Install an instruction selector.
  if (!DisablePPCDAGDAG)
    PM.add(createPPC32ISelDag(*this));
  else
    PM.add(createPPC32ISelPattern(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createPrologEpilogCodeInserter());

  // Must run branch selection immediately preceding the asm printer
  PM.add(createPPCBranchSelectionPass());

  // Decide which asm printer to use.  If the user has not specified one on
  // the command line, choose whichever one matches the default (current host).
  switch (PPCTarget) {
  case TargetAIX:
    PM.add(createAIXAsmPrinter(Out, *this));
    break;
  case TargetDefault:
  case TargetDarwin:
    PM.add(createDarwinAsmPrinter(Out, *this));
    break;
  }

  PM.add(createMachineCodeDeleter());
  return false;
}

void PowerPCJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // The JIT does not support or need PIC.
  PICEnabled = false;

  // Run loop strength reduction before anything else.
  PM.add(createLoopStrengthReducePass());
  PM.add(createCFGSimplificationPass());

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Install an instruction selector.
  PM.add(createPPC32ISelPattern(TM));

  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());

  // Must run branch selection immediately preceding the asm printer
  PM.add(createPPCBranchSelectionPass());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));
}

/// PowerPCTargetMachine ctor - Create an ILP32 architecture model
///
PPC32TargetMachine::PPC32TargetMachine(const Module &M, IntrinsicLowering *IL,
                                       const std::string &FS)
  : PowerPCTargetMachine(PPC32ID, IL, M, FS,
                         TargetData(PPC32ID,false,4,4,4,4,4,4,2,1,1),
                         PowerPCFrameInfo(*this, false)), JITInfo(*this) {}

unsigned PPC32TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "powerpc-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "powerpc-")
    return 20;

  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}
