//===-- PowerPCTargetMachine.cpp - Define TargetMachine for PowerPC -------===//
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

#include "PowerPC.h"
#include "PowerPCTargetMachine.h"
#include "PowerPCFrameInfo.h"
#include "PPC32TargetMachine.h"
#include "PPC64TargetMachine.h"
#include "PPC32JITInfo.h"
#include "PPC64JITInfo.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
using namespace llvm;

bool llvm::GPOPT = false;

namespace llvm {
  cl::opt<bool> AIX("aix",
                    cl::desc("Generate AIX/xcoff instead of Darwin/MachO"),
                    cl::Hidden);
  cl::opt<bool> EnablePPCLSR("enable-lsr-for-ppc",
                             cl::desc("Enable LSR for PPC (beta)"),
                             cl::Hidden);
  cl::opt<bool, true> EnableGPOPT("enable-gpopt", cl::Hidden,
                                  cl::location(GPOPT),
                                  cl::desc("Enable optimizations for GP cpus"));
}

namespace {
  const std::string PPC32ID = "PowerPC/32bit";
  const std::string PPC64ID = "PowerPC/64bit";

  // Register the targets
  RegisterTarget<PPC32TargetMachine>
  X("ppc32", "  PowerPC 32-bit");

#if 0
  RegisterTarget<PPC64TargetMachine>
  Y("ppc64", "  PowerPC 64-bit (unimplemented)");
#endif
}

PowerPCTargetMachine::PowerPCTargetMachine(const std::string &name,
                                           IntrinsicLowering *IL,
                                           const Module &M,
                                           const TargetData &TD,
                                           const PowerPCFrameInfo &TFI)
: TargetMachine(name, IL, TD), FrameInfo(TFI), Subtarget(M) {}

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

  bool LP64 = (0 != dynamic_cast<PPC64TargetMachine *>(this));

  if (EnablePPCLSR) {
    PM.add(createLoopStrengthReducePass());
    PM.add(createCFGSimplificationPass());
  }

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  PM.add(createLowerConstantExpressionsPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Default to pattern ISel
  if (LP64)
    PM.add(createPPC64ISelPattern(*this));
  else if (PatternISelTriState == 0)
    PM.add(createPPC32ISelSimple(*this));
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

  if (AIX)
    PM.add(createAIXAsmPrinter(Out, *this));
  else
    PM.add(createDarwinAsmPrinter(Out, *this));

  PM.add(createMachineCodeDeleter());
  return false;
}

void PowerPCJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // The JIT does not support or need PIC.
  PICEnabled = false;

  bool LP64 = (0 != dynamic_cast<PPC64TargetMachine *>(&TM));

  if (EnablePPCLSR) {
    PM.add(createLoopStrengthReducePass());
    PM.add(createCFGSimplificationPass());
  }

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  PM.add(createLowerConstantExpressionsPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Default to pattern ISel
  if (LP64)
    PM.add(createPPC64ISelPattern(TM));
  else if (PatternISelTriState == 0)
    PM.add(createPPC32ISelSimple(TM));
  else
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
PPC32TargetMachine::PPC32TargetMachine(const Module &M, IntrinsicLowering *IL)
  : PowerPCTargetMachine(PPC32ID, IL, M,
                         TargetData(PPC32ID,false,4,4,4,4,4,4,2,1,1),
                         PowerPCFrameInfo(*this, false)), JITInfo(*this) {}

/// PPC64TargetMachine ctor - Create a LP64 architecture model
///
PPC64TargetMachine::PPC64TargetMachine(const Module &M, IntrinsicLowering *IL)
  : PowerPCTargetMachine(PPC64ID, IL, M,
                         TargetData(PPC64ID,false,8,4,4,4,4,4,2,1,1),
                         PowerPCFrameInfo(*this, true)) {}

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

unsigned PPC64TargetMachine::getModuleMatchQuality(const Module &M) {
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Direct match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}
