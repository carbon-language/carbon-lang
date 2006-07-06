//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
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

#include "PPC.h"
#include "PPCFrameInfo.h"
#include "PPCTargetMachine.h"
#include "PPCJITInfo.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
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
  RegisterTarget<PPC32TargetMachine>
  X("ppc32", "  PowerPC 32");
  RegisterTarget<PPC64TargetMachine>
  Y("ppc64", "  PowerPC 64");
}

unsigned PPC32TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  if (sizeof(void*) == 4)
    return 10;
#else
  return 0;
#endif
}
unsigned PPC64TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  if (sizeof(void*) == 8)
    return 10;
#endif
  return 0;
}

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
  // We strongly match "powerpc64-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 10 && std::string(TT.begin(), TT.begin()+10) == "powerpc64-")
    return 20;
  
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return getJITMatchQuality()/2;
}


PPCTargetMachine::PPCTargetMachine(const Module &M, const std::string &FS,
                                   bool is64Bit)
  : TargetMachine("PowerPC"), Subtarget(M, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameInfo(*this, false), JITInfo(*this), TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {

  if (getRelocationModel() == Reloc::Default)
    if (Subtarget.isDarwin())
      setRelocationModel(Reloc::DynamicNoPIC);
    else
      setRelocationModel(Reloc::PIC);
}

PPC32TargetMachine::PPC32TargetMachine(const Module &M, const std::string &FS) 
  : PPCTargetMachine(M, FS, false) {
}


PPC64TargetMachine::PPC64TargetMachine(const Module &M, const std::string &FS)
  : PPCTargetMachine(M, FS, true) {
}

/// addPassesToEmitFile - Add passes to the specified pass manager to implement
/// a static compiler for this target.
///
bool PPCTargetMachine::addPassesToEmitFile(PassManager &PM,
                                           std::ostream &Out,
                                           CodeGenFileType FileType,
                                           bool Fast) {
  if (FileType != TargetMachine::AssemblyFile) return true;
  
  // Run loop strength reduction before anything else.
  if (!Fast) PM.add(createLoopStrengthReducePass(&TLInfo));

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());
  
  // Clean up after other passes, e.g. merging critical edges.
  if (!Fast) PM.add(createCFGSimplificationPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Install an instruction selector.
  PM.add(createPPCISelDag(*this));

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
  if (Subtarget.isAIX())
    PM.add(createAIXAsmPrinter(Out, *this));
  else
    PM.add(createDarwinAsmPrinter(Out, *this));

  PM.add(createMachineCodeDeleter());
  return false;
}

void PPCJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // The JIT should use the static relocation model.
  TM.setRelocationModel(Reloc::Static);

  // Run loop strength reduction before anything else.
  PM.add(createLoopStrengthReducePass(TM.getTargetLowering()));

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // Clean up after other passes, e.g. merging critical edges.
  PM.add(createCFGSimplificationPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Install an instruction selector.
  PM.add(createPPCISelDag(TM));

  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());

  // Must run branch selection immediately preceding the asm printer
  PM.add(createPPCBranchSelectionPass());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));
}

