//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);  
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
}

PPCTargetMachine::PPCTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameLowering(Subtarget), JITInfo(*this, is64Bit),
    TLInfo(*this), TSInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {
}

/// Override this for PowerPC.  Tail merging happily breaks up instruction issue
/// groups, which typically degrades performance.
bool PPCTargetMachine::getEnableTailMergeDefault() const { return false; }

void PPC32TargetMachine::anchor() { }

PPC32TargetMachine::PPC32TargetMachine(const Target &T, StringRef TT, 
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {
}

void PPC64TargetMachine::anchor() { }

PPC64TargetMachine::PPC64TargetMachine(const Target &T, StringRef TT, 
                                       StringRef CPU,  StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool PPCTargetMachine::addInstSelector(PassManagerBase &PM) {
  // Install an instruction selector.
  PM.add(createPPCISelDag(*this));
  return false;
}

bool PPCTargetMachine::addPreEmitPass(PassManagerBase &PM) {
  // Must run branch selection immediately preceding the asm printer.
  PM.add(createPPCBranchSelectionPass());
  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      JITCodeEmitter &JCE) {
  // FIXME: This should be moved to TargetJITInfo!!
  if (Subtarget.isPPC64())
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    Options.DisableJumpTables = true;      
  
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCJITCodeEmitterPass(*this, JCE));

  return false;
}
