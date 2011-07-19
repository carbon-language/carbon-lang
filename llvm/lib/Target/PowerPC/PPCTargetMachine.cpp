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
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

// This is duplicated code. Refactor this.
static MCStreamer *createMCStreamer(const Target &T, const std::string &TT,
                                    MCContext &Ctx, TargetAsmBackend &TAB,
                                    raw_ostream &OS,
                                    MCCodeEmitter *Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  if (Triple(TT).isOSDarwin())
    return createMachOStreamer(Ctx, TAB, OS, Emitter, RelaxAll);

  return NULL;
}

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);  
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
  
  // Register the MC Code Emitter
  TargetRegistry::RegisterCodeEmitter(ThePPC32Target, createPPCMCCodeEmitter);
  TargetRegistry::RegisterCodeEmitter(ThePPC64Target, createPPCMCCodeEmitter);
  
  
  // Register the asm backend.
  TargetRegistry::RegisterAsmBackend(ThePPC32Target, createPPCAsmBackend);
  TargetRegistry::RegisterAsmBackend(ThePPC64Target, createPPCAsmBackend);
  
  // Register the object streamer.
  TargetRegistry::RegisterObjectStreamer(ThePPC32Target, createMCStreamer);
  TargetRegistry::RegisterObjectStreamer(ThePPC64Target, createMCStreamer);
}

PPCTargetMachine::PPCTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   Reloc::Model RM, bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, RM),
    Subtarget(TT, CPU, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameLowering(Subtarget), JITInfo(*this, is64Bit),
    TLInfo(*this), TSInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {
}

/// Override this for PowerPC.  Tail merging happily breaks up instruction issue
/// groups, which typically degrades performance.
bool PPCTargetMachine::getEnableTailMergeDefault() const { return false; }

PPC32TargetMachine::PPC32TargetMachine(const Target &T, StringRef TT, 
                                       StringRef CPU,
                                       StringRef FS, Reloc::Model RM) 
  : PPCTargetMachine(T, TT, CPU, FS, RM, false) {
}


PPC64TargetMachine::PPC64TargetMachine(const Target &T, StringRef TT, 
                                       StringRef CPU, 
                                       StringRef FS, Reloc::Model RM)
  : PPCTargetMachine(T, TT, CPU, FS, RM, true) {
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool PPCTargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createPPCISelDag(*this));
  return false;
}

bool PPCTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel) {
  // Must run branch selection immediately preceding the asm printer.
  PM.add(createPPCBranchSelectionPass());
  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel,
                                      JITCodeEmitter &JCE) {
  // FIXME: This should be moved to TargetJITInfo!!
  if (Subtarget.isPPC64())
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    DisableJumpTables = true;      
  
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCJITCodeEmitterPass(*this, JCE));

  return false;
}
