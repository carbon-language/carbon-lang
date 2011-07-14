//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ARMTargetMachine.h"
#include "ARMFrameLowering.h"
#include "ARM.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

// This is duplicated code. Refactor this.
static MCStreamer *createMCStreamer(const Target &T, const std::string &TT,
                                    MCContext &Ctx, TargetAsmBackend &TAB,
                                    raw_ostream &OS,
                                    MCCodeEmitter *Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin())
    return createMachOStreamer(Ctx, TAB, OS, Emitter, RelaxAll);

  if (TheTriple.isOSWindows()) {
    llvm_unreachable("ARM does not support Windows COFF format");
    return NULL;
  }

  return createELFStreamer(Ctx, TAB, OS, Emitter, RelaxAll, NoExecStack);
}

extern "C" void LLVMInitializeARMTarget() {
  // Register the target.
  RegisterTargetMachine<ARMTargetMachine> X(TheARMTarget);
  RegisterTargetMachine<ThumbTargetMachine> Y(TheThumbTarget);

  // Register the MC Code Emitter
  TargetRegistry::RegisterCodeEmitter(TheARMTarget, createARMMCCodeEmitter);
  TargetRegistry::RegisterCodeEmitter(TheThumbTarget, createARMMCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterAsmBackend(TheARMTarget, createARMAsmBackend);
  TargetRegistry::RegisterAsmBackend(TheThumbTarget, createARMAsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterObjectStreamer(TheARMTarget, createMCStreamer);
  TargetRegistry::RegisterObjectStreamer(TheThumbTarget, createMCStreamer);

}

/// TargetMachine ctor - Create an ARM architecture model.
///
ARMBaseTargetMachine::ARMBaseTargetMachine(const Target &T,
                                           const std::string &TT,
                                           const std::string &CPU,
                                           const std::string &FS)
  : LLVMTargetMachine(T, TT, CPU, FS),
    Subtarget(TT, CPU, FS),
    JITInfo(),
    InstrItins(Subtarget.getInstrItineraryData()) {
  DefRelocModel = getRelocationModel();

  // Default to soft float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Soft;
}

ARMTargetMachine::ARMTargetMachine(const Target &T, const std::string &TT,
                                   const std::string &CPU,
                                   const std::string &FS)
  : ARMBaseTargetMachine(T, TT, CPU, FS), InstrInfo(Subtarget),
    DataLayout(Subtarget.isAPCS_ABI() ?
               std::string("e-p:32:32-f64:32:64-i64:32:64-"
                           "v128:32:128-v64:32:64-n32") :
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "v128:64:128-v64:64:64-n32")),
    ELFWriterInfo(*this),
    TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget) {
  if (!Subtarget.hasARMOps())
    report_fatal_error("CPU: '" + Subtarget.getCPUString() + "' does not "
                       "support ARM mode execution!");
}

ThumbTargetMachine::ThumbTargetMachine(const Target &T, const std::string &TT,
                                       const std::string &CPU,
                                       const std::string &FS)
  : ARMBaseTargetMachine(T, TT, CPU, FS),
    InstrInfo(Subtarget.hasThumb2()
              ? ((ARMBaseInstrInfo*)new Thumb2InstrInfo(Subtarget))
              : ((ARMBaseInstrInfo*)new Thumb1InstrInfo(Subtarget))),
    DataLayout(Subtarget.isAPCS_ABI() ?
               std::string("e-p:32:32-f64:32:64-i64:32:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:32:128-v64:32:64-a:0:32-n32") :
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:64:128-v64:64:64-a:0:32-n32")),
    ELFWriterInfo(*this),
    TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget.hasThumb2()
              ? new ARMFrameLowering(Subtarget)
              : (ARMFrameLowering*)new Thumb1FrameLowering(Subtarget)) {
}

// Pass Pipeline Configuration
bool ARMBaseTargetMachine::addPreISel(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel) {
  if (OptLevel != CodeGenOpt::None)
    PM.add(createARMGlobalMergePass(getTargetLowering()));

  return false;
}

bool ARMBaseTargetMachine::addInstSelector(PassManagerBase &PM,
                                           CodeGenOpt::Level OptLevel) {
  PM.add(createARMISelDag(*this, OptLevel));
  return false;
}

bool ARMBaseTargetMachine::addPreRegAlloc(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  // FIXME: temporarily disabling load / store optimization pass for Thumb1.
  if (OptLevel != CodeGenOpt::None && !Subtarget.isThumb1Only())
    PM.add(createARMLoadStoreOptimizationPass(true));
  if (OptLevel != CodeGenOpt::None && Subtarget.isCortexA9())
    PM.add(createMLxExpansionPass());

  return true;
}

bool ARMBaseTargetMachine::addPreSched2(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel) {
  // FIXME: temporarily disabling load / store optimization pass for Thumb1.
  if (OptLevel != CodeGenOpt::None) {
    if (!Subtarget.isThumb1Only())
      PM.add(createARMLoadStoreOptimizationPass());
    if (Subtarget.hasNEON())
      PM.add(createNEONMoveFixPass());
  }

  // Expand some pseudo instructions into multiple instructions to allow
  // proper scheduling.
  PM.add(createARMExpandPseudoPass());

  if (OptLevel != CodeGenOpt::None) {
    if (!Subtarget.isThumb1Only())
      PM.add(createIfConverterPass());
  }
  if (Subtarget.isThumb2())
    PM.add(createThumb2ITBlockPass());

  return true;
}

bool ARMBaseTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  if (Subtarget.isThumb2() && !Subtarget.prefers32BitThumb())
    PM.add(createThumb2SizeReductionPass());

  PM.add(createARMConstantIslandPass());
  return true;
}

bool ARMBaseTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel,
                                          JITCodeEmitter &JCE) {
  // FIXME: Move this to TargetJITInfo!
  if (DefRelocModel == Reloc::Default)
    setRelocationModel(Reloc::Static);

  // Machine code emitter pass for ARM.
  PM.add(createARMJITCodeEmitterPass(*this, JCE));
  return false;
}
