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
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static cl::opt<bool>
EnableGlobalMerge("global-merge", cl::Hidden,
                  cl::desc("Enable global merge pass"),
                  cl::init(true));

extern "C" void LLVMInitializeARMTarget() {
  // Register the target.
  RegisterTargetMachine<ARMTargetMachine> X(TheARMTarget);
  RegisterTargetMachine<ThumbTargetMachine> Y(TheThumbTarget);
}

/// TargetMachine ctor - Create an ARM architecture model.
///
ARMBaseTargetMachine::ARMBaseTargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           Reloc::Model RM, CodeModel::Model CM)
  : LLVMTargetMachine(T, TT, CPU, FS, RM, CM),
    Subtarget(TT, CPU, FS),
    JITInfo(),
    InstrItins(Subtarget.getInstrItineraryData()) {
  // Default to soft float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Soft;
}

ARMTargetMachine::ARMTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   Reloc::Model RM, CodeModel::Model CM)
  : ARMBaseTargetMachine(T, TT, CPU, FS, RM, CM), InstrInfo(Subtarget),
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

ThumbTargetMachine::ThumbTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       Reloc::Model RM, CodeModel::Model CM)
  : ARMBaseTargetMachine(T, TT, CPU, FS, RM, CM),
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

bool ARMBaseTargetMachine::addPreISel(PassManagerBase &PM,
                                      CodeGenOpt::Level OptLevel) {
  if (OptLevel != CodeGenOpt::None && EnableGlobalMerge)
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
  if (getMCAsmInfo()->getExceptionHandlingType() == ExceptionHandling::SjLj)
    createARMSjLjLoweringPass();
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
  // Machine code emitter pass for ARM.
  PM.add(createARMJITCodeEmitterPass(*this, JCE));
  return false;
}
