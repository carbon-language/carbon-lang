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
#include "llvm/Transforms/Scalar.h"
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
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS),
    JITInfo(),
    InstrItins(Subtarget.getInstrItineraryData()) {
  // Default to soft float ABI
  if (Options.FloatABIType == FloatABI::Default)
    this->Options.FloatABIType = FloatABI::Soft;
}

void ARMTargetMachine::anchor() { }

ARMTargetMachine::ARMTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
  : ARMBaseTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    InstrInfo(Subtarget),
    DataLayout(Subtarget.isAPCS_ABI() ?
               std::string("e-p:32:32-f64:32:64-i64:32:64-"
                           "v128:32:128-v64:32:64-n32-S32") :
               Subtarget.isAAPCS_ABI() ?
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "v128:64:128-v64:64:64-n32-S64") :
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "v128:64:128-v64:64:64-n32-S32")),
    ELFWriterInfo(*this),
    TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget) {
  if (!Subtarget.hasARMOps())
    report_fatal_error("CPU: '" + Subtarget.getCPUString() + "' does not "
                       "support ARM mode execution!");
}

void ThumbTargetMachine::anchor() { }

ThumbTargetMachine::ThumbTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : ARMBaseTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    InstrInfo(Subtarget.hasThumb2()
              ? ((ARMBaseInstrInfo*)new Thumb2InstrInfo(Subtarget))
              : ((ARMBaseInstrInfo*)new Thumb1InstrInfo(Subtarget))),
    DataLayout(Subtarget.isAPCS_ABI() ?
               std::string("e-p:32:32-f64:32:64-i64:32:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:32:128-v64:32:64-a:0:32-n32-S32") :
               Subtarget.isAAPCS_ABI() ?
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:64:128-v64:64:64-a:0:32-n32-S64") :
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:64:128-v64:64:64-a:0:32-n32-S32")),
    ELFWriterInfo(*this),
    TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget.hasThumb2()
              ? new ARMFrameLowering(Subtarget)
              : (ARMFrameLowering*)new Thumb1FrameLowering(Subtarget)) {
}

bool ARMBaseTargetMachine::addPreISel(PassManagerBase &PM) {
  if (getOptLevel() != CodeGenOpt::None && EnableGlobalMerge)
    PM.add(createGlobalMergePass(getTargetLowering()));

  return false;
}

bool ARMBaseTargetMachine::addInstSelector(PassManagerBase &PM) {
  PM.add(createARMISelDag(*this, getOptLevel()));
  return false;
}

bool ARMBaseTargetMachine::addPreRegAlloc(PassManagerBase &PM) {
  // FIXME: temporarily disabling load / store optimization pass for Thumb1.
  if (getOptLevel() != CodeGenOpt::None && !Subtarget.isThumb1Only())
    PM.add(createARMLoadStoreOptimizationPass(true));
  if (getOptLevel() != CodeGenOpt::None && Subtarget.isCortexA9())
    PM.add(createMLxExpansionPass());
  return true;
}

bool ARMBaseTargetMachine::addPreSched2(PassManagerBase &PM) {
  // FIXME: temporarily disabling load / store optimization pass for Thumb1.
  if (getOptLevel() != CodeGenOpt::None) {
    if (!Subtarget.isThumb1Only())
      PM.add(createARMLoadStoreOptimizationPass());
    if (Subtarget.hasNEON())
      PM.add(createExecutionDependencyFixPass(&ARM::DPRRegClass));
  }

  // Expand some pseudo instructions into multiple instructions to allow
  // proper scheduling.
  PM.add(createARMExpandPseudoPass());

  if (getOptLevel() != CodeGenOpt::None) {
    if (!Subtarget.isThumb1Only())
      PM.add(createIfConverterPass());
  }
  if (Subtarget.isThumb2())
    PM.add(createThumb2ITBlockPass());

  return true;
}

bool ARMBaseTargetMachine::addPreEmitPass(PassManagerBase &PM) {
  if (Subtarget.isThumb2()) {
    if (!Subtarget.prefers32BitThumb())
      PM.add(createThumb2SizeReductionPass());

    // Constant island pass work on unbundled instructions.
    PM.add(createUnpackMachineBundlesPass());
  }

  PM.add(createARMConstantIslandPass());

  return true;
}

bool ARMBaseTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                          JITCodeEmitter &JCE) {
  // Machine code emitter pass for ARM.
  PM.add(createARMJITCodeEmitterPass(*this, JCE));
  return false;
}
