//===-- SparcTargetMachine.cpp - Define TargetMachine for Sparc -----------===//
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

#include "SparcTargetMachine.h"
#include "Sparc.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/PassManager.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSparcTarget() {
  // Register the target.
  RegisterTargetMachine<SparcV8TargetMachine> X(TheSparcTarget);
  RegisterTargetMachine<SparcV9TargetMachine> Y(TheSparcV9Target);
}

static std::string computeDataLayout(const SparcSubtarget &ST) {
  // Sparc is big endian.
  std::string Ret = "E";

  // Some ABIs have 32bit pointers.
  if (!ST.is64Bit())
    Ret += "-p:32:32";

  // Alignments for 64 bit integers.
  Ret += "-i64:64";

  // On SparcV9 128 floats are aligned to 128 bits, on others only to 64.
  // On SparcV9 registers can hold 64 or 32 bits, on others only 32.
  if (ST.is64Bit())
    Ret += "-n32:64";
  else
    Ret += "-f128:64-n32";

  if (ST.is64Bit())
    Ret += "-S128";
  else
    Ret += "-S64";

  return Ret;
}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL,
                                       bool is64bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, is64bit),
    DL(computeDataLayout(Subtarget)),
    InstrInfo(Subtarget),
    TLInfo(*this), TSInfo(*this),
    FrameLowering(Subtarget) {
  initAsmInfo();
}

namespace {
/// Sparc Code Generator Pass Configuration Options.
class SparcPassConfig : public TargetPassConfig {
public:
  SparcPassConfig(SparcTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  SparcTargetMachine &getSparcTargetMachine() const {
    return getTM<SparcTargetMachine>();
  }

  virtual bool addInstSelector();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *SparcTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SparcPassConfig(this, PM);
}

bool SparcPassConfig::addInstSelector() {
  addPass(createSparcISelDag(getSparcTargetMachine()));
  return false;
}

bool SparcTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        JITCodeEmitter &JCE) {
  // Machine code emitter pass for Sparc.
  PM.add(createSparcJITCodeEmitterPass(*this, JCE));
  return false;
}

/// addPreEmitPass - This pass may be implemented by targets that want to run
/// passes immediately before machine code is emitted.  This should return
/// true if -print-machineinstrs should print out the code after the passes.
bool SparcPassConfig::addPreEmitPass(){
  addPass(createSparcDelaySlotFillerPass(getSparcTargetMachine()));
  return true;
}

void SparcV8TargetMachine::anchor() { }

SparcV8TargetMachine::SparcV8TargetMachine(const Target &T,
                                           StringRef TT, StringRef CPU,
                                           StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {
}

void SparcV9TargetMachine::anchor() { }

SparcV9TargetMachine::SparcV9TargetMachine(const Target &T,
                                           StringRef TT,  StringRef CPU,
                                           StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {
}
