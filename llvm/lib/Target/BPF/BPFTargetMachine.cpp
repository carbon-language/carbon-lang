//===-- BPFTargetMachine.cpp - Define TargetMachine for BPF ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about BPF target spec.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFTargetMachine.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

extern "C" void LLVMInitializeBPFTarget() {
  // Register the target.
  RegisterTargetMachine<BPFTargetMachine> X(TheBPFTarget);
}

// DataLayout --> Little-endian, 64-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer.
BPFTargetMachine::BPFTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                                   StringRef FS, const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, "e-m:e-p:64:64-i64:64-n32:64-S128", TT, CPU, FS,
                        Options, RM, CM, OL),
      TLOF(make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, CPU, FS, *this) {
  initAsmInfo();
}
namespace {
// BPF Code Generator Pass Configuration Options.
class BPFPassConfig : public TargetPassConfig {
public:
  BPFPassConfig(BPFTargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  BPFTargetMachine &getBPFTargetMachine() const {
    return getTM<BPFTargetMachine>();
  }

  bool addInstSelector() override;
};
}

TargetPassConfig *BPFTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new BPFPassConfig(this, PM);
}

// Install an instruction selector pass using
// the ISelDag to gen BPF code.
bool BPFPassConfig::addInstSelector() {
  addPass(createBPFISelDag(getBPFTargetMachine()));

  return false;
}
