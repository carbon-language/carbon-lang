//===-- PTXTargetMachine.cpp - Define TargetMachine for PTX ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PTX target.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
  MCStreamer *createPTXAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                   bool isVerboseAsm, bool useLoc,
                                   bool useCFI,
                                   MCInstPrinter *InstPrint,
                                   MCCodeEmitter *CE,
                                   TargetAsmBackend *TAB,
                                   bool ShowInst);
}

extern "C" void LLVMInitializePTXTarget() {

  RegisterTargetMachine<PTX32TargetMachine> X(ThePTX32Target);
  RegisterTargetMachine<PTX64TargetMachine> Y(ThePTX64Target);

  TargetRegistry::RegisterAsmStreamer(ThePTX32Target, createPTXAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(ThePTX64Target, createPTXAsmStreamer);
}

namespace {
  const char* DataLayout32 =
    "e-p:32:32-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
  const char* DataLayout64 =
    "e-p:64:64-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
}

// DataLayout and FrameLowering are filled with dummy data
PTXTargetMachine::PTXTargetMachine(const Target &T,
                                   StringRef TT,
                                   StringRef CPU,
                                   StringRef FS,
                                   Reloc::Model RM, bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, RM),
    DataLayout(is64Bit ? DataLayout64 : DataLayout32),
    Subtarget(TT, CPU, FS, is64Bit),
    FrameLowering(Subtarget),
    InstrInfo(*this),
    TLInfo(*this) {
}

PTX32TargetMachine::PTX32TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       Reloc::Model RM)
  : PTXTargetMachine(T, TT, CPU, FS, RM, false) {
}

PTX64TargetMachine::PTX64TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       Reloc::Model RM)
  : PTXTargetMachine(T, TT, CPU, FS, RM, true) {
}

bool PTXTargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  PM.add(createPTXISelDag(*this, OptLevel));
  return false;
}

bool PTXTargetMachine::addPostRegAlloc(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // PTXMFInfoExtract must after register allocation!
  PM.add(createPTXMFInfoExtract(*this, OptLevel));
  return false;
}
