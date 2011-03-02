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
#include "PTXMCAsmInfo.h"
#include "PTXTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
  MCStreamer *createPTXAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                   bool isVerboseAsm, bool useLoc,
                                   MCInstPrinter *InstPrint,
                                   MCCodeEmitter *CE,
                                   TargetAsmBackend *TAB,
                                   bool ShowInst);
}

extern "C" void LLVMInitializePTXTarget() {
  RegisterTargetMachine<PTXTargetMachine> X(ThePTXTarget);
  RegisterAsmInfo<PTXMCAsmInfo> Y(ThePTXTarget);
  TargetRegistry::RegisterAsmStreamer(ThePTXTarget, createPTXAsmStreamer);
}

namespace {
  const char* DataLayout32 = "e-p:32:32-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
  const char* DataLayout64 = "e-p:64:64-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
}

// DataLayout and FrameLowering are filled with dummy data
PTXTargetMachine::PTXTargetMachine(const Target &T,
                                   const std::string &TT,
                                   const std::string &FS)
  : Subtarget(TT, FS),
    // FIXME: This feels like a dirty hack, but Subtarget does not appear to be
    //        initialized at this point, and we need to finish initialization of
    //        DataLayout.
    DataLayout((FS.find("64bit") != FS.npos) ? DataLayout64 : DataLayout32),
    LLVMTargetMachine(T, TT),
    FrameLowering(Subtarget),
    TLInfo(*this),
    InstrInfo(*this) {
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
