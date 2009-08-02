//===-- BlackfinTargetMachine.cpp - Define TargetMachine for Blackfin -----===//
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

#include "BlackfinTargetMachine.h"
#include "Blackfin.h"
#include "BlackfinTargetAsmInfo.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializeBlackfinTarget() {
  RegisterTargetMachine<BlackfinTargetMachine> X(TheBlackfinTarget);
}

const TargetAsmInfo* BlackfinTargetMachine::createTargetAsmInfo() const {
  return new BlackfinTargetAsmInfo();
}

BlackfinTargetMachine::BlackfinTargetMachine(const Target &T,
                                             const Module &M,
                                             const std::string &FS)
  : LLVMTargetMachine(T),
    DataLayout("e-p:32:32-i64:32-f64:32"),
    Subtarget(M.getTargetTriple(), FS),
    TLInfo(*this),
    InstrInfo(Subtarget),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 4, 0) {
}

unsigned BlackfinTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && std::string(TT.begin(), TT.begin()+5) == "bfin-")
    return 20;

  // Otherwise we don't match.
  return 0;
}

bool BlackfinTargetMachine::addInstSelector(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel) {
  PM.add(createBlackfinISelDag(*this, OptLevel));
  return false;
}
