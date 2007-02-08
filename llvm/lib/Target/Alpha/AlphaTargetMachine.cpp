//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaJITInfo.h"
#include "AlphaTargetAsmInfo.h"
#include "AlphaTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"

using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<AlphaTargetMachine> X("alpha", "  Alpha (incomplete)");
}

const TargetAsmInfo *AlphaTargetMachine::createTargetAsmInfo() const {
  return new AlphaTargetAsmInfo(*this);
}

unsigned AlphaTargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "alpha*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && TT[0] == 'a' && TT[1] == 'l' && TT[2] == 'p' &&
      TT[3] == 'h' && TT[4] == 'a')
    return 20;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

unsigned AlphaTargetMachine::getJITMatchQuality() {
#ifdef __alpha
  return 10;
#else
  return 0;
#endif
}

AlphaTargetMachine::AlphaTargetMachine(const Module &M, const std::string &FS)
  : DataLayout("e"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    JITInfo(*this),
    Subtarget(M, FS),
    TLInfo(*this) {
  setRelocationModel(Reloc::PIC_);
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool AlphaTargetMachine::addInstSelector(FunctionPassManager &PM, bool Fast) {
  PM.add(createAlphaISelDag(*this));
  return false;
}
bool AlphaTargetMachine::addPreEmitPass(FunctionPassManager &PM, bool Fast) {
  // Must run branch selection immediately preceding the asm printer
  PM.add(createAlphaBranchSelectionPass());
  return false;
}
bool AlphaTargetMachine::addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                            std::ostream &Out) {
  PM.add(createAlphaLLRPPass(*this));
  PM.add(createAlphaCodePrinterPass(Out, *this));
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(FunctionPassManager &PM, bool Fast,
                                        MachineCodeEmitter &MCE) {
  PM.add(createAlphaCodeEmitterPass(*this, MCE));
  return false;
}
bool AlphaTargetMachine::addSimpleCodeEmitter(FunctionPassManager &PM,
                                              bool Fast,
                                              MachineCodeEmitter &MCE) {
  return addCodeEmitter(PM, Fast, MCE);
}
