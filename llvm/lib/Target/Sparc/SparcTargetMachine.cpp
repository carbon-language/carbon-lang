//===-- SparcTargetMachine.cpp - Define TargetMachine for Sparc -----------===//
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

#include "SparcTargetAsmInfo.h"
#include "SparcTargetMachine.h"
#include "Sparc.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<SparcTargetMachine> X("sparc", "  SPARC");
}

const TargetAsmInfo *SparcTargetMachine::createTargetAsmInfo() const {
  return new SparcTargetAsmInfo(*this);
}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Module &M, const std::string &FS)
  : DataLayout("E-p:32:32"),
    Subtarget(M, FS), InstrInfo(Subtarget),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
}

unsigned SparcTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "sparc-")
    return 20;

  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
#ifdef __sparc__
    return 20;   // BE/32 ==> Prefer sparc on sparc
#else
    return 5;    // BE/32 ==> Prefer ppc elsewhere
#endif
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return 0;
}

bool SparcTargetMachine::addInstSelector(FunctionPassManager &PM, bool Fast) {
  PM.add(createSparcISelDag(*this));
  return false;
}

/// addPreEmitPass - This pass may be implemented by targets that want to run
/// passes immediately before machine code is emitted.  This should return
/// true if -print-machineinstrs should print out the code after the passes.
bool SparcTargetMachine::addPreEmitPass(FunctionPassManager &PM, bool Fast) {
  PM.add(createSparcFPMoverPass(*this));
  PM.add(createSparcDelaySlotFillerPass(*this));
  return true;
}

bool SparcTargetMachine::addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                            std::ostream &Out) {
  // Output assembly language.
  PM.add(createSparcCodePrinterPass(Out, *this));
  return false;
}
