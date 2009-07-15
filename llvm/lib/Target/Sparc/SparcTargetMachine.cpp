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

#include "SparcTargetAsmInfo.h"
#include "SparcTargetMachine.h"
#include "Sparc.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

// Register the target.
extern Target TheSparcTarget;
static RegisterTarget<SparcTargetMachine> X(TheSparcTarget, "sparc", "SPARC");

// Force static initialization.
extern "C" void LLVMInitializeSparcTarget() { }

const TargetAsmInfo *SparcTargetMachine::createTargetAsmInfo() const {
  // FIXME: Handle Solaris subtarget someday :)
  return new SparcELFTargetAsmInfo(*this);
}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Target &T, const Module &M, 
                                       const std::string &FS)
  : LLVMTargetMachine(T),
    DataLayout("E-p:32:32-f128:128:128"),
    Subtarget(M, FS), TLInfo(*this), InstrInfo(Subtarget),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
}

bool SparcTargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createSparcISelDag(*this));
  return false;
}

/// addPreEmitPass - This pass may be implemented by targets that want to run
/// passes immediately before machine code is emitted.  This should return
/// true if -print-machineinstrs should print out the code after the passes.
bool SparcTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel){
  PM.add(createSparcFPMoverPass(*this));
  PM.add(createSparcDelaySlotFillerPass(*this));
  return true;
}

bool SparcTargetMachine::addAssemblyEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            bool Verbose,
                                            formatted_raw_ostream &Out) {
  FunctionPass *Printer = getTarget().createAsmPrinter(Out, *this, Verbose);
  if (!Printer)
    llvm_report_error("unable to create assembly printer");
  PM.add(Printer);
  return false;
}
