//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -----------===//
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

#include "Alpha.h"
#include "AlphaJITInfo.h"
#include "AlphaTargetAsmInfo.h"
#include "AlphaTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

// Register the targets
extern Target TheAlphaTarget;
static RegisterTarget<AlphaTargetMachine> X(TheAlphaTarget, "alpha", 
                                            "Alpha [experimental]");

// No assembler printer by default
AlphaTargetMachine::AsmPrinterCtorFn AlphaTargetMachine::AsmPrinterCtor = 0;

// Force static initialization.
extern "C" void LLVMInitializeAlphaTarget() { }

const TargetAsmInfo *AlphaTargetMachine::createTargetAsmInfo() const {
  return new AlphaTargetAsmInfo(*this);
}

AlphaTargetMachine::AlphaTargetMachine(const Target &T, const Module &M, 
                                       const std::string &FS)
  : LLVMTargetMachine(T),
    DataLayout("e-f128:128:128"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    JITInfo(*this),
    Subtarget(M, FS),
    TLInfo(*this) {
  setRelocationModel(Reloc::PIC_);
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool AlphaTargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createAlphaISelDag(*this));
  return false;
}
bool AlphaTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel) {
  // Must run branch selection immediately preceding the asm printer
  PM.add(createAlphaBranchSelectionPass());
  PM.add(createAlphaLLRPPass(*this));
  return false;
}
bool AlphaTargetMachine::addAssemblyEmitter(PassManagerBase &PM,
                                            CodeGenOpt::Level OptLevel,
                                            bool Verbose,
                                            formatted_raw_ostream &Out) {
  // Output assembly language.
  assert(AsmPrinterCtor && "AsmPrinter was not linked in");
  if (AsmPrinterCtor)
    PM.add(AsmPrinterCtor(Out, *this, Verbose));
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel,
                                        bool DumpAsm, MachineCodeEmitter &MCE) {
  PM.add(createAlphaCodeEmitterPass(*this, MCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(ferrs(), *this, true));
  }
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel,
                                        bool DumpAsm, JITCodeEmitter &JCE) {
  PM.add(createAlphaJITCodeEmitterPass(*this, JCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(ferrs(), *this, true));
  }
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel,
                                        bool DumpAsm, ObjectCodeEmitter &OCE) {
  PM.add(createAlphaObjectCodeEmitterPass(*this, OCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(ferrs(), *this, true));
  }
  return false;
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              bool DumpAsm,
                                              MachineCodeEmitter &MCE) {
  return addCodeEmitter(PM, OptLevel, DumpAsm, MCE);
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              bool DumpAsm,
                                              JITCodeEmitter &JCE) {
  return addCodeEmitter(PM, OptLevel, DumpAsm, JCE);
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              bool DumpAsm,
                                              ObjectCodeEmitter &OCE) {
  return addCodeEmitter(PM, OptLevel, DumpAsm, OCE);
}

