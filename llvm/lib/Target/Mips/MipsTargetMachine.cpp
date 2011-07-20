//===-- MipsTargetMachine.cpp - Define TargetMachine for Mips -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Mips target spec.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeMipsTarget() {
  // Register the target.
  RegisterTargetMachine<MipsTargetMachine> X(TheMipsTarget);
  RegisterTargetMachine<MipselTargetMachine> Y(TheMipselTarget);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables
// an easier handling.
// Using CodeModel::Large enables different CALL behavior.
MipsTargetMachine::
MipsTargetMachine(const Target &T, StringRef TT,
                  StringRef CPU, StringRef FS,
                  Reloc::Model RM, CodeModel::Model CM,
                  bool isLittle=false):
  LLVMTargetMachine(T, TT, CPU, FS, RM, CM),
  Subtarget(TT, CPU, FS, isLittle),
  DataLayout(isLittle ? 
             std::string("e-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-n32") :
             std::string("E-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-n32")),
  InstrInfo(*this),
  FrameLowering(Subtarget),
  TLInfo(*this), TSInfo(*this) {
}

MipselTargetMachine::
MipselTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS,
                    Reloc::Model RM, CodeModel::Model CM) :
  MipsTargetMachine(T, TT, CPU, FS, RM, CM, true) {}

// Install an instruction selector pass using
// the ISelDag to gen Mips code.
bool MipsTargetMachine::
addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel)
{
  PM.add(createMipsISelDag(*this));
  return false;
}

// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
bool MipsTargetMachine::
addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel)
{
  PM.add(createMipsDelaySlotFillerPass(*this));
  return true;
}

bool MipsTargetMachine::
addPreRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel) {
  PM.add(createMipsEmitGPRestorePass(*this));
  return true;
}

bool MipsTargetMachine::
addPostRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel) {
  PM.add(createMipsExpandPseudoPass(*this));
  return true;
}
