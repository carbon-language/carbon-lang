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
#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeMipsTarget() {
  // Register the target.
  RegisterTargetMachine<MipsTargetMachine> X(TheMipsTarget);
  RegisterTargetMachine<MipselTargetMachine> Y(TheMipselTarget);
}

const TargetAsmInfo *MipsTargetMachine::createTargetAsmInfo() const {
  return new MipsTargetAsmInfo();
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables 
// an easier handling.
// Using CodeModel::Large enables different CALL behavior.
MipsTargetMachine::
MipsTargetMachine(const Target &T, const std::string &TT, const std::string &FS,
                  bool isLittle=false):
  LLVMTargetMachine(T, TT),
  Subtarget(TT, FS, isLittle), 
  DataLayout(isLittle ? std::string("e-p:32:32:32-i8:8:32-i16:16:32") :
                        std::string("E-p:32:32:32-i8:8:32-i16:16:32")), 
  InstrInfo(*this), 
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0),
  TLInfo(*this) {
  // Abicall enables PIC by default
  if (getRelocationModel() == Reloc::Default) {
    if (Subtarget.isABI_O32())
      setRelocationModel(Reloc::PIC_);
    else
      setRelocationModel(Reloc::Static);
  }

  // TODO: create an option to enable long calls, like -mlong-calls, 
  // that would be our CodeModel::Large. It must not work with Abicall.
  if (getCodeModel() == CodeModel::Default)
    setCodeModel(CodeModel::Small);
}

MipselTargetMachine::
MipselTargetMachine(const Target &T, const std::string &TT,
                    const std::string &FS) :
  MipsTargetMachine(T, TT, FS, true) {}

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
