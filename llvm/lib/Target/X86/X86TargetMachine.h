//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "X86.h"
#include "X86ELFWriterInfo.h"
#include "X86InstrInfo.h"
#include "X86ISelLowering.h"
#include "X86FrameLowering.h"
#include "X86JITInfo.h"
#include "X86SelectionDAGInfo.h"
#include "X86Subtarget.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  
class formatted_raw_ostream;
class StringRef;

class X86TargetMachine : public LLVMTargetMachine {
  X86Subtarget      Subtarget;
  X86FrameLowering  FrameLowering;
  X86ELFWriterInfo  ELFWriterInfo;

public:
  X86TargetMachine(const Target &T, StringRef TT, 
                   StringRef CPU, StringRef FS,
                   Reloc::Model RM, CodeModel::Model CM,
                   bool is64Bit);

  virtual const X86InstrInfo     *getInstrInfo() const {
    llvm_unreachable("getInstrInfo not implemented");
  }
  virtual const TargetFrameLowering  *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    llvm_unreachable("getJITInfo not implemented");
  }
  virtual const X86Subtarget     *getSubtargetImpl() const{ return &Subtarget; }
  virtual const X86TargetLowering *getTargetLowering() const {
    llvm_unreachable("getTargetLowering not implemented");
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const { 
    llvm_unreachable("getSelectionDAGInfo not implemented");
  }
  virtual const X86RegisterInfo  *getRegisterInfo() const {
    return &getInstrInfo()->getRegisterInfo();
  }
  virtual const X86ELFWriterInfo *getELFWriterInfo() const {
    return Subtarget.isTargetELF() ? &ELFWriterInfo : 0;
  }

  // Set up the pass pipeline.
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPostRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              JITCodeEmitter &JCE);
};

/// X86_32TargetMachine - X86 32-bit target machine.
///
class X86_32TargetMachine : public X86TargetMachine {
  const TargetData  DataLayout; // Calculates type size & alignment
  X86InstrInfo      InstrInfo;
  X86SelectionDAGInfo TSInfo;
  X86TargetLowering TLInfo;
  X86JITInfo        JITInfo;
public:
  X86_32TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS,
                      Reloc::Model RM, CodeModel::Model CM);
  virtual const TargetData *getTargetData() const { return &DataLayout; }
  virtual const X86TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const { 
    return &TSInfo;
  }
  virtual const X86InstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    return &JITInfo;
  }
};

/// X86_64TargetMachine - X86 64-bit target machine.
///
class X86_64TargetMachine : public X86TargetMachine {
  const TargetData  DataLayout; // Calculates type size & alignment
  X86InstrInfo      InstrInfo;
  X86SelectionDAGInfo TSInfo;
  X86TargetLowering TLInfo;
  X86JITInfo        JITInfo;
public:
  X86_64TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS,
                      Reloc::Model RM, CodeModel::Model CM);
  virtual const TargetData *getTargetData() const { return &DataLayout; }
  virtual const X86TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const { 
    return &TSInfo;
  }
  virtual const X86InstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    return &JITInfo;
  }
};

} // End llvm namespace

#endif
