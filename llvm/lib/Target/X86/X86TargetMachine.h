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

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "X86.h"
#include "X86ELFWriterInfo.h"
#include "X86InstrInfo.h"
#include "X86JITInfo.h"
#include "X86Subtarget.h"
#include "X86ISelLowering.h"

namespace llvm {
  
class formatted_raw_ostream;

class X86TargetMachine : public LLVMTargetMachine {
  X86Subtarget      Subtarget;
  const TargetData  DataLayout; // Calculates type size & alignment
  TargetFrameInfo   FrameInfo;
  X86InstrInfo      InstrInfo;
  X86JITInfo        JITInfo;
  X86TargetLowering TLInfo;
  X86ELFWriterInfo  ELFWriterInfo;
  Reloc::Model      DefRelocModel; // Reloc model before it's overridden.

private:
  // We have specific defaults for X86.
  virtual void setCodeModelForJIT();
  virtual void setCodeModelForStatic();
  
public:
  X86TargetMachine(const Target &T, const std::string &TT, 
                   const std::string &FS, bool is64Bit);

  virtual const X86InstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual       X86JITInfo       *getJITInfo()         { return &JITInfo; }
  virtual const X86Subtarget     *getSubtargetImpl() const{ return &Subtarget; }
  virtual       X86TargetLowering *getTargetLowering() const { 
    return const_cast<X86TargetLowering*>(&TLInfo); 
  }
  virtual const X86RegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  virtual const X86ELFWriterInfo *getELFWriterInfo() const {
    return Subtarget.isTargetELF() ? &ELFWriterInfo : 0;
  }

  virtual DwarfLSDAEncoding::Encoding getLSDAEncoding() const;

  // Set up the pass pipeline.
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPostRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              MachineCodeEmitter &MCE);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              JITCodeEmitter &JCE);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              ObjectCodeEmitter &OCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    MachineCodeEmitter &MCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    JITCodeEmitter &JCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    ObjectCodeEmitter &OCE);
};

/// X86_32TargetMachine - X86 32-bit target machine.
///
class X86_32TargetMachine : public X86TargetMachine {
public:
  X86_32TargetMachine(const Target &T, const std::string &M,
                      const std::string &FS);
};

/// X86_64TargetMachine - X86 64-bit target machine.
///
class X86_64TargetMachine : public X86TargetMachine {
public:
  X86_64TargetMachine(const Target &T, const std::string &TT,
                      const std::string &FS);
};

} // End llvm namespace

#endif
