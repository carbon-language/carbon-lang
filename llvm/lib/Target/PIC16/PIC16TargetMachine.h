//===-- PIC16TargetMachine.h - Define TargetMachine for PIC16 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PIC16 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef PIC16_TARGETMACHINE_H
#define PIC16_TARGETMACHINE_H

#include "PIC16InstrInfo.h"
#include "PIC16ISelLowering.h"
#include "PIC16Subtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// PIC16TargetMachine
///
class PIC16TargetMachine : public LLVMTargetMachine {
  PIC16Subtarget        Subtarget;
  const TargetData      DataLayout;       // Calculates type size & alignment
  PIC16InstrInfo        InstrInfo;
  PIC16TargetLowering   TLInfo;
  TargetFrameInfo       FrameInfo;

protected:
  virtual const TargetAsmInfo *createTargetAsmInfo() const;
  
public:
  PIC16TargetMachine(const Module &M, const std::string &FS);

  virtual const TargetFrameInfo *getFrameInfo() const 
  { return &FrameInfo; }
  virtual const PIC16InstrInfo *getInstrInfo() const 
  { return &InstrInfo; }
  virtual const TargetData *getTargetData() const    
  { return &DataLayout; }
  virtual PIC16TargetLowering *getTargetLowering() const 
  { return const_cast<PIC16TargetLowering*>(&TLInfo); }
  virtual const PIC16RegisterInfo *getRegisterInfo() const 
  { return &InstrInfo.getRegisterInfo(); }
  
  virtual bool addInstSelector(PassManagerBase &PM, bool Fast);
  virtual bool addPrologEpilogInserter(PassManagerBase &PM, bool Fast);
  virtual bool addPreEmitPass(PassManagerBase &PM, bool Fast);
  virtual bool addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                  raw_ostream &Out);
};
} // end namespace llvm

#endif
