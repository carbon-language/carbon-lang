//===-- IA64TargetMachine.h - Define TargetMachine for IA64 ---*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the IA64 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef IA64TARGETMACHINE_H
#define IA64TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "IA64InstrInfo.h"
#include "IA64ISelLowering.h"

namespace llvm {

class IA64TargetMachine : public TargetMachine {
  const TargetData DataLayout;       // Calculates type size & alignment
  IA64InstrInfo      InstrInfo;
  TargetFrameInfo    FrameInfo;
  //IA64JITInfo      JITInfo;
  IA64TargetLowering TLInfo;
public:
  IA64TargetMachine(const Module &M, const std::string &FS);

  virtual const IA64InstrInfo      *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo    *getFrameInfo() const { return &FrameInfo; }
  virtual       IA64TargetLowering *getTargetLowering() const { 
    return const_cast<IA64TargetLowering*>(&TLInfo);
  }
  virtual const MRegisterInfo    *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }

  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast);

  static unsigned getModuleMatchQuality(const Module &M);

};
} // End llvm namespace

#endif


