//===-- AlphaTargetMachine.h - Define TargetMachine for Alpha ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Alpha-specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHA_TARGETMACHINE_H
#define ALPHA_TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "AlphaInstrInfo.h"

namespace llvm {

class GlobalValue;
class IntrinsicLowering;

class AlphaTargetMachine : public TargetMachine {
  AlphaInstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;

public:
  AlphaTargetMachine(const Module &M, IntrinsicLowering *IL);

  virtual const AlphaInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);

  static unsigned getModuleMatchQuality(const Module &M);
};

} // end namespace llvm

#endif
