//===-- SparcV8TargetMachine.h - Define TargetMachine for SparcV8 -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SparcV8 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV8TARGETMACHINE_H
#define SPARCV8TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "SparcV8InstrInfo.h"
#include "SparcV8Subtarget.h"

namespace llvm {

class IntrinsicLowering;
class Module;

class SparcV8TargetMachine : public TargetMachine {
  SparcV8Subtarget Subtarget;
  SparcV8InstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
public:
  SparcV8TargetMachine(const Module &M, IntrinsicLowering *IL,
                       const std::string &FS);

  virtual const SparcV8InstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const TargetSubtarget  *getSubtargetImpl() const{ return &Subtarget; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  static unsigned getModuleMatchQuality(const Module &M);

  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast);
};

} // end namespace llvm

#endif
