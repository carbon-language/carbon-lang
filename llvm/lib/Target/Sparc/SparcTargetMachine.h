//===-- SparcTargetMachine.h - Define TargetMachine for Sparc ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sparc specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCTARGETMACHINE_H
#define SPARCTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "SparcInstrInfo.h"
#include "SparcSubtarget.h"

namespace llvm {

class Module;

class SparcTargetMachine : public TargetMachine {
  SparcSubtarget Subtarget;
  SparcInstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
public:
  SparcTargetMachine(const Module &M, const std::string &FS);

  virtual const SparcInstrInfo *getInstrInfo() const { return &InstrInfo; }
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
