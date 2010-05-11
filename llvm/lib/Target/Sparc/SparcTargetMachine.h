//===-- SparcTargetMachine.h - Define TargetMachine for Sparc ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sparc specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCTARGETMACHINE_H
#define SPARCTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "SparcInstrInfo.h"
#include "SparcSubtarget.h"
#include "SparcISelLowering.h"
#include "SparcSelectionDAGInfo.h"

namespace llvm {

class SparcTargetMachine : public LLVMTargetMachine {
  SparcSubtarget Subtarget;
  const TargetData DataLayout;       // Calculates type size & alignment
  SparcTargetLowering TLInfo;
  SparcSelectionDAGInfo TSInfo;
  SparcInstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
public:
  SparcTargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS, bool is64bit);

  virtual const SparcInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const SparcSubtarget   *getSubtargetImpl() const{ return &Subtarget; }
  virtual const SparcRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const SparcTargetLowering* getTargetLowering() const {
    return &TLInfo;
  }
  virtual const SparcSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
};

/// SparcV8TargetMachine - Sparc 32-bit target machine
///
class SparcV8TargetMachine : public SparcTargetMachine {
public:
  SparcV8TargetMachine(const Target &T, const std::string &TT,
                       const std::string &FS);
};

/// SparcV9TargetMachine - Sparc 64-bit target machine
///
class SparcV9TargetMachine : public SparcTargetMachine {
public:
  SparcV9TargetMachine(const Target &T, const std::string &TT,
                       const std::string &FS);
};

} // end namespace llvm

#endif
