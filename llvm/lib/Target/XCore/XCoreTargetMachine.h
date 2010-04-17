//===-- XCoreTargetMachine.h - Define TargetMachine for XCore ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the XCore specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETMACHINE_H
#define XCORETARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "XCoreFrameInfo.h"
#include "XCoreSubtarget.h"
#include "XCoreInstrInfo.h"
#include "XCoreISelLowering.h"

namespace llvm {

class XCoreTargetMachine : public LLVMTargetMachine {
  XCoreSubtarget Subtarget;
  const TargetData DataLayout;       // Calculates type size & alignment
  XCoreInstrInfo InstrInfo;
  XCoreFrameInfo FrameInfo;
  XCoreTargetLowering TLInfo;
public:
  XCoreTargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);

  virtual const XCoreInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const XCoreFrameInfo *getFrameInfo() const { return &FrameInfo; }
  virtual const XCoreSubtarget *getSubtargetImpl() const { return &Subtarget; }
  virtual const XCoreTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
};

} // end namespace llvm

#endif
