//===-- SkeletonTargetMachine.h - TargetMachine for Skeleton ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Skeleton specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SKELETONTARGETMACHINE_H
#define SKELETONTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "SkeletonInstrInfo.h"
#include "SkeletonJITInfo.h"

namespace llvm {
  class IntrinsicLowering;

  class SkeletonTargetMachine : public TargetMachine {
    SkeletonInstrInfo InstrInfo;
    TargetFrameInfo FrameInfo;
    SkeletonJITInfo JITInfo;
  public:
    SkeletonTargetMachine(const Module &M, IntrinsicLowering *IL,
                          const std::string &FS);

    virtual const SkeletonInstrInfo *getInstrInfo() const { return &InstrInfo; }
    virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
    virtual const MRegisterInfo *getRegisterInfo() const {
      return &InstrInfo.getRegisterInfo();
    }
    virtual TargetJITInfo *getJITInfo() {
      return &JITInfo;
    }

    virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                            MachineCodeEmitter &MCE);

    virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                     CodeGenFileType FileType, bool Fast);
  };

} // end namespace llvm

#endif
