//===-- PowerPCTargetMachine.h - Define TargetMachine for PowerPC -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC-specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_TARGETMACHINE_H
#define POWERPC_TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "PowerPCFrameInfo.h"
#include "PowerPCSubtarget.h"

namespace llvm {

class GlobalValue;
class IntrinsicLowering;

class PowerPCTargetMachine : public TargetMachine {
  PowerPCFrameInfo  FrameInfo;
  PPCSubtarget      Subtarget;
protected:
  PowerPCTargetMachine(const std::string &name, IntrinsicLowering *IL,
                       const Module &M, const TargetData &TD, 
                       const PowerPCFrameInfo &TFI);
public:
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const TargetSubtarget  *getSubtargetImpl() const{ return &Subtarget; }

  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType);
};

} // end namespace llvm

#endif
