//===-- PowerPCTargetMachine.h - Define TargetMachine for PowerPC -*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_TARGETMACHINE_H
#define POWERPC_TARGETMACHINE_H

#include "PowerPCFrameInfo.h"
#include "PowerPCInstrInfo.h"
#include "PowerPCJITInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include <set>

namespace llvm {

class GlobalValue;
class IntrinsicLowering;

class PowerPCTargetMachine : public TargetMachine {
  PowerPCInstrInfo InstrInfo;
  PowerPCFrameInfo FrameInfo;
  PowerPCJITInfo JITInfo;

protected:
  PowerPCTargetMachine(const std::string &name, IntrinsicLowering *IL,
                       const TargetData &TD, const PowerPCFrameInfo &TFI,
                       const PowerPCJITInfo &TJI, bool is64b);
public:
  virtual const PowerPCInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual TargetJITInfo *getJITInfo() {
    return &JITInfo;
  }

  static unsigned getJITMatchQuality();

  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);

  // Two shared sets between the instruction selector and the printer allow for
  // correct linkage on Darwin
  std::set<GlobalValue*> CalledFunctions;
  std::set<GlobalValue*> AddressTaken;
};

} // end namespace llvm

#endif
