//===- PTXInstrInfo.h - PTX Instruction Information -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_INSTR_INFO_H
#define PTX_INSTR_INFO_H

#include "PTXRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {
class PTXTargetMachine;

class PTXInstrInfo : public TargetInstrInfoImpl {
  private:
    const PTXRegisterInfo RI;
    PTXTargetMachine &TM;

  public:
    explicit PTXInstrInfo(PTXTargetMachine &_TM);

    virtual const PTXRegisterInfo &getRegisterInfo() const { return RI; }
  }; // class PTXInstrInfo
} // namespace llvm

#endif // PTX_INSTR_INFO_H
