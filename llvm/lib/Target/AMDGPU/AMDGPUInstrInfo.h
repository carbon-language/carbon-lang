//===-- AMDGPUInstrInfo.h - AMDGPU Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of a TargetInstrInfo class that is common
// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUINSTRUCTIONINFO_H_
#define AMDGPUINSTRUCTIONINFO_H_

#include "AMDGPURegisterInfo.h"
#include "AMDILInstrInfo.h"

#include <map>

namespace llvm {

class AMDGPUTargetMachine;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;

class AMDGPUInstrInfo : public AMDILInstrInfo {

public:
  explicit AMDGPUInstrInfo(AMDGPUTargetMachine &tm);

  virtual const AMDGPURegisterInfo &getRegisterInfo() const = 0;

  /// convertToISA - Convert the AMDIL MachineInstr to a supported ISA
  /// MachineInstr
  virtual void convertToISA(MachineInstr & MI, MachineFunction &MF,
    DebugLoc DL) const;

};

} // End llvm namespace

#endif // AMDGPUINSTRINFO_H_
