//===-- AMDGPURegisterInfo.h - AMDGPURegisterInfo Interface -*- C++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief TargetRegisterInfo interface that is implemented by all hw codegen
/// targets.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUREGISTERINFO_H
#define AMDGPUREGISTERINFO_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

namespace llvm {

class AMDGPUSubtarget;
class TargetInstrInfo;

struct AMDGPURegisterInfo : public AMDGPUGenRegisterInfo {
  static const MCPhysReg CalleeSavedReg;
  const AMDGPUSubtarget &ST;

  AMDGPURegisterInfo(const AMDGPUSubtarget &st);

  BitVector getReservedRegs(const MachineFunction &MF) const override {
    assert(!"Unimplemented");  return BitVector();
  }

  virtual const TargetRegisterClass* getCFGStructurizerRegClass(MVT VT) const {
    assert(!"Unimplemented"); return nullptr;
  }

  virtual unsigned getHWRegIndex(unsigned Reg) const {
    assert(!"Unimplemented"); return 0;
  }

  /// \returns the sub reg enum value for the given \p Channel
  /// (e.g. getSubRegFromChannel(0) -> AMDGPU::sub0)
  unsigned getSubRegFromChannel(unsigned Channel) const;

  const MCPhysReg* getCalleeSavedRegs(const MachineFunction *MF) const override;
  void eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;
  unsigned getFrameRegister(const MachineFunction &MF) const override;

  unsigned getIndirectSubReg(unsigned IndirectIndex) const;

};

} // End namespace llvm

#endif // AMDIDSAREGISTERINFO_H
