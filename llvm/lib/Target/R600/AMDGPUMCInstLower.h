//===- AMDGPUMCInstLower.h MachineInstr Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_MCINSTLOWER_H
#define AMDGPU_MCINSTLOWER_H

namespace llvm {

class AMDGPUSubtarget;
class MachineInstr;
class MCContext;
class MCInst;

class AMDGPUMCInstLower {

  // This must be kept in sync with the SISubtarget class in SIInstrInfo.td
  enum SISubtarget {
    SI = 0
  };

  MCContext &Ctx;
  const AMDGPUSubtarget &ST;

  /// Convert a member of the AMDGPUSubtarget::Generation enum to the
  /// SISubtarget enum.
  enum SISubtarget AMDGPUSubtargetToSISubtarget(unsigned Gen) const;

  /// Get the MC opcode for this MachineInstr.
  unsigned getMCOpcode(unsigned MIOpcode) const;

public:
  AMDGPUMCInstLower(MCContext &ctx, const AMDGPUSubtarget &ST);

  /// \brief Lower a MachineInstr to an MCInst
  void lower(const MachineInstr *MI, MCInst &OutMI) const;

};

} // End namespace llvm

#endif //AMDGPU_MCINSTLOWER_H
