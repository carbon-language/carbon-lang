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

class MCInst;
class MCContext;
class MachineInstr;

class AMDGPUMCInstLower {

  MCContext &Ctx;

public:
  AMDGPUMCInstLower(MCContext &ctx);

  /// \brief Lower a MachineInstr to an MCInst
  void lower(const MachineInstr *MI, MCInst &OutMI) const;

};

} // End namespace llvm

#endif //AMDGPU_MCINSTLOWER_H
