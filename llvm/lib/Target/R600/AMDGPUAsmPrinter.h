//===-- AMDGPUAsmPrinter.h - Print AMDGPU assembly code -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_ASMPRINTER_H
#define AMDGPU_ASMPRINTER_H

#include "llvm/CodeGen/AsmPrinter.h"

namespace llvm {

class AMDGPUAsmPrinter : public AsmPrinter {

public:
  explicit AMDGPUAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "AMDGPU Assembly Printer";
  }

  /// \brief Emit register usage information so that the GPU driver
  /// can correctly setup the GPU state.
  void EmitProgramInfo(MachineFunction &MF);

  /// Implemented in AMDGPUMCInstLower.cpp
  virtual void EmitInstruction(const MachineInstr *MI);
};

} // End anonymous llvm

#endif //AMDGPU_ASMPRINTER_H
