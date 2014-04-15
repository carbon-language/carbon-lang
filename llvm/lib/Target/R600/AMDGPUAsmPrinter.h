//===-- AMDGPUAsmPrinter.h - Print AMDGPU assembly code ---------*- C++ -*-===//
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
#include <string>
#include <vector>

namespace llvm {

class AMDGPUAsmPrinter : public AsmPrinter {
private:
  struct SIProgramInfo {
    SIProgramInfo() :
      CodeLen(0),
      NumSGPR(0),
      NumVGPR(0) {}

    uint64_t CodeLen;
    unsigned NumSGPR;
    unsigned NumVGPR;
  };

  void getSIProgramInfo(SIProgramInfo &Out, MachineFunction &MF) const;
  void findNumUsedRegistersSI(MachineFunction &MF,
                              unsigned &NumSGPR,
                              unsigned &NumVGPR) const;

  /// \brief Emit register usage information so that the GPU driver
  /// can correctly setup the GPU state.
  void EmitProgramInfoR600(MachineFunction &MF);
  void EmitProgramInfoSI(MachineFunction &MF, const SIProgramInfo &KernelInfo);

public:
  explicit AMDGPUAsmPrinter(TargetMachine &TM, MCStreamer &Streamer);

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "AMDGPU Assembly Printer";
  }

  /// Implemented in AMDGPUMCInstLower.cpp
  virtual void EmitInstruction(const MachineInstr *MI);

protected:
  bool DisasmEnabled;
  std::vector<std::string> DisasmLines, HexLines;
  size_t DisasmLineMaxLen;
};

} // End anonymous llvm

#endif //AMDGPU_ASMPRINTER_H
