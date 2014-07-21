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
#include <vector>

namespace llvm {

class AMDGPUAsmPrinter : public AsmPrinter {
private:
  struct SIProgramInfo {
    SIProgramInfo() :
      NumVGPR(0),
      NumSGPR(0),
      Priority(0),
      FloatMode(0),
      Priv(0),
      DX10Clamp(0),
      DebugMode(0),
      IEEEMode(0),
      ScratchSize(0),
      CodeLen(0) {}

    // Fields set in PGM_RSRC1 pm4 packet.
    uint32_t NumVGPR;
    uint32_t NumSGPR;
    uint32_t Priority;
    uint32_t FloatMode;
    uint32_t Priv;
    uint32_t DX10Clamp;
    uint32_t DebugMode;
    uint32_t IEEEMode;
    uint32_t ScratchSize;

    // Bonus information for debugging.
    uint64_t CodeLen;
  };

  void getSIProgramInfo(SIProgramInfo &Out, const MachineFunction &MF) const;
  void findNumUsedRegistersSI(const MachineFunction &MF,
                              unsigned &NumSGPR,
                              unsigned &NumVGPR) const;

  /// \brief Emit register usage information so that the GPU driver
  /// can correctly setup the GPU state.
  void EmitProgramInfoR600(const MachineFunction &MF);
  void EmitProgramInfoSI(const MachineFunction &MF, const SIProgramInfo &KernelInfo);

public:
  explicit AMDGPUAsmPrinter(TargetMachine &TM, MCStreamer &Streamer);

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "AMDGPU Assembly Printer";
  }

  /// Implemented in AMDGPUMCInstLower.cpp
  void EmitInstruction(const MachineInstr *MI) override;

  void EmitEndOfAsmFile(Module &M) override;

protected:
  bool DisasmEnabled;
  std::vector<std::string> DisasmLines, HexLines;
  size_t DisasmLineMaxLen;
};

} // End anonymous llvm

#endif //AMDGPU_ASMPRINTER_H
