//=====-- AMDGPUSubtarget.h - Define Subtarget for the AMDIL ---*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUSUBTARGET_H
#define AMDGPUSUBTARGET_H
#include "AMDGPU.h"
#include "AMDGPUInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "AMDGPUGenSubtargetInfo.inc"

#define MAX_CB_SIZE (1 << 16)

namespace llvm {

class AMDGPUSubtarget : public AMDGPUGenSubtargetInfo {

  std::unique_ptr<AMDGPUInstrInfo> InstrInfo;

public:
  enum Generation {
    R600 = 0,
    R700,
    EVERGREEN,
    NORTHERN_ISLANDS,
    SOUTHERN_ISLANDS,
    SEA_ISLANDS
  };

private:
  std::string DevName;
  bool Is64bit;
  bool DumpCode;
  bool R600ALUInst;
  bool HasVertexCache;
  short TexVTXClauseSize;
  Generation Gen;
  bool FP64;
  bool FP64Denormals;
  bool FP32Denormals;
  bool CaymanISA;
  bool EnableIRStructurizer;
  bool EnablePromoteAlloca;
  bool EnableIfCvt;
  unsigned WavefrontSize;
  bool CFALUBug;
  int LocalMemorySize;

  InstrItineraryData InstrItins;

public:
  AMDGPUSubtarget(StringRef TT, StringRef CPU, StringRef FS);

  const AMDGPUInstrInfo *getInstrInfo() const {
    return InstrInfo.get();
  }

  const InstrItineraryData &getInstrItineraryData() const {
    return InstrItins;
  }

  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

  bool is64bit() const {
    return Is64bit;
  }

  bool hasVertexCache() const {
    return HasVertexCache;
  }

  short getTexVTXClauseSize() const {
    return TexVTXClauseSize;
  }

  Generation getGeneration() const {
    return Gen;
  }

  bool hasHWFP64() const {
    return FP64;
  }

  bool hasCaymanISA() const {
    return CaymanISA;
  }

  bool hasFP32Denormals() const {
    return FP32Denormals;
  }

  bool hasFP64Denormals() const {
    return FP64Denormals;
  }

  bool hasBFE() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBFI() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBFM() const {
    return hasBFE();
  }

  bool hasBCNT(unsigned Size) const {
    if (Size == 32)
      return (getGeneration() >= EVERGREEN);

    if (Size == 64)
      return (getGeneration() >= SOUTHERN_ISLANDS);

    return false;
  }

  bool hasMulU24() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasMulI24() const {
    return (getGeneration() >= SOUTHERN_ISLANDS ||
            hasCaymanISA());
  }

  bool hasFFBL() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasFFBH() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool IsIRStructurizerEnabled() const {
    return EnableIRStructurizer;
  }

  bool isPromoteAllocaEnabled() const {
    return EnablePromoteAlloca;
  }

  bool isIfCvtEnabled() const {
    return EnableIfCvt;
  }

  unsigned getWavefrontSize() const {
    return WavefrontSize;
  }

  unsigned getStackEntrySize() const;

  bool hasCFAluBug() const {
    assert(getGeneration() <= NORTHERN_ISLANDS);
    return CFALUBug;
  }

  int getLocalMemorySize() const {
    return LocalMemorySize;
  }

  bool enableMachineScheduler() const override {
    return getGeneration() <= NORTHERN_ISLANDS;
  }

  // Helper functions to simplify if statements
  bool isTargetELF() const {
    return false;
  }

  StringRef getDeviceName() const {
    return DevName;
  }

  bool dumpCode() const {
    return DumpCode;
  }
  bool r600ALUEncoding() const {
    return R600ALUInst;
  }
};

} // End namespace llvm

#endif // AMDGPUSUBTARGET_H
