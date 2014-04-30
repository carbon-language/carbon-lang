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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "AMDGPUGenSubtargetInfo.inc"

#define MAX_CB_SIZE (1 << 16)

namespace llvm {

class AMDGPUSubtarget : public AMDGPUGenSubtargetInfo {
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
  size_t DefaultSize[3];
  std::string DevName;
  bool Is64bit;
  bool Is32on64bit;
  bool DumpCode;
  bool R600ALUInst;
  bool HasVertexCache;
  short TexVTXClauseSize;
  enum Generation Gen;
  bool FP64;
  bool CaymanISA;
  bool EnableIRStructurizer;
  bool EnableIfCvt;
  unsigned WavefrontSize;
  bool CFALUBug;

  InstrItineraryData InstrItins;

public:
  AMDGPUSubtarget(StringRef TT, StringRef CPU, StringRef FS);

  const InstrItineraryData &getInstrItineraryData() const { return InstrItins; }
  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

  bool is64bit() const;
  bool hasVertexCache() const;
  short getTexVTXClauseSize() const;
  enum Generation getGeneration() const;
  bool hasHWFP64() const;
  bool hasCaymanISA() const;

  bool hasBFE() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasBFM() const {
    return hasBFE();
  }

  bool hasMulU24() const {
    return (getGeneration() >= EVERGREEN);
  }

  bool hasMulI24() const {
    return (getGeneration() >= SOUTHERN_ISLANDS ||
            hasCaymanISA());
  }

  bool IsIRStructurizerEnabled() const;
  bool isIfCvtEnabled() const;
  unsigned getWavefrontSize() const;
  unsigned getStackEntrySize() const;
  bool hasCFAluBug() const;

  bool enableMachineScheduler() const override {
    return getGeneration() <= NORTHERN_ISLANDS;
  }

  // Helper functions to simplify if statements
  bool isTargetELF() const;
  std::string getDeviceName() const;
  virtual size_t getDefaultSize(uint32_t dim) const final;
  bool dumpCode() const { return DumpCode; }
  bool r600ALUEncoding() const { return R600ALUInst; }

};

} // End namespace llvm

#endif // AMDGPUSUBTARGET_H
