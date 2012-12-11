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
#include "AMDILDevice.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "AMDGPUGenSubtargetInfo.inc"

#define MAX_CB_SIZE (1 << 16)

namespace llvm {

class AMDGPUSubtarget : public AMDGPUGenSubtargetInfo {
private:
  bool CapsOverride[AMDGPUDeviceInfo::MaxNumberCapabilities];
  const AMDGPUDevice *Device;
  size_t DefaultSize[3];
  std::string DevName;
  bool Is64bit;
  bool Is32on64bit;
  bool DumpCode;
  bool R600ALUInst;

  InstrItineraryData InstrItins;

public:
  AMDGPUSubtarget(StringRef TT, StringRef CPU, StringRef FS);
  virtual ~AMDGPUSubtarget();

  const InstrItineraryData &getInstrItineraryData() const { return InstrItins; }
  virtual void ParseSubtargetFeatures(llvm::StringRef CPU, llvm::StringRef FS);

  bool isOverride(AMDGPUDeviceInfo::Caps) const;
  bool is64bit() const;

  // Helper functions to simplify if statements
  bool isTargetELF() const;
  const AMDGPUDevice* device() const;
  std::string getDataLayout() const;
  std::string getDeviceName() const;
  virtual size_t getDefaultSize(uint32_t dim) const;
  bool dumpCode() const { return DumpCode; }
  bool r600ALUEncoding() const { return R600ALUInst; }

};

} // End namespace llvm

#endif // AMDGPUSUBTARGET_H
