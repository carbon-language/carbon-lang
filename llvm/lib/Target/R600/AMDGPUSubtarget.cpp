//===-- AMDGPUSubtarget.cpp - AMDGPU Subtarget Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Implements the AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSubtarget.h"

using namespace llvm;

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AMDGPUGenSubtargetInfo.inc"

AMDGPUSubtarget::AMDGPUSubtarget(StringRef TT, StringRef CPU, StringRef FS) :
  AMDGPUGenSubtargetInfo(TT, CPU, FS), DumpCode(false) {
    InstrItins = getInstrItineraryForCPU(CPU);

  // Default card
  StringRef GPU = CPU;
  Is64bit = false;
  DefaultSize[0] = 64;
  DefaultSize[1] = 1;
  DefaultSize[2] = 1;
  HasVertexCache = false;
  TexVTXClauseSize = 0;
  Gen = AMDGPUSubtarget::R600;
  FP64 = false;
  CaymanISA = false;
  EnableIRStructurizer = true;
  EnableIfCvt = true;
  ParseSubtargetFeatures(GPU, FS);
  DevName = GPU;
}

bool
AMDGPUSubtarget::is64bit() const  {
  return Is64bit;
}
bool
AMDGPUSubtarget::hasVertexCache() const {
  return HasVertexCache;
}
short
AMDGPUSubtarget::getTexVTXClauseSize() const {
  return TexVTXClauseSize;
}
enum AMDGPUSubtarget::Generation
AMDGPUSubtarget::getGeneration() const {
  return Gen;
}
bool
AMDGPUSubtarget::hasHWFP64() const {
  return FP64;
}
bool
AMDGPUSubtarget::hasCaymanISA() const {
  return CaymanISA;
}
bool
AMDGPUSubtarget::IsIRStructurizerEnabled() const {
  return EnableIRStructurizer;
}
bool
AMDGPUSubtarget::isIfCvtEnabled() const {
  return EnableIfCvt;
}
bool
AMDGPUSubtarget::isTargetELF() const {
  return false;
}
size_t
AMDGPUSubtarget::getDefaultSize(uint32_t dim) const {
  if (dim > 3) {
    return 1;
  } else {
    return DefaultSize[dim];
  }
}

std::string
AMDGPUSubtarget::getDataLayout() const {
  std::string DataLayout = std::string(
   "e"
   "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32"
   "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
   "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048"
   "-n32:64"
  );

  if (hasHWFP64()) {
    DataLayout.append("-f64:64:64");
  }

  if (is64bit()) {
    DataLayout.append("-p:64:64:64");
  } else {
    DataLayout.append("-p:32:32:32");
  }

  if (Gen >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    DataLayout.append("-p3:32:32:32");
  }

  return DataLayout;
}

std::string
AMDGPUSubtarget::getDeviceName() const {
  return DevName;
}
