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

#define DEBUG_TYPE "amdgpu-subtarget"

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
  WavefrontSize = 0;
  CFALUBug = false;
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
unsigned
AMDGPUSubtarget::getWavefrontSize() const {
  return WavefrontSize;
}
unsigned
AMDGPUSubtarget::getStackEntrySize() const {
  assert(getGeneration() <= NORTHERN_ISLANDS);
  switch(getWavefrontSize()) {
  case 16:
    return 8;
  case 32:
    if (hasCaymanISA())
      return 4;
    else
      return 8;
  case 64:
    return 4;
  default:
    llvm_unreachable("Illegal wavefront size.");
  }
}
bool
AMDGPUSubtarget::hasCFAluBug() const {
  assert(getGeneration() <= NORTHERN_ISLANDS);
  return CFALUBug;
}
bool
AMDGPUSubtarget::isTargetELF() const {
  return false;
}
size_t
AMDGPUSubtarget::getDefaultSize(uint32_t dim) const {
  if (dim > 2) {
    return 1;
  } else {
    return DefaultSize[dim];
  }
}

std::string
AMDGPUSubtarget::getDeviceName() const {
  return DevName;
}
