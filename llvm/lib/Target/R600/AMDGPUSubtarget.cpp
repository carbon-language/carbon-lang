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
#include "R600InstrInfo.h"
#include "SIInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AMDGPUGenSubtargetInfo.inc"

AMDGPUSubtarget::AMDGPUSubtarget(StringRef TT, StringRef GPU, StringRef FS) :
  AMDGPUGenSubtargetInfo(TT, GPU, FS),
  DevName(GPU),
  Is64bit(false),
  DumpCode(false),
  R600ALUInst(false),
  HasVertexCache(false),
  TexVTXClauseSize(0),
  Gen(AMDGPUSubtarget::R600),
  FP64(false),
  CaymanISA(false),
  EnableIRStructurizer(true),
  EnableIfCvt(true),
  WavefrontSize(0),
  CFALUBug(false),
  LocalMemorySize(0),
  InstrItins(getInstrItineraryForCPU(GPU)) {
  ParseSubtargetFeatures(GPU, FS);

  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    InstrInfo.reset(new R600InstrInfo(*this));
  } else {
    InstrInfo.reset(new SIInstrInfo(*this));
  }
}

unsigned AMDGPUSubtarget::getStackEntrySize() const {
  assert(getGeneration() <= NORTHERN_ISLANDS);
  switch(getWavefrontSize()) {
  case 16:
    return 8;
  case 32:
    return hasCaymanISA() ? 4 : 8;
  case 64:
    return 4;
  default:
    llvm_unreachable("Illegal wavefront size.");
  }
}
