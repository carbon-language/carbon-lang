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
#include "llvm/ADT/SmallString.h"

#include "llvm/ADT/SmallString.h"

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
  FP64Denormals(false),
  FP32Denormals(false),
  CaymanISA(false),
  EnableIRStructurizer(true),
  EnablePromoteAlloca(false),
  EnableIfCvt(true),
  WavefrontSize(0),
  CFALUBug(false),
  LocalMemorySize(0),
  InstrItins(getInstrItineraryForCPU(GPU)) {
  // On SI+, we want FP64 denormals to be on by default. FP32 denormals can be
  // enabled, but some instructions do not respect them and they run at the
  // double precision rate, so don't enable by default.
  //
  // We want to be able to turn these off, but making this a subtarget feature
  // for SI has the unhelpful behavior that it unsets everything else if you
  // disable it.

  SmallString<256> FullFS("+promote-alloca,+fp64-denormals,");
  FullFS += FS;

  ParseSubtargetFeatures(GPU, FullFS);

  if (getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS) {
    InstrInfo.reset(new R600InstrInfo(*this));

    // FIXME: I don't think think Evergreen has any useful support for
    // denormals, but should be checked. Should we issue a warning somewhere if
    // someone tries to enable these?
    FP32Denormals = false;
    FP64Denormals = false;
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
