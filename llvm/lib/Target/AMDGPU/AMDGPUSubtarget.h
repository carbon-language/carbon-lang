//=====-- AMDGPUSubtarget.h - Define Subtarget for the AMDIL ---*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file declares the AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef _AMDGPUSUBTARGET_H_
#define _AMDGPUSUBTARGET_H_
#include "AMDILSubtarget.h"

namespace llvm {

class AMDGPUSubtarget : public AMDILSubtarget
{
  InstrItineraryData InstrItins;

public:
  AMDGPUSubtarget(StringRef TT, StringRef CPU, StringRef FS) :
    AMDILSubtarget(TT, CPU, FS)
  {
    InstrItins = getInstrItineraryForCPU(CPU);
  }

  const InstrItineraryData &getInstrItineraryData() const { return InstrItins; }
};

} // End namespace llvm

#endif // AMDGPUSUBTARGET_H_
