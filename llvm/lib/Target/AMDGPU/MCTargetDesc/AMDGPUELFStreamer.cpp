//===-------- AMDGPUELFStreamer.cpp - ELF Object Output -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUELFStreamer.h"
#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;

void AMDGPUELFStreamer::InitSections(bool NoExecStack) {
  // Start with the .hsatext section by default.
  SwitchSection(AMDGPU::getHSATextSection(getContext()));
}

MCELFStreamer *llvm::createAMDGPUELFStreamer(MCContext &Context,
                                           MCAsmBackend &MAB,
                                           raw_pwrite_stream &OS,
                                           MCCodeEmitter *Emitter,
                                           bool RelaxAll) {
  return new AMDGPUELFStreamer(Context, MAB, OS, Emitter);
}
