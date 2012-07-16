//===-- AMDGPUMCTargetDesc.h - AMDGPU Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AMDGPU specific target descriptions.
//
//===----------------------------------------------------------------------===//
//

#ifndef AMDGPUMCTARGETDESC_H
#define AMDGPUMCTARGETDESC_H

namespace llvm {
class MCSubtargetInfo;
class Target;

extern Target TheAMDGPUTarget;

} // End llvm namespace

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "AMDGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"

#endif // AMDGPUMCTARGETDESC_H
