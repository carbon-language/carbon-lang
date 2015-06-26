//===-- AMDGPUBaseInfo.cpp - AMDGPU Base encoding information--------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "AMDGPUBaseInfo.h"
#include "llvm/MC/SubtargetFeature.h"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_ENUM

namespace llvm {
namespace AMDGPU {

IsaVersion getIsaVersion(const FeatureBitset &Features) {

  if (Features.test(FeatureISAVersion7_0_0))
    return {7, 0, 0};

  if (Features.test(FeatureISAVersion7_0_1))
    return {7, 0, 1};

  if (Features.test(FeatureISAVersion8_0_0))
    return {8, 0, 0};

  if (Features.test(FeatureISAVersion8_0_1))
    return {8, 0, 1};

  return {0, 0, 0};
}

} // End namespace AMDGPU
} // End namespace llvm
