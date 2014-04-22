//===- NVPTXSubtarget.cpp - NVPTX Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVPTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "NVPTXSubtarget.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NVPTXGenSubtargetInfo.inc"

// Pin the vtable to this file.
void NVPTXSubtarget::anchor() {}

NVPTXSubtarget::NVPTXSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, bool is64Bit)
    : NVPTXGenSubtargetInfo(TT, CPU, FS), Is64Bit(is64Bit), PTXVersion(0),
      SmVersion(20) {

  Triple T(TT);

  if (T.getOS() == Triple::NVCL)
    drvInterface = NVPTX::NVCL;
  else
    drvInterface = NVPTX::CUDA;

  // Provide the default CPU if none
  std::string defCPU = "sm_20";

  ParseSubtargetFeatures((CPU.empty() ? defCPU : CPU), FS);

  // Get the TargetName from the FS if available
  if (FS.empty() && CPU.empty())
    TargetName = defCPU;
  else if (!CPU.empty())
    TargetName = CPU;
  else
    llvm_unreachable("we are not using FeatureStr");

  // We default to PTX 3.1, but we cannot just default to it in the initializer
  // since the attribute parser checks if the given option is >= the default.
  // So if we set ptx31 as the default, the ptx30 attribute would never match.
  // Instead, we use 0 as the default and manually set 31 if the default is
  // used.
  if (PTXVersion == 0) {
    PTXVersion = 31;
  }
}
