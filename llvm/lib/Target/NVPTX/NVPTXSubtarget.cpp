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

static std::string computeDataLayout(bool is64Bit) {
  std::string Ret = "e";

  if (!is64Bit)
    Ret += "-p:32:32";

  Ret += "-i64:64-v16:16-v32:32-n16:32:64";

  return Ret;
}

NVPTXSubtarget &NVPTXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
    // Provide the default CPU if we don't have one.
  if (CPU.empty() && FS.size())
    llvm_unreachable("we are not using FeatureStr");
  TargetName = CPU.empty() ? "sm_20" : CPU;

  ParseSubtargetFeatures(TargetName, FS);

  // Set default to PTX 3.2 (CUDA 5.5)
  if (PTXVersion == 0) {
    PTXVersion = 32;
  }

  return *this;
}

NVPTXSubtarget::NVPTXSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, const TargetMachine &TM,
                               bool is64Bit)
    : NVPTXGenSubtargetInfo(TT, CPU, FS), Is64Bit(is64Bit), PTXVersion(0),
      SmVersion(20), DL(computeDataLayout(is64Bit)),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      TLInfo((const NVPTXTargetMachine &)TM), TSInfo(&DL),
      FrameLowering(*this) {

  Triple T(TT);

  if (T.getOS() == Triple::NVCL)
    drvInterface = NVPTX::NVCL;
  else
    drvInterface = NVPTX::CUDA;
}
