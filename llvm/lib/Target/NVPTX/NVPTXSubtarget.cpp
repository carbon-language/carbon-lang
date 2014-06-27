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

  // We default to PTX 3.1, but we cannot just default to it in the initializer
  // since the attribute parser checks if the given option is >= the default.
  // So if we set ptx31 as the default, the ptx30 attribute would never match.
  // Instead, we use 0 as the default and manually set 31 if the default is
  // used.
  if (PTXVersion == 0)
    PTXVersion = 31;

  return *this;
}

NVPTXSubtarget::NVPTXSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, const TargetMachine &TM,
                               bool is64Bit)
    : NVPTXGenSubtargetInfo(TT, CPU, FS), Is64Bit(is64Bit), PTXVersion(0),
      SmVersion(20), DL(computeDataLayout(is64Bit)),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      TLInfo((NVPTXTargetMachine &)TM), TSInfo(&DL), FrameLowering(*this) {

  Triple T(TT);

  if (T.getOS() == Triple::NVCL)
    drvInterface = NVPTX::NVCL;
  else
    drvInterface = NVPTX::CUDA;
}
