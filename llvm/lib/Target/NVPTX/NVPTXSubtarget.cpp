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
#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NVPTXGenSubtargetInfo.inc"

using namespace llvm;

// Select Driver Interface
#include "llvm/Support/CommandLine.h"
namespace {
cl::opt<NVPTX::DrvInterface>
DriverInterface(cl::desc("Choose driver interface:"),
                cl::values(
                    clEnumValN(NVPTX::NVCL, "drvnvcl", "Nvidia OpenCL driver"),
                    clEnumValN(NVPTX::CUDA, "drvcuda", "Nvidia CUDA driver"),
                    clEnumValN(NVPTX::TEST, "drvtest", "Plain Test"),
                    clEnumValEnd),
                    cl::init(NVPTX::NVCL));
}

NVPTXSubtarget::NVPTXSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, bool is64Bit)
:NVPTXGenSubtargetInfo(TT, "", FS), // Don't pass CPU to subtarget,
 // because we don't register all
 // nvptx targets.
 Is64Bit(is64Bit) {

  drvInterface = DriverInterface;

  // Provide the default CPU if none
  std::string defCPU = "sm_10";

  // Get the TargetName from the FS if available
  if (FS.empty() && CPU.empty())
    TargetName = defCPU;
  else if (!CPU.empty())
    TargetName = CPU;
  else
    llvm_unreachable("we are not using FeatureStr");

  // Set up the SmVersion
  SmVersion = atoi(TargetName.c_str()+3);
}
