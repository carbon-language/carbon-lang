//===-- AArch64Subtarget.cpp - AArch64 Subtarget Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "AArch64Subtarget.h"
#include "AArch64RegisterInfo.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define DEBUG_TYPE "aarch64-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AArch64GenSubtargetInfo.inc"

using namespace llvm;

enum AlignMode {
  DefaultAlign,
  StrictAlign,
  NoStrictAlign
};

static cl::opt<AlignMode>
Align(cl::desc("Load/store alignment support"),
      cl::Hidden, cl::init(DefaultAlign),
      cl::values(
          clEnumValN(DefaultAlign,  "aarch64-default-align",
                     "Generate unaligned accesses only on hardware/OS "
                     "combinations that are known to support them"),
          clEnumValN(StrictAlign,   "aarch64-strict-align",
                     "Disallow all unaligned memory accesses"),
          clEnumValN(NoStrictAlign, "aarch64-no-strict-align",
                     "Allow unaligned memory accesses"),
          clEnumValEnd));

// Pin the vtable to this file.
void AArch64Subtarget::anchor() {}

AArch64Subtarget::AArch64Subtarget(StringRef TT, StringRef CPU, StringRef FS,
                                   bool LittleEndian)
    : AArch64GenSubtargetInfo(TT, CPU, FS), ARMProcFamily(Others),
      HasFPARMv8(false), HasNEON(false), HasCrypto(false), TargetTriple(TT),
      CPUString(CPU), IsLittleEndian(LittleEndian) {

  initializeSubtargetFeatures(CPU, FS);
}

void AArch64Subtarget::initializeSubtargetFeatures(StringRef CPU,
                                                   StringRef FS) {
  AllowsUnalignedMem = false;

  if (CPU.empty())
    CPUString = "generic";

  std::string FullFS = FS;
  if (CPUString == "generic") {
    // Enable FP by default.
    if (FullFS.empty())
      FullFS = "+fp-armv8";
    else
      FullFS = "+fp-armv8," + FullFS;
  }

  ParseSubtargetFeatures(CPU, FullFS);

  switch (Align) {
    case DefaultAlign:
      // Linux targets support unaligned accesses on AARCH64
      AllowsUnalignedMem = isTargetLinux();
      break;
    case StrictAlign:
      AllowsUnalignedMem = false;
      break;
    case NoStrictAlign:
      AllowsUnalignedMem = true;
      break;
  }
}

bool AArch64Subtarget::GVIsIndirectSymbol(const GlobalValue *GV,
                                          Reloc::Model RelocM) const {
  if (RelocM == Reloc::Static)
    return false;

  return !GV->hasLocalLinkage() && !GV->hasHiddenVisibility();
}
