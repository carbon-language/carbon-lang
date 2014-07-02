//===-- SystemZSubtarget.cpp - SystemZ subtarget information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZSubtarget.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Host.h"

using namespace llvm;

#define DEBUG_TYPE "systemz-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SystemZGenSubtargetInfo.inc"

// Pin the vtable to this file.
void SystemZSubtarget::anchor() {}

SystemZSubtarget &
SystemZSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
#if defined(__linux__) && defined(__s390x__)
  if (CPUName == "generic")
    CPUName = sys::getHostCPUName();
#endif
  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
  return *this;
}

SystemZSubtarget::SystemZSubtarget(const std::string &TT,
                                   const std::string &CPU,
                                   const std::string &FS,
                                   const TargetMachine &TM)
    : SystemZGenSubtargetInfo(TT, CPU, FS), HasDistinctOps(false),
      HasLoadStoreOnCond(false), HasHighWord(false), HasFPExtension(false),
      HasFastSerialization(false), HasInterlockedAccess1(false),
      TargetTriple(TT),
      // Make sure that global data has at least 16 bits of alignment by
      // default, so that we can refer to it using LARL.  We don't have any
      // special requirements for stack variables though.
      DL("E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM),
      TSInfo(DL), FrameLowering() {}

// Return true if GV binds locally under reloc model RM.
static bool bindsLocally(const GlobalValue *GV, Reloc::Model RM) {
  // For non-PIC, all symbols bind locally.
  if (RM == Reloc::Static)
    return true;

  return GV->hasLocalLinkage() || !GV->hasDefaultVisibility();
}

bool SystemZSubtarget::isPC32DBLSymbol(const GlobalValue *GV,
                                       Reloc::Model RM,
                                       CodeModel::Model CM) const {
  // PC32DBL accesses require the low bit to be clear.  Note that a zero
  // value selects the default alignment and is therefore OK.
  if (GV->getAlignment() == 1)
    return false;

  // For the small model, all locally-binding symbols are in range.
  if (CM == CodeModel::Small)
    return bindsLocally(GV, RM);

  // For Medium and above, assume that the symbol is not within the 4GB range.
  // Taking the address of locally-defined text would be OK, but that
  // case isn't easy to detect.
  return false;
}
