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

#define DEBUG_TYPE "systemz-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SystemZGenSubtargetInfo.inc"

using namespace llvm;

// Pin the vtabel to this file.
void SystemZSubtarget::anchor() {}

SystemZSubtarget::SystemZSubtarget(const std::string &TT,
                                   const std::string &CPU,
                                   const std::string &FS)
  : SystemZGenSubtargetInfo(TT, CPU, FS), HasDistinctOps(false),
    HasLoadStoreOnCond(false), HasHighWord(false), HasFPExtension(false),
    HasFastSerialization(false), HasInterlockedAccess1(false),
    TargetTriple(TT) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
#if defined(__linux__) && defined(__s390x__)
  if (CPUName == "generic")
    CPUName = sys::getHostCPUName();
#endif

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
}

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
