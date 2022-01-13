//===-- SystemZSubtarget.cpp - SystemZ subtarget information --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZSubtarget.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "systemz-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SystemZGenSubtargetInfo.inc"

static cl::opt<bool> UseSubRegLiveness(
    "systemz-subreg-liveness",
    cl::desc("Enable subregister liveness tracking for SystemZ (experimental)"),
    cl::Hidden);

// Pin the vtable to this file.
void SystemZSubtarget::anchor() {}

SystemZSubtarget &
SystemZSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  StringRef CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
  // Parse features string.
  ParseSubtargetFeatures(CPUName, /*TuneCPU*/ CPUName, FS);

  // -msoft-float implies -mno-vx.
  if (HasSoftFloat)
    HasVector = false;

  // -mno-vx implicitly disables all vector-related features.
  if (!HasVector) {
    HasVectorEnhancements1 = false;
    HasVectorEnhancements2 = false;
    HasVectorPackedDecimal = false;
    HasVectorPackedDecimalEnhancement = false;
    HasVectorPackedDecimalEnhancement2 = false;
  }

  return *this;
}

SystemZCallingConventionRegisters *
SystemZSubtarget::initializeSpecialRegisters() {
  if (isTargetXPLINK64())
    return new SystemZXPLINK64Registers;
  else if (isTargetELF())
    return new SystemZELFRegisters;
  else {
    llvm_unreachable("Invalid Calling Convention. Cannot initialize Special "
                     "Call Registers!");
  }
}

SystemZSubtarget::SystemZSubtarget(const Triple &TT, const std::string &CPU,
                                   const std::string &FS,
                                   const TargetMachine &TM)
    : SystemZGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      HasDistinctOps(false), HasLoadStoreOnCond(false), HasHighWord(false),
      HasFPExtension(false), HasPopulationCount(false),
      HasMessageSecurityAssist3(false), HasMessageSecurityAssist4(false),
      HasResetReferenceBitsMultiple(false), HasFastSerialization(false),
      HasInterlockedAccess1(false), HasMiscellaneousExtensions(false),
      HasExecutionHint(false), HasLoadAndTrap(false),
      HasTransactionalExecution(false), HasProcessorAssist(false),
      HasDFPZonedConversion(false), HasEnhancedDAT2(false), HasVector(false),
      HasLoadStoreOnCond2(false), HasLoadAndZeroRightmostByte(false),
      HasMessageSecurityAssist5(false), HasDFPPackedConversion(false),
      HasMiscellaneousExtensions2(false), HasGuardedStorage(false),
      HasMessageSecurityAssist7(false), HasMessageSecurityAssist8(false),
      HasVectorEnhancements1(false), HasVectorPackedDecimal(false),
      HasInsertReferenceBitsMultiple(false), HasMiscellaneousExtensions3(false),
      HasMessageSecurityAssist9(false), HasVectorEnhancements2(false),
      HasVectorPackedDecimalEnhancement(false), HasEnhancedSort(false),
      HasDeflateConversion(false), HasVectorPackedDecimalEnhancement2(false),
      HasNNPAssist(false), HasBEAREnhancement(false),
      HasResetDATProtection(false), HasProcessorActivityInstrumentation(false),
      HasSoftFloat(false), TargetTriple(TT),
      SpecialRegisters(initializeSpecialRegisters()),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      TSInfo(), FrameLowering() {}

bool SystemZSubtarget::enableSubRegLiveness() const {
  return UseSubRegLiveness;
}

bool SystemZSubtarget::isPC32DBLSymbol(const GlobalValue *GV,
                                       CodeModel::Model CM) const {
  // PC32DBL accesses require the low bit to be clear.
  //
  // FIXME: Explicitly check for functions: the datalayout is currently
  // missing information about function pointers.
  const DataLayout &DL = GV->getParent()->getDataLayout();
  if (GV->getPointerAlignment(DL) == 1 && !GV->getValueType()->isFunctionTy())
    return false;

  // For the small model, all locally-binding symbols are in range.
  if (CM == CodeModel::Small)
    return TLInfo.getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);

  // For Medium and above, assume that the symbol is not within the 4GB range.
  // Taking the address of locally-defined text would be OK, but that
  // case isn't easy to detect.
  return false;
}
