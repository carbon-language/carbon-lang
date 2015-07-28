//===-- ARMSubtarget.cpp - ARM Subtarget Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "ARMSubtarget.h"
#include "ARMFrameLowering.h"
#include "ARMISelLowering.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSelectionDAGInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "Thumb1FrameLowering.h"
#include "Thumb1InstrInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "arm-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ARMGenSubtargetInfo.inc"

static cl::opt<bool>
UseFusedMulOps("arm-use-mulops",
               cl::init(true), cl::Hidden);

enum ITMode {
  DefaultIT,
  RestrictedIT,
  NoRestrictedIT
};

static cl::opt<ITMode>
IT(cl::desc("IT block support"), cl::Hidden, cl::init(DefaultIT),
   cl::ZeroOrMore,
   cl::values(clEnumValN(DefaultIT, "arm-default-it",
                         "Generate IT block based on arch"),
              clEnumValN(RestrictedIT, "arm-restrict-it",
                         "Disallow deprecated IT based on ARMv8"),
              clEnumValN(NoRestrictedIT, "arm-no-restrict-it",
                         "Allow IT blocks based on ARMv7"),
              clEnumValEnd));

/// initializeSubtargetDependencies - Initializes using a CPU and feature string
/// so that we can use initializer lists for subtarget initialization.
ARMSubtarget &ARMSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  return *this;
}

ARMFrameLowering *ARMSubtarget::initializeFrameLowering(StringRef CPU,
                                                        StringRef FS) {
  ARMSubtarget &STI = initializeSubtargetDependencies(CPU, FS);
  if (STI.isThumb1Only())
    return (ARMFrameLowering *)new Thumb1FrameLowering(STI);

  return new ARMFrameLowering(STI);
}

ARMSubtarget::ARMSubtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS,
                           const ARMBaseTargetMachine &TM, bool IsLittle)
    : ARMGenSubtargetInfo(TT, CPU, FS), ARMProcFamily(Others),
      ARMProcClass(None), stackAlignment(4), CPUString(CPU), IsLittle(IsLittle),
      TargetTriple(TT), Options(TM.Options), TM(TM),
      FrameLowering(initializeFrameLowering(CPU, FS)),
      // At this point initializeSubtargetDependencies has been called so
      // we can query directly.
      InstrInfo(isThumb1Only()
                    ? (ARMBaseInstrInfo *)new Thumb1InstrInfo(*this)
                    : !isThumb()
                          ? (ARMBaseInstrInfo *)new ARMInstrInfo(*this)
                          : (ARMBaseInstrInfo *)new Thumb2InstrInfo(*this)),
      TLInfo(TM, *this) {}

void ARMSubtarget::initializeEnvironment() {
  HasV4TOps = false;
  HasV5TOps = false;
  HasV5TEOps = false;
  HasV6Ops = false;
  HasV6MOps = false;
  HasV6KOps = false;
  HasV6T2Ops = false;
  HasV7Ops = false;
  HasV8Ops = false;
  HasV8_1aOps = false;
  HasVFPv2 = false;
  HasVFPv3 = false;
  HasVFPv4 = false;
  HasFPARMv8 = false;
  HasNEON = false;
  UseNEONForSinglePrecisionFP = false;
  UseMulOps = UseFusedMulOps;
  SlowFPVMLx = false;
  HasVMLxForwarding = false;
  SlowFPBrcc = false;
  InThumbMode = false;
  UseSoftFloat = false;
  HasThumb2 = false;
  NoARM = false;
  ReserveR9 = false;
  NoMovt = false;
  SupportsTailCall = false;
  HasFP16 = false;
  HasD16 = false;
  HasHardwareDivide = false;
  HasHardwareDivideInARM = false;
  HasT2ExtractPack = false;
  HasDataBarrier = false;
  Pref32BitThumb = false;
  AvoidCPSRPartialUpdate = false;
  AvoidMOVsShifterOperand = false;
  HasRAS = false;
  HasMPExtension = false;
  HasVirtualization = false;
  FPOnlySP = false;
  HasPerfMon = false;
  HasTrustZone = false;
  HasCrypto = false;
  HasCRC = false;
  HasZeroCycleZeroing = false;
  StrictAlign = false;
  Thumb2DSP = false;
  UseNaClTrap = false;
  GenLongCalls = false;
  UnsafeFPMath = false;
}

void ARMSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  if (CPUString.empty()) {
    if (isTargetDarwin() && TargetTriple.getArchName().endswith("v7s"))
      // Default to the Swift CPU when targeting armv7s/thumbv7s.
      CPUString = "swift";
    else
      CPUString = "generic";
  }

  // Insert the architecture feature derived from the target triple into the
  // feature string. This is important for setting features that are implied
  // based on the architecture version.
  std::string ArchFS = ARM_MC::ParseARMTriple(TargetTriple, CPUString);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = (Twine(ArchFS) + "," + FS).str();
    else
      ArchFS = FS;
  }
  ParseSubtargetFeatures(CPUString, ArchFS);

  // FIXME: This used enable V6T2 support implicitly for Thumb2 mode.
  // Assert this for now to make the change obvious.
  assert(hasV6T2Ops() || !hasThumb2());

  // Keep a pointer to static instruction cost data for the specified CPU.
  SchedModel = getSchedModelForCPU(CPUString);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  // FIXME: this is invalid for WindowsCE
  if (isTargetWindows())
    NoARM = true;

  if (isAAPCS_ABI())
    stackAlignment = 8;
  if (isTargetNaCl())
    stackAlignment = 16;

  if (isTargetMachO())
    SupportsTailCall = !isTargetIOS() || !getTargetTriple().isOSVersionLT(5, 0);
  else
    SupportsTailCall = !isThumb1Only();

  switch (IT) {
  case DefaultIT:
    RestrictIT = hasV8Ops();
    break;
  case RestrictedIT:
    RestrictIT = true;
    break;
  case NoRestrictedIT:
    RestrictIT = false;
    break;
  }

  // NEON f32 ops are non-IEEE 754 compliant. Darwin is ok with it by default.
  const FeatureBitset &Bits = getFeatureBits();
  if ((Bits[ARM::ProcA5] || Bits[ARM::ProcA8]) && // Where this matters
      (Options.UnsafeFPMath || isTargetDarwin()))
    UseNEONForSinglePrecisionFP = true;
}

bool ARMSubtarget::isAPCS_ABI() const {
  assert(TM.TargetABI != ARMBaseTargetMachine::ARM_ABI_UNKNOWN);
  return TM.TargetABI == ARMBaseTargetMachine::ARM_ABI_APCS;
}
bool ARMSubtarget::isAAPCS_ABI() const {
  assert(TM.TargetABI != ARMBaseTargetMachine::ARM_ABI_UNKNOWN);
  return TM.TargetABI == ARMBaseTargetMachine::ARM_ABI_AAPCS;
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol.
bool
ARMSubtarget::GVIsIndirectSymbol(const GlobalValue *GV,
                                 Reloc::Model RelocM) const {
  if (RelocM == Reloc::Static)
    return false;

  bool isDef = GV->isStrongDefinitionForLinker();

  if (!isTargetMachO()) {
    // Extra load is needed for all externally visible.
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return false;
    return true;
  } else {
    // If this is a strong reference to a definition, it is definitely not
    // through a stub.
    if (isDef)
      return false;

    // Unless we have a symbol with hidden visibility, we have to go through a
    // normal $non_lazy_ptr stub because this symbol might be resolved late.
    if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
      return true;

    if (RelocM == Reloc::PIC_) {
      // If symbol visibility is hidden, we have a stub for common symbol
      // references and external declarations.
      if (GV->isDeclarationForLinker() || GV->hasCommonLinkage())
        // Hidden $non_lazy_ptr reference.
        return true;
    }
  }

  return false;
}

unsigned ARMSubtarget::getMispredictionPenalty() const {
  return SchedModel.MispredictPenalty;
}

bool ARMSubtarget::hasSinCos() const {
  return getTargetTriple().isiOS() && !getTargetTriple().isOSVersionLT(7, 0);
}

bool ARMSubtarget::enableMachineScheduler() const {
  // Enable the MachineScheduler before register allocation for out-of-order
  // architectures where we do not use the PostRA scheduler anymore (for now
  // restricted to swift).
  return getSchedModel().isOutOfOrder() && isSwift();
}

// This overrides the PostRAScheduler bit in the SchedModel for any CPU.
bool ARMSubtarget::enablePostRAScheduler() const {
  // No need for PostRA scheduling on out of order CPUs (for now restricted to
  // swift).
  if (getSchedModel().isOutOfOrder() && isSwift())
    return false;
  return (!isThumb() || hasThumb2());
}

bool ARMSubtarget::enableAtomicExpand() const {
  return hasAnyDataBarrier() && !isThumb1Only();
}

bool ARMSubtarget::useMovt(const MachineFunction &MF) const {
  // NOTE Windows on ARM needs to use mov.w/mov.t pairs to materialise 32-bit
  // immediates as it is inherently position independent, and may be out of
  // range otherwise.
  return !NoMovt && hasV6T2Ops() &&
         (isTargetWindows() ||
          !MF.getFunction()->hasFnAttribute(Attribute::MinSize));
}

bool ARMSubtarget::useFastISel() const {
  // Thumb2 support on iOS; ARM support on iOS, Linux and NaCl.
  return TM.Options.EnableFastISel &&
         ((isTargetMachO() && !isThumb1Only()) ||
          (isTargetLinux() && !isThumb()) || (isTargetNaCl() && !isThumb()));
}
