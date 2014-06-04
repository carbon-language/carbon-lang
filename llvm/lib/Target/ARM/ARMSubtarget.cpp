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
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "arm-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ARMGenSubtargetInfo.inc"

static cl::opt<bool>
ReserveR9("arm-reserve-r9", cl::Hidden,
          cl::desc("Reserve R9, making it unavailable as GPR"));

static cl::opt<bool>
ArmUseMOVT("arm-use-movt", cl::init(true), cl::Hidden);

static cl::opt<bool>
UseFusedMulOps("arm-use-mulops",
               cl::init(true), cl::Hidden);

enum AlignMode {
  DefaultAlign,
  StrictAlign,
  NoStrictAlign
};

static cl::opt<AlignMode>
Align(cl::desc("Load/store alignment support"),
      cl::Hidden, cl::init(DefaultAlign),
      cl::values(
          clEnumValN(DefaultAlign,  "arm-default-align",
                     "Generate unaligned accesses only on hardware/OS "
                     "combinations that are known to support them"),
          clEnumValN(StrictAlign,   "arm-strict-align",
                     "Disallow all unaligned memory accesses"),
          clEnumValN(NoStrictAlign, "arm-no-strict-align",
                     "Allow unaligned memory accesses"),
          clEnumValEnd));

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

ARMSubtarget::ARMSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, bool IsLittle,
                           const TargetOptions &Options)
  : ARMGenSubtargetInfo(TT, CPU, FS)
  , ARMProcFamily(Others)
  , ARMProcClass(None)
  , stackAlignment(4)
  , CPUString(CPU)
  , IsLittle(IsLittle)
  , TargetTriple(TT)
  , Options(Options)
  , TargetABI(ARM_ABI_UNKNOWN) {
  initializeEnvironment();
  resetSubtargetFeatures(CPU, FS);
}

void ARMSubtarget::initializeEnvironment() {
  HasV4TOps = false;
  HasV5TOps = false;
  HasV5TEOps = false;
  HasV6Ops = false;
  HasV6MOps = false;
  HasV6T2Ops = false;
  HasV7Ops = false;
  HasV8Ops = false;
  HasVFPv2 = false;
  HasVFPv3 = false;
  HasVFPv4 = false;
  HasFPARMv8 = false;
  HasNEON = false;
  MinSize = false;
  UseNEONForSinglePrecisionFP = false;
  UseMulOps = UseFusedMulOps;
  SlowFPVMLx = false;
  HasVMLxForwarding = false;
  SlowFPBrcc = false;
  InThumbMode = false;
  HasThumb2 = false;
  NoARM = false;
  PostRAScheduler = false;
  IsR9Reserved = ReserveR9;
  UseMovt = false;
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
  AllowsUnalignedMem = false;
  Thumb2DSP = false;
  UseNaClTrap = false;
  UnsafeFPMath = false;
}

void ARMSubtarget::resetSubtargetFeatures(const MachineFunction *MF) {
  AttributeSet FnAttrs = MF->getFunction()->getAttributes();
  Attribute CPUAttr = FnAttrs.getAttribute(AttributeSet::FunctionIndex,
                                           "target-cpu");
  Attribute FSAttr = FnAttrs.getAttribute(AttributeSet::FunctionIndex,
                                          "target-features");
  std::string CPU =
    !CPUAttr.hasAttribute(Attribute::None) ?CPUAttr.getValueAsString() : "";
  std::string FS =
    !FSAttr.hasAttribute(Attribute::None) ? FSAttr.getValueAsString() : "";
  if (!FS.empty()) {
    initializeEnvironment();
    resetSubtargetFeatures(CPU, FS);
  }

  MinSize =
      FnAttrs.hasAttribute(AttributeSet::FunctionIndex, Attribute::MinSize);
}

void ARMSubtarget::resetSubtargetFeatures(StringRef CPU, StringRef FS) {
  if (CPUString.empty()) {
    if (isTargetIOS() && TargetTriple.getArchName().endswith("v7s"))
      // Default to the Swift CPU when targeting armv7s/thumbv7s.
      CPUString = "swift";
    else
      CPUString = "generic";
  }

  // Insert the architecture feature derived from the target triple into the
  // feature string. This is important for setting features that are implied
  // based on the architecture version.
  std::string ArchFS = ARM_MC::ParseARMTriple(TargetTriple.getTriple(),
                                              CPUString);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS.str();
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

  if (TargetABI == ARM_ABI_UNKNOWN) {
    switch (TargetTriple.getEnvironment()) {
    case Triple::Android:
    case Triple::EABI:
    case Triple::EABIHF:
    case Triple::GNUEABI:
    case Triple::GNUEABIHF:
      TargetABI = ARM_ABI_AAPCS;
      break;
    default:
      if ((isTargetIOS() && isMClass()) ||
          (TargetTriple.isOSBinFormatMachO() &&
           TargetTriple.getOS() == Triple::UnknownOS))
        TargetABI = ARM_ABI_AAPCS;
      else
        TargetABI = ARM_ABI_APCS;
      break;
    }
  }

  // FIXME: this is invalid for WindowsCE
  if (isTargetWindows()) {
    TargetABI = ARM_ABI_AAPCS;
    NoARM = true;
  }

  if (isAAPCS_ABI())
    stackAlignment = 8;
  if (isTargetNaCl())
    stackAlignment = 16;

  UseMovt = hasV6T2Ops() && ArmUseMOVT;

  if (isTargetMachO()) {
    IsR9Reserved = ReserveR9 | !HasV6Ops;
    SupportsTailCall = !isTargetIOS() || !getTargetTriple().isOSVersionLT(5, 0);
  } else {
    IsR9Reserved = ReserveR9;
    SupportsTailCall = !isThumb1Only();
  }

  if (!isThumb() || hasThumb2())
    PostRAScheduler = true;

  switch (Align) {
    case DefaultAlign:
      // Assume pre-ARMv6 doesn't support unaligned accesses.
      //
      // ARMv6 may or may not support unaligned accesses depending on the
      // SCTLR.U bit, which is architecture-specific. We assume ARMv6
      // Darwin and NetBSD targets support unaligned accesses, and others don't.
      //
      // ARMv7 always has SCTLR.U set to 1, but it has a new SCTLR.A bit
      // which raises an alignment fault on unaligned accesses. Linux
      // defaults this bit to 0 and handles it as a system-wide (not
      // per-process) setting. It is therefore safe to assume that ARMv7+
      // Linux targets support unaligned accesses. The same goes for NaCl.
      //
      // The above behavior is consistent with GCC.
      AllowsUnalignedMem =
          (hasV7Ops() && (isTargetLinux() || isTargetNaCl() ||
                          isTargetNetBSD())) ||
          (hasV6Ops() && (isTargetMachO() || isTargetNetBSD()));
      // The one exception is cortex-m0, which despite being v6, does not
      // support unaligned accesses. Rather than make the above boolean
      // expression even more obtuse, just override the value here.
      if (isThumb1Only() && isMClass())
        AllowsUnalignedMem = false;
      break;
    case StrictAlign:
      AllowsUnalignedMem = false;
      break;
    case NoStrictAlign:
      AllowsUnalignedMem = true;
      break;
  }

  switch (IT) {
  case DefaultIT:
    RestrictIT = hasV8Ops() ? true : false;
    break;
  case RestrictedIT:
    RestrictIT = true;
    break;
  case NoRestrictedIT:
    RestrictIT = false;
    break;
  }

  // NEON f32 ops are non-IEEE 754 compliant. Darwin is ok with it by default.
  uint64_t Bits = getFeatureBits();
  if ((Bits & ARM::ProcA5 || Bits & ARM::ProcA8) && // Where this matters
      (Options.UnsafeFPMath || isTargetDarwin()))
    UseNEONForSinglePrecisionFP = true;
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol.
bool
ARMSubtarget::GVIsIndirectSymbol(const GlobalValue *GV,
                                 Reloc::Model RelocM) const {
  if (RelocM == Reloc::Static)
    return false;

  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->hasAvailableExternallyLinkage();
  if (GV->isDeclaration() && !GV->isMaterializable())
    isDecl = true;

  if (!isTargetMachO()) {
    // Extra load is needed for all externally visible.
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return false;
    return true;
  } else {
    if (RelocM == Reloc::PIC_) {
      // If this is a strong reference to a definition, it is definitely not
      // through a stub.
      if (!isDecl && !GV->isWeakForLinker())
        return false;

      // Unless we have a symbol with hidden visibility, we have to go through a
      // normal $non_lazy_ptr stub because this symbol might be resolved late.
      if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
        return true;

      // If symbol visibility is hidden, we have a stub for common symbol
      // references and external declarations.
      if (isDecl || GV->hasCommonLinkage())
        // Hidden $non_lazy_ptr reference.
        return true;

      return false;
    } else {
      // If this is a strong reference to a definition, it is definitely not
      // through a stub.
      if (!isDecl && !GV->isWeakForLinker())
        return false;

      // Unless we have a symbol with hidden visibility, we have to go through a
      // normal $non_lazy_ptr stub because this symbol might be resolved late.
      if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
        return true;
    }
  }

  return false;
}

unsigned ARMSubtarget::getMispredictionPenalty() const {
  return SchedModel->MispredictPenalty;
}

bool ARMSubtarget::hasSinCos() const {
  return getTargetTriple().getOS() == Triple::IOS &&
    !getTargetTriple().isOSVersionLT(7, 0);
}

// Enable the PostMachineScheduler if the target selects it instead of
// PostRAScheduler. Currently only available on the command line via
// -misched-postra.
bool ARMSubtarget::enablePostMachineScheduler() const {
  return PostRAScheduler;
}

bool ARMSubtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtargetInfo::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_NONE;
  return PostRAScheduler && OptLevel >= CodeGenOpt::Default;
}
