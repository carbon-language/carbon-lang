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
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ARMGenSubtargetInfo.inc"

using namespace llvm;

static cl::opt<bool>
ReserveR9("arm-reserve-r9", cl::Hidden,
          cl::desc("Reserve R9, making it unavailable as GPR"));

static cl::opt<bool>
DarwinUseMOVT("arm-darwin-use-movt", cl::init(true), cl::Hidden);

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

ARMSubtarget::ARMSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, const TargetOptions &Options)
  : ARMGenSubtargetInfo(TT, CPU, FS)
  , ARMProcFamily(Others)
  , stackAlignment(4)
  , CPUString(CPU)
  , TargetTriple(TT)
  , Options(Options)
  , TargetABI(ARM_ABI_APCS) {
  initializeEnvironment();
  resetSubtargetFeatures(CPU, FS);
}

void ARMSubtarget::initializeEnvironment() {
  HasV4TOps = false;
  HasV5TOps = false;
  HasV5TEOps = false;
  HasV6Ops = false;
  HasV6T2Ops = false;
  HasV7Ops = false;
  HasV8Ops = false;
  HasVFPv2 = false;
  HasVFPv3 = false;
  HasVFPv4 = false;
  HasV8FP = false;
  HasNEON = false;
  UseNEONForSinglePrecisionFP = false;
  UseMulOps = UseFusedMulOps;
  SlowFPVMLx = false;
  HasVMLxForwarding = false;
  SlowFPBrcc = false;
  InThumbMode = false;
  HasThumb2 = false;
  IsMClass = false;
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
  FPOnlySP = false;
  HasPerfMon = false;
  HasTrustZone = false;
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
}

void ARMSubtarget::resetSubtargetFeatures(StringRef CPU, StringRef FS) {
  if (CPUString.empty())
    CPUString = "generic";

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

  // Thumb2 implies at least V6T2. FIXME: Fix tests to explicitly specify a
  // ARM version or CPU and then remove this.
  if (!HasV6T2Ops && hasThumb2())
    HasV4TOps = HasV5TOps = HasV5TEOps = HasV6Ops = HasV6T2Ops = true;

  // Keep a pointer to static instruction cost data for the specified CPU.
  SchedModel = getSchedModelForCPU(CPUString);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  if ((TargetTriple.getTriple().find("eabi") != std::string::npos) ||
      (isTargetIOS() && isMClass()))
    // FIXME: We might want to separate AAPCS and EABI. Some systems, e.g.
    // Darwin-EABI conforms to AACPS but not the rest of EABI.
    TargetABI = ARM_ABI_AAPCS;

  if (isAAPCS_ABI())
    stackAlignment = 8;

  if (!isTargetIOS())
    UseMovt = hasV6T2Ops();
  else {
    IsR9Reserved = ReserveR9 | !HasV6Ops;
    UseMovt = DarwinUseMOVT && hasV6T2Ops();
    SupportsTailCall = !getTargetTriple().isOSVersionLT(5, 0);
  }

  if (!isThumb() || hasThumb2())
    PostRAScheduler = true;

  switch (Align) {
    case DefaultAlign:
      // Assume pre-ARMv6 doesn't support unaligned accesses.
      //
      // ARMv6 may or may not support unaligned accesses depending on the
      // SCTLR.U bit, which is architecture-specific. We assume ARMv6
      // Darwin targets support unaligned accesses, and others don't.
      //
      // ARMv7 always has SCTLR.U set to 1, but it has a new SCTLR.A bit
      // which raises an alignment fault on unaligned accesses. Linux
      // defaults this bit to 0 and handles it as a system-wide (not
      // per-process) setting. It is therefore safe to assume that ARMv7+
      // Linux targets support unaligned accesses. The same goes for NaCl.
      //
      // The above behavior is consistent with GCC.
      AllowsUnalignedMem = (
          (hasV7Ops() && (isTargetLinux() || isTargetNaCl())) ||
          (hasV6Ops() && isTargetDarwin()));
      break;
    case StrictAlign:
      AllowsUnalignedMem = false;
      break;
    case NoStrictAlign:
      AllowsUnalignedMem = true;
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

  if (!isTargetDarwin()) {
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

bool ARMSubtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtargetInfo::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(&ARM::GPRRegClass);
  return PostRAScheduler && OptLevel >= CodeGenOpt::Default;
}
