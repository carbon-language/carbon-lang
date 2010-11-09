//===-- ARMSubtarget.cpp - ARM Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "ARMSubtarget.h"
#include "ARMGenSubtarget.inc"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

static cl::opt<bool>
ReserveR9("arm-reserve-r9", cl::Hidden,
          cl::desc("Reserve R9, making it unavailable as GPR"));

static cl::opt<bool>
UseMOVT("arm-use-movt",
        cl::init(true), cl::Hidden);

static cl::opt<bool>
StrictAlign("arm-strict-align", cl::Hidden,
            cl::desc("Disallow all unaligned memory accesses"));

ARMSubtarget::ARMSubtarget(const std::string &TT, const std::string &FS,
                           bool isT)
  : ARMArchVersion(V4)
  , ARMProcFamily(Others)
  , ARMFPUType(None)
  , UseNEONForSinglePrecisionFP(false)
  , SlowVMLx(false)
  , SlowFPBrcc(false)
  , IsThumb(isT)
  , ThumbMode(Thumb1)
  , NoARM(false)
  , PostRAScheduler(false)
  , IsR9Reserved(ReserveR9)
  , UseMovt(UseMOVT)
  , HasFP16(false)
  , HasD16(false)
  , HasHardwareDivide(false)
  , HasT2ExtractPack(false)
  , HasDataBarrier(false)
  , Pref32BitThumb(false)
  , HasMPExtension(false)
  , FPOnlySP(false)
  , AllowsUnalignedMem(false)
  , stackAlignment(4)
  , CPUString("generic")
  , TargetType(isELF) // Default to ELF unless otherwise specified.
  , TargetABI(ARM_ABI_APCS) {
  // Default to soft float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Soft;

  // Determine default and user specified characteristics

  // When no arch is specified either by CPU or by attributes, make the default
  // ARMv4T.
  const char *ARMArchFeature = "";
  if (CPUString == "generic" && (FS.empty() || FS == "generic")) {
    ARMArchVersion = V4T;
    ARMArchFeature = ",+v4t";
  }

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  unsigned Len = TT.length();
  unsigned Idx = 0;

  if (Len >= 5 && TT.substr(0, 4) == "armv")
    Idx = 4;
  else if (Len >= 6 && TT.substr(0, 5) == "thumb") {
    IsThumb = true;
    if (Len >= 7 && TT[5] == 'v')
      Idx = 6;
  }
  if (Idx) {
    unsigned SubVer = TT[Idx];
    if (SubVer >= '7' && SubVer <= '9') {
      ARMArchVersion = V7A;
      ARMArchFeature = ",+v7a";
      if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        ARMArchVersion = V7M;
        ARMArchFeature = ",+v7m";
      }
    } else if (SubVer == '6') {
      ARMArchVersion = V6;
      ARMArchFeature = ",+v6";
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == '2') {
        ARMArchVersion = V6T2;
        ARMArchFeature = ",+v6t2";
      }
    } else if (SubVer == '5') {
      ARMArchVersion = V5T;
      ARMArchFeature = ",+v5t";
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == 'e') {
        ARMArchVersion = V5TE;
        ARMArchFeature = ",+v5te";
      }
    } else if (SubVer == '4') {
      if (Len >= Idx+2 && TT[Idx+1] == 't') {
        ARMArchVersion = V4T;
        ARMArchFeature = ",+v4t";
      } else {
        ARMArchVersion = V4;
        ARMArchFeature = "";
      }
    }
  }

  if (Len >= 10) {
    if (TT.find("-darwin") != std::string::npos)
      // arm-darwin
      TargetType = isDarwin;
  }

  if (TT.find("eabi") != std::string::npos)
    TargetABI = ARM_ABI_AAPCS;

  // Parse features string.  If the first entry in FS (the CPU) is missing,
  // insert the architecture feature derived from the target triple.  This is
  // important for setting features that are implied based on the architecture
  // version.
  std::string FSWithArch;
  if (FS.empty())
    FSWithArch = std::string(ARMArchFeature);
  else if (FS.find(',') == 0)
    FSWithArch = std::string(ARMArchFeature) + FS;
  else
    FSWithArch = FS;
  CPUString = ParseSubtargetFeatures(FSWithArch, CPUString);

  // Thumb2 implies at least V6T2.
  if (ARMArchVersion >= V6T2)
    ThumbMode = Thumb2;
  else if (ThumbMode >= Thumb2)
    ARMArchVersion = V6T2;

  if (isAAPCS_ABI())
    stackAlignment = 8;

  if (isTargetDarwin())
    IsR9Reserved = ReserveR9 | (ARMArchVersion < V6);

  if (!isThumb() || hasThumb2())
    PostRAScheduler = true;

  // v6+ may or may not support unaligned mem access depending on the system
  // configuration.
  if (!StrictAlign && hasV6Ops() && isTargetDarwin())
    AllowsUnalignedMem = true;
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol.
bool
ARMSubtarget::GVIsIndirectSymbol(const GlobalValue *GV,
                                 Reloc::Model RelocM) const {
  if (RelocM == Reloc::Static)
    return false;

  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->isDeclaration() && !GV->isMaterializable();

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
  // If we have a reasonable estimate of the pipeline depth, then we can
  // estimate the penalty of a misprediction based on that.
  if (isCortexA8())
    return 13;
  else if (isCortexA9())
    return 8;
  
  // Otherwise, just return a sensible default.
  return 10;
}

bool ARMSubtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtarget::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtarget::ANTIDEP_CRITICAL;
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(&ARM::GPRRegClass);
  return PostRAScheduler && OptLevel >= CodeGenOpt::Default;
}
