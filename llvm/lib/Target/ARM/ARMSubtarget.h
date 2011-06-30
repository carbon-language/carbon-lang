//=====---- ARMSubtarget.h - Define Subtarget for the ARM -----*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ARM specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSUBTARGET_H
#define ARMSUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/ADT/Triple.h"
#include <string>

namespace llvm {
class GlobalValue;

class ARMSubtarget : public TargetSubtarget {
protected:
  enum ARMArchEnum {
    V4, V4T, V5T, V5TE, V6, V6M, V6T2, V7A, V7M
  };

  enum ARMProcFamilyEnum {
    Others, CortexA8, CortexA9
  };

  enum ARMFPEnum {
    None, VFPv2, VFPv3, NEON
  };

  enum ThumbTypeEnum {
    Thumb1,
    Thumb2
  };

  /// ARMArchVersion - ARM architecture version: V4, V4T (base), V5T, V5TE,
  /// V6, V6T2, V7A, V7M.
  ARMArchEnum ARMArchVersion;

  /// ARMProcFamily - ARM processor family: Cortex-A8, Cortex-A9, and others.
  ARMProcFamilyEnum ARMProcFamily;

  /// ARMFPUType - Floating Point Unit type.
  ARMFPEnum ARMFPUType;

  /// UseNEONForSinglePrecisionFP - if the NEONFP attribute has been
  /// specified. Use the method useNEONForSinglePrecisionFP() to
  /// determine if NEON should actually be used.
  bool UseNEONForSinglePrecisionFP;

  /// SlowFPVMLx - If the VFP2 / NEON instructions are available, indicates
  /// whether the FP VML[AS] instructions are slow (if so, don't use them).
  bool SlowFPVMLx;

  /// HasVMLxForwarding - If true, NEON has special multiplier accumulator
  /// forwarding to allow mul + mla being issued back to back.
  bool HasVMLxForwarding;

  /// SlowFPBrcc - True if floating point compare + branch is slow.
  bool SlowFPBrcc;

  /// IsThumb - True if we are in thumb mode, false if in ARM mode.
  bool IsThumb;

  /// ThumbMode - Indicates supported Thumb version.
  ThumbTypeEnum ThumbMode;

  /// NoARM - True if subtarget does not support ARM mode execution.
  bool NoARM;

  /// PostRAScheduler - True if using post-register-allocation scheduler.
  bool PostRAScheduler;

  /// IsR9Reserved - True if R9 is a not available as general purpose register.
  bool IsR9Reserved;

  /// UseMovt - True if MOVT / MOVW pairs are used for materialization of 32-bit
  /// imms (including global addresses).
  bool UseMovt;

  /// HasFP16 - True if subtarget supports half-precision FP (We support VFP+HF
  /// only so far)
  bool HasFP16;

  /// HasD16 - True if subtarget is limited to 16 double precision
  /// FP registers for VFPv3.
  bool HasD16;

  /// HasHardwareDivide - True if subtarget supports [su]div
  bool HasHardwareDivide;

  /// HasT2ExtractPack - True if subtarget supports thumb2 extract/pack
  /// instructions.
  bool HasT2ExtractPack;

  /// HasDataBarrier - True if the subtarget supports DMB / DSB data barrier
  /// instructions.
  bool HasDataBarrier;

  /// Pref32BitThumb - If true, codegen would prefer 32-bit Thumb instructions
  /// over 16-bit ones.
  bool Pref32BitThumb;

  /// AvoidCPSRPartialUpdate - If true, codegen would avoid using instructions
  /// that partially update CPSR and add false dependency on the previous
  /// CPSR setting instruction.
  bool AvoidCPSRPartialUpdate;

  /// HasMPExtension - True if the subtarget supports Multiprocessing
  /// extension (ARMv7 only).
  bool HasMPExtension;

  /// FPOnlySP - If true, the floating point unit only supports single
  /// precision.
  bool FPOnlySP;

  /// AllowsUnalignedMem - If true, the subtarget allows unaligned memory
  /// accesses for some types.  For details, see
  /// ARMTargetLowering::allowsUnalignedMemoryAccesses().
  bool AllowsUnalignedMem;

  /// stackAlignment - The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned stackAlignment;

  /// CPUString - String name of used CPU.
  std::string CPUString;

  /// TargetTriple - What processor and OS we're targeting.
  Triple TargetTriple;

  /// Selected instruction itineraries (one entry per itinerary class.)
  InstrItineraryData InstrItins;

 public:
  enum {
    isELF, isDarwin
  } TargetType;

  enum {
    ARM_ABI_APCS,
    ARM_ABI_AAPCS // ARM EABI
  } TargetABI;

  /// This constructor initializes the data members to match that
  /// of the specified triple.
  ///
  ARMSubtarget(const std::string &TT, const std::string &CPU,
               const std::string &FS, bool isThumb);

  /// getMaxInlineSizeThreshold - Returns the maximum memset / memcpy size
  /// that still makes it profitable to inline the call.
  unsigned getMaxInlineSizeThreshold() const {
    // FIXME: For now, we don't lower memcpy's to loads / stores for Thumb1.
    // Change this once Thumb1 ldmia / stmia support is added.
    return isThumb1Only() ? 0 : 64;
  }
  /// ParseSubtargetFeatures - Parses features string setting specified
  /// subtarget options.  Definition of function is auto generated by tblgen.
  void ParseSubtargetFeatures(const std::string &FS, const std::string &CPU);

  void computeIssueWidth();

  bool hasV4TOps()  const { return ARMArchVersion >= V4T;  }
  bool hasV5TOps()  const { return ARMArchVersion >= V5T;  }
  bool hasV5TEOps() const { return ARMArchVersion >= V5TE; }
  bool hasV6Ops()   const { return ARMArchVersion >= V6;   }
  bool hasV6T2Ops() const { return ARMArchVersion >= V6T2; }
  bool hasV7Ops()   const { return ARMArchVersion >= V7A;  }

  bool isCortexA8() const { return ARMProcFamily == CortexA8; }
  bool isCortexA9() const { return ARMProcFamily == CortexA9; }

  bool hasARMOps() const { return !NoARM; }

  bool hasVFP2() const { return ARMFPUType >= VFPv2; }
  bool hasVFP3() const { return ARMFPUType >= VFPv3; }
  bool hasNEON() const { return ARMFPUType >= NEON;  }
  bool useNEONForSinglePrecisionFP() const {
    return hasNEON() && UseNEONForSinglePrecisionFP; }
  bool hasDivide() const { return HasHardwareDivide; }
  bool hasT2ExtractPack() const { return HasT2ExtractPack; }
  bool hasDataBarrier() const { return HasDataBarrier; }
  bool useFPVMLx() const { return !SlowFPVMLx; }
  bool hasVMLxForwarding() const { return HasVMLxForwarding; }
  bool isFPBrccSlow() const { return SlowFPBrcc; }
  bool isFPOnlySP() const { return FPOnlySP; }
  bool prefers32BitThumb() const { return Pref32BitThumb; }
  bool avoidCPSRPartialUpdate() const { return AvoidCPSRPartialUpdate; }
  bool hasMPExtension() const { return HasMPExtension; }

  bool hasFP16() const { return HasFP16; }
  bool hasD16() const { return HasD16; }

  const Triple &getTargetTriple() const { return TargetTriple; }

  bool isTargetDarwin() const { return TargetTriple.isOSDarwin(); }
  bool isTargetELF() const { return !isTargetDarwin(); }

  bool isAPCS_ABI() const { return TargetABI == ARM_ABI_APCS; }
  bool isAAPCS_ABI() const { return TargetABI == ARM_ABI_AAPCS; }

  bool isThumb() const { return IsThumb; }
  bool isThumb1Only() const { return IsThumb && (ThumbMode == Thumb1); }
  bool isThumb2() const { return IsThumb && (ThumbMode == Thumb2); }
  bool hasThumb2() const { return ThumbMode >= Thumb2; }

  bool isR9Reserved() const { return IsR9Reserved; }

  bool useMovt() const { return UseMovt && hasV6T2Ops(); }

  bool allowsUnalignedMem() const { return AllowsUnalignedMem; }

  const std::string & getCPUString() const { return CPUString; }

  unsigned getMispredictionPenalty() const;

  /// enablePostRAScheduler - True at 'More' optimization.
  bool enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                             TargetSubtarget::AntiDepBreakMode& Mode,
                             RegClassVector& CriticalPathRCs) const;

  /// getInstrItins - Return the instruction itineraies based on subtarget
  /// selection.
  const InstrItineraryData &getInstrItineraryData() const { return InstrItins; }

  /// getStackAlignment - Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return stackAlignment; }

  /// GVIsIndirectSymbol - true if the GV will be accessed via an indirect
  /// symbol.
  bool GVIsIndirectSymbol(const GlobalValue *GV, Reloc::Model RelocM) const;
};
} // End llvm namespace

#endif  // ARMSUBTARGET_H
