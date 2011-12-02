//===-- llvm/Target/TargetOptions.h - Target Options ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines command line option flags that are shared across various
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPTIONS_H
#define LLVM_TARGET_TARGETOPTIONS_H

#include <string>

namespace llvm {
  class MachineFunction;
  class StringRef;

  // Possible float ABI settings. Used with FloatABIType in TargetOptions.h.
  namespace FloatABI {
    enum ABIType {
      Default, // Target-specific (either soft of hard depending on triple, etc).
      Soft, // Soft float.
      Hard  // Hard float.
    };
  }

  /// StrongPHIElim - This flag enables more aggressive PHI elimination
  /// wth earlier copy coalescing.
  extern bool StrongPHIElim;

  class TargetOptions {
  public:
    TargetOptions()
        : PrintMachineCode(false), NoFramePointerElim(false),
          NoFramePointerElimNonLeaf(false), LessPreciseFPMADOption(false),
          NoExcessFPPrecision(false), UnsafeFPMath(false), NoInfsFPMath(false),
          NoNaNsFPMath(false), HonorSignDependentRoundingFPMathOption(false),
          UseSoftFloat(false), NoZerosInBSS(false), JITExceptionHandling(false),
          JITEmitDebugInfo(false), JITEmitDebugInfoToDisk(false),
          GuaranteedTailCallOpt(false), StackAlignmentOverride(0),
          RealignStack(true), DisableJumpTables(false), EnableFastISel(false),
          EnableSegmentedStacks(false), TrapFuncName(""),
          FloatABIType(FloatABI::Default)
    {}

    /// PrintMachineCode - This flag is enabled when the -print-machineinstrs
    /// option is specified on the command line, and should enable debugging
    /// output from the code generator.
    unsigned PrintMachineCode : 1;

    /// NoFramePointerElim - This flag is enabled when the -disable-fp-elim is
    /// specified on the command line.  If the target supports the frame pointer
    /// elimination optimization, this option should disable it.
    unsigned NoFramePointerElim : 1;

    /// NoFramePointerElimNonLeaf - This flag is enabled when the
    /// -disable-non-leaf-fp-elim is specified on the command line. If the
    /// target supports the frame pointer elimination optimization, this option
    /// should disable it for non-leaf functions.
    unsigned NoFramePointerElimNonLeaf : 1;

    /// DisableFramePointerElim - This returns true if frame pointer elimination
    /// optimization should be disabled for the given machine function.
    bool DisableFramePointerElim(const MachineFunction &MF) const;

    /// LessPreciseFPMAD - This flag is enabled when the
    /// -enable-fp-mad is specified on the command line.  When this flag is off
    /// (the default), the code generator is not allowed to generate mad
    /// (multiply add) if the result is "less precise" than doing those
    /// operations individually.
    unsigned LessPreciseFPMADOption : 1;
    bool LessPreciseFPMAD() const;

    /// NoExcessFPPrecision - This flag is enabled when the
    /// -disable-excess-fp-precision flag is specified on the command line.
    /// When this flag is off (the default), the code generator is allowed to
    /// produce results that are "more precise" than IEEE allows.  This includes
    /// use of FMA-like operations and use of the X86 FP registers without
    /// rounding all over the place.
    unsigned NoExcessFPPrecision : 1;

    /// UnsafeFPMath - This flag is enabled when the
    /// -enable-unsafe-fp-math flag is specified on the command line.  When
    /// this flag is off (the default), the code generator is not allowed to
    /// produce results that are "less precise" than IEEE allows.  This includes
    /// use of X86 instructions like FSIN and FCOS instead of libcalls.
    /// UnsafeFPMath implies LessPreciseFPMAD.
    unsigned UnsafeFPMath : 1;

    /// NoInfsFPMath - This flag is enabled when the
    /// -enable-no-infs-fp-math flag is specified on the command line. When
    /// this flag is off (the default), the code generator is not allowed to
    /// assume the FP arithmetic arguments and results are never +-Infs.
    unsigned NoInfsFPMath : 1;

    /// NoNaNsFPMath - This flag is enabled when the
    /// -enable-no-nans-fp-math flag is specified on the command line. When
    /// this flag is off (the default), the code generator is not allowed to
    /// assume the FP arithmetic arguments and results are never NaNs.
    unsigned NoNaNsFPMath : 1;

    /// HonorSignDependentRoundingFPMath - This returns true when the
    /// -enable-sign-dependent-rounding-fp-math is specified.  If this returns
    /// false (the default), the code generator is allowed to assume that the
    /// rounding behavior is the default (round-to-zero for all floating point
    /// to integer conversions, and round-to-nearest for all other arithmetic
    /// truncations).  If this is enabled (set to true), the code generator must
    /// assume that the rounding mode may dynamically change.
    unsigned HonorSignDependentRoundingFPMathOption : 1;
    bool HonorSignDependentRoundingFPMath() const;
  
    /// UseSoftFloat - This flag is enabled when the -soft-float flag is
    /// specified on the command line.  When this flag is on, the code generator
    /// will generate libcalls to the software floating point library instead of
    /// target FP instructions.
    unsigned UseSoftFloat : 1;

    /// NoZerosInBSS - By default some codegens place zero-initialized data to
    /// .bss section. This flag disables such behaviour (necessary, e.g. for
    /// crt*.o compiling).
    unsigned NoZerosInBSS : 1;

    /// JITExceptionHandling - This flag indicates that the JIT should emit
    /// exception handling information.
    unsigned JITExceptionHandling : 1;

    /// JITEmitDebugInfo - This flag indicates that the JIT should try to emit
    /// debug information and notify a debugger about it.
    unsigned JITEmitDebugInfo : 1;

    /// JITEmitDebugInfoToDisk - This flag indicates that the JIT should write
    /// the object files generated by the JITEmitDebugInfo flag to disk.  This
    /// flag is hidden and is only for debugging the debug info.
    unsigned JITEmitDebugInfoToDisk : 1;

    /// GuaranteedTailCallOpt - This flag is enabled when -tailcallopt is
    /// specified on the commandline. When the flag is on, participating targets
    /// will perform tail call optimization on all calls which use the fastcc
    /// calling convention and which satisfy certain target-independent
    /// criteria (being at the end of a function, having the same return type
    /// as their parent function, etc.), using an alternate ABI if necessary.
    unsigned GuaranteedTailCallOpt : 1;

    /// StackAlignmentOverride - Override default stack alignment for target.
    unsigned StackAlignmentOverride;

    /// RealignStack - This flag indicates whether the stack should be
    /// automatically realigned, if needed.
    unsigned RealignStack : 1;

    /// DisableJumpTables - This flag indicates jump tables should not be 
    /// generated.
    unsigned DisableJumpTables : 1;

    /// EnableFastISel - This flag enables fast-path instruction selection
    /// which trades away generated code quality in favor of reducing
    /// compile time.
    unsigned EnableFastISel : 1;
  
    unsigned EnableSegmentedStacks : 1;

    /// getTrapFunctionName - If this returns a non-empty string, this means
    /// isel should lower Intrinsic::trap to a call to the specified function
    /// name instead of an ISD::TRAP node.
    std::string TrapFuncName;
    StringRef getTrapFunctionName() const;

    /// FloatABIType - This setting is set by -float-abi=xxx option is specfied
    /// on the command line. This setting may either be Default, Soft, or Hard.
    /// Default selects the target's default behavior. Soft selects the ABI for
    /// UseSoftFloat, but does not indicate that FP hardware may not be used.
    /// Such a combination is unfortunately popular (e.g. arm-apple-darwin).
    /// Hard presumes that the normal FP ABI is used.
    FloatABI::ABIType FloatABIType;
  };
} // End llvm namespace

#endif
