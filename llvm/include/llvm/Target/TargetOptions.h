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

namespace llvm {
  // Possible float ABI settings. Used with FloatABIType in TargetOptions.h.
  namespace FloatABI {
    enum ABIType {
      Default, // Target-specific (either soft of hard depending on triple, etc).
      Soft, // Soft float.
      Hard  // Hard float.
    };
  }
  
  /// PrintMachineCode - This flag is enabled when the -print-machineinstrs
  /// option is specified on the command line, and should enable debugging
  /// output from the code generator.
  extern bool PrintMachineCode;

  /// NoFramePointerElim - This flag is enabled when the -disable-fp-elim is
  /// specified on the command line.  If the target supports the frame pointer
  /// elimination optimization, this option should disable it.
  extern bool NoFramePointerElim;

  /// LessPreciseFPMAD - This flag is enabled when the
  /// -enable-fp-mad is specified on the command line.  When this flag is off
  /// (the default), the code generator is not allowed to generate mad
  /// (multiply add) if the result is "less precise" than doing those operations
  /// individually.
  extern bool LessPreciseFPMADOption;
  extern bool LessPreciseFPMAD();

  /// NoExcessFPPrecision - This flag is enabled when the
  /// -disable-excess-fp-precision flag is specified on the command line.  When
  /// this flag is off (the default), the code generator is allowed to produce
  /// results that are "more precise" than IEEE allows.  This includes use of
  /// FMA-like operations and use of the X86 FP registers without rounding all
  /// over the place.
  extern bool NoExcessFPPrecision;

  /// UnsafeFPMath - This flag is enabled when the
  /// -enable-unsafe-fp-math flag is specified on the command line.  When
  /// this flag is off (the default), the code generator is not allowed to
  /// produce results that are "less precise" than IEEE allows.  This includes
  /// use of X86 instructions like FSIN and FCOS instead of libcalls.
  /// UnsafeFPMath implies FiniteOnlyFPMath and LessPreciseFPMAD.
  extern bool UnsafeFPMath;

  /// FiniteOnlyFPMath - This returns true when the -enable-finite-only-fp-math
  /// option is specified on the command line. If this returns false (default),
  /// the code generator is not allowed to assume that FP arithmetic arguments
  /// and results are never NaNs or +-Infs.
  extern bool FiniteOnlyFPMathOption;
  extern bool FiniteOnlyFPMath();
  
  /// HonorSignDependentRoundingFPMath - This returns true when the
  /// -enable-sign-dependent-rounding-fp-math is specified.  If this returns
  /// false (the default), the code generator is allowed to assume that the
  /// rounding behavior is the default (round-to-zero for all floating point to
  /// integer conversions, and round-to-nearest for all other arithmetic
  /// truncations).  If this is enabled (set to true), the code generator must
  /// assume that the rounding mode may dynamically change.
  extern bool HonorSignDependentRoundingFPMathOption;
  extern bool HonorSignDependentRoundingFPMath();
  
  /// UseSoftFloat - This flag is enabled when the -soft-float flag is specified
  /// on the command line.  When this flag is on, the code generator will
  /// generate libcalls to the software floating point library instead of
  /// target FP instructions.
  extern bool UseSoftFloat;

  /// FloatABIType - This setting is set by -float-abi=xxx option is specfied
  /// on the command line. This setting may either be Default, Soft, or Hard.
  /// Default selects the target's default behavior. Soft selects the ABI for
  /// UseSoftFloat, but does not inidcate that FP hardware may not be used.
  /// Such a combination is unfortunately popular (e.g. arm-apple-darwin).
  /// Hard presumes that the normal FP ABI is used.
  extern FloatABI::ABIType FloatABIType;

  /// NoZerosInBSS - By default some codegens place zero-initialized data to
  /// .bss section. This flag disables such behaviour (necessary, e.g. for
  /// crt*.o compiling).
  extern bool NoZerosInBSS;

  /// DwarfExceptionHandling - This flag indicates that Dwarf exception
  /// information should be emitted.
  extern bool DwarfExceptionHandling;

  /// SjLjExceptionHandling - This flag indicates that SJLJ exception
  /// information should be emitted.
  extern bool SjLjExceptionHandling;

  /// JITEmitDebugInfo - This flag indicates that the JIT should try to emit
  /// debug information and notify a debugger about it.
  extern bool JITEmitDebugInfo;

  /// JITEmitDebugInfoToDisk - This flag indicates that the JIT should write
  /// the object files generated by the JITEmitDebugInfo flag to disk.  This
  /// flag is hidden and is only for debugging the debug info.
  extern bool JITEmitDebugInfoToDisk;

  /// UnwindTablesMandatory - This flag indicates that unwind tables should
  /// be emitted for all functions.
  extern bool UnwindTablesMandatory;

  /// GuaranteedTailCallOpt - This flag is enabled when -tailcallopt is
  /// specified on the commandline. When the flag is on, participating targets
  /// will perform tail call optimization on all calls which use the fastcc
  /// calling convention and which satisfy certain target-independent
  /// criteria (being at the end of a function, having the same return type
  /// as their parent function, etc.), using an alternate ABI if necessary.
  extern bool GuaranteedTailCallOpt;

  /// StackAlignment - Override default stack alignment for target.
  extern unsigned StackAlignment;

  /// RealignStack - This flag indicates, whether stack should be automatically
  /// realigned, if needed.
  extern bool RealignStack;

  /// DisableJumpTables - This flag indicates jump tables should not be 
  /// generated.
  extern bool DisableJumpTables;

  /// EnableFastISel - This flag enables fast-path instruction selection
  /// which trades away generated code quality in favor of reducing
  /// compile time.
  extern bool EnableFastISel;
  
  /// StrongPHIElim - This flag enables more aggressive PHI elimination
  /// wth earlier copy coalescing.
  extern bool StrongPHIElim;

  /// DisableScheduling - This flag disables instruction scheduling. In
  /// particular, it assigns an ordering to the SDNodes, which the scheduler
  /// uses instead of its normal heuristics to perform scheduling.
  extern bool DisableScheduling;

} // End llvm namespace

#endif
