//===--- CodeGenOptions.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeGenOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_CODEGENOPTIONS_H
#define LLVM_CLANG_FRONTEND_CODEGENOPTIONS_H

#include <string>
#include <vector>

namespace clang {

/// CodeGenOptions - Track various options which control how the code
/// is optimized and passed to the backend.
class CodeGenOptions {
public:
  enum InliningMethod {
    NoInlining,         // Perform no inlining whatsoever.
    NormalInlining,     // Use the standard function inlining pass.
    OnlyAlwaysInlining  // Only run the always inlining pass.
  };

  enum ObjCDispatchMethodKind {
    Legacy = 0,
    NonLegacy = 1,
    Mixed = 2
  };

  unsigned AsmVerbose        : 1; /// -dA, -fverbose-asm.
  unsigned ObjCAutoRefCountExceptions : 1; /// Whether ARC should be EH-safe.
  unsigned CXAAtExit         : 1; /// Use __cxa_atexit for calling destructors.
  unsigned CXXCtorDtorAliases: 1; /// Emit complete ctors/dtors as linker
                                  /// aliases to base ctors when possible.
  unsigned DataSections      : 1; /// Set when -fdata-sections is enabled
  unsigned DebugInfo         : 1; /// Should generate debug info (-g).
  unsigned LimitDebugInfo    : 1; /// Limit generated debug info to reduce size.
  unsigned DisableFPElim     : 1; /// Set when -fomit-frame-pointer is enabled.
  unsigned DisableLLVMOpts   : 1; /// Don't run any optimizations, for use in
                                  /// getting .bc files that correspond to the
                                  /// internal state before optimizations are
                                  /// done.
  unsigned DisableRedZone    : 1; /// Set when -mno-red-zone is enabled.
  unsigned EmitDeclMetadata  : 1; /// Emit special metadata indicating what
                                  /// Decl* various IR entities came from.  Only
                                  /// useful when running CodeGen as a
                                  /// subroutine.
  unsigned EmitGcovArcs      : 1; /// Emit coverage data files, aka. GCDA.
  unsigned EmitGcovNotes     : 1; /// Emit coverage "notes" files, aka GCNO.
  unsigned ForbidGuardVariables : 1; /// Issue errors if C++ guard variables
                                  /// are required
  unsigned FunctionSections  : 1; /// Set when -ffunction-sections is enabled
  unsigned HiddenWeakTemplateVTables : 1; /// Emit weak vtables and RTTI for
                                  /// template classes with hidden visibility
  unsigned HiddenWeakVTables : 1; /// Emit weak vtables, RTTI, and thunks with
                                  /// hidden visibility.
  unsigned InstrumentFunctions : 1; /// Set when -finstrument-functions is
                                    /// enabled.
  unsigned InstrumentForProfiling : 1; /// Set when -pg is enabled
  unsigned LessPreciseFPMAD  : 1; /// Enable less precise MAD instructions to be
                                  /// generated.
  unsigned MergeAllConstants : 1; /// Merge identical constants.
  unsigned NoCommon          : 1; /// Set when -fno-common or C++ is enabled.
  unsigned NoDwarf2CFIAsm    : 1; /// Set when -fno-dwarf2-cfi-asm is enabled.
  unsigned NoExecStack       : 1; /// Set when -Wa,--noexecstack is enabled.
  unsigned NoImplicitFloat   : 1; /// Set when -mno-implicit-float is enabled.
  unsigned NoInfsFPMath      : 1; /// Assume FP arguments, results not +-Inf.
  unsigned NoNaNsFPMath      : 1; /// Assume FP arguments, results not NaN.
  unsigned NoZeroInitializedInBSS : 1; /// -fno-zero-initialized-in-bss
  unsigned ObjCDispatchMethod : 2; /// Method of Objective-C dispatch to use.
  unsigned OmitLeafFramePointer : 1; /// Set when -momit-leaf-frame-pointer is
                                     /// enabled.
  unsigned OptimizationLevel : 3; /// The -O[0-4] option specified.
  unsigned OptimizeSize      : 2; /// If -Os (==1) or -Oz (==2) is specified.
  unsigned RelaxAll          : 1; /// Relax all machine code instructions.
  unsigned RelaxedAliasing   : 1; /// Set when -fno-strict-aliasing is enabled.
  unsigned SaveTempLabels    : 1; /// Save temporary labels.
  unsigned SimplifyLibCalls  : 1; /// Set when -fbuiltin is enabled.
  unsigned SoftFloat         : 1; /// -soft-float.
  unsigned TimePasses        : 1; /// Set when -ftime-report is enabled.
  unsigned UnitAtATime       : 1; /// Unused. For mirroring GCC optimization
                                  /// selection.
  unsigned UnrollLoops       : 1; /// Control whether loops are unrolled.
  unsigned UnsafeFPMath      : 1; /// Allow unsafe floating point optzns.
  unsigned UnwindTables      : 1; /// Emit unwind tables.
  unsigned VerifyModule      : 1; /// Control whether the module should be run
                                  /// through the LLVM Verifier.

  /// The code model to use (-mcmodel).
  std::string CodeModel;

  /// The filename with path we use for coverage files. The extension will be
  /// replaced.
  std::string CoverageFile;

  /// Enable additional debugging information.
  std::string DebugPass;

  /// The string to embed in the debug information for the compile unit, if
  /// non-empty.
  std::string DwarfDebugFlags;

  /// The ABI to use for passing floating point arguments.
  std::string FloatABI;

  /// The float precision limit to use, if non-empty.
  std::string LimitFloatPrecision;

  /// The kind of inlining to perform.
  InliningMethod Inlining;

  /// The user provided name for the "main file", if non-empty. This is useful
  /// in situations where the input file name does not match the original input
  /// file, for example with -save-temps.
  std::string MainFileName;

  /// The name of the relocation model to use.
  std::string RelocationModel;

  /// A list of command-line options to forward to the LLVM backend.
  std::vector<std::string> BackendOptions;

  /// The user specified number of registers to be used for integral arguments,
  /// or 0 if unspecified.
  unsigned NumRegisterParameters;

public:
  CodeGenOptions() {
    AsmVerbose = 0;
    ObjCAutoRefCountExceptions = 0;
    CXAAtExit = 1;
    CXXCtorDtorAliases = 0;
    DataSections = 0;
    DebugInfo = 0;
    LimitDebugInfo = 0;
    DisableFPElim = 0;
    DisableLLVMOpts = 0;
    DisableRedZone = 0;
    EmitDeclMetadata = 0;
    EmitGcovArcs = 0;
    EmitGcovNotes = 0;
    ForbidGuardVariables = 0;
    FunctionSections = 0;
    HiddenWeakTemplateVTables = 0;
    HiddenWeakVTables = 0;
    InstrumentFunctions = 0;
    InstrumentForProfiling = 0;
    LessPreciseFPMAD = 0;
    MergeAllConstants = 1;
    NoCommon = 0;
    NoDwarf2CFIAsm = 0;
    NoImplicitFloat = 0;
    NoInfsFPMath = 0;
    NoNaNsFPMath = 0;
    NoZeroInitializedInBSS = 0;
    NumRegisterParameters = 0;
    ObjCDispatchMethod = Legacy;
    OmitLeafFramePointer = 0;
    OptimizationLevel = 0;
    OptimizeSize = 0;
    RelaxAll = 0;
    RelaxedAliasing = 0;
    SaveTempLabels = 0;
    SimplifyLibCalls = 1;
    SoftFloat = 0;
    TimePasses = 0;
    UnitAtATime = 1;
    UnrollLoops = 0;
    UnsafeFPMath = 0;
    UnwindTables = 0;
    VerifyModule = 1;

    Inlining = NoInlining;
    RelocationModel = "pic";
  }

  ObjCDispatchMethodKind getObjCDispatchMethod() const {
    return ObjCDispatchMethodKind(ObjCDispatchMethod);
  }
};

}  // end namespace clang

#endif
