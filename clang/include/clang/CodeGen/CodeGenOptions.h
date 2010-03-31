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

#ifndef LLVM_CLANG_CODEGEN_CODEGENOPTIONS_H
#define LLVM_CLANG_CODEGEN_CODEGENOPTIONS_H

#include <string>

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

  unsigned AsmVerbose        : 1; /// -dA, -fverbose-asm.
  unsigned CXAAtExit         : 1; /// Use __cxa_atexit for calling destructors.
  unsigned CXXCtorDtorAliases: 1; /// Emit complete ctors/dtors as linker
                                  /// aliases to base ctors when possible.
  unsigned DebugInfo         : 1; /// Should generate deubg info (-g).
  unsigned DisableFPElim     : 1; /// Set when -fomit-frame-pointer is enabled.
  unsigned DisableLLVMOpts   : 1; /// Don't run any optimizations, for use in
                                  /// getting .bc files that correspond to the
                                  /// internal state before optimizations are
                                  /// done.
  unsigned DisableRedZone    : 1; /// Set when -mno-red-zone is enabled.
  unsigned MergeAllConstants : 1; /// Merge identical constants.
  unsigned NoCommon          : 1; /// Set when -fno-common or C++ is enabled.
  unsigned NoImplicitFloat   : 1; /// Set when -mno-implicit-float is enabled.
  unsigned NoZeroInitializedInBSS : 1; /// -fno-zero-initialized-in-bss
  unsigned ObjCLegacyDispatch: 1; /// Use legacy Objective-C dispatch, even with
                                  /// 2.0 runtime.
  unsigned OptimizationLevel : 3; /// The -O[0-4] option specified.
  unsigned OptimizeSize      : 1; /// If -Os is specified.
  unsigned SoftFloat         : 1; /// -soft-float.
  unsigned TimePasses        : 1; /// Set when -ftime-report is enabled.
  unsigned UnitAtATime       : 1; /// Unused. For mirroring GCC optimization
                                  /// selection.
  unsigned UnrollLoops       : 1; /// Control whether loops are unrolled.
  unsigned UnwindTables      : 1; /// Emit unwind tables.
  unsigned VerifyModule      : 1; /// Control whether the module should be run
                                  /// through the LLVM Verifier.

  /// The code model to use (-mcmodel).
  std::string CodeModel;

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

public:
  CodeGenOptions() {
    AsmVerbose = 0;
    CXAAtExit = 1;
    CXXCtorDtorAliases = 0;
    DebugInfo = 0;
    DisableFPElim = 0;
    DisableLLVMOpts = 0;
    DisableRedZone = 0;
    MergeAllConstants = 1;
    NoCommon = 0;
    NoImplicitFloat = 0;
    NoZeroInitializedInBSS = 0;
    ObjCLegacyDispatch = 0;
    OptimizationLevel = 0;
    OptimizeSize = 0;
    SoftFloat = 0;
    TimePasses = 0;
    UnitAtATime = 1;
    UnrollLoops = 0;
    UnwindTables = 0;
    VerifyModule = 1;

    Inlining = NoInlining;
    RelocationModel = "pic";
  }
};

}  // end namespace clang

#endif
