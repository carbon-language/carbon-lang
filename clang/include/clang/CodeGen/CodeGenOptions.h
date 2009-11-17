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

  unsigned DebugInfo         : 1; /// Should generate deubg info (-g).
  unsigned DisableLLVMOpts   : 1; /// Don't run any optimizations, for use in
                                  /// getting .bc files that correspond to the
                                  /// internal state before optimizations are
                                  /// done.
  unsigned DisableRedZone    : 1; /// Set when -mno-red-zone is enabled.
  unsigned MergeAllConstants : 1; /// Merge identical constants.
  unsigned NoCommon          : 1; /// Set when -fno-common or C++ is enabled.
  unsigned NoImplicitFloat   : 1; /// Set when -mno-implicit-float is enabled.
  unsigned OptimizationLevel : 3; /// The -O[0-4] option specified.
  unsigned OptimizeSize      : 1; /// If -Os is specified.
  unsigned SimplifyLibCalls  : 1; /// Should standard library calls be treated
                                  /// specially.
  unsigned TimePasses        : 1; /// Set when -ftime-report is enabled.
  unsigned UnitAtATime       : 1; /// Unused. For mirroring GCC optimization
                                  /// selection.
  unsigned UnrollLoops       : 1; /// Control whether loops are unrolled.
  unsigned VerifyModule      : 1; /// Control whether the module should be run
                                  /// through the LLVM Verifier.

  /// Inlining - The kind of inlining to perform.
  InliningMethod Inlining;

public:
  CodeGenOptions() {
    OptimizationLevel = 0;
    OptimizeSize = 0;
    DebugInfo = 0;
    UnitAtATime = 1;
    SimplifyLibCalls = UnrollLoops = 0;
    VerifyModule = 1;
    TimePasses = 0;
    NoCommon = 0;
    Inlining = NoInlining;
    DisableRedZone = 0;
    NoImplicitFloat = 0;
    MergeAllConstants = 1;
    DisableLLVMOpts = 0;
  }
};

}  // end namespace clang

#endif
