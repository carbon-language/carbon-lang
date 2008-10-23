//===--- CompileOptions.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CompileOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_COMPILEOPTIONS_H
#define LLVM_CLANG_COMPILEOPTIONS_H

namespace clang {

/// CompileOptions - Track various options which control how the code
/// is optimized and passed to the backend.
struct CompileOptions {
  unsigned OptimizationLevel : 3; /// The -O[0-4] option specified.
  unsigned OptimizeSize      : 1; /// If -Os is specified.
  unsigned UnitAtATime       : 1; /// Unused. For mirroring GCC
                                  /// optimization selection.
  unsigned InlineFunctions   : 1; /// Should functions be inlined?
  unsigned SimplifyLibCalls  : 1; /// Should standard library calls be
                                  /// treated specially.
  unsigned UnrollLoops       : 1; /// Control whether loops are unrolled.
  unsigned VerifyModule      : 1; /// Control whether the module
                                  /// should be run through the LLVM Verifier.

public:
  CompileOptions() {
    OptimizationLevel = 0;
    OptimizeSize = 0;
    UnitAtATime = InlineFunctions = SimplifyLibCalls = 1;
    UnrollLoops = 1;
    VerifyModule = 1;
  }
};

}  // end namespace clang

#endif
