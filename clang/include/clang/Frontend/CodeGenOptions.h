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
  unsigned CXAAtExit         : 1; /// Use __cxa_atexit for calling destructors.
  unsigned CXXCtorDtorAliases: 1; /// Emit complete ctors/dtors as linker
                                  /// aliases to base ctors when possible.
  unsigned DataSections      : 1; /// Set when -fdata-sections is enabled
  unsigned DebugInfo         : 1; /// Should generate debug info (-g).
  unsigned DisableFPElim     : 1; /// Set when -fomit-frame-pointer is enabled.
  unsigned DisableLLVMOpts   : 1; /// Don't run any optimizations, for use in
                                  /// getting .bc files that correspond to the
                                  /// internal state before optimizations are
                                  /// done.
  unsigned DisableRedZone    : 1; /// Set when -mno-red-zone is enabled.
  unsigned EmitDeclMetadata  : 1; /// Emit special metadata indicating what Decl*
                                  /// various IR entities came from.  Only useful
                                  /// when running CodeGen as a subroutine.
  unsigned FunctionSections  : 1; /// Set when -ffunction-sections is enabled
  unsigned EmitWeakTemplatesHidden : 1;  /// Emit weak vtables and typeinfo for
                                  /// template classes with hidden visibility
  unsigned InstrumentFunctions : 1; /// Set when -finstrument-functions is enabled
  unsigned MergeAllConstants : 1; /// Merge identical constants.
  unsigned NoCommon          : 1; /// Set when -fno-common or C++ is enabled.
  unsigned NoImplicitFloat   : 1; /// Set when -mno-implicit-float is enabled.
  unsigned NoZeroInitializedInBSS : 1; /// -fno-zero-initialized-in-bss
  unsigned ObjCDispatchMethod : 2; /// Method of Objective-C dispatch to use.
  unsigned OmitLeafFramePointer : 1; /// Set when -momit-leaf-frame-pointer is
                                     /// enabled.
  unsigned OptimizationLevel : 3; /// The -O[0-4] option specified.
  unsigned OptimizeSize      : 1; /// If -Os is specified.
  unsigned RelaxAll          : 1; /// Relax all machine code instructions.
  unsigned SimplifyLibCalls  : 1; /// Set when -fbuiltin is enabled.
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
    DataSections = 0;
    DebugInfo = 0;
    DisableFPElim = 0;
    DisableLLVMOpts = 0;
    DisableRedZone = 0;
    EmitDeclMetadata = 0;
    FunctionSections = 0;
    EmitWeakTemplatesHidden = 0;
    MergeAllConstants = 1;
    NoCommon = 0;
    NoImplicitFloat = 0;
    NoZeroInitializedInBSS = 0;
    ObjCDispatchMethod = Legacy;
    OmitLeafFramePointer = 0;
    OptimizationLevel = 0;
    OptimizeSize = 0;
    RelaxAll = 0;
    SimplifyLibCalls = 1;
    SoftFloat = 0;
    TimePasses = 0;
    UnitAtATime = 1;
    UnrollLoops = 0;
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
