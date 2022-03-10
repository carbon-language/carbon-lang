//===--------- Definition of the AddressSanitizer class ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AddressSanitizer class which is a port of the legacy
// AddressSanitizer pass to use the new PassManager infrastructure.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"

namespace llvm {

/// Frontend-provided metadata for source location.
struct LocationMetadata {
  StringRef Filename;
  int LineNo = 0;
  int ColumnNo = 0;

  LocationMetadata() = default;

  bool empty() const { return Filename.empty(); }
  void parse(MDNode *MDN);
};

/// Frontend-provided metadata for global variables.
class GlobalsMetadata {
public:
  struct Entry {
    LocationMetadata SourceLoc;
    StringRef Name;
    bool IsDynInit = false;
    bool IsExcluded = false;

    Entry() = default;
  };

  /// Create a default uninitialized GlobalsMetadata instance.
  GlobalsMetadata() = default;

  /// Create an initialized GlobalsMetadata instance.
  GlobalsMetadata(Module &M);

  /// Returns metadata entry for a given global.
  Entry get(GlobalVariable *G) const {
    auto Pos = Entries.find(G);
    return (Pos != Entries.end()) ? Pos->second : Entry();
  }

  /// Handle invalidation from the pass manager.
  /// These results are never invalidated.
  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &) {
    return false;
  }
  bool invalidate(Function &, const PreservedAnalyses &,
                  FunctionAnalysisManager::Invalidator &) {
    return false;
  }

private:
  DenseMap<GlobalVariable *, Entry> Entries;
};

/// The ASanGlobalsMetadataAnalysis initializes and returns a GlobalsMetadata
/// object. More specifically, ASan requires looking at all globals registered
/// in 'llvm.asan.globals' before running, which only depends on reading module
/// level metadata. This analysis is required to run before running the
/// AddressSanitizerPass since it collects that metadata.
/// The legacy pass manager equivalent of this is ASanGlobalsMetadataLegacyPass.
class ASanGlobalsMetadataAnalysis
    : public AnalysisInfoMixin<ASanGlobalsMetadataAnalysis> {
public:
  using Result = GlobalsMetadata;

  Result run(Module &, ModuleAnalysisManager &);

private:
  friend AnalysisInfoMixin<ASanGlobalsMetadataAnalysis>;
  static AnalysisKey Key;
};

struct AddressSanitizerOptions {
  bool CompileKernel = false;
  bool Recover = false;
  bool UseAfterScope = false;
  AsanDetectStackUseAfterReturnMode UseAfterReturn =
      AsanDetectStackUseAfterReturnMode::Runtime;
};

/// Public interface to the address sanitizer pass for instrumenting code to
/// check for various memory errors at runtime.
///
/// The sanitizer itself is a function pass that works by inserting various
/// calls to the ASan runtime library functions. The runtime library essentially
/// replaces malloc() and free() with custom implementations that allow regions
/// surrounding requested memory to be checked for invalid accesses.
class AddressSanitizerPass : public PassInfoMixin<AddressSanitizerPass> {
public:
  AddressSanitizerPass(const AddressSanitizerOptions &Options)
      : Options(Options){};
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
  static bool isRequired() { return true; }

private:
  AddressSanitizerOptions Options;
};

/// Public interface to the address sanitizer module pass for instrumenting code
/// to check for various memory errors.
///
/// This adds 'asan.module_ctor' to 'llvm.global_ctors'. This pass may also
/// run intependently of the function address sanitizer.
class ModuleAddressSanitizerPass
    : public PassInfoMixin<ModuleAddressSanitizerPass> {
public:
  ModuleAddressSanitizerPass(
      const AddressSanitizerOptions &Options, bool UseGlobalGC = true,
      bool UseOdrIndicator = false,
      AsanDtorKind DestructorKind = AsanDtorKind::Global);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
  static bool isRequired() { return true; }

private:
  AddressSanitizerOptions Options;
  bool UseGlobalGC;
  bool UseOdrIndicator;
  AsanDtorKind DestructorKind;
};

// Insert AddressSanitizer (address basic correctness checking) instrumentation
FunctionPass *createAddressSanitizerFunctionPass(
    bool CompileKernel = false, bool Recover = false,
    bool UseAfterScope = false,
    AsanDetectStackUseAfterReturnMode UseAfterReturn =
        AsanDetectStackUseAfterReturnMode::Runtime);
ModulePass *createModuleAddressSanitizerLegacyPassPass(
    bool CompileKernel = false, bool Recover = false, bool UseGlobalsGC = true,
    bool UseOdrIndicator = true,
    AsanDtorKind DestructorKind = AsanDtorKind::Global);

struct ASanAccessInfo {
  const int32_t Packed;
  const uint8_t AccessSizeIndex;
  const bool IsWrite;
  const bool CompileKernel;

  explicit ASanAccessInfo(int32_t Packed);
  ASanAccessInfo(bool IsWrite, bool CompileKernel, uint8_t AccessSizeIndex);
};

} // namespace llvm

#endif
