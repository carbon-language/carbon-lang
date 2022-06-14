//====- Internalize.h - Internalization API ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass loops over all of the functions and variables in the input module.
// If the function or variable does not need to be preserved according to the
// client supplied callback, it is marked as internal.
//
// This transformation would not be legal in a regular compilation, but it gets
// extra information from the linker about what is safe.
//
// For example: Internalizing a function with external linkage. Only if we are
// told it is only used from within this module, it is safe to do it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INTERNALIZE_H
#define LLVM_TRANSFORMS_IPO_INTERNALIZE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/PassManager.h"
#include <functional>

namespace llvm {
class Module;
class CallGraph;

/// A pass that internalizes all functions and variables other than those that
/// must be preserved according to \c MustPreserveGV.
class InternalizePass : public PassInfoMixin<InternalizePass> {
  struct ComdatInfo {
    // The number of members. A comdat with one member which is not externally
    // visible can be freely dropped.
    size_t Size = 0;
    // Whether the comdat has an externally visible member.
    bool External = false;
  };

  bool IsWasm = false;

  /// Client supplied callback to control wheter a symbol must be preserved.
  const std::function<bool(const GlobalValue &)> MustPreserveGV;
  /// Set of symbols private to the compiler that this pass should not touch.
  StringSet<> AlwaysPreserved;

  /// Return false if we're allowed to internalize this GV.
  bool shouldPreserveGV(const GlobalValue &GV);
  /// Internalize GV if it is possible to do so, i.e. it is not externally
  /// visible and is not a member of an externally visible comdat.
  bool maybeInternalize(GlobalValue &GV,
                        DenseMap<const Comdat *, ComdatInfo> &ComdatMap);
  /// If GV is part of a comdat and is externally visible, keep track of its
  /// comdat so that we don't internalize any of its members.
  void checkComdat(GlobalValue &GV,
                   DenseMap<const Comdat *, ComdatInfo> &ComdatMap);

public:
  InternalizePass();
  InternalizePass(std::function<bool(const GlobalValue &)> MustPreserveGV)
      : MustPreserveGV(std::move(MustPreserveGV)) {}

  /// Run the internalizer on \p TheModule, returns true if any changes was
  /// made.
  ///
  /// If the CallGraph \p CG is supplied, it will be updated when
  /// internalizing a function (by removing any edge from the "external node")
  bool internalizeModule(Module &TheModule, CallGraph *CG = nullptr);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Helper function to internalize functions and variables in a Module.
inline bool
internalizeModule(Module &TheModule,
                  std::function<bool(const GlobalValue &)> MustPreserveGV,
                  CallGraph *CG = nullptr) {
  return InternalizePass(std::move(MustPreserveGV))
      .internalizeModule(TheModule, CG);
}
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INTERNALIZE_H
