//===--- Passes/PatchEntries.h - pass for patching function entries -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass for patching original function entry points.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_PATCH_ENTRIES_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_PATCH_ENTRIES_H

#include "Passes/BinaryPasses.h"
#include "Relocation.h"
#include "llvm/ADT/SmallString.h"

namespace llvm {
namespace bolt {

/// Pass for patching original function entry points.
class PatchEntries : public BinaryFunctionPass {
  // If the function size is below the threshold, attempt to skip patching it.
  static constexpr uint64_t PatchThreshold = 128;

  struct InstructionPatch {
    uint64_t Offset;
    SmallString<8> Code;
    Relocation Rel;
  };

public:
  explicit PatchEntries() : BinaryFunctionPass(false) {
  }

  const char *getName() const override {
    return "patch-entries";
  }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
