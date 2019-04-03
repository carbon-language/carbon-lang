//===--- Passes/PLTCall.h - PLT call optimization -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_PLTCALL_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_PLTCALL_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryPasses.h"

namespace llvm {
namespace bolt {

class PLTCall : public BinaryFunctionPass {
public:

  /// PLT optimization type
  enum OptType : char {
    OT_NONE = 0,   /// Do not optimize
    OT_HOT  = 1,   /// Optimize hot PLT calls
    OT_ALL  = 2    /// Optimize all PLT calls
  };

  explicit PLTCall(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "PLT call optimization";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF);
 }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
