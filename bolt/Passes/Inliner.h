//===--- Passes/Inliner.h - Inlining infra for BOLT -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The set of optimization/analysis passes that run on BinaryFunctions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_INLINER_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_INLINER_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Inlining of single basic block functions.
/// The pass currently does not handle CFI instructions. This is needed for
/// correctness and we may break exception handling because of this.
class InlineSmallFunctions : public BinaryFunctionPass {
private:
  std::set<const BinaryFunction *> InliningCandidates;

  /// Maximum number of instructions in an inlined function.
  static const unsigned kMaxInstructions = 8;
  /// Maximum code size (in bytes) of inlined function (used by aggressive
  /// inlining).
  static const uint64_t kMaxSize = 60;
  /// Maximum number of functions that will be considered for inlining (in
  /// descending hottness order).
  static const unsigned kMaxFunctions = 30000;

  /// Statistics collected for debugging.
  uint64_t TotalDynamicCalls = 0;
  uint64_t InlinedDynamicCalls = 0;
  uint64_t TotalInlineableCalls = 0;
  std::unordered_set<const BinaryFunction *> Modified;

  static bool mustConsider(const BinaryFunction &BF);

  void findInliningCandidates(BinaryContext &BC,
                              const std::map<uint64_t, BinaryFunction> &BFs);

  /// Inline the call in CallInst to InlinedFunctionBB (the only BB of the
  /// called function).
  void inlineCall(BinaryContext &BC,
                  BinaryBasicBlock &BB,
                  MCInst *CallInst,
                  const BinaryBasicBlock &InlinedFunctionBB);

  bool inlineCallsInFunction(BinaryContext &BC,
                             BinaryFunction &Function);

  /// The following methods do a more aggressive inlining pass, where we
  /// inline calls as well as tail calls and we are not limited to inlining
  /// functions with only one basic block.
  /// FIXME: Currently these are broken since they do not work with the split
  /// function option.
  void findInliningCandidatesAggressive(
      BinaryContext &BC, const std::map<uint64_t, BinaryFunction> &BFs);

  bool inlineCallsInFunctionAggressive(
      BinaryContext &BC, BinaryFunction &Function);

  /// Inline the call in CallInst to InlinedFunction. Inlined function should not
  /// contain any landing pad or thrower edges but can have more than one blocks.
  ///
  /// Return the location (basic block and instruction index) where the code of
  /// the caller function continues after the the inlined code.
  std::pair<BinaryBasicBlock *, unsigned>
  inlineCall(BinaryContext &BC,
             BinaryFunction &CallerFunction,
             BinaryBasicBlock *CallerBB,
             const unsigned CallInstIdex,
             const BinaryFunction &InlinedFunction);

public:
  explicit InlineSmallFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "inlining";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

} // namespace bolt
} // namespace llvm

#endif
