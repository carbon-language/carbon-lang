//===--- BinaryPasses.h - Binary-level analysis/optimization passes -------===//
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

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_PASSES_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_PASSES_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
namespace bolt {

/// An optimization/analysis pass that runs on functions.
class BinaryFunctionPass {
  const cl::opt<bool> &PrintPass;
protected:
  explicit BinaryFunctionPass(const cl::opt<bool> &PrintPass)
    : PrintPass(PrintPass) { }

  /// Control whether a specific function should be skipped during
  /// optimization.
  bool shouldOptimize(const BinaryFunction &BF) const;
public:
  virtual ~BinaryFunctionPass() = default;

  /// The name of this pass
  virtual const char *getName() const = 0;

  /// Control whether debug info is printed after this pass is completed.
  bool printPass() const { return PrintPass; }

  /// Control whether debug info is printed for an individual function after
  /// this pass is completed (printPass() must have returned true).
  virtual bool shouldPrint(const BinaryFunction &BF) const;

  /// Execute this pass on the given functions.
  virtual void runOnFunctions(BinaryContext &BC,
                              std::map<uint64_t, BinaryFunction> &BFs,
                              std::set<uint64_t> &LargeFunctions) = 0;
};

/// Detects functions that simply do a tail call when they are called and
/// optimizes calls to these functions.
class OptimizeBodylessFunctions : public BinaryFunctionPass {
private:
  /// EquivalentCallTarget[F] = G ==> function F is simply a tail call to G,
  /// thus calls to F can be optimized to calls to G.
  std::unordered_map<const MCSymbol *, const BinaryFunction *>
    EquivalentCallTarget;

  void analyze(BinaryFunction &BF,
               BinaryContext &BC,
               std::map<uint64_t, BinaryFunction> &BFs);

  void optimizeCalls(BinaryFunction &BF,
                     BinaryContext &BC);

  /// Stats for eliminated calls.
  uint64_t NumEliminatedCalls{0};
  uint64_t NumOptimizedCallSites{0};

public:
  explicit OptimizeBodylessFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }
  const char *getName() const override {
    return "optimize-bodyless";
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

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

/// Detect and eliminate unreachable basic blocks. We could have those
/// filled with nops and they are used for alignment.
class EliminateUnreachableBlocks : public BinaryFunctionPass {
  std::unordered_set<const BinaryFunction *> Modified;
  unsigned DeletedBlocks{0};
  uint64_t DeletedBytes{0};
  void runOnFunction(BinaryFunction& Function);
 public:
  EliminateUnreachableBlocks(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "eliminate-unreachable";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext&,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

// Reorder the basic blocks for each function based on hotness.
class ReorderBasicBlocks : public BinaryFunctionPass {
 public:
  explicit ReorderBasicBlocks(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "reordering";
  }
  bool shouldPrint(const BinaryFunction &BF) const override;
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// Sync local branches with CFG.
class FixupBranches : public BinaryFunctionPass {
 public:
  explicit FixupBranches(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "fix-branches";
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// Fix the CFI state and exception handling information after all other
/// passes have completed.
class FixupFunctions : public BinaryFunctionPass {
 public:
  explicit FixupFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "fixup-functions";
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// An optimization to simplify conditional tail calls by removing
/// unnecessary branches.
///
/// This optimization considers both of the following cases:
///
/// foo: ...
///      jcc L1   original
///      ...
/// L1:  jmp bar  # TAILJMP
///
/// ->
///
/// foo: ...
///      jcc bar  iff jcc L1 is expected
///      ...
///
/// L1 is unreachable
///
/// OR
///
/// foo: ...
///      jcc  L2
/// L1:  jmp  dest  # TAILJMP
/// L2:  ...
///
/// ->
///
/// foo: jncc dest  # TAILJMP
/// L2:  ...
///
/// L1 is unreachable
///
/// For this particular case, the first basic block ends with
/// a conditional branch and has two successors, one fall-through
/// and one for when the condition is true.
/// The target of the conditional is a basic block with a single
/// unconditional branch (i.e. tail call) to another function.
/// We don't care about the contents of the fall-through block.
/// We assume that the target of the conditional branch is the
/// first successor.
class SimplifyConditionalTailCalls : public BinaryFunctionPass {
  uint64_t NumCandidateTailCalls{0};
  uint64_t NumTailCallsPatched{0};
  uint64_t NumOrigForwardBranches{0};
  uint64_t NumOrigBackwardBranches{0};
  std::unordered_set<const BinaryFunction *> Modified;

  bool shouldRewriteBranch(const BinaryBasicBlock *PredBB,
                           const MCInst &CondBranch,
                           const BinaryBasicBlock *BB,
                           const bool DirectionFlag);

  uint64_t fixTailCalls(BinaryContext &BC, BinaryFunction &BF);
 public:
  explicit SimplifyConditionalTailCalls(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "simplify-conditional-tail-calls";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// Perform simple peephole optimizations.
class Peepholes : public BinaryFunctionPass {
  uint64_t NumDoubleJumps{0};
  uint64_t TailCallTraps{0};

  /// Attempt to use the minimum operand width for arithmetic, branch and
  /// move instructions.
  void shortenInstructions(BinaryContext &BC, BinaryFunction &Function);

  /// Replace double jumps with a jump directly to the target, i.e.
  /// jmp/jcc L1; L1: jmp L2 -> jmp/jcc L2.
  void fixDoubleJumps(BinaryContext &BC, BinaryFunction &Function);

  /// Add trap instructions immediately after indirect tail calls to prevent
  /// the processor from decoding instructions immediate following the
  /// tailcall.
  void addTailcallTraps(BinaryContext &BC, BinaryFunction &Function);
 public:
  explicit Peepholes(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "peepholes";
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// An optimization to simplify loads from read-only sections.The pass converts
/// load instructions with statically computed target address such as:
///
///      mov 0x12f(%rip), %eax
///
/// to their counterparts that use immediate opreands instead of memory loads:
///
///     mov $0x4007dc, %eax
///
/// when the target address points somewhere inside a read-only section.
///
class SimplifyRODataLoads : public BinaryFunctionPass {
  uint64_t NumLoadsSimplified{0};
  uint64_t NumDynamicLoadsSimplified{0};
  uint64_t NumLoadsFound{0};
  uint64_t NumDynamicLoadsFound{0};
  std::unordered_set<const BinaryFunction *> Modified;

  bool simplifyRODataLoads(BinaryContext &BC, BinaryFunction &BF);

public:
  explicit SimplifyRODataLoads(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "simplify-read-only-loads";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// An optimization that replaces references to identical functions with
/// references to a single one of them.
///
class IdenticalCodeFolding : public BinaryFunctionPass {
public:
  explicit IdenticalCodeFolding(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "identical-code-folding";
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

///
/// Prints a list of the top 100 functions sorted by a set of
/// dyno stats categories.
///
class PrintSortedBy : public BinaryFunctionPass {
 public:
  explicit PrintSortedBy(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "print-sorted-by";
  }
  bool shouldPrint(const BinaryFunction &) const override {
    return false;
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// Optimize indirect calls.
/// The indirect call promotion pass visits each indirect call and
/// examines the BranchData for each. If the most frequent targets
/// from that callsite exceed the specified threshold (default 90%),
/// the call is promoted. Otherwise, it is ignored. By default,
/// only one target is considered at each callsite.
/// 
/// When an candidate callsite is processed, we modify the callsite
/// to test for the most common call targets before calling through
/// the original generic call mechanism.
/// 
/// The CFG and layout are modified by ICP.
/// 
/// A few new command line options have been added:
///   -indirect-call-promotion
///   -indirect-call-promotion-threshold=<percentage>
///   -indirect-call-promotion-mispredict-threshold=<percentage>
///   -indirect-call-promotion-topn=<int>
///
/// The threshold is the minimum frequency of a call target needed
/// before ICP is triggered.
///
/// The mispredict threshold is used to disable the optimization at
/// any callsite where the branch predictor does a good enough job
/// that ICP wouldn't help regardless of the frequency of the most
/// common target.
/// 
/// The topn option controls the number of targets to consider for
/// each callsite, e.g. ICP is triggered if topn=2 and the total
/// frequency of the top two call targets exceeds the threshold.
///
/// The minimize code size option controls whether or not the hot
/// calls are to registers (callq %r10) or to function addresses
/// (callq $foo).
///
/// Example of ICP:
/// 
/// C++ code:
/// 
///   int B_count = 0;
///   int C_count = 0;
/// 
///   struct A { virtual void foo() = 0; }
///   struct B : public A { virtual void foo() { ++B_count; }; };
///   struct C : public A { virtual void foo() { ++C_count; }; };
/// 
///   A* a = ...
///   a->foo();
///   ...
/// 
/// original assembly:
/// 
///   B0: 49 8b 07             mov    (%r15),%rax
///       4c 89 ff             mov    %r15,%rdi
///       ff 10                callq  *(%rax)
///       41 83 e6 01          and    $0x1,%r14d
///       4d 89 e6             mov    %r12,%r14
///       4c 0f 44 f5          cmove  %rbp,%r14
///       4c 89 f7             mov    %r14,%rdi
///       ...
/// 
/// after ICP:
/// 
///   B0: 49 8b 07             mov    (%r15),%rax
///       4c 89 ff             mov    %r15,%rdi
///       48 81 38 e0 0b 40 00 cmpq   $B::foo,(%rax)
///       75 29                jne    B3
///   B1: e8 45 03 00 00       callq  $B::foo
///   B2: 41 83 e6 01          and    $0x1,%r14d
///       4d 89 e6             mov    %r12,%r14
///       4c 0f 44 f5          cmove  %rbp,%r14
///       4c 89 f7             mov    %r14,%rdi
///       ...
/// 
///   B3: ff 10                callq  *(%rax)
///       eb d6                jmp    B2
///
class IndirectCallPromotion : public BinaryFunctionPass {
  using BasicBlocksVector = std::vector<std::unique_ptr<BinaryBasicBlock>>;
  std::unordered_set<const BinaryFunction *> Modified;
  // Total number of calls from all callsites.
  uint64_t TotalCalls{0};

  // Total number of indirect calls from all callsites.
  // (a fraction of TotalCalls)
  uint64_t TotalIndirectCalls{0};

  // Total number of callsites that use indirect calls.
  // (the total number of callsites is not recorded)
  uint64_t TotalIndirectCallsites{0};

  // Total number of indirect callsites that are optimized by ICP.
  // (a fraction of TotalIndirectCallsites)
  uint64_t TotalOptimizedIndirectCallsites{0};

  // Total number of indirect calls that are optimized by ICP.
  // (a fraction of TotalCalls)
  uint64_t TotalNumFrequentCalls{0};

  std::vector<BranchInfo> getCallTargets(BinaryContext &BC,
                                         const FuncBranchData &BranchData,
                                         const MCInst &Inst) const;

  size_t canPromoteCallsite(const BinaryBasicBlock *BB,
                            const MCInst &Inst,
                            const std::vector<BranchInfo> &Targets,
                            uint64_t NumCalls);

  void printCallsiteInfo(const BinaryBasicBlock *BB,
                         const MCInst &Inst,
                         const std::vector<BranchInfo> &Targets,
                         const size_t N,
                         uint64_t NumCalls) const;

  std::vector<std::pair<MCSymbol *, uint64_t>>
  findCallTargetSymbols(BinaryContext &BC,
                        const std::vector<BranchInfo> &Targets,
                        const size_t N) const;

  std::vector<std::unique_ptr<BinaryBasicBlock>>
  rewriteCall(BinaryContext &BC,
              BinaryFunction &Function,
              BinaryBasicBlock *IndCallBlock,
              const MCInst &CallInst,
              MCInstrAnalysis::ICPdata &&ICPcode) const;

  BinaryBasicBlock *fixCFG(BinaryContext &BC,
                           BinaryFunction &Function,
                           BinaryBasicBlock *IndCallBlock,
                           const bool IsTailCall,
                           BasicBlocksVector &&NewBBs,
                           const std::vector<BranchInfo> &Targets) const;

 public:
  explicit IndirectCallPromotion(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const {
    return "indirect-call-promotion";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

/// Pass for lowering any instructions that we have raised and that have
/// to be lowered.
class InstructionLowering : public BinaryFunctionPass {
public:
  explicit InstructionLowering(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override {
    return "inst-lowering";
  }

  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

} // namespace bolt
} // namespace llvm

#endif
