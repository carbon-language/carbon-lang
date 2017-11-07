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

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_BINARY_PASSES_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_BINARY_PASSES_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "HFSort.h"
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
protected:
  bool PrintPass;

  explicit BinaryFunctionPass(const bool PrintPass)
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

/// A pass to print program-wide dynostats.
class DynoStatsPrintPass : public BinaryFunctionPass {
protected:
  DynoStats PrevDynoStats;
  std::string Title;

public:
  DynoStatsPrintPass(const DynoStats &PrevDynoStats, const char *Title)
    : BinaryFunctionPass(false)
    , PrevDynoStats(PrevDynoStats)
    , Title(Title) {
  }

  const char *getName() const override {
    return "print dyno-stats after optimizations";
  }

  bool shouldPrint(const BinaryFunction &BF) const override {
    return false;
  }

  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override {
    const auto NewDynoStats = getDynoStats(BFs);
    const auto Changed = (NewDynoStats != PrevDynoStats);
    outs() << "BOLT-INFO: program-wide dynostats "
           << Title << (Changed ? "" : " (no change)") << ":\n\n"
           << PrevDynoStats;
    if (Changed) {
      outs() << '\n';
      NewDynoStats.print(outs(), &PrevDynoStats);
    }
    outs() << '\n';
  }
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
  /// Choose which strategy should the block layout heuristic prioritize when
  /// facing conflicting goals.
  enum LayoutType : char {
    /// LT_NONE - do not change layout of basic blocks
    LT_NONE = 0, /// no reordering
    /// LT_REVERSE - reverse the order of basic blocks, meant for testing
    /// purposes. The first basic block is left intact and the rest are
    /// put in the reverse order.
    LT_REVERSE,
    /// LT_OPTIMIZE - optimize layout of basic blocks based on profile.
    LT_OPTIMIZE,
    /// LT_OPTIMIZE_BRANCH is an implementation of what is suggested in Pettis'
    /// paper (PLDI '90) about block reordering, trying to minimize branch
    /// mispredictions.
    LT_OPTIMIZE_BRANCH,
    /// LT_OPTIMIZE_CACHE piggybacks on the idea from Ispike paper (CGO '04)
    /// that suggests putting frequently executed chains first in the layout.
    LT_OPTIMIZE_CACHE,
    /// Create clusters and use random order for them.
    LT_OPTIMIZE_SHUFFLE,
  };

private:
  // Function size, in number of BBs, above which we fallback to a heuristic
  // solution to the layout problem instead of seeking the optimal one.
  static constexpr uint64_t FUNC_SIZE_THRESHOLD = 10;

  void modifyFunctionLayout(BinaryFunction &Function,
                            LayoutType Type,
                            bool MinBranchClusters,
                            bool Split) const;

  /// Split function in two: a part with warm or hot BBs and a part with never
  /// executed BBs. The cold part is moved to a new BinaryFunction.
  void splitFunction(BinaryFunction &Function) const;

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
class FinalizeFunctions : public BinaryFunctionPass {
 public:
  explicit FinalizeFunctions(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) { }

  const char *getName() const override {
    return "finalize-functions";
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
  uint64_t CTCExecCount{0};
  uint64_t CTCTakenCount{0};
  uint64_t NumOrigForwardBranches{0};
  uint64_t NumOrigBackwardBranches{0};
  uint64_t NumDoubleJumps{0};
  uint64_t DeletedBlocks{0};
  uint64_t DeletedBytes{0};
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
  uint64_t NumUselessCondBranches{0};

  /// Attempt to use the minimum operand width for arithmetic, branch and
  /// move instructions.
  void shortenInstructions(BinaryContext &BC, BinaryFunction &Function);

  /// Add trap instructions immediately after indirect tail calls to prevent
  /// the processor from decoding instructions immediate following the
  /// tailcall.
  void addTailcallTraps(BinaryContext &BC, BinaryFunction &Function);

  /// Remove useless duplicate successors.  When the conditional
  /// successor is the same as the unconditional successor, we can
  /// remove the conditional successor and branch instruction.
  void removeUselessCondBranches(BinaryContext &BC, BinaryFunction &Function);
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

/// Pass for stripping 'repz' from 'repz retq' sequence of instructions.
class StripRepRet : public BinaryFunctionPass {
public:
  explicit StripRepRet(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override {
    return "strip-rep-ret";
  }

  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

enum FrameOptimizationType : char {
  FOP_NONE, /// Don't perform FOP.
  FOP_HOT,  /// Perform FOP on hot functions.
  FOP_ALL   /// Perform FOP on all functions.
};

} // namespace bolt
} // namespace llvm

#endif
