//===--- Passes/FrameAnalysis.h -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_FRAMEANALYSIS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_FRAMEANALYSIS_H

#include "BinaryPasses.h"
#include "StackPointerTracking.h"

namespace llvm {
namespace bolt {

/// Alias analysis information attached to each instruction that accesses a
/// frame position. This is called a "frame index" by LLVM Target libs when
/// it is building a MachineFunction frame, and we use the same name here
/// because we are essentially doing the job of frame reconstruction.
struct FrameIndexEntry {
  /// If both IsLoad and IsStore are set, it means this is an instruction that
  /// reads and updates this frame location.
  bool IsLoad;
  bool IsStore;
  /// If a store, this controls whether the store uses a register os an imm
  /// as the source value.
  bool IsStoreFromReg;
  /// If load, this holds the destination register. If store, this holds
  /// either the source register or source immediate.
  int32_t RegOrImm;

  /// StackOffset and Size are the two aspects that identify this frame access
  /// for the purposes of alias analysis.
  int64_t StackOffset;
  uint8_t Size;

  /// If this is false, we will never atempt to remove or optimize this
  /// instruction. We just use it to keep track of stores we don't fully
  /// understand but we know it may write to a frame position.
  bool IsSimple;

  uint16_t StackPtrReg;
};

/// Record an access to an argument in stack. This should be attached to
/// call instructions, so StackOffset and Size are determined in the context
/// of the caller. This information helps the caller understand how the callee
/// may access its private stack.
struct ArgInStackAccess {
  int64_t StackOffset;
  uint8_t Size;

  bool operator<(const ArgInStackAccess &RHS) const {
    if (StackOffset != RHS.StackOffset)
      return StackOffset < RHS.StackOffset;
    return Size < RHS.Size;
  }
};

/// The set of all args-in-stack accesses for a given instruction. If
/// AssumeEverything is true, then the set should be ignored and the
/// corresponding instruction should be treated as accessing the entire
/// stack for the purposes of analysis and optimization.
struct ArgAccesses {
  bool AssumeEverything;
  std::set<ArgInStackAccess> Set;

  explicit ArgAccesses(bool AssumeEverything)
      : AssumeEverything(AssumeEverything) {}
};

raw_ostream &operator<<(raw_ostream &OS,
                        const FrameIndexEntry &FIE);

/// This pass attaches stack access information to instructions. If a load/store
/// instruction accesses a stack position, it will identify the CFA offset and
/// size information of this access, where CFA is the Canonical Frame Address
/// (using DWARF terminology).
///
/// This pass also computes frame usage information obtained by a bottom-up call
/// graph traversal: which registers are clobbered by functions (including their
/// callees as determined by the call graph), whether a function accesses its
/// caller's stack frame and whether a function demands its stack to be aligned
/// due to the use of SSE aligned load/store operations present in itself or any
/// of its direct or indirect callees.
///
/// Initialization:
///
///   FrameAnalysis FA(PrintPass);
///   FA.runOnFunctions(BC, BFs, LargeFunctions);
///
/// Usage (fetching frame access information about a given instruction):
///
///   auto FIE = FA.getFIEFor(BC, Instruction);
///   if (FIE && FIE->IsSimple) {
///     ... = FIE->StackOffset
///     ... = FIE->Size
///   }
///
/// Usage (determining the set of stack positions accessed by the target of a
/// call:
///
///    auto Args = FA.getArgAccessesFor(BC, CallInst);
///    if (Args && Args->AssumeEverything) {
///      ... callee may access any position of our current stack frame
///    }
///
class FrameAnalysis : public BinaryFunctionPass {
  /// Call graph info
  /// The set of functions analyzed by our call graph
  std::set<BinaryFunction *> Functions;
  /// Model the "function calls function" edges
  std::map<const BinaryFunction *, std::vector<BinaryFunction *>>
      CallGraphEdges;
  /// Model the "function called by function" edges
  std::map<const BinaryFunction *, std::vector<BinaryFunction *>>
      ReverseCallGraphEdges;
  /// DFS or reverse post-ordering of the call graph nodes to allow us to
  /// traverse the call graph bottom-up
  std::deque<BinaryFunction *> TopologicalCGOrder;

  /// Map functions to the set of registers they may overwrite starting at when
  /// it is called until it returns to the caller.
  std::map<const BinaryFunction *, BitVector> RegsKilledMap;

  /// Map functions to the set of <stack offsets, size> tuples representing
  /// accesses to stack positions that belongs to caller
  std::map<const BinaryFunction *, std::set<std::pair<int64_t, uint8_t>>>
      ArgsTouchedMap;

  /// The set of functions we were able to perform the full analysis up to
  /// restoring frame indexes for all load/store instructions.
  DenseSet<const BinaryFunction *> AnalyzedFunctions;

  /// Set of functions that require the stack to be 16B aligned
  DenseSet<const BinaryFunction *> FunctionsRequireAlignment;

  /// Owns ArgAccesses for all instructions. References to elements are
  /// attached to instructions as indexes to this vector, in MCAnnotations.
  std::vector<ArgAccesses> ArgAccessesVector;
  /// Same for FrameIndexEntries.
  std::vector<FrameIndexEntry> FIEVector;

  /// Analysis stats counters
  uint64_t NumFunctionsAllClobber{0};
  uint64_t CountFunctionsAllClobber{0};
  uint64_t NumFunctionsNotOptimized{0};
  uint64_t NumFunctionsFailedRestoreFI{0};
  uint64_t CountFunctionsNotOptimized{0};
  uint64_t CountFunctionsFailedRestoreFI{0};
  uint64_t CountDenominator{0};

  /// If this flag is set to true, the analysis will never run completely,
  /// but will stop after callgraph and a clobber analysis for every function
  /// has been computed.
  bool ClobberAnalysisOnly{false};

  /// Convenience functions for appending MCAnnotations to instructions with
  /// our specific data
  void addArgAccessesFor(const BinaryContext &BC, MCInst &Inst,
                         ArgAccesses &&AA);
  void addArgInStackAccessFor(const BinaryContext &BC, MCInst &Inst,
                              const ArgInStackAccess &Arg);
  void addFIEFor(const BinaryContext &BC, MCInst &Inst,
                 const FrameIndexEntry &FIE);

  /// Perform the initial step of populating CallGraphEdges and
  /// ReverseCallGraphEdges for all functions in BFs.
  void buildCallGraph(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs);

  /// Compute a DFS traversal of the call graph in Functions, CallGraphEdges
  /// and ReverseCallGraphEdges and stores it in TopologicalCGOrder.
  void buildCGTraversalOrder();

  /// Compute the set of registers \p Func may write to during its execution,
  /// starting at the point when it is called up until when it returns. Returns
  /// a BitVector the size of the target number of registers, representing the
  /// set of clobbered registers.
  BitVector getFunctionClobberList(const BinaryContext &BC,
                                   const BinaryFunction *Func);

  /// Perform the step of building the set of registers clobbered by each
  /// function execution, populating RegsKilledMap.
  void buildClobberMap(const BinaryContext &BC);

  /// Analyzes an instruction and if it is a call, checks the called function
  /// to record which args in stack are accessed, if any. Returns true if
  /// the args data associated with this instruction were updated.
  bool updateArgsTouchedFor(const BinaryContext &BC, const BinaryFunction &BF,
                            MCInst &Inst, int CurOffset);

  /// Performs a pass over \p BF to check for accesses to arguments in stack,
  /// flagging those as accessing the caller stack frame. All functions called
  /// by \p BF must have been previously analyzed. Returns true if updated
  /// args data about this function.
  bool computeArgsAccessed(const BinaryContext &BC, BinaryFunction &BF);

  /// Alias analysis to disambiguate which frame position is accessed by each
  /// instruction in function \p BF. Add MCAnnotation<FrameIndexEntry> to
  /// instructions that access a frame position. Return false if it failed
  /// to analyze and this information can't be safely determined for \p BF.
  bool restoreFrameIndex(const BinaryContext &BC, BinaryFunction &BF);

public:
  explicit FrameAnalysis(const cl::opt<bool> &PrintPass,
                         bool ClobberAnalysisOnly=false)
      : BinaryFunctionPass(PrintPass),
        ClobberAnalysisOnly(ClobberAnalysisOnly) {
    // Position 0 of the vector should be always associated with "assume access
    // everything".
    ArgAccessesVector.emplace_back(ArgAccesses(/*AssumeEverything*/ true));
  }

  const char *getName() const override {
    return "frame-analysis";
  }

  /// Return true if we could fully analyze \p Func
  bool hasFrameInfo(const BinaryFunction &Func) const {
    return AnalyzedFunctions.count(&Func);
  }

  /// Return true if \p Func cannot operate with a misaligned CFA
  bool requiresAlignment(const BinaryFunction &Func) const {
    return FunctionsRequireAlignment.count(&Func);
  }

  /// Compute the set of registers \p Inst may write to, marking them in
  /// \p KillSet. If this is a call, try to get the set of registers the call
  /// target will write to.
  void getInstClobberList(const BinaryContext &BC, const MCInst &Inst,
                          BitVector &KillSet) const;

  /// Functions for retrieving our specific MCAnnotation data from instructions
  ErrorOr<ArgAccesses &> getArgAccessesFor(const BinaryContext &BC,
                                           const MCInst &Inst);

  ErrorOr<const ArgAccesses &> getArgAccessesFor(const BinaryContext &BC,
                                                 const MCInst &Inst) const;

  ErrorOr<const FrameIndexEntry &> getFIEFor(const BinaryContext &BC,
                                             const MCInst &Inst) const;

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;

  /// Remove all MCAnnotations attached by this pass
  void cleanAnnotations(const BinaryContext &BC,
                        std::map<uint64_t, BinaryFunction> &BFs);


  /// Print to standard output statistics about the analysis performed by this
  /// pass
  void printStats();
};

} // namespace bolt
} // namespace llvm


#endif
