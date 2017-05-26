//===--- Passes/FrameOptimizer.h ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_FRAMEOPTIMIZER_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_FRAMEOPTIMIZER_H

#include "BinaryPasses.h"
#include "CallGraph.h"

namespace llvm {
namespace bolt {

/// FrameOptimizerPass strives for removing unnecessary stack frame accesses.
/// For example, caller-saved registers may be conservatively pushed to the
/// stack because the callee may write to these registers. But if we can prove
/// the callee will never touch these registers, we can remove this spill.
///
/// This optimization analyzes the call graph and first compute the set of
/// registers that may get overwritten when executing a function (this includes
/// the set of registers touched by all functions this function may call during
/// its execution).
///
/// The second step is to perform an alias analysis to disambiguate which stack
/// position is being accessed by each load/store instruction, and annotate
/// these instructions.
///
/// The third step performs a forward dataflow analysis, using intersection as
/// the confluence operator, to propagate information about available
/// stack definitions at each point of the program. This definition shows
/// an equivalence between the value in a stack position and the value of a
/// register or immediate. To have those preserved, both register and the value
/// in the stack position cannot be touched by another instruction.
/// These definitions we are tracking occur in the form:
///
///     stack def:  MEM[FRAME - 0x5c]  <= RAX
///
/// Any instruction that writes to RAX will kill this definition, meaning RAX
/// cannot be used to recover the same value that is in FRAME - 0x5c. Any memory
/// write instruction to FRAME - 0x5c will also kill this definition.
///
/// If such a definition is available at an instruction that loads from this
/// frame offset, we have detected a redundant load. For example, if the
/// previous stack definition is available at the following instruction, this
/// is an example of a redundant stack load:
///
///     stack load:  RAX  <= MEM[FRAME - 0x5c]
///
/// The fourth step will use this info to actually modify redundant loads. In
/// our running example, we would change the stack load to the following reg
/// move:
///
///     RAX <= RAX  // can be deleted
///
/// In this example, since the store source register is the same as the load
/// destination register, this creates a redundant MOV that can be deleted.
///
class FrameOptimizerPass : public BinaryFunctionPass {
  /// Stats aggregating variables
  uint64_t NumRedundantLoads{0};
  uint64_t NumLoadsChangedToReg{0};
  uint64_t NumLoadsChangedToImm{0};
  uint64_t NumLoadsDeleted{0};
  /// Number of functions we conservatively marked as clobbering the entire set
  /// of registers because we couldn't fully understand it.
  uint64_t NumFunctionsAllClobber{0};
  /// Execution count of those functions to give us an idea of their dynamic
  /// coverage
  uint64_t CountFunctionsAllClobber{0};

  /// Call graph info
  CallGraph Cg;

  /// DFS or reverse post-ordering of the call graph nodes to allow us to
  /// traverse the call graph bottom-up
  std::deque<BinaryFunction *> TopologicalCGOrder;

  /// Map functions to the set of registers they may overwrite starting at when
  /// it is called until it returns to the caller.
  std::map<const BinaryFunction *, BitVector> RegsKilledMap;

public:
  /// Alias analysis information attached to each instruction that accesses a
  /// frame position. This is called a "frame index" by LLVM Target libs when
  /// it is building a MachineFunction frame, and we use the same name here
  /// because we are essentially doing the job of frame reconstruction.
  struct FrameIndexEntry {
    /// If this is false, this instruction is necessarily a store
    bool IsLoad;
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
  };
  typedef std::unordered_map<const MCInst *, const FrameIndexEntry>
      FrameIndexMapTy;
  FrameIndexMapTy FrameIndexMap;

  /// Compute the set of registers \p Inst may write to, marking them in
  /// \p KillSet. If this is a call, try to get the set of registers the call
  /// target will write to.
  void getInstClobberList(const BinaryContext &BC, const MCInst &Inst,
                          BitVector &KillSet) const;
private:
  /// Compute the set of registers \p Func may write to during its execution,
  /// starting at the point when it is called up until when it returns. Returns
  /// a BitVector the size of the target number of registers, representing the
  /// set of clobbered registers.
  BitVector getFunctionClobberList(const BinaryContext &BC,
                                   const BinaryFunction *Func);

  /// Perform the step of building the set of registers clobbered by each
  /// function execution, populating RegsKilledMap.
  void buildClobberMap(const BinaryContext &BC);

  /// Alias analysis to disambiguate which frame position is accessed by each
  /// instruction in function \p BF. Populates FrameIndexMap.
  bool restoreFrameIndex(const BinaryContext &BC, const BinaryFunction &BF);

  /// Uses RegsKilledMap and FrameIndexMap to perform a dataflow analysis in
  /// \p BF to reveal unnecessary reloads from the frame. Use the analysis
  /// to convert memory loads to register moves or immediate loads. Delete
  /// redundant register moves.
  void removeUnnecessarySpills(const BinaryContext &BC,
                               BinaryFunction &BF);

public:
  explicit FrameOptimizerPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override {
    return "frame-optimizer";
  }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};

} // namespace bolt
} // namespace llvm


#endif
