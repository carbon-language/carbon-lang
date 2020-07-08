//===--- Passes/FeatureMiner.h -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// A very simple feature extractor based on Calder's paper
// Evidence-based static branch prediction using machine learning
// https://dl.acm.org/doi/10.1145/239912.239923
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_FEATUREMINER_H_
#define LLVM_TOOLS_LLVM_BOLT_PASSES_FEATUREMINER_H_

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryLoop.h"
#include "DominatorAnalysis.h"
#include "Passes/BinaryPasses.h"
#include "Passes/BranchPredictionInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace bolt {

class FeatureMiner : public BinaryFunctionPass {
private:
  std::unique_ptr<BranchPredictionInfo> BPI;

  /// BasicBlockInfo - This structure holds feature information about the target
  /// BasicBlock of either the taken or the fallthrough paths of a given branch.
  struct BasicBlockInfo {
    Optional<bool> BranchDominates;     // 1 - dominates, 0 - does not dominate
    Optional<bool> BranchPostdominates; // 1 - postdominates, 0 - does not PD
    Optional<bool> LoopHeader; // 1 - loop header, 0 - not a loop header
    Optional<bool> Backedge;   // 1 - loop back, 0 - not a loop back
    Optional<bool> Exit;       // 1 - loop exit, 0 - not a loop exit
    Optional<bool> Call;       // 1 - program call, 0 - not a program call
    Optional<unsigned> NumCalls;
    Optional<unsigned> NumLoads;
    Optional<unsigned> NumStores;
    Optional<int32_t> EndOpcode; // 0 = NOTHING
    StringRef EndOpcodeStr = "UNDEF";
    Optional<int32_t> BasicBlockSize;
    std::string FromFunName = "UNDEF";
    uint32_t FromBb;
    std::string ToFunName = "UNDEF";
    uint32_t ToBb;

    Optional<unsigned> NumCallsExit;
    Optional<unsigned> NumCallsInvoke;
    Optional<unsigned> NumIndirectCalls;
    Optional<unsigned> NumTailCalls;
  };

  typedef std::unique_ptr<struct BasicBlockInfo> BBIPtr;

  /// StaticBranchInfo - This structure holds feature information about each
  /// two-way branch from the program.
  struct StaticBranchInfo {
    StringRef OpcodeStr = "UNDEF";
    StringRef CmpOpcodeStr = "UNDEF";
    bool Simple = 0;

    Optional<int32_t> Opcode;
    Optional<int32_t> CmpOpcode;
    Optional<int64_t> Count;
    Optional<int64_t> MissPredicted;
    Optional<int64_t> FallthroughCount;
    Optional<int64_t> FallthroughMissPredicted;
    BBIPtr TrueSuccessor = std::make_unique<struct BasicBlockInfo>();
    BBIPtr FalseSuccessor = std::make_unique<struct BasicBlockInfo>();
    Optional<int8_t> ProcedureType; // 1 - Leaf, 0 - NonLeaf, 2 - CallSelf
    Optional<bool> LoopHeader;      // 1 — loop header, 0 - not a loop header
    Optional<bool> Direction;       // 1 - Forward Branch, 0 - Backward Branch

    Optional<unsigned> NumOuterLoops;
    Optional<unsigned> TotalLoops;
    Optional<unsigned> MaximumLoopDepth;
    Optional<unsigned> LoopDepth;
    Optional<unsigned> LoopNumExitEdges;
    Optional<unsigned> LoopNumExitBlocks;
    Optional<unsigned> LoopNumExitingBlocks;
    Optional<unsigned> LoopNumLatches;
    Optional<unsigned> LoopNumBlocks;
    Optional<unsigned> LoopNumBackEdges;
    Optional<unsigned> NumLoads;
    Optional<unsigned> NumStores;

    Optional<bool> LocalExitingBlock;
    Optional<bool> LocalLatchBlock;
    Optional<bool> LocalLoopHeader;
    Optional<bool> Call;

    Optional<unsigned> NumCalls;
    Optional<unsigned> NumCallsExit;
    Optional<unsigned> NumCallsInvoke;
    Optional<unsigned> NumIndirectCalls;
    Optional<unsigned> NumTailCalls;
    Optional<unsigned> NumSelfCalls;

    Optional<unsigned> NumBasicBlocks;

    Optional<unsigned> DeltaTaken;

    Optional<int32_t> OperandRAType;
    Optional<int32_t> OperandRBType;

    Optional<int32_t> BasicBlockSize;
  };

  typedef std::unique_ptr<struct StaticBranchInfo> SBIPtr;
  std::vector<SBIPtr> BranchesInfoSet;

  /// getProcedureType - Determines which category the function falls into:
  /// Leaf, Non-leaf or Calls-self.
  int8_t getProcedureType(BinaryFunction &Function, BinaryContext &BC);

  /// addSuccessorInfo - Discovers feature information for the target successor
  /// basic block, and inserts it into the static branch info container.
  void addSuccessorInfo(DominatorAnalysis<false> &DA,
                        DominatorAnalysis<true> &PDA, SBIPtr const &SBI,
                        BinaryFunction &Function, BinaryContext &BC,
                        MCInst &Inst, BinaryBasicBlock &BB, bool Succ);

  /// extractFeatures - Extracts the feature information for each two-way branch
  /// from the program.
  void extractFeatures(BinaryFunction &Function, BinaryContext &BC);

  /// dumpSuccessorFeatures - Dumps the feature information about the target
  /// BasicBlock of either the taken or the fallthrough paths of a given branch.
  void dumpSuccessorFeatures(raw_ostream &Printer, BBIPtr &Successor);

  /// dumpFeatures - Dumps the feature information about each two-way branch
  /// from the program.
  void dumpFeatures(raw_ostream &Printer, uint64_t FunctionAddress);

  /// dumpProfileData - Dumps a limited version of the inout profile data
  /// that contains only profile for conditional branches, unconditional
  /// branches and terminators that aren't branches.
  void dumpProfileData(BinaryFunction &Function, raw_ostream &Printer);

public:
  explicit FeatureMiner(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "feature-miner"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif /* LLVM_TOOLS_LLVM_BOLT_PASSES_FEATUREMINER_H_ */
