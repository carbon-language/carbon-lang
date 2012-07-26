//===- lib/CodeGen/MachineTraceMetrics.h - Super-scalar metrics -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the MachineTraceMetrics analysis pass
// that estimates CPU resource usage and critical data dependency paths through
// preferred traces. This is useful for super-scalar CPUs where execution speed
// can be limited both by data dependencies and by limited execution resources.
//
// Out-of-order CPUs will often be executing instructions from multiple basic
// blocks at the same time. This makes it difficult to estimate the resource
// usage accurately in a single basic block. Resources can be estimated better
// by looking at a trace through the current basic block.
//
// For every block, the MachineTraceMetrics pass will pick a preferred trace
// that passes through the block. The trace is chosen based on loop structure,
// branch probabilities, and resource usage. The intention is to pick likely
// traces that would be the most affected by code transformations.
//
// It is expensive to compute a full arbitrary trace for every block, so to
// save some computations, traces are chosen to be convergent. This means that
// if the traces through basic blocks A and B ever cross when moving away from
// A and B, they never diverge again. This applies in both directions - If the
// traces meet above A and B, they won't diverge when going further back.
//
// Traces tend to align with loops. The trace through a block in an inner loop
// will begin at the loop entry block and end at a back edge. If there are
// nested loops, the trace may begin and end at those instead.
//
// For each trace, we compute the critical path length, which is the number of
// cycles required to execute the trace when execution is limited by data
// dependencies only. We also compute the resource height, which is the number
// of cycles required to execute all instructions in the trace when ignoring
// data dependencies.
//
// Every instruction in the current block has a slack - the number of cycles
// execution of the instruction can be delayed without extending the critical
// path.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_TRACE_METRICS_H
#define LLVM_CODEGEN_MACHINE_TRACE_METRICS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class TargetInstrInfo;
class TargetRegisterInfo;
class MachineBasicBlock;
class MachineRegisterInfo;
class MachineLoopInfo;
class MachineLoop;
class raw_ostream;

class MachineTraceMetrics : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  const MachineLoopInfo *Loops;

public:
  class Ensemble;
  class Trace;
  static char ID;
  MachineTraceMetrics();
  void getAnalysisUsage(AnalysisUsage&) const;
  bool runOnMachineFunction(MachineFunction&);
  void releaseMemory();

  friend class Ensemble;
  friend class Trace;

  /// Per-basic block information that doesn't depend on the trace through the
  /// block.
  struct FixedBlockInfo {
    /// The number of non-trivial instructions in the block.
    /// Doesn't count PHI and COPY instructions that are likely to be removed.
    unsigned InstrCount;

    /// True when the block contains calls.
    bool HasCalls;

    FixedBlockInfo() : InstrCount(~0u), HasCalls(false) {}

    /// Returns true when resource information for this block has been computed.
    bool hasResources() const { return InstrCount != ~0u; }

    /// Invalidate resource information.
    void invalidate() { InstrCount = ~0u; }
  };

  /// Get the fixed resource information about MBB. Compute it on demand.
  const FixedBlockInfo *getResources(const MachineBasicBlock*);

  /// Per-basic block information that relates to a specific trace through the
  /// block. Convergent traces means that only one of these is required per
  /// block in a trace ensemble.
  struct TraceBlockInfo {
    /// Trace predecessor, or NULL for the first block in the trace.
    const MachineBasicBlock *Pred;

    /// Trace successor, or NULL for the last block in the trace.
    const MachineBasicBlock *Succ;

    /// Accumulated number of instructions in the trace above this block.
    /// Does not include instructions in this block.
    unsigned InstrDepth;

    /// Accumulated number of instructions in the trace below this block.
    /// Includes instructions in this block.
    unsigned InstrHeight;

    TraceBlockInfo() : Pred(0), Succ(0), InstrDepth(~0u), InstrHeight(~0u) {}

    /// Returns true if the depth resources have been computed from the trace
    /// above this block.
    bool hasValidDepth() const { return InstrDepth != ~0u; }

    /// Returns true if the height resources have been computed from the trace
    /// below this block.
    bool hasValidHeight() const { return InstrHeight != ~0u; }

    /// Invalidate depth resources when some block above this one has changed.
    void invalidateDepth() { InstrDepth = ~0u; }

    /// Invalidate height resources when a block below this one has changed.
    void invalidateHeight() { InstrHeight = ~0u; }
  };

  /// A trace represents a plausible sequence of executed basic blocks that
  /// passes through the current basic block one. The Trace class serves as a
  /// handle to internal cached data structures.
  class Trace {
    Ensemble &TE;
    TraceBlockInfo &TBI;

  public:
    explicit Trace(Ensemble &te, TraceBlockInfo &tbi) : TE(te), TBI(tbi) {}
    void print(raw_ostream&) const;

    /// Compute the total number of instructions in the trace.
    unsigned getInstrCount() const {
      return TBI.InstrDepth + TBI.InstrHeight;
    }
  };

  /// A trace ensemble is a collection of traces selected using the same
  /// strategy, for example 'minimum resource height'. There is one trace for
  /// every block in the function.
  class Ensemble {
    SmallVector<TraceBlockInfo, 4> BlockInfo;
    friend class Trace;

    void computeTrace(const MachineBasicBlock*);
    void computeDepthResources(const MachineBasicBlock*);
    void computeHeightResources(const MachineBasicBlock*);

  protected:
    MachineTraceMetrics &CT;
    virtual const MachineBasicBlock *pickTracePred(const MachineBasicBlock*) =0;
    virtual const MachineBasicBlock *pickTraceSucc(const MachineBasicBlock*) =0;
    explicit Ensemble(MachineTraceMetrics*);
    MachineLoop *getLoopFor(const MachineBasicBlock*);
    const TraceBlockInfo *getDepthResources(const MachineBasicBlock*) const;
    const TraceBlockInfo *getHeightResources(const MachineBasicBlock*) const;

  public:
    virtual ~Ensemble();
    virtual const char *getName() =0;
    void invalidate(const MachineBasicBlock *MBB);

    /// Get the trace that passes through MBB.
    /// The trace is computed on demand.
    Trace getTrace(const MachineBasicBlock *MBB);
  };

  /// Strategies for selecting traces.
  enum Strategy {
    /// Select the trace through a block that has the fewest instructions.
    TS_MinInstrCount,

    TS_NumStrategies
  };

  /// Get the trace ensemble representing the given trace selection strategy.
  /// The returned Ensemble object is owned by the MachineTraceMetrics analysis,
  /// and valid for the lifetime of the analysis pass.
  Ensemble *getEnsemble(Strategy);

  /// Invalidate cached information about MBB. This must be called *before* MBB
  /// is erased, or the CFG is otherwise changed.
  void invalidate(const MachineBasicBlock *MBB);

private:
  // One entry per basic block, indexed by block number.
  SmallVector<FixedBlockInfo, 4> BlockInfo;

  // One ensemble per strategy.
  Ensemble* Ensembles[TS_NumStrategies];
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               const MachineTraceMetrics::Trace &Tr) {
  Tr.print(OS);
  return OS;
}

} // end namespace llvm

#endif
