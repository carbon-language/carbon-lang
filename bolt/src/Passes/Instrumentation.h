//===--- Passes/Instrumentation.h -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an instrumentation pass that modifies the input binary to generate
// a profile after execution finishes. It can modify branches and calls to
// increment counters stored in the process memory. A runtime library is linked
// into the final binary to handle writing these counters to an fdata file. See
// runtime/instr.cpp
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_INSTRUMENTATION_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_INSTRUMENTATION_H

#include "BinaryPasses.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {
namespace bolt {

class Instrumentation {
public:
  Instrumentation() {}

  /// Modifies all functions by inserting instrumentation code (first step)
  void runOnFunctions(BinaryContext &BC);

  /// Emit data structures that will be necessary during runtime (second step)
  void emit(BinaryContext &BC, MCStreamer &Streamer,
            const BinaryFunction &InitFunction,
            const BinaryFunction &FiniFunction);

  /// Create a non-allocatable ELF section with read-only tables necessary for
  /// writing the instrumented data profile during program finish. The runtime
  /// library needs to open the program executable file and read this data from
  /// disk, this is not loaded by the system.
  void emitTablesAsELFNote(BinaryContext &BC);

private:
  // All structs here are part of the program metadata serialization format and
  // consist of POD types or array of POD types that are trivially mapped from
  // disk to memory. This provides the runtime library with a basic
  // understanding of the program structure, so it can build a CFG for each
  // function and deduce execution counts for edges that don't require explicit
  // counters. It also provides function names and offsets used when writing the
  // fdata file.

  // Location information -- analoguous to the concept of the same name in fdata
  // writing/reading. The difference is that the name is stored as an index to a
  // string table written separately.
  struct LocDescription {
    uint32_t FuncString;
    uint32_t Offset;
  };

  // Inter-function control flow transfer instrumentation
  struct CallDescription {
    LocDescription FromLoc;
    uint32_t FromNode;  // Node refers to the CFG node index of the call site
    LocDescription ToLoc;
    uint32_t Counter;
    const BinaryFunction *Target;
  };

  // Spans multiple counters during runtime - this is an indirect call site
  struct IndCallDescription {
    LocDescription FromLoc;
  };

  // This is an indirect call target (any entry point from any function). This
  // is stored sorted in the binary for fast lookups during data writing.
  struct IndCallTargetDescription {
    LocDescription ToLoc;
    const BinaryFunction *Target;
  };

  // Intra-function control flow transfer instrumentation
  struct EdgeDescription {
    LocDescription FromLoc;
    uint32_t FromNode;
    LocDescription ToLoc;
    uint32_t ToNode;
    uint32_t Counter;
  };

  // Basic block frequency (CFG node) instrumentation - only used for spanning
  // tree leaf nodes.
  struct InstrumentedNode {
    uint32_t Node;
    uint32_t Counter;
  };

  // Entry basic blocks for a function. We record their output addresses to
  // check frequency of this address (via node number) against all tracked calls
  // to this address and discover traffic coming from uninstrumented code.
  struct EntryNode {
    uint64_t Node;
    uint64_t Address;
  };

  // Base struct organizing all metadata pertaining to a single function
  struct FunctionDescription {
    const BinaryFunction *Function;
    std::vector<InstrumentedNode> LeafNodes;
    std::vector<EdgeDescription> Edges;
    DenseSet<std::pair<uint32_t, uint32_t>> EdgesSet;
    std::vector<CallDescription> Calls;
    std::vector<EntryNode> EntryNodes;
  };

  void instrumentFunction(BinaryContext &BC, BinaryFunction &Function,
                          MCPlusBuilder::AllocatorIdTy = 0);

  /// Retrieve the string table index for the name of \p Function. We encode
  /// instrumented locations descriptions with the aid of a string table to
  /// manage memory of the instrumentation runtime in a more efficient way.
  /// If this function name is not represented in the string table yet, it will
  /// be inserted and its index returned.
  uint32_t getFunctionNameIndex(const BinaryFunction &Function);

  /// Metadata creation methods
  void createIndCallDescription(const BinaryFunction &FromFunction,
                                uint32_t From);
  void createIndCallTargetDescription(const BinaryFunction &ToFunction,
                                      uint32_t To);
  bool createCallDescription(FunctionDescription &FuncDesc,
                             const BinaryFunction &FromFunction, uint32_t From,
                             uint32_t FromNodeID,
                             const BinaryFunction &ToFunction, uint32_t To,
                             bool IsInvoke);
  bool createEdgeDescription(FunctionDescription &FuncDesc,
                             const BinaryFunction &FromFunction, uint32_t From,
                             uint32_t FromNodeID,
                             const BinaryFunction &ToFunction, uint32_t To,
                             uint32_t ToNodeID, bool Instrumented);
  void createLeafNodeDescription(FunctionDescription &FuncDesc, uint32_t Node);

  /// Create the sequence of instructions to increment a counter
  std::vector<MCInst> createInstrumentationSnippet(BinaryContext &BC,
                                                   bool IsLeaf);

  // Critical edges worklist
  // This worklist keeps track of CFG edges <From-To> that needs to be split.
  // This task is deferred until we finish processing all BBs because we can't
  // modify the CFG while iterating over it. For each edge, \p SplitInstrsTy
  // stores the list of instrumentation instructions as a vector of MCInsts.
  // instrumentOneTarget() populates this, instrumentFunction() consumes.
  using SplitWorklistTy =
      std::vector<std::pair<BinaryBasicBlock *, BinaryBasicBlock *>>;
  using SplitInstrsTy = std::vector<std::vector<MCInst>>;

  /// Instrument the branch or call in \p Iter. \p TargetBB should be non-null
  /// if this is a local branch and null if it is a call. Return true if the
  /// location was instrumented with an explicit counter or false if it just
  /// created the description, but no explicit counters were necessary.
  bool instrumentOneTarget(SplitWorklistTy &SplitWorklist,
                           SplitInstrsTy &SplitInstrs,
                           BinaryBasicBlock::iterator &Iter,
                           BinaryFunction &FromFunction,
                           BinaryBasicBlock &FromBB, uint32_t From,
                           BinaryFunction &ToFunc, BinaryBasicBlock *TargetBB,
                           uint32_t ToOffset, bool IsLeaf, bool IsInvoke,
                           FunctionDescription *FuncDesc,
                           uint32_t FromNodeID, uint32_t ToNodeID = 0);

  void instrumentLeafNode(BinaryContext &BC, BinaryBasicBlock &BB,
                          BinaryBasicBlock::iterator Iter, bool IsLeaf,
                          FunctionDescription &FuncDesc, uint32_t Node);

  void instrumentIndirectTarget(BinaryBasicBlock &BB,
                                BinaryBasicBlock::iterator &Iter,
                                BinaryFunction &FromFunction, uint32_t From);

  void createAuxiliaryFunctions(BinaryContext &BC);

  uint32_t getFDSize() const;

  /// Stores function names, to be emitted to the runtime
  std::string StringTable;

  /// strtab indices in StringTable for each function name
  std::unordered_map<const BinaryFunction *, uint32_t> FuncToStringIdx;

  /// Intra-function control flow and direct calls
  std::vector<FunctionDescription> FunctionDescriptions;
  mutable std::shared_timed_mutex FDMutex;

  /// Inter-function control flow via indirect calls
  std::vector<IndCallDescription> IndCallDescriptions;
  std::vector<IndCallTargetDescription> IndCallTargetDescriptions;

  /// Identify all counters used in runtime while instrumentation is running
  std::vector<MCSymbol *> Counters;

  /// Our runtime indirect call instrumenter function
  MCSymbol *IndCallHandlerFunc;
  MCSymbol *IndTailCallHandlerFunc;

  /// Our generated initial indirect call handler function that does nothing
  /// except calling the indirect call target. The target program starts
  /// using this no-op instrumentation function until our runtime library
  /// setup runs and installs the correct handler. We need something before
  /// our setup runs in case dyld starts running init code for other libs when
  /// we did not have time to set up our indirect call counters yet.
  BinaryFunction *InitialIndCallHandlerFunction;
  BinaryFunction *InitialIndTailCallHandlerFunction;

  /// Statistics on counters
  uint32_t DirectCallCounters{0};
  uint32_t BranchCounters{0};
  uint32_t LeafNodeCounters{0};
};

}
}

#endif
