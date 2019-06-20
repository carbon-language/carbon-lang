//===--- Passes/Instrumentation.h -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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

/// This is an instrumentation pass that modifies the input binary to generate
/// a profile after execution finishes. It modifies branches to increment
/// counters stored in the process memory and inserts a new function that
/// dumps this data to an fdata file.
///
/// The runtime for instrumentation has a string table that holds function
/// names. It also must include two data structures: the counter values being
/// incremented after each instrumented branch and a description of these
/// counters to be written in a file during dump. The description references
/// string indices in the string table for function names, as well as function
/// offsets locating branch source and destination. The counter values will be
/// converted to decimal form when writing the dumped fdata.
///
/// OPPORTUNITIES ON PERFORMANCE
/// This instrumentation is experimental and currently uses a naive approach
/// where every branch is instrumented. This is not ideal for runtime
/// performance, but should be good enough for us to evaluate/debug LBR profile
/// quality against instrumentation. Hopefully we can make this more efficient
/// in the future, but most optimizations here can cost a lot in BOLT processing
/// time. Keep in mind the instrumentation pass runs on every single BB of the
/// entire input binary, thus it is very expensive to do analyses, such as FLAGS
/// liveness to avoid spilling flags on every branch, if the binary is large.
///
/// MISSING: instrumentation of indirect calls
class Instrumentation {
public:
  Instrumentation() {}

  /// Modifies all functions by inserting instrumentation code (first step)
  void runOnFunctions(BinaryContext &BC);

  /// Emit data structures that will be necessary during runtime (second step)
  void emit(BinaryContext &BC, MCStreamer &Streamer);

  /// Access the function injected by the instrumentation pass necessary to
  /// write profile to a file. This is only valid after instrumentation
  /// finished (step 1).
  BinaryFunction *getDumpFunction() const {
    return DumpFunction;
  }
private:
  // Instrumented branch location information
  struct CounterDescription {
    uint32_t FromFuncStringIdx;
    uint32_t FromOffset;
    uint32_t ToFuncStringIdx;
    uint32_t ToOffset;
  };

  /// Create a new injected function that will be needed at runtime to write
  /// profile
  void createDumpFunction(BinaryContext &BC);

  /// Retrieve the string table index for the name of \p Function. We encode
  /// instrumented locations descriptions with the aid of a string table to
  /// manage memory of the instrumentation runtime in a more efficient way.
  /// If this function name is not represented in the string table yet, it will
  /// be inserted and its index returned.
  uint32_t getFunctionNameIndex(const BinaryFunction &Function);

  /// Populate all information needed to identify an instrumented location:
  /// branch source location in terms of function name plus offset, as well as
  /// branch destination (also name + offset). This will be encoded in the
  /// binary as static data and function name strings will reference a strtab.
  CounterDescription createDescription(const BinaryFunction &FromFunction,
                                       uint32_t From,
                                       const BinaryFunction &ToFunction,
                                       uint32_t To);


  /// Create the sequence of instructions to instrument a branch happening
  /// at \p FromFunction + \p FromOffset to \p ToFunc + \p ToOffset
  std::vector<MCInst> createInstrumentationSnippet(BinaryFunction &FromFunction,
                                                   uint32_t FromOffset,
                                                   BinaryFunction &ToFunc,
                                                   uint32_t ToOffset);

  /// Instrument the branch in \p Iter located at \p FromFunction + \p From,
  /// basic block \p FromBB. The destination of the branch is \p ToFunc +
  /// \p ToOffset. \p TargetBB should be non-null if this is a local branch
  /// and null if it is a call. Return true on success.
  bool instrumentOneTarget(BinaryBasicBlock::iterator &Iter,
                           BinaryFunction &FromFunction,
                           BinaryBasicBlock &FromBB, uint32_t From,
                           BinaryFunction &ToFunc, BinaryBasicBlock *TargetBB,
                           uint32_t ToOffset);

  void emitDescription(const CounterDescription &Desc, MCStreamer &Streamer);

  /// Critical edges worklist
  /// This worklist keeps track of CFG edges <From-To> that needs to be split.
  /// This task is deferred until we finish processing all BBs because we can't
  /// modify the CFG while iterating over it. For each edge, \p SplitInstrs
  /// stores the list of instrumentation instructions as a vector of MCInsts.
  /// instrumentOneTarget() populates this, runOnFunctions() consumes.
  std::vector<std::pair<BinaryBasicBlock *, BinaryBasicBlock *>> SplitWorklist;
  std::vector<std::vector<MCInst>> SplitInstrs;

  /// Stores function names, to be emitted to the runtime
  std::string StringTable;

  /// strtab indices in StringTable for each function name
  std::unordered_map<const BinaryFunction *, uint32_t> FuncToStringIdx;
  std::vector<CounterDescription> Descriptions;

  /// Identify all counters used in runtime while instrumentation is running
  std::vector<MCSymbol *> Labels;

  /// Label marking start of the memory region containing instrumentation
  /// counters, total vector size is Labels.size() 8-byte counters
  MCSymbol *Locs;

  /// Start of the vector with descriptions (one CounterDescription for each
  /// counter), vector size is Labels.size() CounterDescription-sized elmts
  MCSymbol *DescriptionsSym;

  /// Label identifying where our string table was emitted to
  MCSymbol *StringsSym;

  /// File name where profile is going to written to after target binary
  /// finishes a run
  MCSymbol *FilenameSym;

  /// Label for a string containing 8 spaces used by the algorithm that writes
  /// profile during conversion of integer to string. \p Chars stores ASCII
  /// representation of numbers from 0 to F.
  MCSymbol *Spaces;
  MCSymbol *Chars;

  /// We keep a pointer to our injected function whose final address will be
  /// needed later to patch the destructor routines in the binary to call us
  /// upon end of execution
  BinaryFunction *DumpFunction;
};

}
}

#endif
