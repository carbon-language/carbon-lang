//===- Delta.h - Delta Debugging Algorithm Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for the Delta Debugging Algorithm:
// it splits a given set of Targets (i.e. Functions, Instructions, BBs, etc.)
// into chunks and tries to reduce the number chunks that are interesting.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAS_DELTA_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAS_DELTA_H

#include "TestRunner.h"
#include "llvm/ADT/ScopeExit.h"
#include <functional>
#include <utility>
#include <vector>

namespace llvm {

struct Chunk {
  int Begin;
  int End;

  /// Helper function to verify if a given Target-index is inside the Chunk
  bool contains(int Index) const { return Index >= Begin && Index <= End; }

  void print() const {
    errs() << "[" << Begin;
    if (End - Begin != 0)
      errs() << "," << End;
    errs() << "]";
  }

  /// Operator when populating CurrentChunks in Generic Delta Pass
  friend bool operator!=(const Chunk &C1, const Chunk &C2) {
    return C1.Begin != C2.Begin || C1.End != C2.End;
  }

  /// Operator used for sets
  friend bool operator<(const Chunk &C1, const Chunk &C2) {
    return std::tie(C1.Begin, C1.End) < std::tie(C2.Begin, C2.End);
  }
};

/// Provides opaque interface for querying into ChunksToKeep without having to
/// actually understand what is going on.
class Oracle {
  /// Out of all the features that we promised to be,
  /// how many have we already processed? 1-based!
  int Index = 1;

  /// The actual workhorse, contains the knowledge whether or not
  /// some particular feature should be preserved this time.
  ArrayRef<Chunk> ChunksToKeep;

public:
  explicit Oracle(ArrayRef<Chunk> ChunksToKeep) : ChunksToKeep(ChunksToKeep) {}

  /// Should be called for each feature on which we are operating.
  /// Name is self-explanatory - if returns true, then it should be preserved.
  bool shouldKeep() {
    if (ChunksToKeep.empty())
      return false; // All further features are to be discarded.

    // Does the current (front) chunk contain such a feature?
    bool ShouldKeep = ChunksToKeep.front().contains(Index);
    auto _ = make_scope_exit([&]() { ++Index; }); // Next time - next feature.

    // Is this the last feature in the chunk?
    if (ChunksToKeep.front().End == Index)
      ChunksToKeep = ChunksToKeep.drop_front(); // Onto next chunk.

    return ShouldKeep;
  }
};

/// This function implements the Delta Debugging algorithm, it receives a
/// number of Targets (e.g. Functions, Instructions, Basic Blocks, etc.) and
/// splits them in half; these chunks of targets are then tested while ignoring
/// one chunk, if a chunk is proven to be uninteresting (i.e. fails the test)
/// it is removed from consideration. The algorithm will attempt to split the
/// Chunks in half and start the process again until it can't split chunks
/// anymore.
///
/// This function is intended to be called by each specialized delta pass (e.g.
/// RemoveFunctions) and receives three key parameters:
/// * Test: The main TestRunner instance which is used to run the provided
/// interesting-ness test, as well as to store and access the reduced Program.
/// * Targets: The amount of Targets that are going to be reduced by the
/// algorithm, for example, the RemoveGlobalVars pass would send the amount of
/// initialized GVs.
/// * ExtractChunksFromModule: A function used to tailor the main program so it
/// only contains Targets that are inside Chunks of the given iteration.
/// Note: This function is implemented by each specialized Delta pass
///
/// Other implementations of the Delta Debugging algorithm can also be found in
/// the CReduce, Delta, and Lithium projects.
void runDeltaPass(TestRunner &Test, int Targets,
                  std::function<void(const std::vector<Chunk> &, Module *)>
                      ExtractChunksFromModule);
} // namespace llvm

#endif
