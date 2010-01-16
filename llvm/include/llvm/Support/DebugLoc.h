//===---- llvm/DebugLoc.h - Debug Location Information ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used
// to describe and track debug location information.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGLOC_H
#define LLVM_DEBUGLOC_H

#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {
  class MDNode;

  /// DebugLoc - Debug location id. This is carried by SDNode and MachineInstr
  /// to index into a vector of unique debug location tuples.
  class DebugLoc {
    unsigned Idx;

  public:
    DebugLoc() : Idx(~0U) {}  // Defaults to invalid.

    static DebugLoc getUnknownLoc()   { DebugLoc L; L.Idx = ~0U; return L; }
    static DebugLoc get(unsigned idx) { DebugLoc L; L.Idx = idx; return L; }

    unsigned getIndex() const { return Idx; }

    /// isUnknown - Return true if there is no debug info for the SDNode /
    /// MachineInstr.
    bool isUnknown() const { return Idx == ~0U; }

    bool operator==(const DebugLoc &DL) const { return Idx == DL.Idx; }
    bool operator!=(const DebugLoc &DL) const { return !(*this == DL); }
  };

    /// DebugLocTracker - This class tracks debug location information.
  ///
  struct DebugLocTracker {
    /// DebugLocations - A vector of unique DebugLocTuple's.
    ///
    std::vector<MDNode *> DebugLocations;

    /// DebugIdMap - This maps DebugLocTuple's to indices into the
    /// DebugLocations vector.
    DenseMap<MDNode *, unsigned> DebugIdMap;

    DebugLocTracker() {}
  };
  
} // end namespace llvm

#endif /* LLVM_DEBUGLOC_H */
