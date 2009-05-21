//===---- llvm/CodeGen/DebugLoc.h - Debug Location Information --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used by the code
// generator to describe and track debug location information.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DEBUGLOC_H
#define LLVM_CODEGEN_DEBUGLOC_H

#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {
  class GlobalVariable;

  /// DebugLocTuple - Debug location tuple of filename id, line and column.
  ///
  struct DebugLocTuple {
    GlobalVariable *CompileUnit;
    unsigned Line, Col;

    DebugLocTuple(GlobalVariable *v, unsigned l, unsigned c)
      : CompileUnit(v), Line(l), Col(c) {};

    bool operator==(const DebugLocTuple &DLT) const {
      return CompileUnit == DLT.CompileUnit &&
             Line == DLT.Line && Col == DLT.Col;
    }
    bool operator!=(const DebugLocTuple &DLT) const {
      return !(*this == DLT);
    }
  };

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

  // Partially specialize DenseMapInfo for DebugLocTyple.
  template<>  struct DenseMapInfo<DebugLocTuple> {
    static inline DebugLocTuple getEmptyKey() {
      return DebugLocTuple(0, ~0U, ~0U);
    }
    static inline DebugLocTuple getTombstoneKey() {
      return DebugLocTuple((GlobalVariable*)~1U, ~1U, ~1U);
    }
    static unsigned getHashValue(const DebugLocTuple &Val) {
      return DenseMapInfo<GlobalVariable*>::getHashValue(Val.CompileUnit) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Line) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Col);
    }
    static bool isEqual(const DebugLocTuple &LHS, const DebugLocTuple &RHS) {
      return LHS.CompileUnit == RHS.CompileUnit &&
             LHS.Line        == RHS.Line &&
             LHS.Col         == RHS.Col;
    }

    static bool isPod() { return true; }
  };

  /// DebugLocTracker - This class tracks debug location information.
  ///
  struct DebugLocTracker {
    /// DebugLocations - A vector of unique DebugLocTuple's.
    ///
    std::vector<DebugLocTuple> DebugLocations;

    /// DebugIdMap - This maps DebugLocTuple's to indices into the
    /// DebugLocations vector.
    DenseMap<DebugLocTuple, unsigned> DebugIdMap;

    DebugLocTracker() {}
  };
  
} // end namespace llvm

#endif /* LLVM_CODEGEN_DEBUGLOC_H */
