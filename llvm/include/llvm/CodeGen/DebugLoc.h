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

#ifndef LLVM_CODEGEN_DEBUGLOC_H
#define LLVM_CODEGEN_DEBUGLOC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <vector>

namespace llvm {

  /// DebugLocTuple - Debug location tuple of filename id, line and column.
  ///
  struct DebugLocTuple {
    unsigned FileId, Line, Col;

    DebugLocTuple(unsigned fi, unsigned l, unsigned c)
      : FileId(fi), Line(l), Col(c) {};
  };

  /// DebugLoc - Debug location id. This is carried by SDNode and
  /// MachineInstr to index into a vector of unique debug location tuples. 
  class DebugLoc {
    unsigned Idx;

  public:
    DebugLoc() : Idx(~0U) {}

    static DebugLoc getNoDebugLoc()   { DebugLoc L; L.Idx = 0;   return L; }
    static DebugLoc get(unsigned idx) { DebugLoc L; L.Idx = idx; return L; }

    bool isInvalid() { return Idx == ~0U; }
    bool isUnknown() { return Idx == 0; }
  };

  struct DebugLocTupleDenseMapInfo {
    static inline DebugLocTuple getEmptyKey() {
      return DebugLocTuple(~0U, ~0U, ~0U);
    }
    static inline DebugLocTuple getTombstoneKey() {
      return DebugLocTuple(~1U, ~1U, ~1U);
    }
    static unsigned getHashValue(const DebugLocTuple &Val) {
      return DenseMapInfo<unsigned>::getHashValue(Val.FileId) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Line) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Col);
    }
    static bool isEqual(const DebugLocTuple &LHS, const DebugLocTuple &RHS) {
      return LHS.FileId == RHS.FileId &&
             LHS.Line == RHS.Line &&
             LHS.Col == RHS.Col;
    }

    static bool isPod() { return true; }
  };

  typedef DenseMap<DebugLocTuple, unsigned, DebugLocTupleDenseMapInfo>
    DebugIdMapType;
    
  /// DebugLocTracker - This class tracks debug location information.
  ///
  struct DebugLocTracker {
    // NumFilenames - Size of the DebugFilenames vector.
    //
    unsigned NumFilenames;
    
    // DebugFilenames - A vector of unique file names.
    //
    std::vector<std::string> DebugFilenames;

    // DebugFilenamesMap - File name to DebugFilenames index map.
    //
    StringMap<unsigned> DebugFilenamesMap;

    // NumDebugLocations - Size of the DebugLocations vector.
    unsigned NumDebugLocations;

    // DebugLocations - A vector of unique DebugLocTuple's.
    //
    std::vector<DebugLocTuple> DebugLocations;

    // DebugIdsMap - This maps DebugLocTuple's to indices into
    // DebugLocations vector.
    DebugIdMapType DebugIdMap;

    DebugLocTracker() : NumFilenames(0), NumDebugLocations(0) {}

    ~DebugLocTracker() {
      NumFilenames = 0;
      DebugFilenames.clear();
      DebugFilenamesMap.clear();
      DebugLocations.clear();
      DebugIdMap.clear();
    }
  };
  
} // end namespace llvm

#endif /* LLVM_CODEGEN_DEBUGLOC_H */
