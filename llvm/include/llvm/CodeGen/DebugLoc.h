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
  class MachineFunction;

  /// DebugScope - Debug scope id. This is carried into DebugLocTuple to index
  /// into a vector of DebugScopeInfos. Provides the ability to associate a
  /// SDNode/MachineInstr with the debug scope that it belongs to.
  ///
  class DebugScope {
    unsigned Idx;

  public:
    DebugScope() : Idx(~0U) {}  // Defaults to null.

    static DebugScope getInvalid()   { DebugScope S; S.Idx = ~0U; return S; }
    static DebugScope get(unsigned idx) { DebugScope S; S.Idx = idx; return S; }

    unsigned getIndex() const { return Idx; }

    /// isInvalid - Return true if it doesn't refer to any scope.
    bool isInvalid() const { return Idx == ~0U; }

    bool operator==(const DebugScope &DS) const { return Idx == DS.Idx; }
    bool operator!=(const DebugScope &DS) const { return !(*this == DS); }
  };

  /// DebugScopeInfo - Contains info about the scope global variable and the
  /// parent debug scope. DebugScope is only a "cookie" that can point to a
  /// specific DebugScopeInfo.
  ///
  struct DebugScopeInfo {
    GlobalVariable *GV;
    DebugScope Parent;

    DebugScopeInfo(GlobalVariable *gv, DebugScope parent)
      : GV(gv), Parent(parent) {}
    DebugScopeInfo()
      : GV(0), Parent(DebugScope::getInvalid()) {}
  };

  /// DebugScopeTracker - Create and keep track of the debug scope while
  /// entering/exiting subprograms and blocks. Used by the instruction
  /// selectors.
  ///
  class DebugScopeTracker {
    DebugScope CurScope;

  public:
    /// getCurScope - The current debug scope that we "entered" through
    /// EnterDebugScope.
    DebugScope getCurScope() const { return CurScope; }

    /// EnterDebugScope - Start a new debug scope. ScopeGV can be a DISubprogram
    /// or a DIBlock.
    void EnterDebugScope(GlobalVariable *ScopeGV, MachineFunction &MF);

    /// ExitDebugScope - "Pop" a DISubprogram or a DIBlock.
    void ExitDebugScope(GlobalVariable *ScopeGV, MachineFunction &MF);
  };

  /// DebugLocTuple - Debug location tuple of a DICompileUnit global variable,
  /// debug scope, line and column.
  ///
  struct DebugLocTuple {
    GlobalVariable *CompileUnit;
    DebugScope Scope;
    unsigned Line, Col;

    DebugLocTuple(GlobalVariable *v, DebugScope s, unsigned l, unsigned c)
      : CompileUnit(v), Scope(s), Line(l), Col(c) {};

    bool operator==(const DebugLocTuple &DLT) const {
      return CompileUnit == DLT.CompileUnit && Scope == DLT.Scope &&
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
      return DebugLocTuple(0, DebugScope::getInvalid(), ~0U, ~0U);
    }
    static inline DebugLocTuple getTombstoneKey() {
      return DebugLocTuple((GlobalVariable*)~1U,DebugScope::get(~1U), ~1U, ~1U);
    }
    static unsigned getHashValue(const DebugLocTuple &Val) {
      return DenseMapInfo<GlobalVariable*>::getHashValue(Val.CompileUnit) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Scope.getIndex()) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Line) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Col);
    }
    static bool isEqual(const DebugLocTuple &LHS, const DebugLocTuple &RHS) {
      return LHS.CompileUnit == RHS.CompileUnit &&
             LHS.Scope       == RHS.Scope &&
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
