//== ConstraintManager.cpp - Constraints on symbolic values -----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the interface to manage constraints on symbolic values.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace ento;

ConstraintManager::~ConstraintManager() {}

static DefinedSVal getLocFromSymbol(const ProgramStateRef &State,
                                    SymbolRef Sym) {
  const MemRegion *R = State->getStateManager().getRegionManager()
                                               .getSymbolicRegion(Sym);
  return loc::MemRegionVal(R);
}

/// Convenience method to query the state to see if a symbol is null or
/// not null, or neither assumption can be made.
ConditionTruthVal ConstraintManager::isNull(ProgramStateRef State,
                                            SymbolRef Sym) {
  // Disable recursive notification of clients.
  llvm::SaveAndRestore<bool> DisableNotify(NotifyAssumeClients, false);
  
  QualType Ty = Sym->getType();
  DefinedSVal V = Loc::isLocType(Ty) ? getLocFromSymbol(State, Sym)
                                     : nonloc::SymbolVal(Sym);
  const ProgramStatePair &P = assumeDual(State, V);
  if (P.first && !P.second)
    return ConditionTruthVal(false);
  if (!P.first && P.second)
    return ConditionTruthVal(true);
  return ConditionTruthVal();
}
