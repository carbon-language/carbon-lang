//===-- DebugLoc.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation for DebugScopeTracker.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DebugLoc.h"
#include "llvm/CodeGen/MachineFunction.h"
using namespace llvm;

/// EnterDebugScope - Start a new debug scope. ScopeGV can be a DISubprogram
/// or a DIBlock.
void DebugScopeTracker::EnterDebugScope(GlobalVariable *ScopeGV,
                                        MachineFunction &MF) {
  assert(ScopeGV && "GlobalVariable for scope is null!");
  CurScope = MF.CreateDebugScope(ScopeGV, CurScope);
}

/// ExitDebugScope - "Pop" a DISubprogram or a DIBlock.
void DebugScopeTracker::ExitDebugScope(GlobalVariable *ScopeGV,
                                       MachineFunction &MF) {
  assert(ScopeGV && "GlobalVariable for scope is null!");
  assert(!CurScope.isInvalid() && "Mismatched region.end ?");
  // We may have skipped a region.end because it was in an unreachable block.
  // Go up the scope chain until we reach the scope that ScopeGV points to.
  DebugScopeInfo DSI;
  do {
    DSI =  MF.getDebugScopeInfo(CurScope);
    CurScope = DSI.Parent;
  } while (!DSI.Parent.isInvalid() && DSI.GV != ScopeGV);
}
