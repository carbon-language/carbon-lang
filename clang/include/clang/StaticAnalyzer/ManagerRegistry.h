//===-- ManagerRegistry.h - Pluggable analyzer module registry --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ManagerRegistry and Register* classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_MANAGER_REGISTRY_H
#define LLVM_CLANG_GR_MANAGER_REGISTRY_H

#include "clang/StaticAnalyzer/PathSensitive/GRState.h"

namespace clang {

namespace ento {

/// ManagerRegistry - This class records manager creators registered at
/// runtime. The information is communicated to AnalysisManager through static
/// members. Better design is expected.

class ManagerRegistry {
public:
  static StoreManagerCreator StoreMgrCreator;
  static ConstraintManagerCreator ConstraintMgrCreator;
};

/// RegisterConstraintManager - This class is used to setup the constraint
/// manager of the static analyzer. The constructor takes a creator function
/// pointer for creating the constraint manager.
///
/// It is used like this:
///
/// class MyConstraintManager {};
/// ConstraintManager* CreateMyConstraintManager(GRStateManager& statemgr) {
///  return new MyConstraintManager(statemgr);
/// }
/// RegisterConstraintManager X(CreateMyConstraintManager);

class RegisterConstraintManager {
public:
  RegisterConstraintManager(ConstraintManagerCreator CMC) {
    assert(ManagerRegistry::ConstraintMgrCreator == 0
           && "ConstraintMgrCreator already set!");
    ManagerRegistry::ConstraintMgrCreator = CMC;
  }
};

} // end GR namespace

} // end clang namespace

#endif
