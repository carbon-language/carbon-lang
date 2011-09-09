//===- ThreadSafety.h ------------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// A intra-procedural analysis for thread safety (e.g. deadlocks and race
// conditions), based off of an annotation system.
//
// See http://gcc.gnu.org/wiki/ThreadSafetyAnnotation for the gcc version.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_THREADSAFETY_H
#define LLVM_CLANG_THREADSAFETY_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Sema/SemaInternal.h"

namespace clang {
namespace thread_safety {

enum ProtectedOperationKind {
  POK_VarDereference,
  POK_VarAccess,
  POK_FunctionCall
};

enum LockKind {
  LK_Shared,
  LK_Exclusive
};

enum AccessKind {
  AK_Read,
  AK_Written
};

class ThreadSafetyHandler {
public:
  typedef llvm::StringRef Name;
  ThreadSafetyHandler() {}
  virtual ~ThreadSafetyHandler() {}
  virtual void handleInvalidLockExp(SourceLocation Loc) {}
  virtual void handleUnmatchedUnlock(Name LockName, SourceLocation Loc) {}
  virtual void handleDoubleLock(Name LockName, SourceLocation Loc) {}
  virtual void handleMutexHeldEndOfScope(Name LockName, SourceLocation Loc){}
  virtual void handleNoLockLoopEntry(Name LockName, SourceLocation Loc) {}
  virtual void handleNoUnlock(Name LockName, Name FunName,
                              SourceLocation Loc) {}
  virtual void handleExclusiveAndShared(Name LockName, SourceLocation Loc1,
                                        SourceLocation Loc2) {}
  virtual void handleNoMutexHeld(const NamedDecl *D, ProtectedOperationKind POK,
                                 AccessKind AK, SourceLocation Loc) {}
  virtual void handleMutexNotHeld(const NamedDecl *D,
                                  ProtectedOperationKind POK, Name LockName,
                                  LockKind LK, SourceLocation Loc) {}
  virtual void handleFunExcludesLock(Name FunName, Name LockName,
                                     SourceLocation Loc) {}
};

void runThreadSafetyAnalysis(AnalysisContext &AC, ThreadSafetyHandler &Handler);
LockKind getLockKindFromAccessKind(AccessKind AK);

}} // end namespace clang::thread_safety
#endif
