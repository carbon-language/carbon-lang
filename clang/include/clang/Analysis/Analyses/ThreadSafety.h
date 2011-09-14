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
// See http://clang.llvm.org/docs/LanguageExtensions.html#threadsafety for more
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_THREADSAFETY_H
#define LLVM_CLANG_THREADSAFETY_H

#include "clang/Analysis/AnalysisContext.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace thread_safety {

/// This enum distinguishes between different kinds of operations that may
/// need to be protected by locks. We use this enum in error handling.
enum ProtectedOperationKind {
  POK_VarDereference, /// Dereferencing a variable (e.g. p in *p = 5;)
  POK_VarAccess, /// Reading or writing a variable (e.g. x in x = 5;)
  POK_FunctionCall /// Making a function call (e.g. fool())
};

/// This enum distinguishes between different kinds of lock actions. For
/// example, it is an error to write a variable protected by shared version of a
/// mutex.
enum LockKind {
  LK_Shared, /// Shared/reader lock of a mutex
  LK_Exclusive /// Exclusive/writer lock of a mutex
};

/// This enum distinguishes between different ways to access (read or write) a
/// variable.
enum AccessKind {
  AK_Read, /// Reading a variable
  AK_Written /// Writing a variable
};

/// Handler class for thread safety warnings.
class ThreadSafetyHandler {
public:
  typedef llvm::StringRef Name;
  virtual ~ThreadSafetyHandler() = 0;

  /// Warn about lock expressions which fail to resolve to lockable objects.
  /// \param Loc -- the SourceLocation of the unresolved expression.
  virtual void handleInvalidLockExp(SourceLocation Loc) {}

  /// Warn about unlock function calls that do not have a prior matching lock
  /// expression.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param Loc -- The SourceLocation of the Unlock
  virtual void handleUnmatchedUnlock(Name LockName, SourceLocation Loc) {}

  /// Warn about lock function calls for locks which are already held.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param Loc -- The location of the second lock expression.
  virtual void handleDoubleLock(Name LockName, SourceLocation Loc) {}

  /// Warn about situations where a mutex is sometimes held and sometimes not.
  /// For example, a mutex is locked on an "if" branch but not the "else"
  /// branch.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param Loc -- The location of the lock expression where the mutex is
  /// locked
  virtual void handleMutexHeldEndOfScope(Name LockName, SourceLocation Loc){}

  /// Warn when a mutex is only held at the start of some loop iterations.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param Loc -- The Loc of the lock expression.
  virtual void handleNoLockLoopEntry(Name LockName, SourceLocation Loc) {}

  /// Warn when a mutex is locked but not unlocked inside a function.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param FunName -- The name of the function
  /// \param Loc -- The location of the lock expression
  virtual void handleNoUnlock(Name LockName, Name FunName,
                              SourceLocation Loc) {}

  /// Warn when a mutex is held exclusively and shared at the same point. For
  /// example, if a mutex is locked exclusively during an if branch and shared
  /// during the else branch.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param Loc1 -- The location of the first lock expression.
  /// \param Loc2 -- The location of the second lock expression.
  virtual void handleExclusiveAndShared(Name LockName, SourceLocation Loc1,
                                        SourceLocation Loc2) {}

  /// Warn when a protected operation occurs while no locks are held.
  /// \param D -- The decl for the protected variable or function
  /// \param POK -- The kind of protected operation (e.g. variable access)
  /// \param AK -- The kind of access (i.e. read or write) that occurred
  /// \param Loc -- The location of the protected operation.
  virtual void handleNoMutexHeld(const NamedDecl *D, ProtectedOperationKind POK,
                                 AccessKind AK, SourceLocation Loc) {}

  /// Warn when a protected operation occurs while the specific mutex protecting
  /// the operation is not locked.
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param D -- The decl for the protected variable or function
  /// \param POK -- The kind of protected operation (e.g. variable access)
  /// \param AK -- The kind of access (i.e. read or write) that occurred
  /// \param Loc -- The location of the protected operation.
  virtual void handleMutexNotHeld(const NamedDecl *D,
                                  ProtectedOperationKind POK, Name LockName,
                                  LockKind LK, SourceLocation Loc) {}

  /// Warn when a function is called while an excluded mutex is locked. For
  /// example, the mutex may be locked inside the function.
  /// \param FunName -- The name of the function
  /// \param LockName -- A StringRef name for the lock expression, to be printed
  /// in the error message.
  /// \param Loc -- The location of the function call.
  virtual void handleFunExcludesLock(Name FunName, Name LockName,
                                     SourceLocation Loc) {}
};

/// \brief Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void runThreadSafetyAnalysis(AnalysisContext &AC, ThreadSafetyHandler &Handler);

/// \brief Helper function that returns a LockKind required for the given level
/// of access.
LockKind getLockKindFromAccessKind(AccessKind AK);

}} // end namespace clang::thread_safety
#endif
