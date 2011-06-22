//=== DelayedCleanupPool.h - Delayed Clean-up Pool Implementation *- C++ -*===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a facility to delay calling cleanup methods until specific
// points.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DELAYEDCLEANUPPOOL_H
#define LLVM_CLANG_BASIC_DELAYEDCLEANUPPOOL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

/// \brief Gathers pairs of pointer-to-object/pointer-to-cleanup-function
/// allowing the cleanup functions to get called (with the pointer as parameter)
/// at specific points.
///
/// The use case is to simplify clean-up of certain resources that, while their
/// lifetime is well-known and restricted, cleaning them up manually is easy to
/// miss and cause a leak.
///
/// The same pointer can be added multiple times; its clean-up function will
/// only be called once.
class DelayedCleanupPool {
public:
  typedef void (*CleanupFn)(void *ptr);

  /// \brief Adds a pointer and its associated cleanup function to be called
  /// at a later point.
  ///
  /// \returns false if the pointer is already added, true otherwise.
  bool delayCleanup(void *ptr, CleanupFn fn) {
    assert(ptr && "Expected valid pointer to object");
    assert(fn && "Expected valid pointer to function");

    CleanupFn &mapFn = Ptrs[ptr];
    assert((!mapFn || mapFn == fn) &&
           "Adding a pointer with different cleanup function!");

    if (!mapFn) {
      mapFn = fn;
      Cleanups.push_back(std::make_pair(ptr, fn));
      return true;
    }

    return false;
  }

  template <typename T>
  bool delayDelete(T *ptr) {
    return delayCleanup(ptr, cleanupWithDelete<T>);
  }

  template <typename T, void (T::*Fn)()>
  bool delayMemberFunc(T *ptr) {
    return delayCleanup(ptr, cleanupWithMemberFunc<T, Fn>);
  }

  void doCleanup() {
    for (llvm::SmallVector<std::pair<void *, CleanupFn>, 8>::reverse_iterator
           I = Cleanups.rbegin(), E = Cleanups.rend(); I != E; ++I)
      I->second(I->first);
    Cleanups.clear();
    Ptrs.clear();
  }

  ~DelayedCleanupPool() {
    doCleanup();
  }

private:
  llvm::DenseMap<void *, CleanupFn> Ptrs;
  llvm::SmallVector<std::pair<void *, CleanupFn>, 8> Cleanups;

  template <typename T>
  static void cleanupWithDelete(void *ptr) {
    delete static_cast<T *>(ptr);
  }

  template <typename T, void (T::*Fn)()>
  static void cleanupWithMemberFunc(void *ptr) {
    (static_cast<T *>(ptr)->*Fn)();
  }
};

/// \brief RAII object for triggering a cleanup of a DelayedCleanupPool.
class DelayedCleanupPoint {
  DelayedCleanupPool &Pool;

public:
  DelayedCleanupPoint(DelayedCleanupPool &pool) : Pool(pool) { }

  ~DelayedCleanupPoint() {
    Pool.doCleanup();
  }
};

} // end namespace clang

#endif
