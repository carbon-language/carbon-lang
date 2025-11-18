// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_LATCH_H_
#define CARBON_COMMON_LATCH_H_

#include <atomic>

#include "llvm/ADT/FunctionExtras.h"

namespace Carbon {

// A synchronization primitive similar to `std::latch` to coordinate starting
// some action once all of a set of other actions complete.
//
// Users initialize the latch (with `Init`), and receive a handle RAII object.
// This handle can be copied, and the latch is satisfied when the last copy of
// the handle returned by `Init` is destroyed.
//
// The latch synchronizes between every destruction of a handle and the
// destruction of the last handle, allowing code that runs after the latch is
// satisfied to access everything written by any thread that destroyed a handle.
// For more details of the synchronization mechanics, see the comments on `Inc`
// and `Dec` that implement this logic.
//
// This type also supports holding a closure to run when satisfied to simplify
// patterns where that body of code is easier to express at the start of work
// being synchronized instead of as each work item completes.
//
// The initialization API is separate from the constructor both for convenience
// and to enable it to provide the initial handle. This makes it easy to build
// constructively correct code where each work unit holds a handle until
// finished, including the initializer of the latch, often using by-value
// captures in a lambda that does the work.
class Latch {
 public:
  class Handle;

  Latch() = default;
  Latch(const Latch&) = delete;
  Latch(Latch&&) = delete;

  // Initialize a latch and get the initial handle to it.
  //
  // When the last copy of the returned handle is destroyed, the latch will be
  // satisfied.
  //
  // A closure may be provided which will be called when that last handle is
  // destroyed. Note that the closure will run on whatever thread executes the
  // last handle destruction. Typically, the closure here should _schedule_ the
  // next step of work on some thread pool rather than performing it directly.
  //
  // Once this method is called, it cannot be called again until all handles are
  // destroyed and the latch is satisfied. It can then be called again to get a
  // fresh handle (and provide a new closure if desired).
  auto Init(llvm::unique_function<auto()->void> on_zero = [] {}) -> Handle;

 private:
  // Increments the latch's counter.
  //
  // This is thread-safe, and may be called concurrently on multiple threads,
  // and may be called concurrently with `Dec`. However, the caller _must_ call
  // `Inc` and then `Dec`, and provide some happens-before relationship between
  // the `Inc` and `Dec`. Typically, this is done with either same-thread
  // happens-before, or because some other synchronization event such as
  // starting a thread or popping a task from a thread pool provides the
  // inter-thread happens-before relationship.
  auto Inc() -> void;

  // Decrements the latch's counter, and returns true when it reaches zero.
  //
  // This is thread-safe, and may be called concurrently with other calls to
  // `Dec` or `Inc`.
  //
  // It also ensures that all threads which call `Dec` and receive `false`
  // synchronize-with the thread that calls `Dec` and receives `true`. As a
  // consequence everything that happens-before the call to `Dec` has an
  // inter-thread happens-before for any code when `Dec` returns `true`.
  //
  // Note that there is no guarantee of inter-thread happens-before to
  // operations after a `Dec` call that returns `false`.
  auto Dec() -> bool;

  std::atomic<int> count_;
  llvm::unique_function<auto()->void> on_zero_;
};

// A copyable RAII handle around a `Latch`.
//
// When the last copy of a handle returned by `Latch::Init` is destroyed, the
// latch is considered satisfied. Copying is supported by incrementing the
// count of the latch. That increment can always be performed because it starts
// from a live handle and so the count cannot have reached zero.
//
// For more details, see the `Latch` class.
class Latch::Handle {
 public:
  Handle(const Handle& arg) : latch_(arg.latch_) {
    if (latch_) {
      arg.latch_->Inc();
    }
  }
  Handle(Handle&& arg) noexcept : latch_(std::exchange(arg.latch_, nullptr)) {}

  ~Handle() {
    if (latch_) {
      latch_->Dec();
    }
  }

  // Drops a handle explicitly, rather than waiting for it to fall out of scope.
  //
  // This also allows observing whether the underlying latch is satisfied.
  // Calls to this function synchronize with all other drops or destructions of
  // latch handles when it returns true, and only the last will return true.
  auto Drop() && -> bool {
    bool last = latch_->Dec();
    latch_ = nullptr;
    return last;
  }

 private:
  friend Latch;

  // Private constructor used by `Latch::Init` to create the initial handle for
  // a latch.
  explicit Handle(Latch* latch) : latch_(latch) { latch_->Inc(); }

  Latch* latch_ = nullptr;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_LATCH_H_
