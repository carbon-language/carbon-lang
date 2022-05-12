//===--- Cancellation.h -------------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Cancellation mechanism for long-running tasks.
//
// This manages interactions between:
//
// 1. Client code that starts some long-running work, and maybe cancels later.
//
//   std::pair<Context, Canceler> Task = cancelableTask();
//   {
//     WithContext Cancelable(std::move(Task.first));
//     Expected
//     deepThoughtAsync([](int answer){ errs() << answer; });
//   }
//   // ...some time later...
//   if (User.fellAsleep())
//     Task.second();
//
//  (This example has an asynchronous computation, but synchronous examples
//  work similarly - the Canceler should be invoked from another thread).
//
// 2. Library code that executes long-running work, and can exit early if the
//   result is not needed.
//
//   void deepThoughtAsync(std::function<void(int)> Callback) {
//     runAsync([Callback]{
//       int A = ponder(6);
//       if (isCancelled())
//         return;
//       int B = ponder(9);
//       if (isCancelled())
//         return;
//       Callback(A * B);
//     });
//   }
//
//   (A real example may invoke the callback with an error on cancellation,
//   the CancelledError is provided for this purpose).
//
// Cancellation has some caveats:
//   - the work will only stop when/if the library code next checks for it.
//     Code outside clangd such as Sema will not do this.
//   - it's inherently racy: client code must be prepared to accept results
//     even after requesting cancellation.
//   - it's Context-based, so async work must be dispatched to threads in
//     ways that preserve the context. (Like runAsync() or TUScheduler).
//
// FIXME: We could add timestamps to isCancelled() and CancelledError.
//        Measuring the start -> cancel -> acknowledge -> finish timeline would
//        help find where libraries' cancellation should be improved.

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_CANCELLATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_CANCELLATION_H

#include "support/Context.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <system_error>

namespace clang {
namespace clangd {

/// A canceller requests cancellation of a task, when called.
/// Calling it again has no effect.
using Canceler = std::function<void()>;

/// Defines a new task whose cancellation may be requested.
/// The returned Context defines the scope of the task.
/// When the context is active, isCancelled() is 0 until the Canceler is
/// invoked, and equal to Reason afterwards.
/// Conventionally, Reason may be the LSP error code to return.
std::pair<Context, Canceler> cancelableTask(int Reason = 1);

/// If the current context is within a cancelled task, returns the reason.
/// (If the context is within multiple nested tasks, true if any are cancelled).
/// Always zero if there is no active cancelable task.
/// This isn't free (context lookup) - don't call it in a tight loop.
int isCancelled(const Context &Ctx = Context::current());

/// Conventional error when no result is returned due to cancellation.
class CancelledError : public llvm::ErrorInfo<CancelledError> {
public:
  static char ID;
  const int Reason;

  CancelledError(int Reason) : Reason(Reason) {}

  void log(llvm::raw_ostream &OS) const override {
    OS << "Task was cancelled.";
  }
  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::operation_canceled);
  }
};

} // namespace clangd
} // namespace clang

#endif
