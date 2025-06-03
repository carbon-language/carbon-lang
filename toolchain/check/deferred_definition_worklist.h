// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DEFERRED_DEFINITION_WORKLIST_H_
#define CARBON_TOOLCHAIN_CHECK_DEFERRED_DEFINITION_WORKLIST_H_

#include <optional>
#include <variant>

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/function.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Check {

// A worklist of pending tasks to perform to check deferred function definitions
// in the right order.
class DeferredDefinitionWorklist {
 public:
  // A worklist task that indicates we should check a deferred function
  // definition that we previously skipped.
  struct CheckSkippedDefinition {
    // The definition that we skipped.
    Parse::DeferredDefinitionIndex definition_index;
    // The suspended function.
    SuspendedFunction suspended_fn;
  };

  // A worklist task that indicates we should enter a nested deferred definition
  // scope. We delay processing the contents of nested deferred definition
  // scopes until we reach the end of the parent scope. For example:
  //
  // ```
  // class A {
  //   class B {
  //     fn F() -> A { return {}; }
  //   }
  // } // A.B.F is type-checked here, with A complete.
  //
  // fn F() {
  //   class C {
  //     fn G() {}
  //   } // C.G is type-checked here.
  // }
  // ```
  struct EnterNestedDeferredDefinitionScope {
    // The suspended scope. This is only set once we reach the end of the scope.
    std::optional<DeclNameStack::SuspendedName> suspended_name;
  };

  // A worklist task that indicates we should leave a nested deferred definition
  // scope.
  struct LeaveNestedDeferredDefinitionScope {};

  // A pending type-checking task.
  using Task =
      std::variant<CheckSkippedDefinition, EnterNestedDeferredDefinitionScope,
                   LeaveNestedDeferredDefinitionScope>;

  explicit DeferredDefinitionWorklist(llvm::raw_ostream* vlog_stream);

  // Suspends the current function definition and push a task onto the worklist
  // to finish it later.
  auto SuspendFunctionAndPush(Context& context,
                              Parse::DeferredDefinitionIndex index,
                              Parse::FunctionDefinitionStartId node_id) -> void;

  // Pushes a task to re-enter a function scope, so that functions defined
  // within it are type-checked in the right context. Returns whether a
  // non-nested scope was entered.
  auto PushEnterDeferredDefinitionScope(Context& context) -> bool;

  // The kind of scope that we just finished.
  enum class FinishedScopeKind {
    // We finished a nested scope. No further action is taken at this point.
    Nested,
    // We finished a non-nested scope that has no further actions to perform.
    NonNestedEmpty,
    // We finished a non-nested scope that has further actions to perform.
    NonNestedWithWork,
  };

  // Suspends the current deferred definition scope, which is finished but still
  // on the decl_name_stack, and pushes a task to leave the scope when we're
  // type-checking deferred definitions. Returns `true` if the current list of
  // deferred definitions should be type-checked immediately.
  auto SuspendFinishedScopeAndPush(Context& context) -> FinishedScopeKind;

  // Returns the current size of the worklist.
  auto size() const -> size_t { return worklist_.size(); }

  // Truncates the worklist to the given size.
  auto truncate(int new_size) -> void { worklist_.truncate(new_size); }

  // Gets the given item on the worklist.
  auto operator[](int index) -> Task& { return worklist_[index]; }

  // CHECK that the work list has no further work.
  auto VerifyEmpty() {
    CARBON_CHECK(worklist_.empty() && entered_scopes_.empty(),
                 "Tasks left behind on worklist.");
  }

 private:
  // A deferred definition scope that is currently still open.
  struct EnteredScope {
    // Whether this scope is nested immediately within the enclosing scope. If
    // so, deferred definitions are not processed at the end of this scope.
    bool nested;
    // The index in worklist_ of the first task in this scope. For a nested
    // scope, this is a EnterNestedDeferredDefinitionScope task.
    size_t worklist_start_index;
    // The corresponding lexical scope index.
    ScopeIndex scope_index;
  };

  llvm::raw_ostream* vlog_stream_;

  // A worklist of type-checking tasks we'll need to do later.
  //
  // Don't allocate any inline storage here. A Task is fairly large, so we never
  // want this to live on the stack. Instead, we reserve space in the
  // constructor for a fairly large number of deferred definitions.
  llvm::SmallVector<Task, 0> worklist_;

  // The deferred definition scopes for the current checking actions.
  llvm::SmallVector<EnteredScope> entered_scopes_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DEFERRED_DEFINITION_WORKLIST_H_
