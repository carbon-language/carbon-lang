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
  // scope.
  struct EnterDeferredDefinitionScope {
    // The suspended scope. This is only set once we reach the end of the scope.
    std::optional<DeclNameStack::SuspendedName> suspended_name;
    // Whether this scope is itself within an outer deferred definition scope.
    // If so, we'll delay processing its contents until we reach the end of the
    // parent scope. For example:
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
    bool in_deferred_definition_scope;
  };

  // A worklist task that indicates we should leave a deferred definition scope.
  struct LeaveDeferredDefinitionScope {
    // Whether this scope is within another deferred definition scope.
    bool in_deferred_definition_scope;
  };

  // A pending type-checking task.
  using Task =
      std::variant<CheckSkippedDefinition, EnterDeferredDefinitionScope,
                   LeaveDeferredDefinitionScope>;

  explicit DeferredDefinitionWorklist(llvm::raw_ostream* vlog_stream);

  // Suspend the current function definition and push a task onto the worklist
  // to finish it later.
  auto SuspendFunctionAndPush(Context& context,
                              Parse::DeferredDefinitionIndex index,
                              Parse::FunctionDefinitionStartId node_id) -> void;

  // Push a task to re-enter a function scope, so that functions defined within
  // it are type-checked in the right context.
  auto PushEnterDeferredDefinitionScope(Context& context) -> void;

  // Suspend the current deferred definition scope, which is finished but still
  // on the decl_name_stack, and push a task to leave the scope when we're
  // type-checking deferred definitions. Returns `true` if the current list of
  // deferred definitions should be type-checked immediately.
  auto SuspendFinishedScopeAndPush(Context& context) -> bool;

  // Pop the next task off the worklist.
  auto Pop() -> Task;

  // CHECK that the work list has no further work.
  auto VerifyEmpty() {
    CARBON_CHECK(worklist_.empty() && entered_scopes_.empty(),
                 "Tasks left behind on worklist.");
  }

 private:
  // A deferred definition scope that is currently still open.
  struct EnteredScope {
    // The index in worklist_ of the EnterDeferredDefinitionScope task.
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
