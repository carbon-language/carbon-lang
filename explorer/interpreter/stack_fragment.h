// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_STACK_FRAGMENT_H_
#define CARBON_EXPLORER_INTERPRETER_STACK_FRAGMENT_H_

#include <memory>
#include <vector>

#include "explorer/ast/value.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/stack.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// A continuation value behaves like a pointer to the underlying stack
// fragment, which is exposed by `Stack()`.
class StackFragment : public ContinuationValue::Representation {
 public:
  // Constructs an empty StackFragment.
  StackFragment() = default;

  // Requires *this to be empty, because by the time we're tearing down the
  // Arena, it's no longer safe to invoke ~Action.
  ~StackFragment() override;

  // Store the given partial todo stack in *this, which must currently be
  // empty. The stack is represented with the top of the stack at the
  // beginning of the vector, the reverse of the usual order.
  void StoreReversed(std::vector<std::unique_ptr<Action>> reversed_todo);

  // Restore the currently stored stack fragment to the top of `todo`,
  // leaving *this empty.
  void RestoreTo(Stack<std::unique_ptr<Action>>& todo);

  // Destroy the currently stored stack fragment.
  void Clear();

  void Print(llvm::raw_ostream& out) const override;

 private:
  // The todo stack of a suspended continuation, starting with the top
  // Action.
  std::vector<std::unique_ptr<Action>> reversed_todo_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_STACK_FRAGMENT_H_
