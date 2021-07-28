// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_FRAME_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_FRAME_H_

#include <list>
#include <string>

#include "common/ostream.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

using Env = Dictionary<std::string, Address>;

struct Scope {
  Scope(Env values, std::list<std::string> l)
      : values(values), locals(std::move(l)) {}
  Env values;
  std::list<std::string> locals;
};

// A frame represents either a function call or a delimited continuation.
struct Frame {
  // The name of the function.
  std::string name;
  // If the frame represents a function call, the bottom scope
  // contains the parameter-argument bindings for this function
  // call. The rest of the scopes contain local variables defined by
  // blocks within the function. The scope at the top of the stack is
  // the current scope and its environment is the one used for looking
  // up the value associated with a variable.
  Stack<Scope*> scopes;
  // The actions that need to be executed in the future of the
  // current function call. The top of the stack is the action
  // that is executed first.
  Stack<Action*> todo;
  // If this frame is the bottom frame of a continuation, then it stores
  // the address of the continuation.
  std::optional<Address> continuation;

  Frame(const Frame&) = delete;
  Frame& operator=(const Frame&) = delete;

  Frame(std::string n, Stack<Scope*> s, Stack<Action*> c)
      : name(std::move(std::move(n))), scopes(s), todo(c), continuation() {}

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::outs()); }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_FRAME_H_
