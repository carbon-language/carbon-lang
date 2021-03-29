// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_

#include <list>
#include <utility>
#include <vector>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

using Env = Dictionary<std::string, Address>;

/***** Scopes *****/

struct Scope {
  Scope(Env e, std::list<std::string> l) : env(e), locals(std::move(l)) {}
  Env env;
  std::list<std::string> locals;
};

/***** Frames and State *****/

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
  // Otherwise the `continuation` field is the sentinel UINT_MAX.
  Address continuation;
  // Returns whether this frame is the bottom frame of a continuation.
  auto IsContinuation() -> bool { return continuation != UINT_MAX; }

  Frame(std::string n, Stack<Scope*> s, Stack<Action*> c)
      : name(std::move(std::move(n))),
        scopes(s),
        todo(c),
        continuation(UINT_MAX) {}
};

struct State {
  Stack<Frame*> stack;
  std::vector<const Value*> heap;
  std::vector<bool> alive;
};

extern State* state;

auto PrintFrame(Frame* frame, std::ostream& out) -> void;
void PrintStack(Stack<Frame*> ls, std::ostream& out);
void PrintEnv(Env env);
auto AllocateValue(const Value* v) -> Address;
auto CopyVal(const Value* val, int line_num) -> const Value*;
auto ToInteger(const Value* v) -> int;

/***** Interpreters *****/

auto InterpProgram(std::list<Declaration>* fs) -> int;
auto InterpExp(Env env, Expression* e) -> const Value*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
