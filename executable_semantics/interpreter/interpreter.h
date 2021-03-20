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

struct Frame {
  std::string name;
  Stack<Scope*> scopes;
  Stack<Action*> todo;

  Frame(std::string n, Stack<Scope*> s, Stack<Action*> c)
      : name(std::move(std::move(n))), scopes(s), todo(c) {}
};

struct State {
  Stack<Frame*> stack;
  std::vector<Value*> heap;
  std::vector<bool> alive;
};

extern State* state;

void PrintEnv(Env env);
auto AllocateValue(Value* v) -> Address;
auto CopyVal(Value* val, int line_num) -> Value*;
auto ToInteger(Value* v) -> int;

/***** Interpreters *****/

auto InterpProgram(std::list<Declaration>* fs) -> int;
auto InterpExp(Env env, Expression* e) -> Value*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
