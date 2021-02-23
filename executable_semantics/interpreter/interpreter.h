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
#include "executable_semantics/interpreter/assoc_list.h"
#include "executable_semantics/interpreter/cons_list.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

using Env = AssocList<std::string, Address>;

/***** Scopes *****/

struct Scope {
  Scope(Env* e, std::list<std::string> l) : env(e), locals(std::move(l)) {}
  Env* env;
  std::list<std::string> locals;
};

/***** Frames and State *****/

struct Frame {
  std::string name;
  Cons<Scope*>* scopes;
  Cons<Action*>* todo;

  Frame(std::string n, Cons<Scope*>* s, Cons<Action*>* c)
      : name(std::move(std::move(n))), scopes(s), todo(c) {}
};

struct State {
  Cons<Frame*>* stack;
  std::vector<Value*> heap;
};

extern State* state;

void PrintEnv(Env* env);
auto AllocateValue(Value* v) -> Address;
auto CopyVal(Value* val, int line_num) -> Value*;
auto ToInteger(Value* v) -> int;

/***** Interpreters *****/

auto InterpProgram(std::list<Declaration*>* fs) -> int;
auto InterpExp(Env* env, Expression* e) -> Value*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
