// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_

#include <list>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/interpreter/frame.h"
#include "executable_semantics/interpreter/heap.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

using Env = Dictionary<std::string, Address>;

struct State {
  Stack<Frame*> stack;
  Heap heap;
};

extern State* state;

void InitEnv(const Declaration& d, Env* env);
void PrintStack(const Stack<Frame*>& ls, llvm::raw_ostream& out);
void PrintEnv(Env values);

/***** Interpreters *****/

auto InterpProgram(const std::list<Ptr<const Declaration>>& fs) -> int;
auto InterpExp(Env values, const Expression* e) -> const Value*;
auto InterpPattern(Env values, const Pattern* p) -> const Value*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
