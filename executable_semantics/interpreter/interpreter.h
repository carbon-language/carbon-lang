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
  Scope(Env values, std::list<std::string> l)
      : values(values), locals(std::move(l)) {}
  Env values;
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

// A Heap represents the abstract machine's dynamically allocated memory.
class Heap {
 public:
  // Constructs an empty Heap.
  Heap() = default;

  Heap(const Heap&) = delete;
  Heap& operator=(const Heap&) = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  auto Read(Address a, int line_num) -> const Value*;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  auto Write(Address a, const Value* v, int line_num) -> void;

  // Put the given value on the heap and mark it as alive.
  auto AllocateValue(const Value* v) -> Address;

  // Marks the object at this address, and all of its sub-objects, as dead.
  auto Deallocate(Address address) -> void;

  // Print the value at the given address to the stream `out`.
  auto PrintAddress(Address a, std::ostream& out) -> void;

  // Print all the values on the heap to the stream `out`.
  auto PrintHeap(std::ostream& out) -> void;

 private:
  // Signal an error if the address is no longer alive.
  void CheckAlive(Address address, int line_num);

  // Marks all sub-objects of this value as dead.
  void DeallocateSubObjects(const Value* val);

  std::vector<const Value*> values_;
  std::vector<bool> alive_;
};

struct State {
  Stack<Frame*> stack;
  Heap heap;
};

extern State* state;

auto PrintFrame(Frame* frame, std::ostream& out) -> void;
void PrintStack(Stack<Frame*> ls, std::ostream& out);
void PrintEnv(Env values, std::ostream& out);
auto CopyVal(const Value* val, int line_num) -> const Value*;
auto ToInteger(const Value* v) -> int;

/***** Interpreters *****/

auto InterpProgram(std::list<Declaration>* fs) -> int;
auto InterpExp(Env values, const Expression* e) -> const Value*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
