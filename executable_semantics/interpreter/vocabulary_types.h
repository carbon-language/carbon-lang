// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_VOCABULARY_TYPES_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_VOCABULARY_TYPES_H_

#include "executable_semantics/interpreter/dictionary.h"

namespace Carbon {

struct Value;
using Address = unsigned int;
using TypeEnv = Dictionary<std::string, Value*>;
using Env = Dictionary<std::string, Address>;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_VOCABULARY_TYPES_H_
