// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This library contains functions to assist dumping objects to stderr during
// interactive debugging. Functions named `Dump` are intended for direct use by
// developers, and should use overload resolution to determine which will be
// invoked. The debugger should do namespace resolution automatically. For
// example:
//
// - lldb: `expr Dump(tokens, id)`
// - gdb: `call Dump(tokens, id)`
//
// The `DumpNoNewline` functions are helpers that exclude a trailing newline.
// They're intended to be composed by `Dump` function implementations.

#ifndef CARBON_TOOLCHAIN_SEM_IR_DUMP_H_
#define CARBON_TOOLCHAIN_SEM_IR_DUMP_H_

#ifndef NDEBUG

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

// Just the instruction itself
auto DumpNoNewline(const File& file, InstId inst_id) -> void;

auto Dump(const File& file, ConstantId const_id) -> void;
// The instruction, its type, and its constant value (if any).
auto Dump(const File& file, InstId inst_id) -> void;
auto Dump(const File& file, NameId name_id) -> void;
auto Dump(const File& file, TypeId type_id) -> void;

}  // namespace Carbon::SemIR

#endif  // NDEBUG

#endif  // CARBON_TOOLCHAIN_SEM_IR_DUMP_H_
