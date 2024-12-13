// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This library contains functions to assist dumping objects to stderr during
// interactive debugging. Functions named `Dump` are intended for direct use by
// developers, and should use overload resolution to determine which will be
// invoked. The debugger should do namespace resolution automatically. For
// example:
//
// - lldb: `expr Dump(context, id)`
// - gdb: `call Dump(context, id)`
//
// The `DumpNoNewline` functions are helpers that exclude a trailing newline.
// They're intended to be composed by `Dump` function implementations.

#ifndef NDEBUG

#include "toolchain/lex/dump.h"

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/dump.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

static auto DumpNoNewline(const Context& context, SemIR::LocId loc_id) -> void {
  if (!loc_id.is_valid()) {
    llvm::errs() << "LocId(invalid)";
    return;
  }

  if (loc_id.is_node_id()) {
    auto token = context.parse_tree().node_token(loc_id.node_id());
    auto line = context.tokens().GetLineNumber(token);
    auto col = context.tokens().GetColumnNumber(token);
    const char* implicit = loc_id.is_implicit() ? " implicit" : "";
    llvm::errs() << "LocId(";
    llvm::errs().write_escaped(context.sem_ir().filename());
    llvm::errs() << ":" << line << ":" << col << implicit << ")";
  } else {
    CARBON_CHECK(loc_id.is_import_ir_inst_id());

    auto import_ir_id = context.sem_ir()
                            .import_ir_insts()
                            .Get(loc_id.import_ir_inst_id())
                            .ir_id;
    const auto* import_file =
        context.sem_ir().import_irs().Get(import_ir_id).sem_ir;
    llvm::errs() << "LocId(import from \"";
    llvm::errs().write_escaped(import_file->filename());
    llvm::errs() << "\")";
  }
}

LLVM_DUMP_METHOD auto Dump(const Context& context, Lex::TokenIndex token)
    -> void {
  Parse::Dump(context.parse_tree(), token);
}

LLVM_DUMP_METHOD auto Dump(const Context& context, Parse::NodeId node_id)
    -> void {
  Parse::Dump(context.parse_tree(), node_id);
}

LLVM_DUMP_METHOD auto Dump(const Context& context, SemIR::LocId loc_id)
    -> void {
  DumpNoNewline(context, loc_id);
  llvm::errs() << '\n';
}

}  // namespace Carbon::Check

#endif  // NDEBUG
