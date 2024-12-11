// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/lex/dump_id.h"

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/dump_id.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

static auto DumpIdImpl(const Context& context, SemIR::LocId loc_id) -> void {
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

// A set of DumpId() overloads that dump an object to stderr, useful for
// calling inside a debugger.
LLVM_DUMP_METHOD auto DumpId(const Context& context, Lex::TokenIndex token)
    -> void {
  Lex::DumpId(context.tokens(), token);
}
LLVM_DUMP_METHOD auto DumpId(const Context& context, Parse::NodeId node_id)
    -> void {
  Parse::DumpId(context.parse_tree(), node_id);
}
LLVM_DUMP_METHOD auto DumpId(const Context& context, SemIR::LocId loc_id)
    -> void {
  DumpIdImpl(context, loc_id);
  llvm::errs() << '\n';
}

}  // namespace Carbon::Check

#endif  // NDEBUG
