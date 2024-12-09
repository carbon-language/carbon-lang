// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/dump_id.h"

#include "common/check.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

auto DumpIdImpl(SemIR::LocId loc_id, const Context& context) -> void {
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

template <>
auto DumpIdMethods<Context>::WriteNewline() const -> void {
  llvm::errs() << "\n";
}

}  // namespace Carbon::Check
