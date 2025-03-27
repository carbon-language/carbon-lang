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
#include "toolchain/sem_ir/dump.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

static auto DumpNoNewline(const Context& context, SemIR::LocId loc_id) -> void {
  if (!loc_id.has_value()) {
    llvm::errs() << "LocId(<none>)";
    return;
  }

  if (loc_id.is_node_id()) {
    auto token = context.parse_tree().node_token(loc_id.node_id());
    auto line = context.tokens().GetLineNumber(token);
    auto col = context.tokens().GetColumnNumber(token);
    const char* implicit = loc_id.is_implicit() ? " implicit" : "";
    llvm::errs() << "LocId(" << FormatEscaped(context.sem_ir().filename())
                 << ":" << line << ":" << col << implicit << ")";
  } else {
    CARBON_CHECK(loc_id.is_import_ir_inst_id());

    auto import_ir_id = context.sem_ir()
                            .import_ir_insts()
                            .Get(loc_id.import_ir_inst_id())
                            .ir_id;
    const auto* import_file =
        context.sem_ir().import_irs().Get(import_ir_id).sem_ir;
    llvm::errs() << "LocId(import from \""
                 << FormatEscaped(import_file->filename()) << "\")";
  }
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, Lex::TokenIndex token)
    -> void {
  Parse::Dump(context.parse_tree(), token);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, Parse::NodeId node_id)
    -> void {
  Parse::Dump(context.parse_tree(), node_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::ClassId class_id) -> void {
  SemIR::Dump(context.sem_ir(), class_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::ConstantId const_id) -> void {
  SemIR::Dump(context.sem_ir(), const_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::EntityNameId entity_name_id) -> void {
  SemIR::Dump(context.sem_ir(), entity_name_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::FacetTypeId facet_type_id) -> void {
  SemIR::Dump(context.sem_ir(), facet_type_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::FunctionId function_id) -> void {
  SemIR::Dump(context.sem_ir(), function_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::GenericId generic_id) -> void {
  SemIR::Dump(context.sem_ir(), generic_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::ImplId impl_id)
    -> void {
  SemIR::Dump(context.sem_ir(), impl_id);
  if (impl_id.has_value()) {
    const auto& impl = context.sem_ir().impls().Get(impl_id);
    auto loc_id = context.sem_ir().insts().GetLocId(impl.witness_id);
    llvm::errs() << "witness loc: ";
    DumpNoNewline(context, loc_id);
    llvm::errs() << '\n';
  }
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::InstBlockId inst_block_id) -> void {
  SemIR::Dump(context.sem_ir(), inst_block_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::InstId inst_id)
    -> void {
  SemIR::Dump(context.sem_ir(), inst_id);
  auto loc_id = context.sem_ir().insts().GetLocId(inst_id);
  llvm::errs() << "  - ";
  DumpNoNewline(context, loc_id);
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::InterfaceId interface_id) -> void {
  SemIR::Dump(context.sem_ir(), interface_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::LocId loc_id)
    -> void {
  DumpNoNewline(context, loc_id);
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::NameId name_id)
    -> void {
  SemIR::Dump(context.sem_ir(), name_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::NameScopeId name_scope_id) -> void {
  SemIR::Dump(context.sem_ir(), name_scope_id);
}

LLVM_DUMP_METHOD static auto Dump(
    const Context& context, SemIR::CompleteFacetTypeId complete_facet_type_id)
    -> void {
  SemIR::Dump(context.sem_ir(), complete_facet_type_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::SpecificId specific_id) -> void {
  SemIR::Dump(context.sem_ir(), specific_id);
}

LLVM_DUMP_METHOD static auto Dump(
    const Context& context, SemIR::SpecificInterfaceId specific_interface_id)
    -> void {
  SemIR::Dump(context.sem_ir(), specific_interface_id);
}

LLVM_DUMP_METHOD static auto Dump(
    const Context& context, SemIR::StructTypeFieldsId struct_type_fields_id)
    -> void {
  SemIR::Dump(context.sem_ir(), struct_type_fields_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::TypeBlockId type_block_id) -> void {
  SemIR::Dump(context.sem_ir(), type_block_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::TypeId type_id)
    -> void {
  SemIR::Dump(context.sem_ir(), type_id);
}

}  // namespace Carbon::Check

#endif  // NDEBUG
