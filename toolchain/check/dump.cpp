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

#ifndef NDEBUG

#include "toolchain/lex/dump.h"

#include <string>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/dump.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/dump.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

static auto Dump(const Context& context, SemIR::LocId loc_id) -> std::string;

LLVM_DUMP_METHOD static auto Dump(const Context& context, Lex::TokenIndex token)
    -> std::string {
  return Parse::Dump(context.parse_tree(), token);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, Parse::NodeId node_id)
    -> std::string {
  return Parse::Dump(context.parse_tree(), node_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::ClassId class_id) -> std::string {
  return SemIR::Dump(context.sem_ir(), class_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::ConstantId const_id) -> std::string {
  return SemIR::Dump(context.sem_ir(), const_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::EntityNameId entity_name_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), entity_name_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::FacetTypeId facet_type_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), facet_type_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::FunctionId function_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), function_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::GenericId generic_id) -> std::string {
  return SemIR::Dump(context.sem_ir(), generic_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::ImplId impl_id)
    -> std::string {
  RawStringOstream out;
  out << SemIR::Dump(context.sem_ir(), impl_id);
  if (!impl_id.has_value()) {
    return out.TakeStr();
  }
  const auto& impl = context.sem_ir().impls().Get(impl_id);
  out << "\nwitness loc: " << Dump(context, SemIR::LocId(impl.witness_id));
  return out.TakeStr();
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::InstBlockId inst_block_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), inst_block_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::InstId inst_id)
    -> std::string {
  RawStringOstream out;
  out << SemIR::Dump(context.sem_ir(), inst_id) << '\n'
      << "  - " << Dump(context, SemIR::LocId(inst_id));
  return out.TakeStr();
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::InterfaceId interface_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), interface_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::LocId loc_id)
    -> std::string {
  RawStringOstream out;
  // TODO: If the canonical location is None but the original is an InstId,
  // should we dump the InstId anyway even though it has no location? Is that
  // ever useful?
  loc_id = context.sem_ir().insts().GetCanonicalLocId(loc_id);
  switch (loc_id.kind()) {
    case SemIR::LocId::Kind::None: {
      out << "LocId(<none>)";
      break;
    }

    case SemIR::LocId::Kind::ImportIRInstId: {
      auto import_ir_id = context.sem_ir()
                              .import_ir_insts()
                              .Get(loc_id.import_ir_inst_id())
                              .ir_id();
      const auto* import_file =
          context.sem_ir().import_irs().Get(import_ir_id).sem_ir;
      out << "LocId(import from \"" << FormatEscaped(import_file->filename())
          << "\")";
      break;
    }

    case SemIR::LocId::Kind::NodeId: {
      auto token = context.parse_tree().node_token(loc_id.node_id());
      auto line = context.tokens().GetLineNumber(token);
      auto col = context.tokens().GetColumnNumber(token);
      const char* implicit = loc_id.is_implicit() ? " implicit" : "";
      out << "LocId(" << FormatEscaped(context.sem_ir().filename()) << ":"
          << line << ":" << col << implicit << ")";
      break;
    }

    case SemIR::LocId::Kind::InstId:
      CARBON_FATAL("unexpected LocId kind");
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::NameId name_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), name_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::NameScopeId name_scope_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), name_scope_id);
}

LLVM_DUMP_METHOD static auto Dump(
    const Context& context,
    SemIR::IdentifiedFacetTypeId identified_facet_type_id) -> std::string {
  return SemIR::Dump(context.sem_ir(), identified_facet_type_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context,
                                  SemIR::SpecificId specific_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), specific_id);
}

LLVM_DUMP_METHOD static auto Dump(
    const Context& context, SemIR::SpecificInterfaceId specific_interface_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), specific_interface_id);
}

LLVM_DUMP_METHOD static auto Dump(
    const Context& context, SemIR::StructTypeFieldsId struct_type_fields_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), struct_type_fields_id);
}

LLVM_DUMP_METHOD static auto Dump(const Context& context, SemIR::TypeId type_id)
    -> std::string {
  return SemIR::Dump(context.sem_ir(), type_id);
}

}  // namespace Carbon::Check

#endif  // NDEBUG
