// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/sem_ir_diagnostic_converter.h"

#include "toolchain/sem_ir/stringify_type.h"
namespace Carbon::Check {

auto SemIRDiagnosticConverter::ConvertLoc(SemIRLoc loc,
                                          ContextFnT context_fn) const
    -> DiagnosticLoc {
  // Cursors for the current IR and instruction in that IR.
  const auto* cursor_ir = sem_ir_;
  auto cursor_inst_id = SemIR::InstId::Invalid;

  // Notes an import on the diagnostic and updates cursors to point at the
  // imported IR.
  auto follow_import_ref = [&](SemIR::ImportIRInstId import_ir_inst_id) {
    auto import_ir_inst = cursor_ir->import_ir_insts().Get(import_ir_inst_id);
    const auto& import_ir = cursor_ir->import_irs().Get(import_ir_inst.ir_id);
    CARBON_CHECK(import_ir.decl_id.is_valid(),
                 "If we get invalid locations here, we may need to more "
                 "thoroughly track ImportDecls.");

    DiagnosticLoc in_import_loc;
    auto import_loc_id = cursor_ir->insts().GetLocId(import_ir.decl_id);
    if (import_loc_id.is_node_id()) {
      // For imports in the current file, the location is simple.
      in_import_loc = ConvertLocInFile(cursor_ir, import_loc_id.node_id(),
                                       loc.token_only, context_fn);
    } else if (import_loc_id.is_import_ir_inst_id()) {
      // For implicit imports, we need to unravel the location a little
      // further.
      auto implicit_import_ir_inst =
          cursor_ir->import_ir_insts().Get(import_loc_id.import_ir_inst_id());
      const auto& implicit_ir =
          cursor_ir->import_irs().Get(implicit_import_ir_inst.ir_id);
      auto implicit_loc_id =
          implicit_ir.sem_ir->insts().GetLocId(implicit_import_ir_inst.inst_id);
      CARBON_CHECK(implicit_loc_id.is_node_id(),
                   "Should only be one layer of implicit imports");
      in_import_loc =
          ConvertLocInFile(implicit_ir.sem_ir, implicit_loc_id.node_id(),
                           loc.token_only, context_fn);
    }

    // TODO: Add an "In implicit import of prelude." note for the case where we
    // don't have a location.
    if (import_loc_id.is_valid()) {
      // TODO: Include the name of the imported library in the diagnostic.
      CARBON_DIAGNOSTIC(InImport, LocationInfo, "in import");
      context_fn(in_import_loc, InImport);
    }

    cursor_ir = import_ir.sem_ir;
    cursor_inst_id = import_ir_inst.inst_id;
  };

  // If the location is is an import, follows it and returns nullopt.
  // Otherwise, it's a parse node, so return the final location.
  auto handle_loc = [&](SemIR::LocId loc_id) -> std::optional<DiagnosticLoc> {
    if (loc_id.is_import_ir_inst_id()) {
      follow_import_ref(loc_id.import_ir_inst_id());
      return std::nullopt;
    } else {
      // Parse nodes always refer to the current IR.
      return ConvertLocInFile(cursor_ir, loc_id.node_id(), loc.token_only,
                              context_fn);
    }
  };

  // Handle the base location.
  if (loc.is_inst_id) {
    cursor_inst_id = loc.inst_id;
  } else {
    if (auto diag_loc = handle_loc(loc.loc_id)) {
      return *diag_loc;
    }
    CARBON_CHECK(cursor_inst_id.is_valid(), "Should have been set");
  }

  while (true) {
    if (cursor_inst_id.is_valid()) {
      auto cursor_inst = cursor_ir->insts().Get(cursor_inst_id);
      if (auto bind_ref = cursor_inst.TryAs<SemIR::ExportDecl>();
          bind_ref && bind_ref->value_id.is_valid()) {
        cursor_inst_id = bind_ref->value_id;
        continue;
      }

      // If the parse node is valid, use it for the location.
      if (auto loc_id = cursor_ir->insts().GetLocId(cursor_inst_id);
          loc_id.is_valid()) {
        if (auto diag_loc = handle_loc(loc_id)) {
          return *diag_loc;
        }
        continue;
      }

      // If a namespace has an instruction for an import, switch to looking at
      // it.
      if (auto ns = cursor_inst.TryAs<SemIR::Namespace>()) {
        if (ns->import_id.is_valid()) {
          cursor_inst_id = ns->import_id;
          continue;
        }
      }
    }

    // Invalid parse node but not an import; just nothing to point at.
    return ConvertLocInFile(cursor_ir, Parse::NodeId::Invalid, loc.token_only,
                            context_fn);
  }
}

auto SemIRDiagnosticConverter::ConvertArg(llvm::Any arg) const -> llvm::Any {
  if (auto* library_name_id = llvm::any_cast<SemIR::LibraryNameId>(&arg)) {
    std::string library_name;
    if (*library_name_id == SemIR::LibraryNameId::Default) {
      library_name = "default library";
    } else if (!library_name_id->is_valid()) {
      library_name = "library <invalid>";
    } else {
      llvm::raw_string_ostream stream(library_name);
      stream << "library \"";
      stream << sem_ir_->string_literal_values().Get(
          library_name_id->AsStringLiteralValueId());
      stream << "\"";
    }
    return library_name;
  }
  if (auto* name_id = llvm::any_cast<SemIR::NameId>(&arg)) {
    return sem_ir_->names().GetFormatted(*name_id).str();
  }
  if (auto* type_of_expr = llvm::any_cast<TypeOfInstId>(&arg)) {
    if (!type_of_expr->inst_id.is_valid()) {
      return "<none>";
    }
    // TODO: Where possible, produce a better description of the type based on
    // the expression.
    return "`" +
           StringifyTypeExpr(
               *sem_ir_,
               sem_ir_->types().GetInstId(
                   sem_ir_->insts().Get(type_of_expr->inst_id).type_id())) +
           "`";
  }
  if (auto* type_expr = llvm::any_cast<InstIdAsType>(&arg)) {
    return "`" + StringifyTypeExpr(*sem_ir_, type_expr->inst_id) + "`";
  }
  if (auto* type_expr = llvm::any_cast<InstIdAsRawType>(&arg)) {
    return StringifyTypeExpr(*sem_ir_, type_expr->inst_id);
  }
  if (auto* type = llvm::any_cast<TypeIdAsRawType>(&arg)) {
    return StringifyTypeExpr(*sem_ir_,
                             sem_ir_->types().GetInstId(type->type_id));
  }
  if (auto* type_id = llvm::any_cast<SemIR::TypeId>(&arg)) {
    return "`" +
           StringifyTypeExpr(*sem_ir_, sem_ir_->types().GetInstId(*type_id)) +
           "`";
  }
  if (auto* typed_int = llvm::any_cast<TypedInt>(&arg)) {
    return llvm::APSInt(typed_int->value,
                        !sem_ir_->types().IsSignedInt(typed_int->type));
  }
  return DiagnosticConverter<SemIRLoc>::ConvertArg(arg);
}

auto SemIRDiagnosticConverter::ConvertLocInFile(const SemIR::File* sem_ir,
                                                Parse::NodeId node_id,
                                                bool token_only,
                                                ContextFnT context_fn) const
    -> DiagnosticLoc {
  return node_converters_[sem_ir->check_ir_id().index]->ConvertLoc(
      Parse::NodeLoc(node_id, token_only), context_fn);
}

}  // namespace Carbon::Check
