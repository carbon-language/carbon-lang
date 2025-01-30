// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/sem_ir_diagnostic_converter.h"

#include "common/raw_string_ostream.h"
#include "toolchain/sem_ir/stringify_type.h"

namespace Carbon::Check {

auto SemIRDiagnosticConverter::ConvertLoc(SemIRLoc loc,
                                          ContextFnT context_fn) const
    -> ConvertedDiagnosticLoc {
  auto converted = ConvertLocImpl(loc, context_fn);

  // Use the token when possible, but -1 is the default value.
  auto last_offset = -1;
  if (last_token_.has_value()) {
    last_offset = sem_ir_->parse_tree().tokens().GetByteOffset(last_token_);
  }

  // When the diagnostic is in the same file, we use the last possible offset;
  // otherwise, we ignore the offset because it's probably in that file.
  if (converted.loc.filename == sem_ir_->filename()) {
    converted.last_byte_offset =
        std::max(converted.last_byte_offset, last_offset);
  } else {
    converted.last_byte_offset = last_offset;
  }

  return converted;
}

auto SemIRDiagnosticConverter::ConvertLocImpl(SemIRLoc loc,
                                              ContextFnT context_fn) const
    -> ConvertedDiagnosticLoc {
  // Cursors for the current IR and instruction in that IR.
  const auto* cursor_ir = sem_ir_;
  auto cursor_inst_id = SemIR::InstId::None;

  // Notes an import on the diagnostic and updates cursors to point at the
  // imported IR.
  auto follow_import_ref = [&](SemIR::ImportIRInstId import_ir_inst_id) {
    auto import_ir_inst = cursor_ir->import_ir_insts().Get(import_ir_inst_id);
    const auto& import_ir = cursor_ir->import_irs().Get(import_ir_inst.ir_id);
    CARBON_CHECK(import_ir.decl_id.has_value(),
                 "If we get `None` locations here, we may need to more "
                 "thoroughly track ImportDecls.");

    ConvertedDiagnosticLoc in_import_loc;
    auto import_loc_id = cursor_ir->insts().GetLocId(import_ir.decl_id);
    if (import_loc_id.is_node_id()) {
      // For imports in the current file, the location is simple.
      in_import_loc = ConvertLocInFile(cursor_ir, import_loc_id.node_id(),
                                       loc.token_only_, context_fn);
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
                           loc.token_only_, context_fn);
    }

    // TODO: Add an "In implicit import of prelude." note for the case where we
    // don't have a location.
    if (import_loc_id.has_value()) {
      // TODO: Include the name of the imported library in the diagnostic.
      CARBON_DIAGNOSTIC(InImport, LocationInfo, "in import");
      context_fn(in_import_loc.loc, InImport);
    }

    cursor_ir = import_ir.sem_ir;
    cursor_inst_id = import_ir_inst.inst_id;
  };

  // If the location is is an import, follows it and returns nullopt.
  // Otherwise, it's a parse node, so return the final location.
  auto handle_loc =
      [&](SemIR::LocId loc_id) -> std::optional<ConvertedDiagnosticLoc> {
    if (loc_id.is_import_ir_inst_id()) {
      follow_import_ref(loc_id.import_ir_inst_id());
      return std::nullopt;
    } else {
      // Parse nodes always refer to the current IR.
      return ConvertLocInFile(cursor_ir, loc_id.node_id(), loc.token_only_,
                              context_fn);
    }
  };

  // Handle the base location.
  if (loc.is_inst_id_) {
    cursor_inst_id = loc.inst_id_;
  } else {
    if (auto diag_loc = handle_loc(loc.loc_id_)) {
      return *diag_loc;
    }
    CARBON_CHECK(cursor_inst_id.has_value(), "Should have been set");
  }

  while (true) {
    if (cursor_inst_id.has_value()) {
      auto cursor_inst = cursor_ir->insts().Get(cursor_inst_id);
      if (auto bind_ref = cursor_inst.TryAs<SemIR::ExportDecl>();
          bind_ref && bind_ref->value_id.has_value()) {
        cursor_inst_id = bind_ref->value_id;
        continue;
      }

      // If the parse node has a value, use it for the location.
      if (auto loc_id = cursor_ir->insts().GetLocId(cursor_inst_id);
          loc_id.has_value()) {
        if (auto diag_loc = handle_loc(loc_id)) {
          return *diag_loc;
        }
        continue;
      }

      // If a namespace has an instruction for an import, switch to looking at
      // it.
      if (auto ns = cursor_inst.TryAs<SemIR::Namespace>()) {
        if (ns->import_id.has_value()) {
          cursor_inst_id = ns->import_id;
          continue;
        }
      }
    }

    // `None` parse node but not an import; just nothing to point at.
    return ConvertLocInFile(cursor_ir, Parse::NodeId::None, loc.token_only_,
                            context_fn);
  }
}

auto SemIRDiagnosticConverter::ConvertArg(llvm::Any arg) const -> llvm::Any {
  if (auto* library_name_id = llvm::any_cast<SemIR::LibraryNameId>(&arg)) {
    std::string library_name;
    if (*library_name_id == SemIR::LibraryNameId::Default) {
      library_name = "default library";
    } else if (!library_name_id->has_value()) {
      library_name = "library <none>";
    } else {
      RawStringOstream stream;
      stream << "library \""
             << sem_ir_->string_literal_values().Get(
                    library_name_id->AsStringLiteralValueId())
             << "\"";
      library_name = stream.TakeStr();
    }
    return library_name;
  }
  if (auto* name_id = llvm::any_cast<SemIR::NameId>(&arg)) {
    return sem_ir_->names().GetFormatted(*name_id).str();
  }
  if (auto* type_of_expr = llvm::any_cast<TypeOfInstId>(&arg)) {
    if (!type_of_expr->inst_id.has_value()) {
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
                                                ContextFnT /*context_fn*/) const
    -> ConvertedDiagnosticLoc {
  const auto& tree_and_subtrees =
      imported_trees_and_subtrees_[sem_ir->check_ir_id().index]();
  return tree_and_subtrees.NodeToDiagnosticLoc(node_id, token_only);
}

}  // namespace Carbon::Check
