// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import.h"

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/merge.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns name information for the entity, corresponding to IDs in the import
// IR rather than the current IR.
static auto GetImportName(const SemIR::File& import_sem_ir,
                          SemIR::Inst import_inst)
    -> std::pair<SemIR::NameId, SemIR::NameScopeId> {
  CARBON_KIND_SWITCH(import_inst) {
    case SemIR::BindAlias::Kind:
    case SemIR::BindExport::Kind:
    case SemIR::BindName::Kind:
    case SemIR::BindSymbolicName::Kind: {
      auto bind_inst = import_inst.As<SemIR::AnyBindName>();
      const auto& bind_name =
          import_sem_ir.bind_names().Get(bind_inst.bind_name_id);
      return {bind_name.name_id, bind_name.enclosing_scope_id};
    }

    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      const auto& class_info = import_sem_ir.classes().Get(class_decl.class_id);
      return {class_info.name_id, class_info.enclosing_scope_id};
    }

    case CARBON_KIND(SemIR::FunctionDecl function_decl): {
      const auto& function =
          import_sem_ir.functions().Get(function_decl.function_id);
      return {function.name_id, function.enclosing_scope_id};
    }

    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      const auto& interface =
          import_sem_ir.interfaces().Get(interface_decl.interface_id);
      return {interface.name_id, interface.enclosing_scope_id};
    }

    case CARBON_KIND(SemIR::Namespace ns): {
      const auto& scope = import_sem_ir.name_scopes().Get(ns.name_scope_id);
      return {scope.name_id, scope.enclosing_scope_id};
    }

    default:
      CARBON_FATAL() << "Unsupported export kind: " << import_inst;
  }
}

// Translate the name to the current IR. It will usually be an identifier, but
// could also be a builtin name ID which is equivalent cross-IR.
static auto CopyNameFromImportIR(Context& context,
                                 const SemIR::File& import_sem_ir,
                                 SemIR::NameId import_name_id) {
  if (auto import_identifier_id = import_name_id.AsIdentifierId();
      import_identifier_id.is_valid()) {
    auto name = import_sem_ir.identifiers().Get(import_identifier_id);
    return SemIR::NameId::ForIdentifier(context.identifiers().Add(name));
  }
  return import_name_id;
}

// Adds a namespace to the IR. The bool on return is true if there was a name
// conflict. diagnose_duplicate_namespace is used when handling a cross-package
// import, where an existing namespace is in the current package and the new
// namespace is a different package.
static auto AddNamespace(
    Context& context, SemIR::TypeId namespace_type_id,
    Parse::ImportDirectiveId node_id, SemIR::NameId name_id,
    SemIR::NameScopeId enclosing_scope_id, bool diagnose_duplicate_namespace,
    std::optional<llvm::function_ref<SemIR::InstId()>> make_import_id)
    -> std::tuple<SemIR::NameScopeId, SemIR::ConstantId, bool> {
  auto& enclosing_scope = context.name_scopes().Get(enclosing_scope_id);
  auto [it, success] =
      enclosing_scope.names.insert({name_id, SemIR::InstId::Invalid});
  if (!success) {
    if (auto namespace_inst =
            context.insts().TryGetAs<SemIR::Namespace>(it->second)) {
      if (diagnose_duplicate_namespace) {
        context.DiagnoseDuplicateName(node_id, it->second);
      }
      return {namespace_inst->name_scope_id,
              context.constant_values().Get(it->second), true};
    }
  }

  auto import_id =
      make_import_id ? (*make_import_id)() : SemIR::InstId::Invalid;
  auto namespace_inst = SemIR::Namespace{
      namespace_type_id, SemIR::NameScopeId::Invalid, import_id};
  // Use the invalid node because there's no node to associate with.
  auto namespace_id = context.AddPlaceholderInst({node_id, namespace_inst});
  namespace_inst.name_scope_id =
      context.name_scopes().Add(namespace_id, name_id, enclosing_scope_id);
  context.ReplaceInstBeforeConstantUse(namespace_id, namespace_inst);

  // Diagnose if there's a name conflict, but still produce the namespace to
  // supersede the name conflict in order to avoid repeat diagnostics.
  if (!success) {
    context.DiagnoseDuplicateName(namespace_id, it->second);
  }

  it->second = namespace_id;
  return {namespace_inst.name_scope_id,
          context.constant_values().Get(namespace_id), false};
}

// Adds a copied namespace to the cache.
static auto CacheCopiedNamespace(
    llvm::DenseMap<SemIR::NameScopeId, SemIR::NameScopeId>& copied_namespaces,
    SemIR::NameScopeId import_scope_id, SemIR::NameScopeId to_scope_id)
    -> void {
  auto [it, success] = copied_namespaces.insert({import_scope_id, to_scope_id});
  CARBON_CHECK(success || it->second == to_scope_id)
      << "Copy result for namespace changed from " << import_scope_id << " to "
      << to_scope_id;
}

// Copies a namespace from the import IR, returning its ID. This may diagnose
// name conflicts, but that won't change the result because namespaces supersede
// other names in conflicts.
static auto CopySingleNameScopeFromImportIR(
    Context& context, SemIR::TypeId namespace_type_id,
    llvm::DenseMap<SemIR::NameScopeId, SemIR::NameScopeId>& copied_namespaces,
    SemIR::ImportIRId ir_id, SemIR::InstId import_inst_id,
    SemIR::NameScopeId import_scope_id, SemIR::NameScopeId enclosing_scope_id,
    SemIR::NameId name_id) -> SemIR::NameScopeId {
  // Produce the namespace for the entry.
  auto make_import_id = [&]() {
    auto bind_name_id = context.bind_names().Add(
        {.name_id = name_id,
         .enclosing_scope_id = enclosing_scope_id,
         .bind_index = SemIR::CompileTimeBindIndex::Invalid});
    auto import_ir_inst_id = context.import_ir_insts().Add(
        {.ir_id = ir_id, .inst_id = import_inst_id});
    return context.AddInst(
        {import_ir_inst_id,
         SemIR::ImportRefLoaded{.type_id = namespace_type_id,
                                .import_ir_inst_id = import_ir_inst_id,
                                .bind_name_id = bind_name_id}});
  };
  auto [namespace_scope_id, namespace_const_id, _] =
      AddNamespace(context, namespace_type_id, Parse::NodeId::Invalid, name_id,
                   enclosing_scope_id, /*diagnose_duplicate_namespace=*/false,
                   make_import_id);

  context.import_ir_constant_values()[ir_id.index].Set(import_inst_id,
                                                       namespace_const_id);

  CacheCopiedNamespace(copied_namespaces, import_scope_id, namespace_scope_id);
  return namespace_scope_id;
}

// Copies enclosing name scopes from the import IR. Handles the parent
// traversal. Returns the NameScope corresponding to the copied
// import_enclosing_scope_id.
static auto CopyEnclosingNameScopesFromImportIR(
    Context& context, SemIR::TypeId namespace_type_id,
    const SemIR::File& import_sem_ir, SemIR::ImportIRId ir_id,
    SemIR::NameScopeId import_enclosing_scope_id,
    llvm::DenseMap<SemIR::NameScopeId, SemIR::NameScopeId>& copied_namespaces)
    -> SemIR::NameScopeId {
  // Package-level names don't need work.
  if (import_enclosing_scope_id == SemIR::NameScopeId::Package) {
    return import_enclosing_scope_id;
  }

  // The scope to add namespaces to. Note this may change while looking at
  // enclosing scopes, if we encounter a namespace that's already added.
  auto scope_cursor = SemIR::NameScopeId::Package;

  // Build a stack of enclosing namespace names, with innermost first.
  llvm::SmallVector<SemIR::NameScopeId> new_namespaces;
  while (import_enclosing_scope_id != SemIR::NameScopeId::Package) {
    // If the namespace was already copied, reuse the results.
    if (auto it = copied_namespaces.find(import_enclosing_scope_id);
        it != copied_namespaces.end()) {
      // We inject names at the provided scope, and don't need to keep
      // traversing parents.
      scope_cursor = it->second;
      break;
    }

    // The namespace hasn't been copied yet, so add it to our list.
    const auto& scope =
        import_sem_ir.name_scopes().Get(import_enclosing_scope_id);
    auto scope_inst =
        import_sem_ir.insts().GetAs<SemIR::Namespace>(scope.inst_id);
    new_namespaces.push_back(scope_inst.name_scope_id);
    import_enclosing_scope_id = scope.enclosing_scope_id;
  }

  // Add enclosing namespace names, starting with the outermost.
  for (auto import_scope_id : llvm::reverse(new_namespaces)) {
    auto import_scope = import_sem_ir.name_scopes().Get(import_scope_id);
    auto name_id =
        CopyNameFromImportIR(context, import_sem_ir, import_scope.name_id);
    scope_cursor = CopySingleNameScopeFromImportIR(
        context, namespace_type_id, copied_namespaces, ir_id,
        import_scope.inst_id, import_scope_id, scope_cursor, name_id);
  }

  return scope_cursor;
}

// Returns the canonical IR inst for an entity. Returns an invalid ir_id for the
// current IR.
static auto GetCanonicalImportIRInst(Context& context,
                                     const SemIR::File* cursor_ir,
                                     SemIR::InstId cursor_inst_id)
    -> SemIR::ImportIRInst {
  for (;;) {
    auto inst = cursor_ir->insts().Get(cursor_inst_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(SemIR::BindExport bind_export): {
        cursor_inst_id = bind_export.value_id;
        continue;
      }
      case SemIR::ImportRefLoaded::Kind:
      case SemIR::ImportRefUnloaded::Kind: {
        auto import_ref = inst.As<SemIR::AnyImportRef>();
        auto import_ir_inst =
            cursor_ir->import_ir_insts().Get(import_ref.import_ir_inst_id);
        cursor_ir = cursor_ir->import_irs().Get(import_ir_inst.ir_id).sem_ir;
        cursor_inst_id = import_ir_inst.inst_id;
        continue;
      }
      default: {
        auto ir_id = SemIR::ImportIRId::Invalid;
        if (cursor_ir != &context.sem_ir()) {
          // This uses AddImportIR in case it was indirectly found, which can
          // happen with two or more steps of exports.
          ir_id = AddImportIR(context, {.node_id = Parse::NodeId::Invalid,
                                        .sem_ir = cursor_ir,
                                        .is_export = false});
        }
        return {.ir_id = ir_id, .inst_id = cursor_inst_id};
      }
    }
  }
}

// Adds an ImportRef for an entity, handling merging if needed.
static auto AddImportRefOrMerge(Context& context, SemIR::ImportIRId ir_id,
                                const SemIR::File& import_sem_ir,
                                SemIR::InstId import_inst_id,
                                SemIR::NameScopeId enclosing_scope_id,
                                SemIR::NameId name_id) -> void {
  // Leave a placeholder that the inst comes from the other IR.
  auto& names = context.name_scopes().Get(enclosing_scope_id).names;
  auto [it, success] = names.insert({name_id, SemIR::InstId::Invalid});
  if (success) {
    auto bind_name_id = context.bind_names().Add(
        {.name_id = name_id,
         .enclosing_scope_id = enclosing_scope_id,
         .bind_index = SemIR::CompileTimeBindIndex::Invalid});
    it->second = AddImportRef(
        context, {.ir_id = ir_id, .inst_id = import_inst_id}, bind_name_id);
    return;
  }

  auto prev_ir_inst =
      GetCanonicalImportIRInst(context, &context.sem_ir(), it->second);
  auto new_ir_inst =
      GetCanonicalImportIRInst(context, &import_sem_ir, import_inst_id);

  // Diagnose if the imported instructions aren't equal. However, then we need
  // to form an instruction for the duplicate diagnostic.
  if (prev_ir_inst != new_ir_inst) {
    auto conflict_id =
        AddImportRef(context, {.ir_id = ir_id, .inst_id = import_inst_id},
                     SemIR::BindNameId::Invalid);
    context.DiagnoseDuplicateName(conflict_id, it->second);
  }
}

auto ImportLibraryFromCurrentPackage(Context& context,
                                     SemIR::TypeId namespace_type_id,
                                     Parse::ImportDirectiveId node_id,
                                     const SemIR::File& import_sem_ir,
                                     bool is_export) -> void {
  auto ir_id = AddImportIR(
      context,
      {.node_id = node_id, .sem_ir = &import_sem_ir, .is_export = is_export});

  context.import_ir_constant_values()[ir_id.index].Set(
      SemIR::InstId::PackageNamespace,
      context.constant_values().Get(SemIR::InstId::PackageNamespace));

  for (const auto import_inst_id :
       import_sem_ir.inst_blocks().Get(SemIR::InstBlockId::Exports)) {
    auto import_inst = import_sem_ir.insts().Get(import_inst_id);
    auto [import_name_id, import_enclosing_scope_id] =
        GetImportName(import_sem_ir, import_inst);

    llvm::DenseMap<SemIR::NameScopeId, SemIR::NameScopeId> copied_namespaces;

    auto name_id = CopyNameFromImportIR(context, import_sem_ir, import_name_id);
    SemIR::NameScopeId enclosing_scope_id = CopyEnclosingNameScopesFromImportIR(
        context, namespace_type_id, import_sem_ir, ir_id,
        import_enclosing_scope_id, copied_namespaces);

    if (auto import_namespace_inst = import_inst.TryAs<SemIR::Namespace>()) {
      // Namespaces are always imported because they're essential for
      // qualifiers, and the type is simple.
      CopySingleNameScopeFromImportIR(
          context, namespace_type_id, copied_namespaces, ir_id, import_inst_id,
          import_namespace_inst->name_scope_id, enclosing_scope_id, name_id);
    } else {
      AddImportRefOrMerge(context, ir_id, import_sem_ir, import_inst_id,
                          enclosing_scope_id, name_id);
    }
  }

  // If an import of the current package caused an error for the imported
  // file, it transitively affects the current file too.
  if (import_sem_ir.name_scopes().Get(SemIR::NameScopeId::Package).has_error) {
    context.name_scopes().Get(SemIR::NameScopeId::Package).has_error = true;
  }
}

auto ImportLibrariesFromOtherPackage(Context& context,
                                     SemIR::TypeId namespace_type_id,
                                     Parse::ImportDirectiveId node_id,
                                     IdentifierId package_id,
                                     llvm::ArrayRef<SemIR::ImportIR> import_irs,
                                     bool has_load_error) -> void {
  CARBON_CHECK(has_load_error || !import_irs.empty())
      << "There should be either a load error or at least one IR.";

  auto name_id = SemIR::NameId::ForIdentifier(package_id);

  auto [namespace_scope_id, namespace_const_id, is_duplicate] = AddNamespace(
      context, namespace_type_id, node_id, name_id, SemIR::NameScopeId::Package,
      /*diagnose_duplicate_namespace=*/true, /*make_import_id=*/std::nullopt);

  auto& scope = context.name_scopes().Get(namespace_scope_id);
  scope.is_closed_import = !is_duplicate;
  for (auto import_ir : import_irs) {
    auto ir_id = AddImportIR(context, import_ir);
    scope.import_ir_scopes.push_back({ir_id, SemIR::NameScopeId::Package});
    context.import_ir_constant_values()[ir_id.index].Set(
        SemIR::InstId::PackageNamespace, namespace_const_id);
  }
  if (has_load_error) {
    scope.has_error = has_load_error;
  }
}

}  // namespace Carbon::Check
