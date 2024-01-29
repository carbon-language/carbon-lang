// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import.h"

#include "common/check.h"
#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/value_stores.h"

namespace Carbon::Check {

// Returns name information for the entity, corresponding to IDs in the import
// IR rather than the current IR. May return Invalid for a TODO.
static auto GetImportName(const SemIR::File& import_sem_ir,
                          SemIR::Inst import_inst)
    -> std::pair<SemIR::NameId, SemIR::NameScopeId> {
  switch (import_inst.kind()) {
    case SemIR::InstKind::BindName: {
      const auto& bind_name = import_sem_ir.bind_names().Get(
          import_inst.As<SemIR::BindName>().bind_name_id);
      return {bind_name.name_id, bind_name.enclosing_scope_id};
    }

    case SemIR::InstKind::ClassDecl: {
      const auto& class_info = import_sem_ir.classes().Get(
          import_inst.As<SemIR::ClassDecl>().class_id);
      return {class_info.name_id, class_info.enclosing_scope_id};
    }

    case SemIR::InstKind::FunctionDecl: {
      const auto& function = import_sem_ir.functions().Get(
          import_inst.As<SemIR::FunctionDecl>().function_id);
      return {function.name_id, function.enclosing_scope_id};
    }

    case SemIR::InstKind::InterfaceDecl: {
      const auto& interface = import_sem_ir.interfaces().Get(
          import_inst.As<SemIR::InterfaceDecl>().interface_id);
      return {interface.name_id, interface.enclosing_scope_id};
    }

    case SemIR::InstKind::Namespace: {
      const auto& scope = import_sem_ir.name_scopes().Get(
          import_inst.As<SemIR::Namespace>().name_scope_id);
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
    Context& context,
    llvm::DenseMap<SemIR::NameScopeId, SemIR::NameScopeId>& copied_namespaces,
    SemIR::ImportIRId ir_id, SemIR::InstId import_inst_id,
    SemIR::NameScopeId import_scope_id, SemIR::NameScopeId enclosing_scope_id,
    SemIR::NameId name_id, SemIR::TypeId namespace_type_id)
    -> SemIR::NameScopeId {
  auto& scope = context.name_scopes().Get(enclosing_scope_id);
  auto [it, success] = scope.names.insert({name_id, SemIR::InstId::Invalid});
  if (!success) {
    if (auto namespace_inst =
            context.insts().TryGetAs<SemIR::Namespace>(it->second)) {
      // Namespaces are open, so we can append to the existing one even if it
      // comes from a different file.
      CacheCopiedNamespace(copied_namespaces, import_scope_id,
                           namespace_inst->name_scope_id);
      return namespace_inst->name_scope_id;
    }
  }

  // Produce the namespace for the entry.
  auto ref_id = context.AddInst(SemIR::ImportRefUsed{
      .type_id = namespace_type_id, .ir_id = ir_id, .inst_id = import_inst_id});
  auto namespace_inst =
      SemIR::Namespace{namespace_type_id, SemIR::NameScopeId::Invalid, ref_id};
  // Use the invalid node because there's no node to associate with.
  auto namespace_id =
      context.AddPlaceholderInst({Parse::NodeId::Invalid, namespace_inst});
  namespace_inst.name_scope_id =
      context.name_scopes().Add(namespace_id, name_id, enclosing_scope_id);
  context.ReplaceInstBeforeConstantUse(
      namespace_id, {Parse::NodeId::Invalid, namespace_inst});

  // Diagnose if there's a name conflict, but still produce the namespace to
  // supersede the name conflict in order to avoid repeat diagnostics.
  if (!success) {
    context.DiagnoseDuplicateName(namespace_id, it->second);
  }

  it->second = namespace_id;
  CacheCopiedNamespace(copied_namespaces, import_scope_id,
                       namespace_inst.name_scope_id);
  return namespace_inst.name_scope_id;
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
        context, copied_namespaces, ir_id, import_scope.inst_id,
        import_scope_id, scope_cursor, name_id, namespace_type_id);
  }

  return scope_cursor;
}

auto Import(Context& context, SemIR::TypeId namespace_type_id,
            const SemIR::File& import_sem_ir) -> void {
  auto ir_id = context.import_irs().Add(&import_sem_ir);

  for (const auto import_inst_id :
       import_sem_ir.inst_blocks().Get(SemIR::InstBlockId::Exports)) {
    auto import_inst = import_sem_ir.insts().Get(import_inst_id);
    auto [import_name_id, import_enclosing_scope_id] =
        GetImportName(import_sem_ir, import_inst);
    // TODO: This should only be invalid when GetImportName for an inst
    // isn't yet implemented. Long-term this should be removed.
    if (!import_name_id.is_valid()) {
      continue;
    }

    llvm::DenseMap<SemIR::NameScopeId, SemIR::NameScopeId> copied_namespaces;

    auto name_id = CopyNameFromImportIR(context, import_sem_ir, import_name_id);
    SemIR::NameScopeId enclosing_scope_id = CopyEnclosingNameScopesFromImportIR(
        context, namespace_type_id, import_sem_ir, ir_id,
        import_enclosing_scope_id, copied_namespaces);

    if (auto import_namespace_inst = import_inst.TryAs<SemIR::Namespace>()) {
      // Namespaces are always imported because they're essential for
      // qualifiers, and the type is simple.
      CopySingleNameScopeFromImportIR(
          context, copied_namespaces, ir_id, import_inst_id,
          import_namespace_inst->name_scope_id, enclosing_scope_id, name_id,
          namespace_type_id);
    } else {
      // Leave a placeholder that the inst comes from the other IR.
      auto target_id = context.AddPlaceholderInst(
          {SemIR::ImportRefUnused{.ir_id = ir_id, .inst_id = import_inst_id}});
      // TODO: When importing from other packages, the scope's names should
      // be changed to allow for ambiguous names. When importing from the
      // current package, as is currently being done, we should issue a
      // diagnostic on conflicts.
      auto [it, success] = context.name_scopes()
                               .Get(enclosing_scope_id)
                               .names.insert({name_id, target_id});
      if (!success) {
        context.DiagnoseDuplicateName(target_id, it->second);
      }
    }
  }
}

}  // namespace Carbon::Check
