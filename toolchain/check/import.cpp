// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import.h"

#include "common/check.h"
#include "common/map.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/merge.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns name information for an EntityWithParamsBase.
template <typename T>
static auto GetImportNameForEntity(const T& entity)
    -> std::pair<SemIR::NameId, SemIR::NameScopeId> {
  return {entity.name_id, entity.parent_scope_id};
}

// Returns name information for the entity, corresponding to IDs in the import
// IR rather than the current IR.
static auto GetImportName(const SemIR::File& import_sem_ir,
                          SemIR::Inst import_inst)
    -> std::pair<SemIR::NameId, SemIR::NameScopeId> {
  CARBON_KIND_SWITCH(import_inst) {
    case SemIR::BindAlias::Kind:
    case SemIR::BindName::Kind:
    case SemIR::BindSymbolicName::Kind:
    case SemIR::ExportDecl::Kind: {
      auto bind_inst = import_inst.As<SemIR::AnyBindNameOrExportDecl>();
      return GetImportNameForEntity(
          import_sem_ir.entity_names().Get(bind_inst.entity_name_id));
    }

    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      return GetImportNameForEntity(
          import_sem_ir.classes().Get(class_decl.class_id));
    }

    case CARBON_KIND(SemIR::FunctionDecl function_decl): {
      return GetImportNameForEntity(
          import_sem_ir.functions().Get(function_decl.function_id));
    }

    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      return GetImportNameForEntity(
          import_sem_ir.interfaces().Get(interface_decl.interface_id));
    }

    case CARBON_KIND(SemIR::Namespace ns): {
      return GetImportNameForEntity(
          import_sem_ir.name_scopes().Get(ns.name_scope_id));
    }

    default:
      CARBON_FATAL("Unsupported export kind: {0}", import_inst);
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

namespace {
struct NamespaceResult {
  SemIR::NameScopeId name_scope_id;
  SemIR::InstId inst_id;
  bool is_duplicate_of_namespace_in_current_package;
};
}  // namespace

// Adds a namespace to the IR. The bool on return is true if there was a name
// conflict. diagnose_duplicate_namespace is used when handling a cross-package
// import, where an existing namespace is in the current package and the new
// namespace is a different package.
static auto AddNamespace(Context& context, SemIR::TypeId namespace_type_id,
                         SemIR::NameId name_id,
                         SemIR::NameScopeId parent_scope_id,
                         bool diagnose_duplicate_namespace,
                         llvm::function_ref<SemIR::InstId()> make_import_id)
    -> NamespaceResult {
  auto* parent_scope = &context.name_scopes().Get(parent_scope_id);
  auto insert_result =
      parent_scope->name_map.Insert(name_id, parent_scope->names.size());
  if (!insert_result.is_inserted()) {
    auto prev_inst_id = parent_scope->names[insert_result.value()].inst_id;
    if (auto namespace_inst =
            context.insts().TryGetAs<SemIR::Namespace>(prev_inst_id)) {
      if (diagnose_duplicate_namespace) {
        auto import_id = make_import_id();
        CARBON_CHECK(import_id.is_valid());
        context.DiagnoseDuplicateName(import_id, prev_inst_id);
      }
      return {namespace_inst->name_scope_id, prev_inst_id, true};
    }
  }

  auto import_id = make_import_id();
  CARBON_CHECK(import_id.is_valid());
  auto import_loc_id = context.insts().GetLocId(import_id);

  auto namespace_inst = SemIR::Namespace{
      namespace_type_id, SemIR::NameScopeId::Invalid, import_id};
  auto namespace_inst_and_loc =
      import_loc_id.is_import_ir_inst_id()
          ? context.MakeImportedLocAndInst(import_loc_id.import_ir_inst_id(),
                                           namespace_inst)
          // TODO: Check that this actually is an `AnyNamespaceId`.
          : SemIR::LocIdAndInst(Parse::AnyNamespaceId(import_loc_id.node_id()),
                                namespace_inst);
  auto namespace_id =
      context.AddPlaceholderInstInNoBlock(namespace_inst_and_loc);
  context.import_ref_ids().push_back(namespace_id);
  namespace_inst.name_scope_id =
      context.name_scopes().Add(namespace_id, name_id, parent_scope_id);
  context.ReplaceInstBeforeConstantUse(namespace_id, namespace_inst);

  // Note we have to get the parent scope freshly, creating the imported
  // namespace may invalidate the pointer above.
  parent_scope = &context.name_scopes().Get(parent_scope_id);

  // Diagnose if there's a name conflict, but still produce the namespace to
  // supersede the name conflict in order to avoid repeat diagnostics.
  if (!insert_result.is_inserted()) {
    auto& entry = parent_scope->names[insert_result.value()];
    context.DiagnoseDuplicateName(namespace_id, entry.inst_id);
    entry.inst_id = namespace_id;
    entry.access_kind = SemIR::AccessKind::Public;
  } else {
    parent_scope->names.push_back({.name_id = name_id,
                                   .inst_id = namespace_id,
                                   .access_kind = SemIR::AccessKind::Public});
  }

  return {namespace_inst.name_scope_id, namespace_id, false};
}

// Adds a copied namespace to the cache.
static auto CacheCopiedNamespace(
    Map<SemIR::NameScopeId, SemIR::NameScopeId>& copied_namespaces,
    SemIR::NameScopeId import_scope_id, SemIR::NameScopeId to_scope_id)
    -> void {
  auto result = copied_namespaces.Insert(import_scope_id, to_scope_id);
  CARBON_CHECK(result.is_inserted() || result.value() == to_scope_id,
               "Copy result for namespace changed from {0} to {1}",
               import_scope_id, to_scope_id);
}

// Copies a namespace from the import IR, returning its ID. This may diagnose
// name conflicts, but that won't change the result because namespaces supersede
// other names in conflicts. The bool on return is true if there was a name
// conflict. copied_namespaces is optional.
static auto CopySingleNameScopeFromImportIR(
    Context& context, SemIR::TypeId namespace_type_id,
    Map<SemIR::NameScopeId, SemIR::NameScopeId>* copied_namespaces,
    SemIR::ImportIRId ir_id, SemIR::InstId import_inst_id,
    SemIR::NameScopeId import_scope_id, SemIR::NameScopeId parent_scope_id,
    SemIR::NameId name_id) -> NamespaceResult {
  // Produce the namespace for the entry.
  auto make_import_id = [&]() {
    auto entity_name_id = context.entity_names().Add(
        {.name_id = name_id,
         .parent_scope_id = parent_scope_id,
         .bind_index = SemIR::CompileTimeBindIndex::Invalid});
    auto import_ir_inst_id = context.import_ir_insts().Add(
        {.ir_id = ir_id, .inst_id = import_inst_id});
    auto inst_id = context.AddInstInNoBlock(
        context.MakeImportedLocAndInst<SemIR::ImportRefLoaded>(
            import_ir_inst_id, {.type_id = namespace_type_id,
                                .import_ir_inst_id = import_ir_inst_id,
                                .entity_name_id = entity_name_id}));
    context.import_ref_ids().push_back(inst_id);
    return inst_id;
  };
  NamespaceResult result =
      AddNamespace(context, namespace_type_id, name_id, parent_scope_id,
                   /*diagnose_duplicate_namespace=*/false, make_import_id);

  auto namespace_const_id = context.constant_values().Get(result.inst_id);
  context.import_ir_constant_values()[ir_id.index].Set(import_inst_id,
                                                       namespace_const_id);

  if (copied_namespaces) {
    CacheCopiedNamespace(*copied_namespaces, import_scope_id,
                         result.name_scope_id);
  }
  return result;
}

// Copies ancestor name scopes from the import IR. Handles the parent traversal.
// Returns the NameScope corresponding to the copied import_parent_scope_id.
static auto CopyAncestorNameScopesFromImportIR(
    Context& context, SemIR::TypeId namespace_type_id,
    const SemIR::File& import_sem_ir, SemIR::ImportIRId ir_id,
    SemIR::NameScopeId import_parent_scope_id,
    Map<SemIR::NameScopeId, SemIR::NameScopeId>& copied_namespaces)
    -> SemIR::NameScopeId {
  // Package-level names don't need work.
  if (import_parent_scope_id == SemIR::NameScopeId::Package) {
    return import_parent_scope_id;
  }

  // The scope to add namespaces to. Note this may change while looking at
  // parent scopes, if we encounter a namespace that's already added.
  auto scope_cursor = SemIR::NameScopeId::Package;

  // Build a stack of ancestor namespace names, with the immediate parent first.
  llvm::SmallVector<SemIR::NameScopeId> new_namespaces;
  while (import_parent_scope_id != SemIR::NameScopeId::Package) {
    // If the namespace was already copied, reuse the results.
    if (auto result = copied_namespaces.Lookup(import_parent_scope_id)) {
      // We inject names at the provided scope, and don't need to keep
      // traversing parents.
      scope_cursor = result.value();
      break;
    }

    // The namespace hasn't been copied yet, so add it to our list.
    const auto& scope = import_sem_ir.name_scopes().Get(import_parent_scope_id);
    auto scope_inst =
        import_sem_ir.insts().GetAs<SemIR::Namespace>(scope.inst_id);
    new_namespaces.push_back(scope_inst.name_scope_id);
    import_parent_scope_id = scope.parent_scope_id;
  }

  // Add ancestor namespace names, starting with the outermost.
  for (auto import_scope_id : llvm::reverse(new_namespaces)) {
    auto import_scope = import_sem_ir.name_scopes().Get(import_scope_id);
    auto name_id =
        CopyNameFromImportIR(context, import_sem_ir, import_scope.name_id);
    scope_cursor =
        CopySingleNameScopeFromImportIR(
            context, namespace_type_id, &copied_namespaces, ir_id,
            import_scope.inst_id, import_scope_id, scope_cursor, name_id)
            .name_scope_id;
  }

  return scope_cursor;
}

// Adds an ImportRef for an entity, handling merging if needed.
static auto AddImportRefOrMerge(Context& context, SemIR::ImportIRId ir_id,
                                const SemIR::File& import_sem_ir,
                                SemIR::InstId import_inst_id,
                                SemIR::NameScopeId parent_scope_id,
                                SemIR::NameId name_id) -> void {
  // Leave a placeholder that the inst comes from the other IR.
  auto& parent_scope = context.name_scopes().Get(parent_scope_id);
  auto insert = parent_scope.name_map.Insert(name_id, [&] {
    auto entity_name_id = context.entity_names().Add(
        {.name_id = name_id,
         .parent_scope_id = parent_scope_id,
         .bind_index = SemIR::CompileTimeBindIndex::Invalid});
    int index = parent_scope.names.size();
    parent_scope.names.push_back(
        {.name_id = name_id,
         .inst_id =
             AddImportRef(context, {.ir_id = ir_id, .inst_id = import_inst_id},
                          entity_name_id),
         .access_kind = SemIR::AccessKind::Public});
    return index;
  });
  if (insert.is_inserted()) {
    return;
  }

  auto inst_id = parent_scope.names[insert.value()].inst_id;
  auto prev_ir_inst =
      GetCanonicalImportIRInst(context, &context.sem_ir(), inst_id);
  VerifySameCanonicalImportIRInst(context, inst_id, prev_ir_inst, ir_id,
                                  &import_sem_ir, import_inst_id);
}

namespace {
// A scope in the API file that still needs to be copied to the implementation
// file. Only used for API file imports.
struct TodoScope {
  // The scope's instruction in the API file.
  SemIR::InstId api_inst_id;
  // The scope in the API file.
  SemIR::NameScopeId api_scope_id;
  // The already-translated scope name in the implementation file.
  SemIR::NameId impl_name_id;
  // The already-copied parent scope in the implementation file.
  SemIR::NameScopeId impl_parent_scope_id;
};
}  // namespace

// Adds an ImportRef to a name scope.
static auto AddScopedImportRef(Context& context,
                               SemIR::NameScopeId parent_scope_id,
                               SemIR::NameScope& parent_scope,
                               SemIR::NameId name_id,
                               SemIR::ImportIRInst import_inst,
                               SemIR::AccessKind access_kind) -> SemIR::InstId {
  // Add an ImportRef for other instructions.
  auto impl_entity_name_id = context.entity_names().Add(
      {.name_id = name_id,
       .parent_scope_id = parent_scope_id,
       .bind_index = SemIR::CompileTimeBindIndex::Invalid});
  auto import_ref_id = AddImportRef(context, import_inst, impl_entity_name_id);
  parent_scope.AddRequired({.name_id = name_id,
                            .inst_id = import_ref_id,
                            .access_kind = access_kind});
  return import_ref_id;
}

// Imports entries in a specific scope into the current file.
static auto ImportScopeFromApiFile(Context& context,
                                   const SemIR::File& api_sem_ir,
                                   SemIR::NameScopeId api_scope_id,
                                   SemIR::NameScopeId impl_scope_id,
                                   llvm::SmallVector<TodoScope>& todo_scopes)
    -> void {
  const auto& api_scope = api_sem_ir.name_scopes().Get(api_scope_id);
  auto& impl_scope = context.name_scopes().Get(impl_scope_id);

  for (const auto& api_entry : api_scope.names) {
    auto impl_name_id =
        CopyNameFromImportIR(context, api_sem_ir, api_entry.name_id);
    if (auto ns =
            api_sem_ir.insts().TryGetAs<SemIR::Namespace>(api_entry.inst_id)) {
      // Ignore cross-package imports. These will be handled through
      // ImportLibrariesFromOtherPackage.
      if (api_scope_id == SemIR::NameScopeId::Package) {
        const auto& ns_scope = api_sem_ir.name_scopes().Get(ns->name_scope_id);
        if (!ns_scope.import_ir_scopes.empty()) {
          continue;
        }
      }

      // Namespaces will be recursed into. Name scope creation is delayed in
      // order to avoid invalidating api_scope/impl_scope.
      todo_scopes.push_back({.api_inst_id = api_entry.inst_id,
                             .api_scope_id = ns->name_scope_id,
                             .impl_name_id = impl_name_id,
                             .impl_parent_scope_id = impl_scope_id});
    } else {
      // Add an ImportRef for other instructions.
      AddScopedImportRef(context, impl_scope_id, impl_scope, impl_name_id,
                         {.ir_id = SemIR::ImportIRId::ApiForImpl,
                          .inst_id = api_entry.inst_id},
                         api_entry.access_kind);
    }
  }
}

auto ImportApiFile(Context& context, SemIR::TypeId namespace_type_id,
                   const SemIR::File& api_sem_ir) -> void {
  context.import_ir_constant_values()[SemIR::ImportIRId::ApiForImpl.index].Set(
      SemIR::InstId::PackageNamespace,
      context.constant_values().Get(SemIR::InstId::PackageNamespace));

  llvm::SmallVector<TodoScope> todo_scopes = {};
  ImportScopeFromApiFile(context, api_sem_ir, SemIR::NameScopeId::Package,
                         SemIR::NameScopeId::Package, todo_scopes);
  while (!todo_scopes.empty()) {
    auto todo_scope = todo_scopes.pop_back_val();
    auto impl_scope_id =
        CopySingleNameScopeFromImportIR(
            context, namespace_type_id, /*copied_namespaces=*/nullptr,
            SemIR::ImportIRId::ApiForImpl, todo_scope.api_inst_id,
            todo_scope.api_scope_id, todo_scope.impl_parent_scope_id,
            todo_scope.impl_name_id)
            .name_scope_id;
    ImportScopeFromApiFile(context, api_sem_ir, todo_scope.api_scope_id,
                           impl_scope_id, todo_scopes);
  }
}

auto ImportLibrariesFromCurrentPackage(
    Context& context, SemIR::TypeId namespace_type_id,
    llvm::ArrayRef<SemIR::ImportIR> import_irs) -> void {
  for (auto import_ir : import_irs) {
    auto ir_id = AddImportIR(context, import_ir);

    context.import_ir_constant_values()[ir_id.index].Set(
        SemIR::InstId::PackageNamespace,
        context.constant_values().Get(SemIR::InstId::PackageNamespace));

    for (const auto import_inst_id :
         import_ir.sem_ir->inst_blocks().Get(SemIR::InstBlockId::Exports)) {
      auto import_inst = import_ir.sem_ir->insts().Get(import_inst_id);
      auto [import_name_id, import_parent_scope_id] =
          GetImportName(*import_ir.sem_ir, import_inst);

      Map<SemIR::NameScopeId, SemIR::NameScopeId> copied_namespaces;

      auto name_id =
          CopyNameFromImportIR(context, *import_ir.sem_ir, import_name_id);
      SemIR::NameScopeId parent_scope_id = CopyAncestorNameScopesFromImportIR(
          context, namespace_type_id, *import_ir.sem_ir, ir_id,
          import_parent_scope_id, copied_namespaces);

      if (auto import_namespace_inst = import_inst.TryAs<SemIR::Namespace>()) {
        // Namespaces are always imported because they're essential for
        // qualifiers, and the type is simple.
        CopySingleNameScopeFromImportIR(
            context, namespace_type_id, &copied_namespaces, ir_id,
            import_inst_id, import_namespace_inst->name_scope_id,
            parent_scope_id, name_id);
      } else {
        AddImportRefOrMerge(context, ir_id, *import_ir.sem_ir, import_inst_id,
                            parent_scope_id, name_id);
      }
    }

    // If an import of the current package caused an error for the imported
    // file, it transitively affects the current file too.
    if (import_ir.sem_ir->name_scopes()
            .Get(SemIR::NameScopeId::Package)
            .has_error) {
      context.name_scopes().Get(SemIR::NameScopeId::Package).has_error = true;
    }
  }
}

auto ImportLibrariesFromOtherPackage(Context& context,
                                     SemIR::TypeId namespace_type_id,
                                     SemIR::InstId import_decl_id,
                                     IdentifierId package_id,
                                     llvm::ArrayRef<SemIR::ImportIR> import_irs,
                                     bool has_load_error) -> void {
  CARBON_CHECK(has_load_error || !import_irs.empty(),
               "There should be either a load error or at least one IR.");

  auto name_id = SemIR::NameId::ForIdentifier(package_id);

  NamespaceResult result = AddNamespace(
      context, namespace_type_id, name_id, SemIR::NameScopeId::Package,
      /*diagnose_duplicate_namespace=*/true, [&] { return import_decl_id; });
  auto namespace_const_id = context.constant_values().Get(result.inst_id);

  auto& scope = context.name_scopes().Get(result.name_scope_id);
  scope.is_closed_import = !result.is_duplicate_of_namespace_in_current_package;
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

// Looks up a name in a scope imported from another package. An `identifier` is
// provided if `name_id` corresponds to an identifier in the current file;
// otherwise, `name_id` is file-agnostic and can be used directly.
static auto LookupNameInImport(const SemIR::File& import_ir,
                               SemIR::NameScopeId import_scope_id,
                               SemIR::NameId name_id,
                               llvm::StringRef identifier)
    -> const Carbon::SemIR::NameScope::Entry* {
  // Determine the NameId in the import IR.
  SemIR::NameId import_name_id = name_id;
  if (!identifier.empty()) {
    auto import_identifier_id = import_ir.identifiers().Lookup(identifier);
    if (!import_identifier_id.is_valid()) {
      // Name doesn't exist in the import IR.
      return nullptr;
    }
    import_name_id = SemIR::NameId::ForIdentifier(import_identifier_id);
  }

  // Look up the name in the import scope.
  const auto& import_scope = import_ir.name_scopes().Get(import_scope_id);
  auto lookup = import_scope.name_map.Lookup(import_name_id);
  if (!lookup) {
    // Name doesn't exist in the import scope.
    return nullptr;
  }

  const auto& import_scope_entry = import_scope.names[lookup.value()];
  if (import_scope_entry.access_kind != SemIR::AccessKind::Public) {
    // Ignore cross-package non-public names.
    return nullptr;
  }

  return &import_scope_entry;
}

// Adds a namespace that points to one in another package.
static auto AddNamespaceFromOtherPackage(Context& context,
                                         SemIR::ImportIRId import_ir_id,
                                         SemIR::InstId import_inst_id,
                                         SemIR::Namespace import_ns,
                                         SemIR::NameScopeId parent_scope_id,
                                         SemIR::NameId name_id)
    -> SemIR::InstId {
  auto namespace_type_id =
      context.GetBuiltinType(SemIR::BuiltinInstKind::NamespaceType);
  NamespaceResult result = CopySingleNameScopeFromImportIR(
      context, namespace_type_id, /*copied_namespaces=*/nullptr, import_ir_id,
      import_inst_id, import_ns.name_scope_id, parent_scope_id, name_id);
  auto& scope = context.name_scopes().Get(result.name_scope_id);
  scope.is_closed_import = !result.is_duplicate_of_namespace_in_current_package;
  scope.import_ir_scopes.push_back({import_ir_id, import_ns.name_scope_id});
  return result.inst_id;
}

auto ImportNameFromOtherPackage(
    Context& context, SemIRLoc loc, SemIR::NameScopeId scope_id,
    llvm::ArrayRef<std::pair<SemIR::ImportIRId, SemIR::NameScopeId>>
        import_ir_scopes,
    SemIR::NameId name_id) -> SemIR::InstId {
  // If the name is an identifier, get the string first so that it can be shared
  // when there are multiple IRs.
  llvm::StringRef identifier;
  if (auto identifier_id = name_id.AsIdentifierId(); identifier_id.is_valid()) {
    identifier = context.identifiers().Get(identifier_id);
    CARBON_CHECK(!identifier.empty());
  }

  // Annotate diagnostics as occurring during this name lookup.
  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InNameLookup, Note, "in name lookup for `{0}`",
                          SemIR::NameId);
        builder.Note(loc, InNameLookup, name_id);
      });

  // Although we track the result here and look in each IR, we pretty much use
  // the first result.
  auto result_id = SemIR::InstId::Invalid;
  // The canonical IR and inst_id for where `result_id` came from, which may be
  // indirectly imported. This is only resolved on a conflict, when it can be
  // used to determine the conflict is actually the same instruction.
  std::optional<SemIR::ImportIRInst> canonical_result_inst;

  for (auto [import_ir_id, import_scope_id] : import_ir_scopes) {
    auto& import_ir = context.import_irs().Get(import_ir_id);

    const auto* import_scope_entry = LookupNameInImport(
        *import_ir.sem_ir, import_scope_id, name_id, identifier);
    if (!import_scope_entry) {
      continue;
    }
    auto import_inst =
        import_ir.sem_ir->insts().Get(import_scope_entry->inst_id);
    if (import_inst.Is<SemIR::AnyImportRef>()) {
      // This entity was added to name lookup by using an import, and is not
      // exported.
      continue;
    }

    // Add the first result found.
    if (!result_id.is_valid()) {
      // If the imported instruction is a namespace, we add it directly instead
      // of as an ImportRef.
      if (auto import_ns = import_inst.TryAs<SemIR::Namespace>()) {
        result_id = AddNamespaceFromOtherPackage(context, import_ir_id,
                                                 import_scope_entry->inst_id,
                                                 *import_ns, scope_id, name_id);
      } else {
        result_id = AddScopedImportRef(
            context, scope_id, context.name_scopes().Get(scope_id), name_id,
            {.ir_id = import_ir_id, .inst_id = import_scope_entry->inst_id},
            SemIR::AccessKind::Public);
        LoadImportRef(context, result_id);
      }
      continue;
    }

    // When namespaces collide between files, merge lookup in the scopes.
    if (auto import_ns = import_inst.TryAs<SemIR::Namespace>()) {
      if (auto ns = context.insts().TryGetAs<SemIR::Namespace>(result_id)) {
        auto& name_scope = context.name_scopes().Get(ns->name_scope_id);
        name_scope.import_ir_scopes.push_back(
            {import_ir_id, import_ns->name_scope_id});
        continue;
      }
    }

    // When there's a name collision, they need to either be the same canonical
    // instruction, or we'll diagnose.
    if (!canonical_result_inst) {
      canonical_result_inst =
          GetCanonicalImportIRInst(context, &context.sem_ir(), result_id);
    }
    VerifySameCanonicalImportIRInst(context, result_id, *canonical_result_inst,
                                    import_ir_id, import_ir.sem_ir,
                                    import_scope_entry->inst_id);
  }

  return result_id;
}

}  // namespace Carbon::Check
