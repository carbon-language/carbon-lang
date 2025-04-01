// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check_unit.h"

#include <string>

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_cpp.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/node_id_traversal.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"

namespace Carbon::Check {

// Returns the number of imported IRs, to assist in Context construction.
static auto GetImportedIRCount(UnitAndImports* unit_and_imports) -> int {
  int count = 0;
  for (auto& package_imports : unit_and_imports->package_imports) {
    count += package_imports.imports.size();
  }
  if (!unit_and_imports->api_for_impl) {
    // Leave an empty slot for ImportIRId::ApiForImpl.
    ++count;
  }
  return count;
}

CheckUnit::CheckUnit(
    UnitAndImports* unit_and_imports,
    llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::raw_ostream* vlog_stream)
    : unit_and_imports_(unit_and_imports),
      total_ir_count_(tree_and_subtrees_getters.size()),
      fs_(std::move(fs)),
      vlog_stream_(vlog_stream),
      emitter_(&unit_and_imports_->err_tracker, tree_and_subtrees_getters,
               unit_and_imports_->unit->sem_ir),
      context_(&emitter_, unit_and_imports_->unit->tree_and_subtrees_getter,
               unit_and_imports_->unit->sem_ir,
               GetImportedIRCount(unit_and_imports),
               tree_and_subtrees_getters.size(), vlog_stream) {}

auto CheckUnit::Run() -> void {
  Timings::ScopedTiming timing(unit_and_imports_->unit->timings, "check");

  // We can safely mark this as checked at the start.
  unit_and_imports_->is_checked = true;

  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context_.PrintForStackDump(output); });

  // Add a block for the file.
  context_.inst_block_stack().Push();

  // TODO: Remove this and the pop in `FinishRun` once we properly push and pop
  // in the right places.
  context_.generic_region_stack().Push();

  InitPackageScopeAndImports();

  // Eagerly import the impls declared in the api file to prepare to redeclare
  // them.
  ImportImplsFromApiFile(context_);

  if (!ProcessNodeIds()) {
    context_.sem_ir().set_has_errors(true);
    return;
  }

  FinishRun();
}

auto CheckUnit::InitPackageScopeAndImports() -> void {
  // Importing makes many namespaces, so only canonicalize the type once.
  auto namespace_type_id =
      GetSingletonType(context_, SemIR::NamespaceType::SingletonInstId);

  // Define the package scope, with an instruction for `package` expressions to
  // reference.
  auto package_scope_id = context_.name_scopes().Add(
      SemIR::Namespace::PackageInstId, SemIR::NameId::PackageNamespace,
      SemIR::NameScopeId::None);
  CARBON_CHECK(package_scope_id == SemIR::NameScopeId::Package);

  auto package_inst_id =
      AddInst<SemIR::Namespace>(context_, Parse::NodeId::None,
                                {.type_id = namespace_type_id,
                                 .name_scope_id = SemIR::NameScopeId::Package,
                                 .import_id = SemIR::InstId::None});
  CARBON_CHECK(package_inst_id == SemIR::Namespace::PackageInstId);

  // If there is an implicit `api` import, set it first so that it uses the
  // ImportIRId::ApiForImpl when processed for imports.
  if (unit_and_imports_->api_for_impl) {
    const auto& names = context_.parse_tree().packaging_decl()->names;
    auto import_decl_id = AddInst<SemIR::ImportDecl>(
        context_, names.node_id,
        {.package_id = SemIR::NameId::ForPackageName(names.package_id)});
    SetApiImportIR(context_,
                   {.decl_id = import_decl_id,
                    .is_export = false,
                    .sem_ir = unit_and_imports_->api_for_impl->unit->sem_ir});
  } else {
    SetApiImportIR(context_,
                   {.decl_id = SemIR::InstId::None, .sem_ir = nullptr});
  }

  // Add import instructions for everything directly imported. Implicit imports
  // are handled separately.
  for (auto& package_imports : unit_and_imports_->package_imports) {
    CARBON_CHECK(!package_imports.import_decl_id.has_value());
    package_imports.import_decl_id = AddInst<SemIR::ImportDecl>(
        context_, package_imports.node_id,
        {.package_id =
             SemIR::NameId::ForPackageName(package_imports.package_id)});
  }
  // Process the imports.
  if (unit_and_imports_->api_for_impl) {
    ImportApiFile(context_, namespace_type_id,
                  *unit_and_imports_->api_for_impl->unit->sem_ir);
  }
  ImportCurrentPackage(package_inst_id, namespace_type_id);
  CARBON_CHECK(context_.scope_stack().PeekIndex() == ScopeIndex::Package);
  ImportOtherPackages(namespace_type_id);

  const auto& cpp_import_names = unit_and_imports_->cpp_import_names;
  if (!cpp_import_names.empty()) {
    auto* cpp_ast = unit_and_imports_->unit->cpp_ast;
    CARBON_CHECK(cpp_ast);
    CARBON_CHECK(!cpp_ast->get());
    *cpp_ast =
        ImportCppFiles(context_, unit_and_imports_->unit->sem_ir->filename(),
                       cpp_import_names, fs_);
  }
}

auto CheckUnit::CollectDirectImports(
    llvm::SmallVector<SemIR::ImportIR>& results,
    llvm::MutableArrayRef<int> ir_to_result_index, SemIR::InstId import_decl_id,
    const PackageImports& imports, bool is_local) -> void {
  for (const auto& import : imports.imports) {
    const auto& direct_ir = *import.unit_info->unit->sem_ir;
    auto& index = ir_to_result_index[direct_ir.check_ir_id().index];
    if (index != -1) {
      // This should only happen when doing API imports for an implementation
      // file. Don't change the entry; is_export doesn't matter.
      continue;
    }
    index = results.size();
    results.push_back({.decl_id = import_decl_id,
                       // Only tag exports in API files, ignoring the value in
                       // implementation files.
                       .is_export = is_local && import.names.is_export,
                       .sem_ir = &direct_ir});
  }
}

auto CheckUnit::CollectTransitiveImports(SemIR::InstId import_decl_id,
                                         const PackageImports* local_imports,
                                         const PackageImports* api_imports)
    -> llvm::SmallVector<SemIR::ImportIR> {
  llvm::SmallVector<SemIR::ImportIR> results;

  // Track whether an IR was imported in full, including `export import`. This
  // distinguishes from IRs that are indirectly added without all names being
  // exported to this IR.
  llvm::SmallVector<int> ir_to_result_index(total_ir_count_, -1);

  // First add direct imports. This means that if an entity is imported both
  // directly and indirectly, the import path will reflect the direct import.
  if (local_imports) {
    CollectDirectImports(results, ir_to_result_index, import_decl_id,
                         *local_imports,
                         /*is_local=*/true);
  }
  if (api_imports) {
    CollectDirectImports(results, ir_to_result_index, import_decl_id,
                         *api_imports,
                         /*is_local=*/false);
  }

  // Loop through direct imports for any indirect exports. The underlying vector
  // is appended during iteration, so take the size first.
  const int direct_imports = results.size();
  for (int direct_index : llvm::seq(direct_imports)) {
    bool is_export = results[direct_index].is_export;

    for (const auto& indirect_ir :
         results[direct_index].sem_ir->import_irs().array_ref()) {
      if (!indirect_ir.is_export) {
        continue;
      }

      auto& indirect_index =
          ir_to_result_index[indirect_ir.sem_ir->check_ir_id().index];
      if (indirect_index == -1) {
        indirect_index = results.size();
        // TODO: In the case of a recursive `export import`, this only points at
        // the outermost import. May want something that better reflects the
        // recursion.
        results.push_back({.decl_id = results[direct_index].decl_id,
                           .is_export = is_export,
                           .sem_ir = indirect_ir.sem_ir});
      } else if (is_export) {
        results[indirect_index].is_export = true;
      }
    }
  }

  return results;
}

auto CheckUnit::ImportCurrentPackage(SemIR::InstId package_inst_id,
                                     SemIR::TypeId namespace_type_id) -> void {
  // Add imports from the current package.
  auto import_map_lookup =
      unit_and_imports_->package_imports_map.Lookup(PackageNameId::None);
  if (!import_map_lookup) {
    // Push the scope; there are no names to add.
    context_.scope_stack().Push(package_inst_id, SemIR::NameScopeId::Package);
    return;
  }
  PackageImports& self_import =
      unit_and_imports_->package_imports[import_map_lookup.value()];

  if (self_import.has_load_error) {
    context_.name_scopes().Get(SemIR::NameScopeId::Package).set_has_error();
  }

  ImportLibrariesFromCurrentPackage(
      context_, namespace_type_id,
      CollectTransitiveImports(self_import.import_decl_id, &self_import,
                               /*api_imports=*/nullptr));

  context_.scope_stack().Push(
      package_inst_id, SemIR::NameScopeId::Package, SemIR::SpecificId::None,
      context_.name_scopes().Get(SemIR::NameScopeId::Package).has_error());
}

auto CheckUnit::ImportOtherPackages(SemIR::TypeId namespace_type_id) -> void {
  // api_imports_list is initially the size of the current file's imports,
  // including for API files, for simplicity in iteration. It's only really used
  // when processing an implementation file, in order to combine the API file
  // imports.
  //
  // For packages imported by the API file, the PackageNameId is the package
  // name and the index is into the API's import list. Otherwise, the initial
  // {None, -1} state remains.
  llvm::SmallVector<std::pair<PackageNameId, int32_t>> api_imports_list;
  api_imports_list.resize(unit_and_imports_->package_imports.size(),
                          {PackageNameId::None, -1});

  // When there's an API file, add the mapping to api_imports_list.
  if (unit_and_imports_->api_for_impl) {
    const auto& api_identifiers =
        unit_and_imports_->api_for_impl->unit->value_stores->identifiers();
    auto& impl_identifiers =
        unit_and_imports_->unit->value_stores->identifiers();

    for (auto [api_imports_index, api_imports] :
         llvm::enumerate(unit_and_imports_->api_for_impl->package_imports)) {
      // Skip the current package.
      if (!api_imports.package_id.has_value()) {
        continue;
      }

      // Translate the package ID from the API file to the implementation file.
      auto impl_package_id = api_imports.package_id;
      if (auto package_identifier_id = impl_package_id.AsIdentifierId();
          package_identifier_id.has_value()) {
        impl_package_id = PackageNameId::ForIdentifier(
            impl_identifiers.Add(api_identifiers.Get(package_identifier_id)));
      }

      if (auto lookup =
              unit_and_imports_->package_imports_map.Lookup(impl_package_id)) {
        // On a hit, replace the entry to unify the API and implementation
        // imports.
        api_imports_list[lookup.value()] = {impl_package_id, api_imports_index};
      } else {
        // On a miss, add the package as API-only.
        api_imports_list.push_back({impl_package_id, api_imports_index});
      }
    }
  }

  for (auto [i, api_imports_entry] : llvm::enumerate(api_imports_list)) {
    // These variables are updated after figuring out which imports are present.
    auto import_decl_id = SemIR::InstId::None;
    PackageNameId package_id = PackageNameId::None;
    bool has_load_error = false;

    // Identify the local package imports if present.
    PackageImports* local_imports = nullptr;
    if (i < unit_and_imports_->package_imports.size()) {
      local_imports = &unit_and_imports_->package_imports[i];
      if (!local_imports->package_id.has_value()) {
        // Skip the current package.
        continue;
      }
      import_decl_id = local_imports->import_decl_id;

      package_id = local_imports->package_id;
      has_load_error |= local_imports->has_load_error;
    }

    // Identify the API package imports if present.
    PackageImports* api_imports = nullptr;
    if (api_imports_entry.second != -1) {
      api_imports = &unit_and_imports_->api_for_impl
                         ->package_imports[api_imports_entry.second];

      if (local_imports) {
        CARBON_CHECK(package_id == api_imports_entry.first);
      } else {
        auto import_ir_inst_id = context_.import_ir_insts().Add(
            {.ir_id = SemIR::ImportIRId::ApiForImpl,
             .inst_id = api_imports->import_decl_id});
        import_decl_id =
            AddInst(context_, MakeImportedLocIdAndInst<SemIR::ImportDecl>(
                                  context_, import_ir_inst_id,
                                  {.package_id = SemIR::NameId::ForPackageName(
                                       api_imports_entry.first)}));
        package_id = api_imports_entry.first;
      }
      has_load_error |= api_imports->has_load_error;
    }

    // Do the actual import.
    ImportLibrariesFromOtherPackage(
        context_, namespace_type_id, import_decl_id, package_id,
        CollectTransitiveImports(import_decl_id, local_imports, api_imports),
        has_load_error);
  }
}

// Loops over all nodes in the tree. On some errors, this may return early,
// for example if an unrecoverable state is encountered.
// NOLINTNEXTLINE(readability-function-size)
auto CheckUnit::ProcessNodeIds() -> bool {
  NodeIdTraversal traversal(&context_, vlog_stream_);

  Parse::NodeId node_id = Parse::NodeId::None;

  // On crash, report which token we were handling.
  PrettyStackTraceFunction node_dumper([&](llvm::raw_ostream& output) {
    const auto& tree = unit_and_imports_->unit->tree_and_subtrees_getter();
    auto converted = tree.NodeToDiagnosticLoc(node_id, /*token_only=*/false);
    converted.loc.FormatLocation(output);
    output << "checking " << context_.parse_tree().node_kind(node_id) << "\n";
    // Crash output has a tab indent; try to indent slightly past that.
    converted.loc.FormatSnippet(output, /*indent=*/10);
  });

  while (auto maybe_node_id = traversal.Next()) {
    node_id = *maybe_node_id;

    emitter_.AdvanceToken(context_.parse_tree().node_token(node_id));

    if (context_.parse_tree().node_has_error(node_id)) {
      context_.TODO(node_id, "handle invalid parse trees in `check`");
      return false;
    }

    bool result;
    auto parse_kind = context_.parse_tree().node_kind(node_id);
    switch (parse_kind) {
#define CARBON_PARSE_NODE_KIND(Name)                                   \
  case Parse::NodeKind::Name: {                                        \
    result = HandleParseNode(                                          \
        context_, context_.parse_tree().As<Parse::Name##Id>(node_id)); \
    break;                                                             \
  }
#include "toolchain/parse/node_kind.def"
    }

    if (!result) {
      CARBON_CHECK(
          unit_and_imports_->err_tracker.seen_error(),
          "HandleParseNode for `{0}` returned false without diagnosing.",
          parse_kind);
      return false;
    }
    traversal.Handle(parse_kind);
  }
  return true;
}

auto CheckUnit::CheckRequiredDeclarations() -> void {
  for (const auto& function : context_.functions().array_ref()) {
    if (!function.first_owning_decl_id.has_value() &&
        function.extern_library_id == context_.sem_ir().library_id()) {
      auto function_loc_id =
          context_.insts().GetLocId(function.non_owning_decl_id);
      CARBON_CHECK(function_loc_id.is_import_ir_inst_id());
      auto import_ir_id = context_.sem_ir()
                              .import_ir_insts()
                              .Get(function_loc_id.import_ir_inst_id())
                              .ir_id;
      auto& import_ir = context_.import_irs().Get(import_ir_id);
      if (import_ir.sem_ir->package_id().has_value() !=
          context_.sem_ir().package_id().has_value()) {
        continue;
      }

      CARBON_DIAGNOSTIC(
          MissingOwningDeclarationInApi, Error,
          "owning declaration required for non-owning declaration");
      if (!import_ir.sem_ir->package_id().has_value() &&
          !context_.sem_ir().package_id().has_value()) {
        emitter_.Emit(function.non_owning_decl_id,
                      MissingOwningDeclarationInApi);
        continue;
      }

      if (import_ir.sem_ir->identifiers().Get(
              import_ir.sem_ir->package_id().AsIdentifierId()) ==
          context_.sem_ir().identifiers().Get(
              context_.sem_ir().package_id().AsIdentifierId())) {
        emitter_.Emit(function.non_owning_decl_id,
                      MissingOwningDeclarationInApi);
      }
    }
  }
}

auto CheckUnit::CheckRequiredDefinitions() -> void {
  CARBON_DIAGNOSTIC(MissingDefinitionInImpl, Error,
                    "no definition found for declaration in impl file");

  // Note that more required definitions can be added during this loop.
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (size_t i = 0; i != context_.definitions_required_by_decl().size(); ++i) {
    SemIR::InstId decl_inst_id = context_.definitions_required_by_decl()[i];
    SemIR::Inst decl_inst = context_.insts().Get(decl_inst_id);
    CARBON_KIND_SWITCH(context_.insts().Get(decl_inst_id)) {
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        if (!context_.classes().Get(class_decl.class_id).is_complete()) {
          emitter_.Emit(decl_inst_id, MissingDefinitionInImpl);
        }
        break;
      }
      case CARBON_KIND(SemIR::FunctionDecl function_decl): {
        if (context_.functions().Get(function_decl.function_id).definition_id ==
            SemIR::InstId::None) {
          emitter_.Emit(decl_inst_id, MissingDefinitionInImpl);
        }
        break;
      }
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        auto& impl = context_.impls().Get(impl_decl.impl_id);
        if (!impl.is_complete()) {
          FillImplWitnessWithErrors(context_, impl);
          CARBON_DIAGNOSTIC(ImplMissingDefinition, Error,
                            "impl declared but not defined");
          emitter_.Emit(decl_inst_id, ImplMissingDefinition);
        }
        break;
      }
      case SemIR::InterfaceDecl::Kind: {
        // TODO: Handle `interface` as well, once we can test it without
        // triggering
        // https://github.com/carbon-language/carbon-lang/issues/4071.
        CARBON_FATAL("TODO: Support interfaces in DiagnoseMissingDefinitions");
      }
      default: {
        CARBON_FATAL("Unexpected inst in definitions_required: {0}", decl_inst);
      }
    }
  }

  // Note that more required definitions can be added during this loop.
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (size_t i = 0; i != context_.definitions_required_by_use().size(); ++i) {
    // This is using the location for the use. We could track the
    // list of enclosing locations if this was used from a generic.
    auto [loc, specific_id] = context_.definitions_required_by_use()[i];
    if (!ResolveSpecificDefinition(context_, loc, specific_id)) {
      CARBON_DIAGNOSTIC(MissingGenericFunctionDefinition, Error,
                        "use of undefined generic function");
      CARBON_DIAGNOSTIC(MissingGenericFunctionDefinitionHere, Note,
                        "generic function declared here");
      auto generic_decl_id =
          context_.generics()
              .Get(context_.specifics().Get(specific_id).generic_id)
              .decl_id;
      emitter_.Build(loc, MissingGenericFunctionDefinition)
          .Note(generic_decl_id, MissingGenericFunctionDefinitionHere)
          .Emit();
    }
  }
}

auto CheckUnit::FinishRun() -> void {
  // TODO: Remove this once we properly push and pop in the right places.
  context_.generic_region_stack().Pop();

  CheckRequiredDeclarations();
  CheckRequiredDefinitions();

  // Pop information for the file-level scope.
  context_.sem_ir().set_top_inst_block_id(context_.inst_block_stack().Pop());
  context_.scope_stack().Pop();

  // Finalizes the list of exports on the IR.
  context_.inst_blocks().ReplacePlaceholder(SemIR::InstBlockId::Exports,
                                            context_.exports());
  // Finalizes the ImportRef inst block.
  context_.inst_blocks().ReplacePlaceholder(SemIR::InstBlockId::ImportRefs,
                                            context_.import_ref_ids());
  // Finalizes __global_init.
  context_.global_init().Finalize();

  context_.sem_ir().set_has_errors(unit_and_imports_->err_tracker.seen_error());

  // Verify that Context cleanly finished.
  context_.VerifyOnFinish();
}

}  // namespace Carbon::Check
