// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check.h"

#include "common/check.h"
#include "common/map.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/node_id_traversal.h"
#include "toolchain/check/sem_ir_diagnostic_converter.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_node_diagnostic_converter.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace {
struct UnitInfo {
  // A given import within the file, with its destination.
  struct Import {
    Parse::Tree::PackagingNames names;
    UnitInfo* unit_info;
  };
  // A file's imports corresponding to a single package, for the map.
  struct PackageImports {
    // Use the constructor so that the SmallVector is only constructed
    // as-needed.
    explicit PackageImports(IdentifierId package_id,
                            Parse::ImportDeclId node_id)
        : package_id(package_id), node_id(node_id) {}

    // The identifier of the imported package.
    IdentifierId package_id;
    // The first `import` declaration in the file, which declared the package's
    // identifier (even if the import failed). Used for associating diagnostics
    // not specific to a single import.
    Parse::ImportDeclId node_id;
    // The associated `import` instruction. Only valid once a file is checked.
    SemIR::InstId import_decl_id = SemIR::InstId::Invalid;
    // Whether there's an import that failed to load.
    bool has_load_error = false;
    // The list of valid imports.
    llvm::SmallVector<Import> imports;
  };

  explicit UnitInfo(SemIR::CheckIRId check_ir_id, Unit& unit)
      : check_ir_id(check_ir_id),
        unit(&unit),
        err_tracker(*unit.consumer),
        emitter(*unit.node_converter, err_tracker) {}

  SemIR::CheckIRId check_ir_id;
  Unit* unit;

  // Emitter information.
  ErrorTrackingDiagnosticConsumer err_tracker;
  DiagnosticEmitter<Parse::NodeLoc> emitter;

  // List of the outgoing imports. If a package includes unavailable library
  // imports, it has an entry with has_load_error set. Invalid imports (for
  // example, `import Main;`) aren't added because they won't add identifiers to
  // name lookup.
  llvm::SmallVector<PackageImports> package_imports;

  // A map of the package names to the outgoing imports above.
  Map<IdentifierId, int32_t> package_imports_map;

  // The remaining number of imports which must be checked before this unit can
  // be processed.
  int32_t imports_remaining = 0;

  // A list of incoming imports. This will be empty for `impl` files, because
  // imports only touch `api` files.
  llvm::SmallVector<UnitInfo*> incoming_imports;

  // The corresponding `api` unit if this is an `impl` file. The entry should
  // also be in the corresponding `PackageImports`.
  UnitInfo* api_for_impl = nullptr;

  // Whether the unit has been checked.
  bool is_checked = false;
};
}  // namespace

// Collects direct imports, for CollectTransitiveImports.
static auto CollectDirectImports(llvm::SmallVector<SemIR::ImportIR>& results,
                                 llvm::MutableArrayRef<int> ir_to_result_index,
                                 SemIR::InstId import_decl_id,
                                 const UnitInfo::PackageImports& imports,
                                 bool is_local) -> void {
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

// Collects transitive imports, handling deduplication. These will be unified
// between local_imports and api_imports.
static auto CollectTransitiveImports(
    SemIR::InstId import_decl_id, const UnitInfo::PackageImports* local_imports,
    const UnitInfo::PackageImports* api_imports, int total_ir_count)
    -> llvm::SmallVector<SemIR::ImportIR> {
  llvm::SmallVector<SemIR::ImportIR> results;

  // Track whether an IR was imported in full, including `export import`. This
  // distinguishes from IRs that are indirectly added without all names being
  // exported to this IR.
  llvm::SmallVector<int> ir_to_result_index(total_ir_count, -1);

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

// Imports the current package.
static auto ImportCurrentPackage(Context& context, UnitInfo& unit_info,
                                 int total_ir_count,
                                 SemIR::InstId package_inst_id,
                                 SemIR::TypeId namespace_type_id) -> void {
  // Add imports from the current package.
  auto import_map_lookup =
      unit_info.package_imports_map.Lookup(IdentifierId::Invalid);
  if (!import_map_lookup) {
    // Push the scope; there are no names to add.
    context.scope_stack().Push(package_inst_id, SemIR::NameScopeId::Package);
    return;
  }
  UnitInfo::PackageImports& self_import =
      unit_info.package_imports[import_map_lookup.value()];

  if (self_import.has_load_error) {
    context.name_scopes().Get(SemIR::NameScopeId::Package).has_error = true;
  }

  ImportLibrariesFromCurrentPackage(
      context, namespace_type_id,
      CollectTransitiveImports(self_import.import_decl_id, &self_import,
                               /*api_imports=*/nullptr, total_ir_count));

  context.scope_stack().Push(
      package_inst_id, SemIR::NameScopeId::Package, SemIR::SpecificId::Invalid,
      context.name_scopes().Get(SemIR::NameScopeId::Package).has_error);
}

// Imports all other packages (excluding the current package).
static auto ImportOtherPackages(Context& context, UnitInfo& unit_info,
                                int total_ir_count,
                                SemIR::TypeId namespace_type_id) -> void {
  // api_imports_list is initially the size of the current file's imports,
  // including for API files, for simplicity in iteration. It's only really used
  // when processing an implementation file, in order to combine the API file
  // imports.
  //
  // For packages imported by the API file, the IdentifierId is the package name
  // and the index is into the API's import list. Otherwise, the initial
  // {Invalid, -1} state remains.
  llvm::SmallVector<std::pair<IdentifierId, int32_t>> api_imports_list;
  api_imports_list.resize(unit_info.package_imports.size(),
                          {IdentifierId::Invalid, -1});

  // When there's an API file, add the mapping to api_imports_list.
  if (unit_info.api_for_impl) {
    const auto& api_identifiers =
        unit_info.api_for_impl->unit->value_stores->identifiers();
    auto& impl_identifiers = unit_info.unit->value_stores->identifiers();

    for (auto [api_imports_index, api_imports] :
         llvm::enumerate(unit_info.api_for_impl->package_imports)) {
      // Skip the current package.
      if (!api_imports.package_id.is_valid()) {
        continue;
      }
      // Translate the package ID from the API file to the implementation file.
      auto impl_package_id =
          impl_identifiers.Add(api_identifiers.Get(api_imports.package_id));
      if (auto lookup = unit_info.package_imports_map.Lookup(impl_package_id)) {
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
    auto import_decl_id = SemIR::InstId::Invalid;
    IdentifierId package_id = IdentifierId::Invalid;
    bool has_load_error = false;

    // Identify the local package imports if present.
    UnitInfo::PackageImports* local_imports = nullptr;
    if (i < unit_info.package_imports.size()) {
      local_imports = &unit_info.package_imports[i];
      if (!local_imports->package_id.is_valid()) {
        // Skip the current package.
        continue;
      }
      import_decl_id = local_imports->import_decl_id;

      package_id = local_imports->package_id;
      has_load_error |= local_imports->has_load_error;
    }

    // Identify the API package imports if present.
    UnitInfo::PackageImports* api_imports = nullptr;
    if (api_imports_entry.second != -1) {
      api_imports =
          &unit_info.api_for_impl->package_imports[api_imports_entry.second];

      if (local_imports) {
        CARBON_CHECK(package_id == api_imports_entry.first);
      } else {
        auto import_ir_inst_id = context.import_ir_insts().Add(
            {.ir_id = SemIR::ImportIRId::ApiForImpl,
             .inst_id = api_imports->import_decl_id});
        import_decl_id =
            context.AddInst(context.MakeImportedLocAndInst<SemIR::ImportDecl>(
                import_ir_inst_id, {.package_id = SemIR::NameId::ForIdentifier(
                                        api_imports_entry.first)}));
        package_id = api_imports_entry.first;
      }
      has_load_error |= api_imports->has_load_error;
    }

    // Do the actual import.
    ImportLibrariesFromOtherPackage(
        context, namespace_type_id, import_decl_id, package_id,
        CollectTransitiveImports(import_decl_id, local_imports, api_imports,
                                 total_ir_count),
        has_load_error);
  }
}

// Add imports to the root block.
static auto InitPackageScopeAndImports(Context& context, UnitInfo& unit_info,
                                       int total_ir_count) -> void {
  // First create the constant values map for all imported IRs. We'll populate
  // these with mappings for namespaces as we go.
  size_t num_irs = 0;
  for (auto& package_imports : unit_info.package_imports) {
    num_irs += package_imports.imports.size();
  }
  if (!unit_info.api_for_impl) {
    // Leave an empty slot for ImportIRId::ApiForImpl.
    ++num_irs;
  }

  context.import_irs().Reserve(num_irs);
  context.import_ir_constant_values().reserve(num_irs);

  context.SetTotalIRCount(total_ir_count);

  // Importing makes many namespaces, so only canonicalize the type once.
  auto namespace_type_id =
      context.GetBuiltinType(SemIR::BuiltinInstKind::NamespaceType);

  // Define the package scope, with an instruction for `package` expressions to
  // reference.
  auto package_scope_id = context.name_scopes().Add(
      SemIR::InstId::PackageNamespace, SemIR::NameId::PackageNamespace,
      SemIR::NameScopeId::Invalid);
  CARBON_CHECK(package_scope_id == SemIR::NameScopeId::Package);

  auto package_inst_id = context.AddInst<SemIR::Namespace>(
      Parse::NodeId::Invalid, {.type_id = namespace_type_id,
                               .name_scope_id = SemIR::NameScopeId::Package,
                               .import_id = SemIR::InstId::Invalid});
  CARBON_CHECK(package_inst_id == SemIR::InstId::PackageNamespace);

  // If there is an implicit `api` import, set it first so that it uses the
  // ImportIRId::ApiForImpl when processed for imports.
  if (unit_info.api_for_impl) {
    const auto& names = context.parse_tree().packaging_decl()->names;
    auto import_decl_id = context.AddInst<SemIR::ImportDecl>(
        names.node_id,
        {.package_id = SemIR::NameId::ForIdentifier(names.package_id)});
    SetApiImportIR(context, {.decl_id = import_decl_id,
                             .is_export = false,
                             .sem_ir = unit_info.api_for_impl->unit->sem_ir});
  } else {
    SetApiImportIR(context,
                   {.decl_id = SemIR::InstId::Invalid, .sem_ir = nullptr});
  }

  // Add import instructions for everything directly imported. Implicit imports
  // are handled separately.
  for (auto& package_imports : unit_info.package_imports) {
    CARBON_CHECK(!package_imports.import_decl_id.is_valid());
    package_imports.import_decl_id = context.AddInst<SemIR::ImportDecl>(
        package_imports.node_id, {.package_id = SemIR::NameId::ForIdentifier(
                                      package_imports.package_id)});
  }

  // Process the imports.
  if (unit_info.api_for_impl) {
    ImportApiFile(context, namespace_type_id,
                  *unit_info.api_for_impl->unit->sem_ir);
  }
  ImportCurrentPackage(context, unit_info, total_ir_count, package_inst_id,
                       namespace_type_id);
  CARBON_CHECK(context.scope_stack().PeekIndex() == ScopeIndex::Package);
  ImportOtherPackages(context, unit_info, total_ir_count, namespace_type_id);
}

// Checks that each required definition is available. If the definition can be
// generated by resolving a specific, does so, otherwise emits a diagnostic for
// each declaration in context.definitions_required() that doesn't have a
// definition.
static auto CheckRequiredDefinitions(Context& context,
                                     Context::DiagnosticEmitter& emitter)
    -> void {
  CARBON_DIAGNOSTIC(MissingDefinitionInImpl, Error,
                    "no definition found for declaration in impl file");
  // Note that more required definitions can be added during this loop.
  for (size_t i = 0; i != context.definitions_required().size(); ++i) {
    SemIR::InstId decl_inst_id = context.definitions_required()[i];
    SemIR::Inst decl_inst = context.insts().Get(decl_inst_id);
    CARBON_KIND_SWITCH(context.insts().Get(decl_inst_id)) {
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        if (!context.classes().Get(class_decl.class_id).is_defined()) {
          emitter.Emit(decl_inst_id, MissingDefinitionInImpl);
        }
        break;
      }
      case CARBON_KIND(SemIR::FunctionDecl function_decl): {
        if (context.functions().Get(function_decl.function_id).definition_id ==
            SemIR::InstId::Invalid) {
          emitter.Emit(decl_inst_id, MissingDefinitionInImpl);
        }
        break;
      }
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        if (!context.impls().Get(impl_decl.impl_id).is_defined()) {
          emitter.Emit(decl_inst_id, MissingDefinitionInImpl);
        }
        break;
      }
      case SemIR::InterfaceDecl::Kind: {
        // TODO: Handle `interface` as well, once we can test it without
        // triggering
        // https://github.com/carbon-language/carbon-lang/issues/4071.
        CARBON_FATAL("TODO: Support interfaces in DiagnoseMissingDefinitions");
      }
      case CARBON_KIND(SemIR::SpecificFunction specific_function): {
        if (!ResolveSpecificDefinition(context,
                                       specific_function.specific_id)) {
          CARBON_DIAGNOSTIC(MissingGenericFunctionDefinition, Error,
                            "use of undefined generic function");
          CARBON_DIAGNOSTIC(MissingGenericFunctionDefinitionHere, Note,
                            "generic function declared here");
          auto generic_decl_id =
              context.generics()
                  .Get(context.specifics()
                           .Get(specific_function.specific_id)
                           .generic_id)
                  .decl_id;
          emitter.Build(decl_inst_id, MissingGenericFunctionDefinition)
              .Note(generic_decl_id, MissingGenericFunctionDefinitionHere)
              .Emit();
        }
        break;
      }
      default: {
        CARBON_FATAL("Unexpected inst in definitions_required: {0}", decl_inst);
      }
    }
  }
}

// Loops over all nodes in the tree. On some errors, this may return early,
// for example if an unrecoverable state is encountered.
// NOLINTNEXTLINE(readability-function-size)
static auto ProcessNodeIds(Context& context, llvm::raw_ostream* vlog_stream,
                           ErrorTrackingDiagnosticConsumer& err_tracker,
                           Parse::NodeLocConverter& converter) -> bool {
  NodeIdTraversal traversal(context, vlog_stream);

  Parse::NodeId node_id = Parse::NodeId::Invalid;

  // On crash, report which token we were handling.
  PrettyStackTraceFunction node_dumper([&](llvm::raw_ostream& output) {
    auto loc = converter.ConvertLoc(
        node_id, [](DiagnosticLoc, const DiagnosticBase<>&) {});
    loc.FormatLocation(output);
    output << ": checking " << context.parse_tree().node_kind(node_id) << "\n";
    // Crash output has a tab indent; try to indent slightly past that.
    loc.FormatSnippet(output, /*indent=*/10);
  });

  while (auto maybe_node_id = traversal.Next()) {
    node_id = *maybe_node_id;
    auto parse_kind = context.parse_tree().node_kind(node_id);

    switch (parse_kind) {
#define CARBON_PARSE_NODE_KIND(Name)                                 \
  case Parse::NodeKind::Name: {                                      \
    if (!HandleParseNode(context, Parse::Name##Id(node_id))) {       \
      CARBON_CHECK(err_tracker.seen_error(),                         \
                   "Handle" #Name                                    \
                   " returned false without printing a diagnostic"); \
      return false;                                                  \
    }                                                                \
    break;                                                           \
  }
#include "toolchain/parse/node_kind.def"
    }

    traversal.Handle(parse_kind);
  }
  return true;
}

// Produces and checks the IR for the provided Parse::Tree.
static auto CheckParseTree(UnitInfo& unit_info, int total_ir_count,
                           llvm::raw_ostream* vlog_stream) -> void {
  Timings::ScopedTiming timing(unit_info.unit->timings, "check");

  // We can safely mark this as checked at the start.
  unit_info.is_checked = true;

  SemIR::File& sem_ir = *unit_info.unit->sem_ir;
  Context::DiagnosticEmitter emitter(*unit_info.unit->sem_ir_converter,
                                     unit_info.err_tracker);
  Context context(*unit_info.unit->tokens, emitter, *unit_info.unit->parse_tree,
                  unit_info.unit->get_parse_tree_and_subtrees, sem_ir,
                  vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the file.
  context.inst_block_stack().Push();

  InitPackageScopeAndImports(context, unit_info, total_ir_count);

  // Eagerly import the impls declared in the api file to prepare to redeclare
  // them.
  ImportImplsFromApiFile(context);

  if (!ProcessNodeIds(context, vlog_stream, unit_info.err_tracker,
                      *unit_info.unit->node_converter)) {
    context.sem_ir().set_has_errors(true);
    return;
  }

  CheckRequiredDefinitions(context, emitter);

  context.Finalize();

  context.VerifyOnFinish();

  sem_ir.set_has_errors(unit_info.err_tracker.seen_error());

#ifndef NDEBUG
  if (auto verify = sem_ir.Verify(); !verify.ok()) {
    CARBON_FATAL("{0}Built invalid semantics IR: {1}\n", sem_ir,
                 verify.error());
  }
#endif
}

// The package and library names, used as map keys.
using ImportKey = std::pair<llvm::StringRef, llvm::StringRef>;

// Returns a key form of the package object. file_package_id is only used for
// imports, not the main package declaration; as a consequence, it will be
// invalid for the main package declaration.
static auto GetImportKey(UnitInfo& unit_info, IdentifierId file_package_id,
                         Parse::Tree::PackagingNames names) -> ImportKey {
  auto* stores = unit_info.unit->value_stores;
  llvm::StringRef package_name =
      names.package_id.is_valid()  ? stores->identifiers().Get(names.package_id)
      : file_package_id.is_valid() ? stores->identifiers().Get(file_package_id)
                                   : "";
  llvm::StringRef library_name =
      names.library_id.is_valid()
          ? stores->string_literal_values().Get(names.library_id)
          : "";
  return {package_name, library_name};
}

static constexpr llvm::StringLiteral CppPackageName = "Cpp";
static constexpr llvm::StringLiteral MainPackageName = "Main";

static auto RenderImportKey(ImportKey import_key) -> std::string {
  if (import_key.first.empty()) {
    import_key.first = MainPackageName;
  }
  if (import_key.second.empty()) {
    return import_key.first.str();
  }
  return llvm::formatv("{0}//{1}", import_key.first, import_key.second).str();
}

// Marks an import as required on both the source and target file.
//
// The ID comparisons between the import and unit are okay because they both
// come from the same file.
static auto TrackImport(Map<ImportKey, UnitInfo*>& api_map,
                        Map<ImportKey, Parse::NodeId>* explicit_import_map,
                        UnitInfo& unit_info, Parse::Tree::PackagingNames import)
    -> void {
  const auto& packaging = unit_info.unit->parse_tree->packaging_decl();

  IdentifierId file_package_id =
      packaging ? packaging->names.package_id : IdentifierId::Invalid;
  auto import_key = GetImportKey(unit_info, file_package_id, import);

  // True if the import has `Main` as the package name, even if it comes from
  // the file's packaging (diagnostics may differentiate).
  bool is_explicit_main = import_key.first == MainPackageName;

  // Explicit imports need more validation than implicit ones. We try to do
  // these in an order of imports that should be removed, followed by imports
  // that might be valid with syntax fixes.
  if (explicit_import_map) {
    // Diagnose redundant imports.
    if (auto insert_result =
            explicit_import_map->Insert(import_key, import.node_id);
        !insert_result.is_inserted()) {
      CARBON_DIAGNOSTIC(RepeatedImport, Error,
                        "library imported more than once");
      CARBON_DIAGNOSTIC(FirstImported, Note, "first import here");
      unit_info.emitter.Build(import.node_id, RepeatedImport)
          .Note(insert_result.value(), FirstImported)
          .Emit();
      return;
    }

    // True if the file's package is implicitly `Main` (by omitting an explicit
    // package name).
    bool is_file_implicit_main =
        !packaging || !packaging->names.package_id.is_valid();
    // True if the import is using implicit "current package" syntax (by
    // omitting an explicit package name).
    bool is_import_implicit_current_package = !import.package_id.is_valid();
    // True if the import is using `default` library syntax.
    bool is_import_default_library = !import.library_id.is_valid();
    // True if the import and file point at the same package, even by
    // incorrectly specifying the current package name to `import`.
    bool is_same_package = is_import_implicit_current_package ||
                           import.package_id == file_package_id;
    // True if the import points at the same library as the file's library.
    bool is_same_library =
        is_same_package &&
        (packaging ? import.library_id == packaging->names.library_id
                   : is_import_default_library);

    // Diagnose explicit imports of the same library, whether from `api` or
    // `impl`.
    if (is_same_library) {
      CARBON_DIAGNOSTIC(ExplicitImportApi, Error,
                        "explicit import of `api` from `impl` file is "
                        "redundant with implicit import");
      CARBON_DIAGNOSTIC(ImportSelf, Error, "file cannot import itself");
      bool is_impl = !packaging || packaging->is_impl;
      unit_info.emitter.Emit(import.node_id,
                             is_impl ? ExplicitImportApi : ImportSelf);
      return;
    }

    // Diagnose explicit imports of `Main//default`. There is no `api` for it.
    // This lets other diagnostics handle explicit `Main` package naming.
    if (is_file_implicit_main && is_import_implicit_current_package &&
        is_import_default_library) {
      CARBON_DIAGNOSTIC(ImportMainDefaultLibrary, Error,
                        "cannot import `Main//default`");
      unit_info.emitter.Emit(import.node_id, ImportMainDefaultLibrary);

      return;
    }

    if (!is_import_implicit_current_package) {
      // Diagnose explicit imports of the same package that use the package
      // name.
      if (is_same_package || (is_file_implicit_main && is_explicit_main)) {
        CARBON_DIAGNOSTIC(
            ImportCurrentPackageByName, Error,
            "imports from the current package must omit the package name");
        unit_info.emitter.Emit(import.node_id, ImportCurrentPackageByName);
        return;
      }

      // Diagnose explicit imports from `Main`.
      if (is_explicit_main) {
        CARBON_DIAGNOSTIC(ImportMainPackage, Error,
                          "cannot import `Main` from other packages");
        unit_info.emitter.Emit(import.node_id, ImportMainPackage);
        return;
      }
    }
  } else if (is_explicit_main) {
    // An implicit import with an explicit `Main` occurs when a `package` rule
    // has bad syntax, which will have been diagnosed when building the API map.
    // As a consequence, we return silently.
    return;
  }

  // Get the package imports, or create them if this is the first.
  auto create_imports = [&]() -> int32_t {
    int32_t index = unit_info.package_imports.size();
    unit_info.package_imports.push_back(
        UnitInfo::PackageImports(import.package_id, import.node_id));
    return index;
  };
  auto insert_result =
      unit_info.package_imports_map.Insert(import.package_id, create_imports);
  UnitInfo::PackageImports& package_imports =
      unit_info.package_imports[insert_result.value()];

  if (auto api_lookup = api_map.Lookup(import_key)) {
    // Add references between the file and imported api.
    UnitInfo* api = api_lookup.value();
    package_imports.imports.push_back({import, api});
    ++unit_info.imports_remaining;
    api->incoming_imports.push_back(&unit_info);

    // If this is the implicit import, note we have it.
    if (!explicit_import_map) {
      CARBON_CHECK(!unit_info.api_for_impl);
      unit_info.api_for_impl = api;
    }
  } else {
    // The imported api is missing.
    package_imports.has_load_error = true;
    if (!explicit_import_map && import_key.first == CppPackageName) {
      // Don't diagnose the implicit import in `impl package Cpp`, because we'll
      // have diagnosed the use of `Cpp` in the declaration.
      return;
    }
    CARBON_DIAGNOSTIC(LibraryApiNotFound, Error,
                      "corresponding API for '{0}' not found", std::string);
    CARBON_DIAGNOSTIC(ImportNotFound, Error, "imported API '{0}' not found",
                      std::string);
    unit_info.emitter.Emit(
        import.node_id,
        explicit_import_map ? ImportNotFound : LibraryApiNotFound,
        RenderImportKey(import_key));
  }
}

// Builds a map of `api` files which might be imported. Also diagnoses issues
// related to the packaging because the strings are loaded as part of getting
// the ImportKey (which we then do for `impl` files too).
static auto BuildApiMapAndDiagnosePackaging(
    llvm::MutableArrayRef<UnitInfo> unit_infos) -> Map<ImportKey, UnitInfo*> {
  Map<ImportKey, UnitInfo*> api_map;
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.unit->parse_tree->packaging_decl();
    // An import key formed from the `package` or `library` declaration. Or, for
    // Main//default, a placeholder key.
    auto import_key = packaging ? GetImportKey(unit_info, IdentifierId::Invalid,
                                               packaging->names)
                                // Construct a boring key for Main//default.
                                : ImportKey{"", ""};

    // Diagnose restricted package names before they become marked as possible
    // APIs.
    if (import_key.first == MainPackageName) {
      CARBON_DIAGNOSTIC(ExplicitMainPackage, Error,
                        "`Main//default` must omit `package` declaration");
      CARBON_DIAGNOSTIC(
          ExplicitMainLibrary, Error,
          "use `library` declaration in `Main` package libraries");
      unit_info.emitter.Emit(packaging->names.node_id,
                             import_key.second.empty() ? ExplicitMainPackage
                                                       : ExplicitMainLibrary);
      continue;
    } else if (import_key.first == CppPackageName) {
      CARBON_DIAGNOSTIC(CppPackageDeclaration, Error,
                        "`Cpp` cannot be used by a `package` declaration");
      unit_info.emitter.Emit(packaging->names.node_id, CppPackageDeclaration);
      continue;
    }

    bool is_impl = packaging && packaging->is_impl;

    // Add to the `api` map and diagnose duplicates. This occurs before the
    // file extension check because we might emit both diagnostics in situations
    // where the user forgets (or has syntax errors with) a package line
    // multiple times.
    if (!is_impl) {
      auto insert_result = api_map.Insert(import_key, &unit_info);
      if (!insert_result.is_inserted()) {
        llvm::StringRef prev_filename =
            insert_result.value()->unit->tokens->source().filename();
        if (packaging) {
          CARBON_DIAGNOSTIC(DuplicateLibraryApi, Error,
                            "library's API previously provided by `{0}`",
                            std::string);
          unit_info.emitter.Emit(packaging->names.node_id, DuplicateLibraryApi,
                                 prev_filename.str());
        } else {
          CARBON_DIAGNOSTIC(DuplicateMainApi, Error,
                            "`Main//default` previously provided by `{0}`",
                            std::string);
          // Use the invalid node because there's no node to associate with.
          unit_info.emitter.Emit(Parse::NodeId::Invalid, DuplicateMainApi,
                                 prev_filename.str());
        }
      }
    }

    // Validate file extensions. Note imports rely the packaging declaration,
    // not the extension. If the input is not a regular file, for example
    // because it is stdin, no filename checking is performed.
    if (unit_info.unit->tokens->source().is_regular_file()) {
      auto filename = unit_info.unit->tokens->source().filename();
      static constexpr llvm::StringLiteral ApiExt = ".carbon";
      static constexpr llvm::StringLiteral ImplExt = ".impl.carbon";
      bool is_api_with_impl_ext = !is_impl && filename.ends_with(ImplExt);
      auto want_ext = is_impl ? ImplExt : ApiExt;
      if (is_api_with_impl_ext || !filename.ends_with(want_ext)) {
        CARBON_DIAGNOSTIC(
            IncorrectExtension, Error,
            "file extension of `{0:.impl|}.carbon` required for {0:`impl`|api}",
            BoolAsSelect);
        auto diag = unit_info.emitter.Build(
            packaging ? packaging->names.node_id : Parse::NodeId::Invalid,
            IncorrectExtension, is_impl);
        if (is_api_with_impl_ext) {
          CARBON_DIAGNOSTIC(
              IncorrectExtensionImplNote, Note,
              "file extension of `.impl.carbon` only allowed for `impl`");
          diag.Note(Parse::NodeId::Invalid, IncorrectExtensionImplNote);
        }
        diag.Emit();
      }
    }
  }
  return api_map;
}

auto CheckParseTrees(llvm::MutableArrayRef<Unit> units, bool prelude_import,
                     llvm::raw_ostream* vlog_stream) -> void {
  // UnitInfo is big due to its SmallVectors, so we default to 0 on the
  // stack.
  llvm::SmallVector<UnitInfo, 0> unit_infos;
  unit_infos.reserve(units.size());
  for (auto [i, unit] : llvm::enumerate(units)) {
    unit_infos.emplace_back(SemIR::CheckIRId(i), unit);
  }

  Map<ImportKey, UnitInfo*> api_map =
      BuildApiMapAndDiagnosePackaging(unit_infos);

  // Mark down imports for all files.
  llvm::SmallVector<UnitInfo*> ready_to_check;
  ready_to_check.reserve(units.size());
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.unit->parse_tree->packaging_decl();
    if (packaging && packaging->is_impl) {
      // An `impl` has an implicit import of its `api`.
      auto implicit_names = packaging->names;
      implicit_names.package_id = IdentifierId::Invalid;
      TrackImport(api_map, nullptr, unit_info, implicit_names);
    }

    Map<ImportKey, Parse::NodeId> explicit_import_map;

    // Add the prelude import. It's added to explicit_import_map so that it can
    // conflict with an explicit import of the prelude.
    IdentifierId core_ident_id =
        unit_info.unit->value_stores->identifiers().Add("Core");
    if (prelude_import &&
        !(packaging && packaging->names.package_id == core_ident_id)) {
      auto prelude_id =
          unit_info.unit->value_stores->string_literal_values().Add("prelude");
      TrackImport(api_map, &explicit_import_map, unit_info,
                  {.node_id = Parse::InvalidNodeId(),
                   .package_id = core_ident_id,
                   .library_id = prelude_id});
    }

    for (const auto& import : unit_info.unit->parse_tree->imports()) {
      TrackImport(api_map, &explicit_import_map, unit_info, import);
    }

    // If there were no imports, mark the file as ready to check for below.
    if (unit_info.imports_remaining == 0) {
      ready_to_check.push_back(&unit_info);
    }
  }

  // Check everything with no dependencies. Earlier entries with dependencies
  // will be checked as soon as all their dependencies have been checked.
  for (int check_index = 0;
       check_index < static_cast<int>(ready_to_check.size()); ++check_index) {
    auto* unit_info = ready_to_check[check_index];
    CheckParseTree(*unit_info, units.size(), vlog_stream);
    for (auto* incoming_import : unit_info->incoming_imports) {
      --incoming_import->imports_remaining;
      if (incoming_import->imports_remaining == 0) {
        ready_to_check.push_back(incoming_import);
      }
    }
  }

  // If there are still units with remaining imports, it means there's a
  // dependency loop.
  if (ready_to_check.size() < unit_infos.size()) {
    // Go through units and mask out unevaluated imports. This breaks everything
    // associated with a loop equivalently, whether it's part of it or depending
    // on a part of it.
    // TODO: Better identify cycles, maybe try to untangle them.
    for (auto& unit_info : unit_infos) {
      if (unit_info.imports_remaining > 0) {
        for (auto& package_imports : unit_info.package_imports) {
          for (auto* import_it = package_imports.imports.begin();
               import_it != package_imports.imports.end();) {
            if (import_it->unit_info->is_checked) {
              // The import is checked, so continue.
              ++import_it;
            } else {
              // The import hasn't been checked, indicating a cycle.
              CARBON_DIAGNOSTIC(ImportCycleDetected, Error,
                                "import cannot be used due to a cycle; cycle "
                                "must be fixed to import");
              unit_info.emitter.Emit(import_it->names.node_id,
                                     ImportCycleDetected);
              // Make this look the same as an import which wasn't found.
              package_imports.has_load_error = true;
              if (unit_info.api_for_impl == import_it->unit_info) {
                unit_info.api_for_impl = nullptr;
              }
              import_it = package_imports.imports.erase(import_it);
            }
          }
        }
      }
    }

    // Check the remaining file contents, which are probably broken due to
    // incomplete imports.
    for (auto& unit_info : unit_infos) {
      if (unit_info.imports_remaining > 0) {
        CheckParseTree(unit_info, units.size(), vlog_stream);
      }
    }
  }
}

}  // namespace Carbon::Check
