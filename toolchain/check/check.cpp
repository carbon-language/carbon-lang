// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check.h"

#include "common/check.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/context.h"
#include "toolchain/check/import.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_node_location_translator.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name) \
  auto Handle##Name(Context& context, Parse::Name##Id parse_node) -> bool;
#include "toolchain/parse/node_kind.def"

// Handles the transformation of a SemIRLocation to a DiagnosticLocation.
class SemIRLocationTranslator
    : public DiagnosticLocationTranslator<SemIRLocation> {
 public:
  explicit SemIRLocationTranslator(
      const llvm::DenseMap<const SemIR::File*, Parse::NodeLocationTranslator*>*
          node_translators,
      const SemIR::File* sem_ir)
      : node_translators_(node_translators), sem_ir_(sem_ir) {}

  auto GetLocation(SemIRLocation loc) -> DiagnosticLocation override {
    // Parse nodes always refer to the current IR.
    if (!loc.is_inst_id) {
      return GetLocationInFile(sem_ir_, loc.node_location);
    }

    const auto* cursor_ir = sem_ir_;
    auto cursor_inst_id = loc.inst_id;
    while (true) {
      // If the parse node is valid, use it for the location.
      if (auto parse_node = cursor_ir->insts().GetParseNode(cursor_inst_id);
          parse_node.is_valid()) {
        return GetLocationInFile(cursor_ir, parse_node);
      }

      // If the parse node was invalid, recurse through import references when
      // possible.
      if (auto import_ref = cursor_ir->insts().TryGetAs<SemIR::AnyImportRef>(
              cursor_inst_id)) {
        cursor_ir = cursor_ir->import_irs().Get(import_ref->ir_id);
        cursor_inst_id = import_ref->inst_id;
        continue;
      }

      // If a namespace has an instruction for an import, switch to looking at
      // it.
      if (auto ns =
              cursor_ir->insts().TryGetAs<SemIR::Namespace>(cursor_inst_id)) {
        if (ns->import_id.is_valid()) {
          cursor_inst_id = ns->import_id;
          continue;
        }
      }

      // Invalid parse node but not an import; just nothing to point at.
      return GetLocationInFile(cursor_ir, Parse::NodeId::Invalid);
    }
  }

 private:
  auto GetLocationInFile(const SemIR::File* sem_ir,
                         Parse::NodeLocation node_location) const
      -> DiagnosticLocation {
    auto it = node_translators_->find(sem_ir);
    CARBON_CHECK(it != node_translators_->end());
    return it->second->GetLocation(node_location);
  }

  const llvm::DenseMap<const SemIR::File*, Parse::NodeLocationTranslator*>*
      node_translators_;
  const SemIR::File* sem_ir_;
};

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
    explicit PackageImports(Parse::NodeId node) : node(node) {}

    // The first `import` directive in the file, which declared the package's
    // identifier (even if the import failed). Used for associating diagnostics
    // not specific to a single import.
    Parse::NodeId node;
    // Whether there's an import that failed to load.
    bool has_load_error = false;
    // The list of valid imports.
    llvm::SmallVector<Import> imports;
  };

  explicit UnitInfo(Unit& unit)
      : unit(&unit),
        translator(unit.tokens, unit.tokens->source().filename(),
                   unit.parse_tree),
        err_tracker(*unit.consumer),
        emitter(translator, err_tracker) {}

  Unit* unit;

  // Emitter information.
  Parse::NodeLocationTranslator translator;
  ErrorTrackingDiagnosticConsumer err_tracker;
  DiagnosticEmitter<Parse::NodeLocation> emitter;

  // A map of package names to outgoing imports. If the
  // import's target isn't available, the unit will be nullptr to assist with
  // name lookup. Invalid imports (for example, `import Main;`) aren't added
  // because they won't add identifiers to name lookup.
  llvm::DenseMap<IdentifierId, PackageImports> package_imports_map;

  // The remaining number of imports which must be checked before this unit can
  // be processed.
  int32_t imports_remaining = 0;

  // A list of incoming imports. This will be empty for `impl` files, because
  // imports only touch `api` files.
  llvm::SmallVector<UnitInfo*> incoming_imports;
};

// Add imports to the root block.
static auto InitPackageScopeAndImports(Context& context, UnitInfo& unit_info)
    -> void {
  // Importing makes many namespaces, so only canonicalize the type once.
  auto namespace_type_id =
      context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType);

  // Define the package scope, with an instruction for `package` expressions to
  // reference.
  auto package_scope_id = context.name_scopes().Add(
      SemIR::InstId::PackageNamespace, SemIR::NameId::PackageNamespace,
      SemIR::NameScopeId::Invalid);
  CARBON_CHECK(package_scope_id == SemIR::NameScopeId::Package);

  auto package_inst_id = context.AddInst(
      {Parse::NodeId::Invalid,
       SemIR::Namespace{namespace_type_id, SemIR::NameScopeId::Package,
                        SemIR::InstId::Invalid}});
  CARBON_CHECK(package_inst_id == SemIR::InstId::PackageNamespace);

  // Add imports from the current package.
  auto self_import = unit_info.package_imports_map.find(IdentifierId::Invalid);
  if (self_import != unit_info.package_imports_map.end()) {
    bool error_in_import = self_import->second.has_load_error;
    for (const auto& import : self_import->second.imports) {
      const auto& import_sem_ir = **import.unit_info->unit->sem_ir;
      Import(context, namespace_type_id, import_sem_ir);
      error_in_import = error_in_import || import_sem_ir.name_scopes()
                                               .Get(SemIR::NameScopeId::Package)
                                               .has_error;
    }

    // If an import of the current package caused an error for the imported
    // file, it transitively affects the current file too.
    if (error_in_import) {
      context.name_scopes().Get(SemIR::NameScopeId::Package).has_error = true;
    }
    context.PushScope(package_inst_id, SemIR::NameScopeId::Package,
                      error_in_import);
  } else {
    // Push the scope; there are no names to add.
    context.PushScope(package_inst_id, SemIR::NameScopeId::Package);
  }
  CARBON_CHECK(context.current_scope_index() == ScopeIndex::Package);

  for (auto& [package_id, package_imports] : unit_info.package_imports_map) {
    if (!package_id.is_valid()) {
      // Current package is handled above.
      continue;
    }

    llvm::SmallVector<const SemIR::File*> sem_irs;
    for (auto import : package_imports.imports) {
      sem_irs.push_back(&**import.unit_info->unit->sem_ir);
    }
    context.AddPackageImports(package_imports.node, package_id, sem_irs,
                              package_imports.has_load_error);
  }
}

// Loops over all nodes in the tree. On some errors, this may return early,
// for example if an unrecoverable state is encountered.
// NOLINTNEXTLINE(readability-function-size)
static auto ProcessParseNodes(Context& context,
                              ErrorTrackingDiagnosticConsumer& err_tracker)
    -> bool {
  for (auto parse_node : context.parse_tree().postorder()) {
    switch (auto parse_kind = context.parse_tree().node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name)                                         \
  case Parse::NodeKind::Name: {                                              \
    if (!Check::Handle##Name(context, Parse::Name##Id(parse_node))) {        \
      CARBON_CHECK(err_tracker.seen_error())                                 \
          << "Handle" #Name " returned false without printing a diagnostic"; \
      return false;                                                          \
    }                                                                        \
    break;                                                                   \
  }
#include "toolchain/parse/node_kind.def"
    }
  }
  return true;
}

// Produces and checks the IR for the provided Parse::Tree.
static auto CheckParseTree(
    llvm::DenseMap<const SemIR::File*, Parse::NodeLocationTranslator*>*
        node_translators,
    const SemIR::File& builtin_ir, UnitInfo& unit_info,
    llvm::raw_ostream* vlog_stream) -> void {
  unit_info.unit->sem_ir->emplace(
      *unit_info.unit->value_stores,
      unit_info.unit->tokens->source().filename().str(), &builtin_ir);

  // For ease-of-access.
  SemIR::File& sem_ir = **unit_info.unit->sem_ir;
  CARBON_CHECK(
      node_translators->insert({&sem_ir, &unit_info.translator}).second);

  SemIRLocationTranslator translator(node_translators, &sem_ir);
  Context::DiagnosticEmitter emitter(translator, unit_info.err_tracker);
  Context context(*unit_info.unit->tokens, emitter, *unit_info.unit->parse_tree,
                  sem_ir, vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the file.
  context.inst_block_stack().Push();

  InitPackageScopeAndImports(context, unit_info);

  if (!ProcessParseNodes(context, unit_info.err_tracker)) {
    context.sem_ir().set_has_errors(true);
    return;
  }

  // Pop information for the file-level scope.
  sem_ir.set_top_inst_block_id(context.inst_block_stack().Pop());
  context.PopScope();
  context.FinalizeExports();

  context.VerifyOnFinish();

  sem_ir.set_has_errors(unit_info.err_tracker.seen_error());

#ifndef NDEBUG
  if (auto verify = sem_ir.Verify(); !verify.ok()) {
    CARBON_FATAL() << sem_ir << "Built invalid semantics IR: " << verify.error()
                   << "\n";
  }
#endif
}

// The package and library names, used as map keys.
using ImportKey = std::pair<llvm::StringRef, llvm::StringRef>;

// Returns a key form of the package object. file_package_id is only used for
// imports, not the main package directive; as a consequence, it will be invalid
// for the main package directive.
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

static constexpr llvm::StringLiteral ExplicitMainName = "Main";

// Marks an import as required on both the source and target file.
//
// The ID comparisons between the import and unit are okay because they both
// come from the same file.
static auto TrackImport(
    llvm::DenseMap<ImportKey, UnitInfo*>& api_map,
    llvm::DenseMap<ImportKey, Parse::NodeId>* explicit_import_map,
    UnitInfo& unit_info, Parse::Tree::PackagingNames import) -> void {
  const auto& packaging = unit_info.unit->parse_tree->packaging_directive();

  IdentifierId file_package_id =
      packaging ? packaging->names.package_id : IdentifierId::Invalid;
  auto import_key = GetImportKey(unit_info, file_package_id, import);

  // True if the import has `Main` as the package name, even if it comes from
  // the file's packaging (diagnostics may differentiate).
  bool is_explicit_main = import_key.first == ExplicitMainName;

  // Explicit imports need more validation than implicit ones. We try to do
  // these in an order of imports that should be removed, followed by imports
  // that might be valid with syntax fixes.
  if (explicit_import_map) {
    // Diagnose redundant imports.
    if (auto [insert_it, success] =
            explicit_import_map->insert({import_key, import.node});
        !success) {
      CARBON_DIAGNOSTIC(RepeatedImport, Error,
                        "Library imported more than once.");
      CARBON_DIAGNOSTIC(FirstImported, Note, "First import here.");
      unit_info.emitter.Build(import.node, RepeatedImport)
          .Note(insert_it->second, FirstImported)
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
                        "Explicit import of `api` from `impl` file is "
                        "redundant with implicit import.");
      CARBON_DIAGNOSTIC(ImportSelf, Error, "File cannot import itself.");
      bool is_impl =
          !packaging || packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl;
      unit_info.emitter.Emit(import.node,
                             is_impl ? ExplicitImportApi : ImportSelf);
      return;
    }

    // Diagnose explicit imports of `Main//default`. There is no `api` for it.
    // This lets other diagnostics handle explicit `Main` package naming.
    if (is_file_implicit_main && is_import_implicit_current_package &&
        is_import_default_library) {
      CARBON_DIAGNOSTIC(ImportMainDefaultLibrary, Error,
                        "Cannot import `Main//default`.");
      unit_info.emitter.Emit(import.node, ImportMainDefaultLibrary);

      return;
    }

    if (!is_import_implicit_current_package) {
      // Diagnose explicit imports of the same package that use the package
      // name.
      if (is_same_package || (is_file_implicit_main && is_explicit_main)) {
        CARBON_DIAGNOSTIC(
            ImportCurrentPackageByName, Error,
            "Imports from the current package must omit the package name.");
        unit_info.emitter.Emit(import.node, ImportCurrentPackageByName);
        return;
      }

      // Diagnose explicit imports from `Main`.
      if (is_explicit_main) {
        CARBON_DIAGNOSTIC(ImportMainPackage, Error,
                          "Cannot import `Main` from other packages.");
        unit_info.emitter.Emit(import.node, ImportMainPackage);
        return;
      }
    }
  } else if (is_explicit_main) {
    // An implicit import with an explicit `Main` occurs when a `package` rule
    // has bad syntax, which will have been diagnosed when building the API map.
    // As a consequence, we return silently.
    return;
  }

  // Get the package imports.
  auto package_imports_it =
      unit_info.package_imports_map.try_emplace(import.package_id, import.node)
          .first;

  if (auto api = api_map.find(import_key); api != api_map.end()) {
    // Add references between the file and imported api.
    package_imports_it->second.imports.push_back({import, api->second});
    ++unit_info.imports_remaining;
    api->second->incoming_imports.push_back(&unit_info);
  } else {
    // The imported api is missing.
    package_imports_it->second.has_load_error = true;
    CARBON_DIAGNOSTIC(LibraryApiNotFound, Error,
                      "Corresponding API not found.");
    CARBON_DIAGNOSTIC(ImportNotFound, Error, "Imported API not found.");
    unit_info.emitter.Emit(
        import.node, explicit_import_map ? ImportNotFound : LibraryApiNotFound);
  }
}

// Builds a map of `api` files which might be imported. Also diagnoses issues
// related to the packaging because the strings are loaded as part of getting
// the ImportKey (which we then do for `impl` files too).
static auto BuildApiMapAndDiagnosePackaging(
    llvm::SmallVector<UnitInfo, 0>& unit_infos)
    -> llvm::DenseMap<ImportKey, UnitInfo*> {
  llvm::DenseMap<ImportKey, UnitInfo*> api_map;
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.unit->parse_tree->packaging_directive();
    // An import key formed from the `package` or `library` directive. Or, for
    // Main//default, a placeholder key.
    auto import_key = packaging ? GetImportKey(unit_info, IdentifierId::Invalid,
                                               packaging->names)
                                // Construct a boring key for Main//default.
                                : ImportKey{"", ""};

    // Diagnose explicit `Main` uses before they become marked as possible
    // APIs.
    if (import_key.first == ExplicitMainName) {
      CARBON_DIAGNOSTIC(ExplicitMainPackage, Error,
                        "`Main//default` must omit `package` directive.");
      CARBON_DIAGNOSTIC(ExplicitMainLibrary, Error,
                        "Use `library` directive in `Main` package libraries.");
      unit_info.emitter.Emit(packaging->names.node, import_key.second.empty()
                                                        ? ExplicitMainPackage
                                                        : ExplicitMainLibrary);
      continue;
    }

    bool is_impl =
        packaging && packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl;

    // Add to the `api` map and diagnose duplicates. This occurs before the
    // file extension check because we might emit both diagnostics in situations
    // where the user forgets (or has syntax errors with) a package line
    // multiple times.
    if (!is_impl) {
      auto [entry, success] = api_map.insert({import_key, &unit_info});
      if (!success) {
        llvm::StringRef prev_filename =
            entry->second->unit->tokens->source().filename();
        if (packaging) {
          CARBON_DIAGNOSTIC(DuplicateLibraryApi, Error,
                            "Library's API previously provided by `{0}`.",
                            std::string);
          unit_info.emitter.Emit(packaging->names.node, DuplicateLibraryApi,
                                 prev_filename.str());
        } else {
          CARBON_DIAGNOSTIC(DuplicateMainApi, Error,
                            "Main//default previously provided by `{0}`.",
                            std::string);
          // Use the invalid node because there's no node to associate with.
          unit_info.emitter.Emit(Parse::NodeId::Invalid, DuplicateMainApi,
                                 prev_filename.str());
        }
      }
    }

    // Validate file extensions. Note imports rely the packaging directive, not
    // the extension. If the input is not a regular file, for example because it
    // is stdin, no filename checking is performed.
    if (unit_info.unit->tokens->source().is_regular_file()) {
      auto filename = unit_info.unit->tokens->source().filename();
      static constexpr llvm::StringLiteral ApiExt = ".carbon";
      static constexpr llvm::StringLiteral ImplExt = ".impl.carbon";
      bool is_api_with_impl_ext = !is_impl && filename.ends_with(ImplExt);
      auto want_ext = is_impl ? ImplExt : ApiExt;
      if (is_api_with_impl_ext || !filename.ends_with(want_ext)) {
        CARBON_DIAGNOSTIC(IncorrectExtension, Error,
                          "File extension of `{0}` required for `{1}`.",
                          llvm::StringLiteral, Lex::TokenKind);
        auto diag = unit_info.emitter.Build(
            packaging ? packaging->names.node : Parse::NodeId::Invalid,
            IncorrectExtension, want_ext,
            is_impl ? Lex::TokenKind::Impl : Lex::TokenKind::Api);
        if (is_api_with_impl_ext) {
          CARBON_DIAGNOSTIC(IncorrectExtensionImplNote, Note,
                            "File extension of `{0}` only allowed for `{1}`.",
                            llvm::StringLiteral, Lex::TokenKind);
          diag.Note(Parse::NodeId::Invalid, IncorrectExtensionImplNote, ImplExt,
                    Lex::TokenKind::Impl);
        }
        diag.Emit();
      }
    }
  }
  return api_map;
}

auto CheckParseTrees(const SemIR::File& builtin_ir,
                     llvm::MutableArrayRef<Unit> units,
                     llvm::raw_ostream* vlog_stream) -> void {
  // Prepare diagnostic emitters in case we run into issues during package
  // checking.
  //
  // UnitInfo is big due to its SmallVectors, so we default to 0 on the stack.
  llvm::SmallVector<UnitInfo, 0> unit_infos;
  unit_infos.reserve(units.size());
  for (auto& unit : units) {
    unit_infos.emplace_back(unit);
  }

  llvm::DenseMap<ImportKey, UnitInfo*> api_map =
      BuildApiMapAndDiagnosePackaging(unit_infos);

  // Mark down imports for all files.
  llvm::SmallVector<UnitInfo*> ready_to_check;
  ready_to_check.reserve(units.size());
  for (auto& unit_info : unit_infos) {
    if (const auto& packaging =
            unit_info.unit->parse_tree->packaging_directive()) {
      if (packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl) {
        // An `impl` has an implicit import of its `api`.
        auto implicit_names = packaging->names;
        implicit_names.package_id = IdentifierId::Invalid;
        TrackImport(api_map, nullptr, unit_info, implicit_names);
      }
    }

    llvm::DenseMap<ImportKey, Parse::NodeId> explicit_import_map;
    for (const auto& import : unit_info.unit->parse_tree->imports()) {
      TrackImport(api_map, &explicit_import_map, unit_info, import);
    }

    // If there were no imports, mark the file as ready to check for below.
    if (unit_info.imports_remaining == 0) {
      ready_to_check.push_back(&unit_info);
    }
  }

  llvm::DenseMap<const SemIR::File*, Parse::NodeLocationTranslator*>
      node_translators;

  // Check everything with no dependencies. Earlier entries with dependencies
  // will be checked as soon as all their dependencies have been checked.
  for (int check_index = 0;
       check_index < static_cast<int>(ready_to_check.size()); ++check_index) {
    auto* unit_info = ready_to_check[check_index];
    CheckParseTree(&node_translators, builtin_ir, *unit_info, vlog_stream);
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
        for (auto& [package_id, package_imports] :
             unit_info.package_imports_map) {
          for (auto* import_it = package_imports.imports.begin();
               import_it != package_imports.imports.end();) {
            if (*import_it->unit_info->unit->sem_ir) {
              // The import is checked, so continue.
              ++import_it;
            } else {
              // The import hasn't been checked, indicating a cycle.
              CARBON_DIAGNOSTIC(ImportCycleDetected, Error,
                                "Import cannot be used due to a cycle. Cycle "
                                "must be fixed to import.");
              unit_info.emitter.Emit(import_it->names.node,
                                     ImportCycleDetected);
              // Make this look the same as an import which wasn't found.
              package_imports.has_load_error = true;
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
        CheckParseTree(&node_translators, builtin_ir, unit_info, vlog_stream);
      }
    }
  }
}

}  // namespace Carbon::Check
