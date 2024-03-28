// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check.h"

#include "common/check.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/import.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_node_diagnostic_converter.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name) \
  auto Handle##Name(Context& context, Parse::Name##Id node_id) -> bool;
#include "toolchain/parse/node_kind.def"

// Handles the transformation of a SemIRLoc to a DiagnosticLoc.
//
// TODO: Move this to diagnostic_helpers.cpp.
class SemIRDiagnosticConverter : public DiagnosticConverter<SemIRLoc> {
 public:
  explicit SemIRDiagnosticConverter(
      const llvm::DenseMap<const SemIR::File*, Parse::NodeLocConverter*>*
          node_converters,
      const SemIR::File* sem_ir)
      : node_converters_(node_converters), sem_ir_(sem_ir) {}

  // Converts an instruction's location to a diagnostic location, which will be
  // the underlying line of code. Adds context for any imports used in the
  // current SemIR to get to the underlying code.
  auto ConvertLoc(SemIRLoc loc, ContextFnT context_fn) const
      -> DiagnosticLoc override {
    // Cursors for the current IR and instruction in that IR.
    const auto* cursor_ir = sem_ir_;
    auto cursor_inst_id = SemIR::InstId::Invalid;

    // Notes an import on the diagnostic and updates cursors to point at the
    // imported IR.
    auto follow_import_ref = [&](SemIR::ImportIRId ir_id,
                                 SemIR::InstId inst_id) {
      const auto& import_ir = cursor_ir->import_irs().Get(ir_id);
      auto context_loc = ConvertLocInFile(cursor_ir, import_ir.node_id,
                                          loc.token_only, context_fn);
      CARBON_DIAGNOSTIC(InImport, Note, "In import.");
      context_fn(context_loc, InImport);
      cursor_ir = import_ir.sem_ir;
      cursor_inst_id = inst_id;
    };

    // If the location is is an import, follows it and returns nullopt.
    // Otherwise, it's a parse node, so return the final location.
    auto handle_loc = [&](SemIR::LocId loc_id) -> std::optional<DiagnosticLoc> {
      if (loc_id.is_import_ir_inst_id()) {
        auto import_ir_inst =
            cursor_ir->import_ir_insts().Get(loc_id.import_ir_inst_id());
        follow_import_ref(import_ir_inst.ir_id, import_ir_inst.inst_id);
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
      CARBON_CHECK(cursor_inst_id.is_valid()) << "Should have been set";
    }

    while (true) {
      // If the parse node is valid, use it for the location.
      if (auto loc_id = cursor_ir->insts().GetLocId(cursor_inst_id);
          loc_id.is_valid()) {
        if (auto diag_loc = handle_loc(loc_id)) {
          return *diag_loc;
        }
        continue;
      }

      // If the parse node was invalid, recurse through import references when
      // possible.
      if (auto import_ref = cursor_ir->insts().TryGetAs<SemIR::AnyImportRef>(
              cursor_inst_id)) {
        follow_import_ref(import_ref->ir_id, import_ref->inst_id);
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
      return ConvertLocInFile(cursor_ir, Parse::NodeId::Invalid, loc.token_only,
                              context_fn);
    }
  }

  auto ConvertArg(llvm::Any arg) const -> llvm::Any override {
    if (auto* name_id = llvm::any_cast<SemIR::NameId>(&arg)) {
      return sem_ir_->names().GetFormatted(*name_id).str();
    }
    if (auto* type_id = llvm::any_cast<SemIR::TypeId>(&arg)) {
      return sem_ir_->StringifyType(*type_id);
    }
    if (auto* typed_int = llvm::any_cast<TypedInt>(&arg)) {
      return llvm::APSInt(typed_int->value,
                          !sem_ir_->types().IsSignedInt(typed_int->type));
    }
    return DiagnosticConverter<SemIRLoc>::ConvertArg(arg);
  }

 private:
  auto ConvertLocInFile(const SemIR::File* sem_ir, Parse::NodeId node_id,
                        bool token_only, ContextFnT context_fn) const
      -> DiagnosticLoc {
    auto it = node_converters_->find(sem_ir);
    CARBON_CHECK(it != node_converters_->end());
    return it->second->ConvertLoc(Parse::NodeLoc(node_id, token_only),
                                  context_fn);
  }

  const llvm::DenseMap<const SemIR::File*, Parse::NodeLocConverter*>*
      node_converters_;
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
    explicit PackageImports(Parse::ImportDirectiveId node_id)
        : node_id(node_id) {}

    // The first `import` directive in the file, which declared the package's
    // identifier (even if the import failed). Used for associating diagnostics
    // not specific to a single import.
    Parse::ImportDirectiveId node_id;
    // Whether there's an import that failed to load.
    bool has_load_error = false;
    // The list of valid imports.
    llvm::SmallVector<Import> imports;
  };

  explicit UnitInfo(Unit& unit)
      : unit(&unit),
        converter(unit.tokens, unit.tokens->source().filename(),
                  unit.parse_tree),
        err_tracker(*unit.consumer),
        emitter(converter, err_tracker) {}

  Unit* unit;

  // Emitter information.
  Parse::NodeLocConverter converter;
  ErrorTrackingDiagnosticConsumer err_tracker;
  DiagnosticEmitter<Parse::NodeLoc> emitter;

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
  // First create the constant values map for all imported IRs. We'll populate
  // these with mappings for namespaces as we go.
  size_t num_irs = context.import_irs().size();
  for (auto& [_, package_imports] : unit_info.package_imports_map) {
    num_irs += package_imports.imports.size();
  }
  context.import_ir_constant_values().resize(
      num_irs, SemIR::ConstantValueStore(SemIR::ConstantId::Invalid));

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
      ImportLibraryFromCurrentPackage(context, namespace_type_id,
                                      import.names.node_id, import_sem_ir);
      error_in_import |= import_sem_ir.name_scopes()
                             .Get(SemIR::NameScopeId::Package)
                             .has_error;
    }

    // If an import of the current package caused an error for the imported
    // file, it transitively affects the current file too.
    if (error_in_import) {
      context.name_scopes().Get(SemIR::NameScopeId::Package).has_error = true;
    }
    context.scope_stack().Push(package_inst_id, SemIR::NameScopeId::Package,
                               error_in_import);
  } else {
    // Push the scope; there are no names to add.
    context.scope_stack().Push(package_inst_id, SemIR::NameScopeId::Package);
  }
  CARBON_CHECK(context.scope_stack().PeekIndex() == ScopeIndex::Package);

  for (auto& [package_id, package_imports] : unit_info.package_imports_map) {
    if (!package_id.is_valid()) {
      // Current package is handled above.
      continue;
    }

    llvm::SmallVector<SemIR::ImportIR> import_irs;
    for (auto import : package_imports.imports) {
      import_irs.push_back({.node_id = import.names.node_id,
                            .sem_ir = &**import.unit_info->unit->sem_ir});
    }
    ImportLibrariesFromOtherPackage(context, namespace_type_id,
                                    package_imports.node_id, package_id,
                                    import_irs, package_imports.has_load_error);
  }

  CARBON_CHECK(context.import_irs().size() == num_irs)
      << "Created an unexpected number of IRs";
}

// Loops over all nodes in the tree. On some errors, this may return early,
// for example if an unrecoverable state is encountered.
// NOLINTNEXTLINE(readability-function-size)
static auto ProcessNodeIds(Context& context,
                           ErrorTrackingDiagnosticConsumer& err_tracker)
    -> bool {
  for (auto node_id : context.parse_tree().postorder()) {
    switch (auto parse_kind = context.parse_tree().node_kind(node_id)) {
#define CARBON_PARSE_NODE_KIND(Name)                                         \
  case Parse::NodeKind::Name: {                                              \
    if (!Check::Handle##Name(context, Parse::Name##Id(node_id))) {           \
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
    llvm::DenseMap<const SemIR::File*, Parse::NodeLocConverter*>*
        node_converters,
    const SemIR::File& builtin_ir, UnitInfo& unit_info,
    llvm::raw_ostream* vlog_stream) -> void {
  unit_info.unit->sem_ir->emplace(
      *unit_info.unit->value_stores,
      unit_info.unit->tokens->source().filename().str(), &builtin_ir);

  // For ease-of-access.
  SemIR::File& sem_ir = **unit_info.unit->sem_ir;
  CARBON_CHECK(node_converters->insert({&sem_ir, &unit_info.converter}).second);

  SemIRDiagnosticConverter converter(node_converters, &sem_ir);
  Context::DiagnosticEmitter emitter(converter, unit_info.err_tracker);
  Context context(*unit_info.unit->tokens, emitter, *unit_info.unit->parse_tree,
                  sem_ir, vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the file.
  context.inst_block_stack().Push();

  InitPackageScopeAndImports(context, unit_info);

  if (!ProcessNodeIds(context, unit_info.err_tracker)) {
    context.sem_ir().set_has_errors(true);
    return;
  }

  // Pop information for the file-level scope.
  sem_ir.set_top_inst_block_id(context.inst_block_stack().Pop());
  context.scope_stack().Pop();
  context.FinalizeExports();
  context.FinalizeGlobalInit();

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
            explicit_import_map->insert({import_key, import.node_id});
        !success) {
      CARBON_DIAGNOSTIC(RepeatedImport, Error,
                        "Library imported more than once.");
      CARBON_DIAGNOSTIC(FirstImported, Note, "First import here.");
      unit_info.emitter.Build(import.node_id, RepeatedImport)
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
      unit_info.emitter.Emit(import.node_id,
                             is_impl ? ExplicitImportApi : ImportSelf);
      return;
    }

    // Diagnose explicit imports of `Main//default`. There is no `api` for it.
    // This lets other diagnostics handle explicit `Main` package naming.
    if (is_file_implicit_main && is_import_implicit_current_package &&
        is_import_default_library) {
      CARBON_DIAGNOSTIC(ImportMainDefaultLibrary, Error,
                        "Cannot import `Main//default`.");
      unit_info.emitter.Emit(import.node_id, ImportMainDefaultLibrary);

      return;
    }

    if (!is_import_implicit_current_package) {
      // Diagnose explicit imports of the same package that use the package
      // name.
      if (is_same_package || (is_file_implicit_main && is_explicit_main)) {
        CARBON_DIAGNOSTIC(
            ImportCurrentPackageByName, Error,
            "Imports from the current package must omit the package name.");
        unit_info.emitter.Emit(import.node_id, ImportCurrentPackageByName);
        return;
      }

      // Diagnose explicit imports from `Main`.
      if (is_explicit_main) {
        CARBON_DIAGNOSTIC(ImportMainPackage, Error,
                          "Cannot import `Main` from other packages.");
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

  // Get the package imports.
  auto package_imports_it = unit_info.package_imports_map
                                .try_emplace(import.package_id, import.node_id)
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
    unit_info.emitter.Emit(import.node_id, explicit_import_map
                                               ? ImportNotFound
                                               : LibraryApiNotFound);
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
      unit_info.emitter.Emit(packaging->names.node_id,
                             import_key.second.empty() ? ExplicitMainPackage
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
          unit_info.emitter.Emit(packaging->names.node_id, DuplicateLibraryApi,
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
            packaging ? packaging->names.node_id : Parse::NodeId::Invalid,
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

  llvm::DenseMap<const SemIR::File*, Parse::NodeLocConverter*> node_converters;

  // Check everything with no dependencies. Earlier entries with dependencies
  // will be checked as soon as all their dependencies have been checked.
  for (int check_index = 0;
       check_index < static_cast<int>(ready_to_check.size()); ++check_index) {
    auto* unit_info = ready_to_check[check_index];
    CheckParseTree(&node_converters, builtin_ir, *unit_info, vlog_stream);
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
              unit_info.emitter.Emit(import_it->names.node_id,
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
        CheckParseTree(&node_converters, builtin_ir, unit_info, vlog_stream);
      }
    }
  }
}

}  // namespace Carbon::Check
