// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check.h"

#include "common/check.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/base/value_store.h"
#include "toolchain/check/context.h"
#include "toolchain/parse/tree_node_location_translator.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

struct UnitInfo {
  explicit UnitInfo(Unit& unit)
      : unit(&unit),
        translator(unit.tokens, unit.parse_tree),
        err_tracker(*unit.consumer),
        emitter(translator, err_tracker) {}

  Unit* unit;

  // Emitter information.
  Parse::NodeLocationTranslator translator;
  ErrorTrackingDiagnosticConsumer err_tracker;
  DiagnosticEmitter<Parse::Node> emitter;

  // A list of outgoing imports.
  llvm::SmallVector<std::pair<Parse::Node, UnitInfo*>> imports;

  // The remaining number of imports which must be checked before this unit can
  // be processed.
  int32_t imports_remaining = 0;

  // A list of incoming imports. This will be empty for `impl` files, because
  // imports only touch `api` files.
  llvm::SmallVector<UnitInfo*> incoming_imports;
};

// Produces and checks the IR for the provided Parse::Tree.
// TODO: Both valid and invalid imports should be recorded on the SemIR. Invalid
// imports should suppress errors where it makes sense.
static auto CheckParseTree(const SemIR::File& builtin_ir, UnitInfo& unit_info,
                           llvm::raw_ostream* vlog_stream) -> void {
  unit_info.unit->sem_ir->emplace(*unit_info.unit->value_stores,
                                  unit_info.unit->tokens->filename().str(),
                                  &builtin_ir);

  // For ease-of-access.
  SemIR::File& sem_ir = **unit_info.unit->sem_ir;
  const Parse::Tree& parse_tree = *unit_info.unit->parse_tree;

  Check::Context context(*unit_info.unit->tokens, unit_info.emitter, parse_tree,
                         sem_ir, vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the Parse::Tree.
  context.inst_block_stack().Push();
  context.PushScope();

  // Loops over all nodes in the tree. On some errors, this may return early,
  // for example if an unrecoverable state is encountered.
  for (auto parse_node : parse_tree.postorder()) {
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (auto parse_kind = parse_tree.node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name)                                         \
  case Parse::NodeKind::Name: {                                              \
    if (!Check::Handle##Name(context, parse_node)) {                         \
      CARBON_CHECK(unit_info.err_tracker.seen_error())                       \
          << "Handle" #Name " returned false without printing a diagnostic"; \
      sem_ir.set_has_errors(true);                                           \
      return;                                                                \
    }                                                                        \
    break;                                                                   \
  }
#include "toolchain/parse/node_kind.def"
    }
  }

  // Pop information for the file-level scope.
  sem_ir.set_top_inst_block_id(context.inst_block_stack().Pop());
  context.PopScope();

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

// Returns a key form of the package object.
static auto GetImportKey(UnitInfo& unit_info,
                         Parse::Tree::PackagingNames package) -> ImportKey {
  auto* stores = unit_info.unit->value_stores;
  return {package.package_id.is_valid()
              ? stores->identifiers().Get(package.package_id)
              : "",
          package.library_id.is_valid()
              ? stores->string_literals().Get(package.library_id)
              : ""};
}

static constexpr llvm::StringLiteral ExplicitMainName = "Main";

// Marks an import as required on both the source and target file.
// TODO: When importing without a package name is supported, check that it's
// used correctly.
static auto TrackImport(
    llvm::DenseMap<ImportKey, UnitInfo*>& api_map,
    llvm::DenseMap<ImportKey, Parse::Node>* explicit_import_map,
    UnitInfo& unit_info, Parse::Tree::PackagingNames import) -> void {
  auto import_key = GetImportKey(unit_info, import);

  // Specialize the error for imports from `Main`.
  if (import_key.first == ExplicitMainName) {
    // Implicit imports will have already warned.
    if (explicit_import_map) {
      CARBON_DIAGNOSTIC(ImportMainPackage, Error,
                        "Cannot import `Main` from other packages.");
      unit_info.emitter.Emit(import.node, ImportMainPackage);
    }
    return;
  }

  if (explicit_import_map) {
    // Check for redundant imports.
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

    // Check for explicit imports of the same library. The ID comparison is okay
    // in this case because both come from the same file.
    auto packaging = unit_info.unit->parse_tree->packaging_directive();
    if (packaging && import.package_id == packaging->names.package_id &&
        import.library_id == packaging->names.library_id) {
      CARBON_DIAGNOSTIC(ExplicitImportApi, Error,
                        "Explicit import `api` of `impl` file is redundant "
                        "with implicit import.");
      CARBON_DIAGNOSTIC(ImportSelf, Error, "File cannot import itself.");
      unit_info.emitter.Emit(
          import.node, packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl
                           ? ExplicitImportApi
                           : ImportSelf);
      return;
    }
  }

  if (auto api = api_map.find(import_key); api != api_map.end()) {
    unit_info.imports.push_back({import.node, api->second});
    ++unit_info.imports_remaining;
    api->second->incoming_imports.push_back(&unit_info);
  } else {
    CARBON_DIAGNOSTIC(LibraryApiNotFound, Error,
                      "Corresponding API not found.");
    CARBON_DIAGNOSTIC(ImportNotFound, Error, "Imported API not found.");
    unit_info.emitter.Emit(
        import.node, explicit_import_map ? ImportNotFound : LibraryApiNotFound);
  }
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

  // Create a map of APIs which might be imported.
  llvm::DenseMap<ImportKey, UnitInfo*> api_map;
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.unit->parse_tree->packaging_directive();
    if (packaging) {
      auto import_key = GetImportKey(unit_info, packaging->names);
      // Catch explicit `Main` errors before they become marked as possible
      // APIs.
      if (import_key.first == ExplicitMainName) {
        CARBON_DIAGNOSTIC(
            ExplicitMainPackage, Error,
            "Default `Main` library must omit `package` directive.");
        CARBON_DIAGNOSTIC(
            ExplicitMainLibrary, Error,
            "Use `library` directive in `Main` package libraries.");
        unit_info.emitter.Emit(packaging->names.node,
                               import_key.second.empty() ? ExplicitMainPackage
                                                         : ExplicitMainLibrary);
        continue;
      }

      if (packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl) {
        continue;
      }

      auto [entry, success] = api_map.insert({import_key, &unit_info});
      if (!success) {
        // TODO: Cross-reference the source, deal with library, etc.
        CARBON_DIAGNOSTIC(DuplicateLibraryApi, Error,
                          "Library's API declared in more than one file.");
        unit_info.emitter.Emit(packaging->names.node, DuplicateLibraryApi);
      }
    }
  }

  // Mark down imports for all files.
  llvm::SmallVector<UnitInfo*> ready_to_check;
  ready_to_check.reserve(units.size());
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.unit->parse_tree->packaging_directive();
    if (packaging && packaging->api_or_impl == Parse::Tree::ApiOrImpl::Impl) {
      // An `impl` has an implicit import of its `api`.
      TrackImport(api_map, nullptr, unit_info, packaging->names);
    }

    llvm::DenseMap<ImportKey, Parse::Node> explicit_import_map;
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
    CheckParseTree(builtin_ir, *unit_info, vlog_stream);
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
        for (auto* import_it = unit_info.imports.begin();
             import_it != unit_info.imports.end();) {
          auto* import_unit = import_it->second->unit;
          if (*import_unit->sem_ir) {
            ++import_it;
          } else {
            CARBON_DIAGNOSTIC(ImportCycleDetected, Error,
                              "Import cannot be used due to a cycle. Cycle "
                              "must be fixed to import.");
            unit_info.emitter.Emit(import_it->first, ImportCycleDetected);
            import_it = unit_info.imports.erase(import_it);
          }
        }
      }
    }

    // Check the remaining file contents, which are probably broken due to
    // incomplete imports.
    for (auto& unit_info : unit_infos) {
      if (unit_info.imports_remaining > 0) {
        CheckParseTree(builtin_ir, unit_info, vlog_stream);
      }
    }
  }
}

}  // namespace Carbon::Check
