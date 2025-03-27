// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/check.h"

#include "common/check.h"
#include "common/map.h"
#include "toolchain/check/check_unit.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_emitter.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// The package and library names, used as map keys.
using ImportKey = std::pair<llvm::StringRef, llvm::StringRef>;

// Returns a key form of the package object. file_package_id is only used for
// imports, not the main package declaration; as a consequence, it will be
// `None` for the main package declaration.
static auto GetImportKey(UnitAndImports& unit_info,
                         PackageNameId file_package_id,
                         Parse::Tree::PackagingNames names) -> ImportKey {
  auto* stores = unit_info.unit->value_stores;
  PackageNameId package_id =
      names.package_id.has_value() ? names.package_id : file_package_id;
  llvm::StringRef package_name;
  if (package_id.has_value()) {
    auto package_ident_id = package_id.AsIdentifierId();
    package_name = package_ident_id.has_value()
                       ? stores->identifiers().Get(package_ident_id)
                       : package_id.AsSpecialName();
  }
  llvm::StringRef library_name =
      names.library_id.has_value()
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
static auto TrackImport(Map<ImportKey, UnitAndImports*>& api_map,
                        Map<ImportKey, Parse::NodeId>* explicit_import_map,
                        UnitAndImports& unit_info,
                        Parse::Tree::PackagingNames import, bool fuzzing)
    -> void {
  const auto& packaging = unit_info.parse_tree().packaging_decl();

  PackageNameId file_package_id =
      packaging ? packaging->names.package_id : PackageNameId::None;
  const auto import_key = GetImportKey(unit_info, file_package_id, import);
  const auto& [import_package_name, import_library_name] = import_key;

  if (import_package_name == CppPackageName) {
    if (import_library_name.empty()) {
      CARBON_DIAGNOSTIC(CppInteropMissingLibrary, Error,
                        "`Cpp` import missing library");
      unit_info.emitter.Emit(import.node_id, CppInteropMissingLibrary);
      return;
    }
    if (fuzzing) {
      // Clang is not crash-resilient.
      CARBON_DIAGNOSTIC(CppInteropFuzzing, Error,
                        "`Cpp` import found during fuzzing");
      unit_info.emitter.Emit(import.node_id, CppInteropFuzzing);
      return;
    }
    unit_info.cpp_import_names.push_back(import);
    return;
  }

  // True if the import has `Main` as the package name, even if it comes from
  // the file's packaging (diagnostics may differentiate).
  bool is_explicit_main = import_package_name == MainPackageName;

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
        !packaging || !packaging->names.package_id.has_value();
    // True if the import is using implicit "current package" syntax (by
    // omitting an explicit package name).
    bool is_import_implicit_current_package = !import.package_id.has_value();
    // True if the import is using `default` library syntax.
    bool is_import_default_library = !import.library_id.has_value();
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
        PackageImports(import.package_id, import.node_id));
    return index;
  };
  auto insert_result =
      unit_info.package_imports_map.Insert(import.package_id, create_imports);
  PackageImports& package_imports =
      unit_info.package_imports[insert_result.value()];

  if (auto api_lookup = api_map.Lookup(import_key)) {
    // Add references between the file and imported api.
    UnitAndImports* api = api_lookup.value();
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
    if (!explicit_import_map && import_package_name == CppPackageName) {
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
    llvm::MutableArrayRef<UnitAndImports> unit_infos)
    -> Map<ImportKey, UnitAndImports*> {
  Map<ImportKey, UnitAndImports*> api_map;
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.parse_tree().packaging_decl();
    // An import key formed from the `package` or `library` declaration. Or, for
    // Main//default, a placeholder key.
    auto import_key = packaging ? GetImportKey(unit_info, PackageNameId::None,
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
            insert_result.value()->source().filename();
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
          // Use `NodeId::None` because there's no node to associate with.
          unit_info.emitter.Emit(Parse::NodeId::None, DuplicateMainApi,
                                 prev_filename.str());
        }
      }
    }

    // Validate file extensions. Note imports rely the packaging declaration,
    // not the extension. If the input is not a regular file, for example
    // because it is stdin, no filename checking is performed.
    if (unit_info.source().is_regular_file()) {
      auto filename = unit_info.source().filename();
      static constexpr llvm::StringLiteral ApiExt = ".carbon";
      static constexpr llvm::StringLiteral ImplExt = ".impl.carbon";
      bool is_api_with_impl_ext = !is_impl && filename.ends_with(ImplExt);
      auto want_ext = is_impl ? ImplExt : ApiExt;
      if (is_api_with_impl_ext || !filename.ends_with(want_ext)) {
        CARBON_DIAGNOSTIC(
            IncorrectExtension, Error,
            "file extension of `{0:.impl|}.carbon` required for {0:`impl`|api}",
            Diagnostics::BoolAsSelect);
        auto diag = unit_info.emitter.Build(
            packaging ? packaging->names.node_id : Parse::NodeId::None,
            IncorrectExtension, is_impl);
        if (is_api_with_impl_ext) {
          CARBON_DIAGNOSTIC(
              IncorrectExtensionImplNote, Note,
              "file extension of `.impl.carbon` only allowed for `impl`");
          diag.Note(Parse::NodeId::None, IncorrectExtensionImplNote);
        }
        diag.Emit();
      }
    }
  }
  return api_map;
}

auto CheckParseTrees(llvm::MutableArrayRef<Unit> units, bool prelude_import,
                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                     llvm::raw_ostream* vlog_stream, bool fuzzing) -> void {
  // UnitAndImports is big due to its SmallVectors, so we default to 0 on the
  // stack.
  llvm::SmallVector<UnitAndImports, 0> unit_infos;
  llvm::SmallVector<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters;
  unit_infos.reserve(units.size());
  tree_and_subtrees_getters.reserve(units.size());
  for (auto [i, unit] : llvm::enumerate(units)) {
    unit_infos.emplace_back(SemIR::CheckIRId(i), unit);
    tree_and_subtrees_getters.push_back(unit.tree_and_subtrees_getter);
  }

  Map<ImportKey, UnitAndImports*> api_map =
      BuildApiMapAndDiagnosePackaging(unit_infos);

  // Mark down imports for all files.
  llvm::SmallVector<UnitAndImports*> ready_to_check;
  ready_to_check.reserve(units.size());
  for (auto& unit_info : unit_infos) {
    const auto& packaging = unit_info.parse_tree().packaging_decl();
    if (packaging && packaging->is_impl) {
      // An `impl` has an implicit import of its `api`.
      auto implicit_names = packaging->names;
      implicit_names.package_id = PackageNameId::None;
      TrackImport(api_map, nullptr, unit_info, implicit_names, fuzzing);
    }

    Map<ImportKey, Parse::NodeId> explicit_import_map;

    // Add the prelude import. It's added to explicit_import_map so that it can
    // conflict with an explicit import of the prelude.
    if (prelude_import &&
        !(packaging && packaging->names.package_id == PackageNameId::Core)) {
      auto prelude_id =
          unit_info.unit->value_stores->string_literal_values().Add("prelude");
      TrackImport(api_map, &explicit_import_map, unit_info,
                  {.node_id = Parse::NoneNodeId(),
                   .package_id = PackageNameId::Core,
                   .library_id = prelude_id},
                  fuzzing);
    }

    for (const auto& import : unit_info.parse_tree().imports()) {
      TrackImport(api_map, &explicit_import_map, unit_info, import, fuzzing);
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
    CheckUnit(unit_info, tree_and_subtrees_getters, fs, vlog_stream).Run();
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
        CheckUnit(&unit_info, tree_and_subtrees_getters, fs, vlog_stream).Run();
      }
    }
  }
}

}  // namespace Carbon::Check
