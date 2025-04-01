// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CHECK_UNIT_H_
#define CARBON_TOOLCHAIN_CHECK_CHECK_UNIT_H_

#include "common/map.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/check.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_emitter.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

struct UnitAndImports;

// A file's imports corresponding to a single package, for
// `UnitAndImports::package_imports`.
struct PackageImports {
  // A given import within the file, with its destination.
  struct Import {
    Parse::Tree::PackagingNames names;
    UnitAndImports* unit_info;
  };

  // Use the constructor so that the SmallVector is only constructed
  // as-needed.
  explicit PackageImports(PackageNameId package_id,
                          Parse::AnyPackagingDeclId node_id)
      : package_id(package_id), node_id(node_id) {}

  // The identifier of the imported package.
  PackageNameId package_id;
  // The first `package` or `import` declaration in the file, which declared the
  // package's identifier (even if the import failed). Used for associating
  // diagnostics not specific to a single import.
  Parse::AnyPackagingDeclId node_id;
  // The associated `import` instruction. Has a value after a file is checked.
  SemIR::InstId import_decl_id = SemIR::InstId::None;
  // Whether there's an import that failed to load.
  bool has_load_error = false;
  // The list of valid imports.
  llvm::SmallVector<Import> imports;
};

// Contains information accumulated while checking a `Unit` (primarily import
// information), in addition to the `Unit` itself.
struct UnitAndImports {
  // Converts a `NodeId` to a diagnostic location for `UnitAndImports`.
  class NodeEmitter : public Diagnostics::Emitter<Parse::NodeId> {
   public:
    explicit NodeEmitter(Diagnostics::Consumer* consumer,
                         Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter)
        : Emitter(consumer),
          tree_and_subtrees_getter_(tree_and_subtrees_getter) {}

   protected:
    auto ConvertLoc(Parse::NodeId node_id, ContextFnT /*context_fn*/) const
        -> Diagnostics::ConvertedLoc override {
      return tree_and_subtrees_getter_().NodeToDiagnosticLoc(
          node_id, /*token_only=*/false);
    }

   private:
    Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter_;
  };

  explicit UnitAndImports(SemIR::CheckIRId check_ir_id, Unit& unit)
      : check_ir_id(check_ir_id),
        unit(&unit),
        err_tracker(*unit.consumer),
        emitter(&err_tracker, unit.tree_and_subtrees_getter) {}

  auto parse_tree() -> const Parse::Tree& { return unit->sem_ir->parse_tree(); }
  auto source() -> const SourceBuffer& {
    return parse_tree().tokens().source();
  }

  SemIR::CheckIRId check_ir_id;
  Unit* unit;

  // Emitter information.
  Diagnostics::ErrorTrackingConsumer err_tracker;
  NodeEmitter emitter;

  // List of the outgoing imports. If a package includes unavailable library
  // imports, it has an entry with has_load_error set. Invalid imports (for
  // example, `import Main;`) aren't added because they won't add identifiers to
  // name lookup.
  llvm::SmallVector<PackageImports> package_imports;

  // A map of the package names to the outgoing imports above.
  Map<PackageNameId, int32_t> package_imports_map;

  // List of the `import Cpp` imports.
  llvm::SmallVector<Parse::Tree::PackagingNames> cpp_import_names;

  // The remaining number of imports which must be checked before this unit can
  // be processed.
  int32_t imports_remaining = 0;

  // A list of incoming imports. This will be empty for `impl` files, because
  // imports only touch `api` files.
  llvm::SmallVector<UnitAndImports*> incoming_imports;

  // The corresponding `api` unit if this is an `impl` file. The entry should
  // also be in the corresponding `PackageImports`.
  UnitAndImports* api_for_impl = nullptr;

  // Whether the unit has been checked.
  bool is_checked = false;
};

// Handles checking of a single unit. Requires that all dependencies have been
// checked.
//
// This mainly splits out the single-unit logic from the higher level cross-unit
// logic in check.cpp.
class CheckUnit {
 public:
  explicit CheckUnit(
      UnitAndImports* unit_and_imports,
      llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
      llvm::raw_ostream* vlog_stream);

  // Produces and checks the IR for the provided unit.
  auto Run() -> void;

 private:
  // Add imports to the root block.
  auto InitPackageScopeAndImports() -> void;

  // Collects direct imports, for CollectTransitiveImports.
  auto CollectDirectImports(llvm::SmallVector<SemIR::ImportIR>& results,
                            llvm::MutableArrayRef<int> ir_to_result_index,
                            SemIR::InstId import_decl_id,
                            const PackageImports& imports, bool is_local)
      -> void;

  // Collects transitive imports, handling deduplication. These will be unified
  // between local_imports and api_imports.
  auto CollectTransitiveImports(SemIR::InstId import_decl_id,
                                const PackageImports* local_imports,
                                const PackageImports* api_imports)
      -> llvm::SmallVector<SemIR::ImportIR>;

  // Imports the current package.
  auto ImportCurrentPackage(SemIR::InstId package_inst_id,
                            SemIR::TypeId namespace_type_id) -> void;

  // Imports all other Carbon packages (excluding the current package).
  auto ImportOtherPackages(SemIR::TypeId namespace_type_id) -> void;

  // Checks that each required declaration is available. This applies for
  // declarations that should exist in an owning library, for which an extern
  // declaration exists that assigns ownership to the current API.
  auto CheckRequiredDeclarations() -> void;

  // Checks that each required definition is available. If the definition can be
  // generated by resolving a specific, does so, otherwise emits a diagnostic
  // for each declaration in context.definitions_required_by_decl() and
  // context.definitions_required_by_use that doesn't have a definition.
  auto CheckRequiredDefinitions() -> void;

  // Does work after processing the parse tree, such as finishing the IR and
  // checking for missing contents.
  auto FinishRun() -> void;

  // Loops over all nodes in the tree. On some errors, this may return early,
  // for example if an unrecoverable state is encountered.
  // NOLINTNEXTLINE(readability-function-size)
  auto ProcessNodeIds() -> bool;

  UnitAndImports* unit_and_imports_;
  // The number of IRs being checked in total.
  int total_ir_count_;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs_;
  llvm::raw_ostream* vlog_stream_;

  DiagnosticEmitter emitter_;
  Context context_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CHECK_UNIT_H_
