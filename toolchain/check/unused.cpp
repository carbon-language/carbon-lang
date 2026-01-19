// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/unused.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"

namespace Carbon::Check {

auto CheckUnusedBindings(Context& context, ScopeStack::ScopeView scope)
    -> void {
  // Don't warn about unused names in the prelude.
  if (context.sem_ir().filename().contains("prelude")) {
    return;
  }

  // We sort the diagnostics to ensure deterministic output.
  struct UnusedDiag {
    SemIR::LocId loc_id;
    SemIR::NameId name_id;
    SemIR::LocId use_loc;
  };
  llvm::SmallVector<UnusedDiag> unused_defs;
  llvm::SmallVector<UnusedDiag> unused_but_used_defs;

  scope.names.ForEach([&](SemIR::NameId name_id) {
    auto& results = context.scope_stack().lexical_lookup().Get(name_id);
    CARBON_CHECK(!results.empty());
    auto& result = results.back();

    // We only care about the result in the current scope.
    if (result.scope_index != scope.index) {
      return;
    }

    // Don't warn about names that aren't in the current file.
    auto decl_loc = context.insts().GetCanonicalLocId(result.inst_id);
    if (decl_loc.kind() == SemIR::LocId::Kind::ImportIRInstId) {
      return;
    }

    auto inst = context.insts().Get(result.inst_id);
    std::optional<SemIR::EntityNameId> entity_name_id;

    if (auto binding = inst.TryAs<SemIR::AnyBinding>()) {
      entity_name_id = binding->entity_name_id;
    }

    if (!entity_name_id) {
      return;
    }

    // Don't warn about special names.
    if (name_id.AsSpecialNameId().has_value()) {
      return;
    }

    const auto& entity_name = context.entity_names().Get(*entity_name_id);
    if (entity_name.is_unused) {
      if (result.is_used) {
        unused_but_used_defs.push_back(
            {decl_loc, name_id, result.first_use_loc});
      }
    } else {
      if (!result.is_used && result.is_declared_reachable) {
        unused_defs.push_back({decl_loc, name_id, SemIR::LocId::None});
      }
    }
  });

  auto sort_diags = [&](const UnusedDiag& a, const UnusedDiag& b) {
    return a.loc_id.index < b.loc_id.index;
  };
  llvm::sort(unused_defs, sort_diags);
  llvm::sort(unused_but_used_defs, sort_diags);

  for (const auto& diag_data : unused_but_used_defs) {
    CARBON_DIAGNOSTIC(UnusedButUsed, Error,
                      "variable `{0}` is marked `unused` but is used",
                      SemIR::NameId);
    CARBON_DIAGNOSTIC(UnusedButUsedHere, Note, "usage is here");
    auto diag = context.emitter().Build(LocIdForDiagnostics(diag_data.loc_id),
                                        UnusedButUsed, diag_data.name_id);
    if (diag_data.use_loc.has_value()) {
      diag.Note(LocIdForDiagnostics(diag_data.use_loc), UnusedButUsedHere);
    }
    diag.Emit();
  }

  for (const auto& diag_data : unused_defs) {
    CARBON_DIAGNOSTIC(UnusedBinding, Warning, "binding `{0}` is unused",
                      SemIR::NameId);
    context.emitter().Emit(LocIdForDiagnostics(diag_data.loc_id), UnusedBinding,
                           diag_data.name_id);
  }
}

}  // namespace Carbon::Check
