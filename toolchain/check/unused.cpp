// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/unused.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"

namespace Carbon::Check {

auto CheckUnusedBinding(Context& context, SemIR::NameId name_id,
                        const LexicalLookup::Result& result) -> void {
  // Don't warn about the special name `_`. Other special names, such as `self`,
  // are warned if unused. `.Self` is also excluded as it is often implicit.
  if (name_id == SemIR::NameId::Underscore ||
      name_id == SemIR::NameId::PeriodSelf) {
    return;
  }

  // Don't warn about names that aren't in the current file.
  auto decl_loc = context.insts().GetCanonicalLocId(result.inst_id);
  if (decl_loc.kind() == SemIR::LocId::Kind::ImportIRInstId) {
    return;
  }

  auto binding = context.insts().TryGetAs<SemIR::AnyBinding>(result.inst_id);
  if (!binding) {
    return;
  }

  const auto& entity_name = context.entity_names().Get(binding->entity_name_id);
  if (entity_name.is_unused) {
    if (result.use_loc_id.has_value()) {
      CARBON_DIAGNOSTIC_ON_SCOPE(UnusedButUsed, Error,
                                 "variable `{0}` marked `unused` but used",
                                 SemIR::NameId);
      CARBON_DIAGNOSTIC(UnusedButUsedHere, Note, "usage here");
      context.emitter()
          .Build(decl_loc, UnusedButUsed, name_id)
          .Note(result.use_loc_id, UnusedButUsedHere)
          .Emit();
    }
  } else {
    if (!result.use_loc_id.has_value() && result.is_decl_reachable) {
      CARBON_DIAGNOSTIC_ON_SCOPE(UnusedBinding, Warning, "binding `{0}` unused",
                                 SemIR::NameId);
      context.emitter().Emit(decl_loc, UnusedBinding, name_id);
    }
  }
}

}  // namespace Carbon::Check
