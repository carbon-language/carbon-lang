// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl_lookup.h"

#include "toolchain/check/deduce.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"

namespace Carbon::Check {

auto LookupInterfaceWitness(Context& context, SemIR::LocId loc_id,
                            SemIR::ConstantId type_const_id,
                            SemIR::ConstantId interface_const_id)
    -> SemIR::InstId {
  // TODO: Add a better impl lookup system. At the very least, we should only be
  // considering impls that are for the same interface we're querying. We can
  // also skip impls that mention any types that aren't part of our impl query.
  for (const auto& impl : context.impls().array_ref()) {
    auto specific_id = SemIR::SpecificId::Invalid;
    if (impl.generic_id.is_valid()) {
      specific_id = DeduceImplArguments(context, loc_id, impl, type_const_id,
                                        interface_const_id);
      if (!specific_id.is_valid()) {
        continue;
      }
    }
    if (!context.constant_values().AreEqualAcrossDeclarations(
            SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                              impl.self_id),
            type_const_id)) {
      continue;
    }
    if (!context.constant_values().AreEqualAcrossDeclarations(
            SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                              impl.constraint_id),
            interface_const_id)) {
      // TODO: An impl of a constraint type should be treated as implementing
      // the constraint's interfaces.
      continue;
    }
    if (!impl.witness_id.is_valid()) {
      // TODO: Diagnose if the impl isn't defined yet?
      return SemIR::InstId::Invalid;
    }
    LoadImportRef(context, impl.witness_id);
    if (specific_id.is_valid()) {
      // We need a definition of the specific `impl` so we can access its
      // witness.
      ResolveSpecificDefinition(context, specific_id);
    }
    return context.constant_values().GetInstId(
        SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                          impl.witness_id));
  }
  return SemIR::InstId::Invalid;
}

}  // namespace Carbon::Check
