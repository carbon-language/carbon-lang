// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"

namespace Carbon::SemIR {

ImplStore::ImplStore(File& sem_ir)
    : sem_ir_(sem_ir), values_(sem_ir.check_ir_id()) {}

auto ImplStore::GetOrAddLookupBucket(const Impl& impl) -> LookupBucketRef {
  auto self_id = sem_ir_.constant_values().GetConstantInstId(impl.self_id);
  InterfaceId interface_id = InterfaceId::None;
  SpecificId specific_id = SpecificId::None;
  auto facet_type_id = TypeId::ForTypeConstant(
      sem_ir_.constant_values().Get(impl.constraint_id));
  if (auto facet_type = sem_ir_.types().TryGetAs<FacetType>(facet_type_id)) {
    const FacetTypeInfo& facet_type_info =
        sem_ir_.facet_types().Get(facet_type->facet_type_id);
    if (auto single = facet_type_info.TryAsSingleExtend()) {
      CARBON_KIND_SWITCH(*single) {
        case CARBON_KIND(SemIR::SpecificInterface interface): {
          interface_id = interface.interface_id;
          specific_id = interface.specific_id;
          break;
        }
        case CARBON_KIND(SemIR::SpecificNamedConstraint _): {
          // TODO: Handle named constraints which resolve to a single interface
          // in the IdentifiedFacetType.
          break;
        }
      }
    }
  }
  return LookupBucketRef(
      *this, lookup_
                 .Insert(std::tuple{self_id, interface_id, specific_id},
                         [] { return ImplOrLookupBucketId::None; })
                 .value());
}

}  // namespace Carbon::SemIR
