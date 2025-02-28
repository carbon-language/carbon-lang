// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/impl.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ImplStore::GetOrAddLookupBucket(const Impl& impl) -> LookupBucketRef {
  auto self_id = sem_ir_.constant_values().GetConstantInstId(impl.self_id);
  InterfaceId interface_id = InterfaceId::None;
  SpecificId specific_id = SpecificId::None;
  auto facet_type_id = TypeId::ForTypeConstant(
      sem_ir_.constant_values().Get(impl.constraint_id));
  if (auto facet_type =
          sem_ir_.types().TryGetAs<SemIR::FacetType>(facet_type_id)) {
    const SemIR::FacetTypeInfo& facet_type_info =
        sem_ir_.facet_types().Get(facet_type->facet_type_id);
    if (auto interface_type = facet_type_info.TryAsSingleInterface()) {
      interface_id = interface_type->interface_id;
      specific_id = interface_type->specific_id;
    }
  }
  return LookupBucketRef(
      *this, lookup_
                 .Insert(std::tuple{self_id, interface_id, specific_id},
                         [] { return ImplOrLookupBucketId::None; })
                 .value());
}

}  // namespace Carbon::SemIR
