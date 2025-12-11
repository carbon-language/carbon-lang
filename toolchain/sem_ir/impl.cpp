// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"

namespace Carbon::SemIR {

ImplStore::ImplStore(File& sem_ir)
    : sem_ir_(sem_ir), values_(sem_ir.check_ir_id()) {}

auto ImplStore::GetOrAddLookupBucket(const Impl& impl) -> LookupBucketRef {
  auto self_id = sem_ir_.constant_values().GetConstantInstId(impl.self_id);
  auto impl_as_interface = SpecificInterface::None;
  auto facet_type_type_id = TypeId::ForTypeConstant(
      sem_ir_.constant_values().Get(impl.constraint_id));
  if (auto facet_type =
          sem_ir_.types().TryGetAs<FacetType>(facet_type_type_id)) {
    auto identified_id =
        sem_ir_.identified_facet_types().TryGetId(facet_type->facet_type_id);
    if (identified_id.has_value()) {
      const auto& identified =
          sem_ir_.identified_facet_types().Get(identified_id);
      if (identified.is_valid_impl_as_target()) {
        impl_as_interface = identified.impl_as_target_interface();
      }
    }
  }
  return LookupBucketRef(*this,
                         lookup_
                             .Insert(std::pair{self_id, impl_as_interface},
                                     [] { return ImplOrLookupBucketId::None; })
                             .value());
}

}  // namespace Carbon::SemIR
