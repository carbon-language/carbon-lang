// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"

namespace Carbon::SemIR {

ImplStore::ImplStore(File& sem_ir)
    : sem_ir_(sem_ir), values_(sem_ir.check_ir_id()) {}

auto ImplStore::GetOrAddLookupBucket(const Impl& impl) -> LookupBucketRef {
  auto self_const_id = sem_ir_.constant_values().Get(impl.self_id);
  return LookupBucketRef(*this,
                         lookup_
                             .Insert(std::pair{self_const_id, impl.interface},
                                     [] { return ImplOrLookupBucketId::None; })
                             .value());
}

}  // namespace Carbon::SemIR

namespace Carbon {
template class ValueStore<SemIR::ImplId, SemIR::Impl, Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
