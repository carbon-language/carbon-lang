// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/impl.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ImplStore::GetOrAddLookupBucket(const Impl& impl) -> LookupBucketRef {
  auto self_id = sem_ir_.constant_values().GetConstantInstId(impl.self_id);
  auto constraint_id =
      sem_ir_.constant_values().GetConstantInstId(impl.constraint_id);
  return LookupBucketRef(
      *this, lookup_
                 .Insert(std::pair{self_id, constraint_id},
                         [] { return ImplOrLookupBucketId::Invalid; })
                 .value());
}

}  // namespace Carbon::SemIR
