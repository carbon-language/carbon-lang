// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto GenericInstanceStore::GetOrAdd(GenericId generic_id,
                                    InstBlockId args_id)
    -> GenericInstanceId {
  return lookup_table_
      .Insert(
          Key{.generic_id = generic_id, .args_id = args_id},
          [&] {
            return generic_instances_.Add(
                {.generic_id = generic_id, .args_id = args_id});
          },
          KeyContext{.instances = generic_instances_.array_ref()})
      .key();
}

}  // namespace Carbon::SemIR
