// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/generic.h"

namespace Carbon::SemIR {

class GenericInstanceStore::KeyContext
    : public TranslatingKeyContext<KeyContext> {
 public:
  // A lookup key for a generic instance.
  struct Key {
    GenericId generic_id;
    InstBlockId args_id;

    friend auto operator==(const Key&, const Key&) -> bool = default;
  };

  explicit KeyContext(llvm::ArrayRef<GenericInstance> instances)
      : instances_(instances) {}

  auto TranslateKey(GenericInstanceId id) const -> Key {
    const auto& instance = instances_[id.index];
    return {.generic_id = instance.generic_id, .args_id = instance.args_id};
  }

 private:
  llvm::ArrayRef<GenericInstance> instances_;
};

auto GenericInstanceStore::GetOrAdd(GenericId generic_id, InstBlockId args_id)
    -> GenericInstanceId {
  return lookup_table_
      .Insert(
          KeyContext::Key{.generic_id = generic_id, .args_id = args_id},
          [&] {
            return generic_instances_.Add(
                {.generic_id = generic_id, .args_id = args_id});
          },
          KeyContext(generic_instances_.array_ref()))
      .key();
}

}  // namespace Carbon::SemIR
