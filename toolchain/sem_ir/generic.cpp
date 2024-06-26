// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

struct GenericInstanceStore::KeyContext {
  // A lookup key for a generic instance.
  struct Key {
    GenericId generic_id;
    InstBlockId args_id;

    friend auto operator==(const Key&, const Key&) -> bool = default;
  };

  llvm::ArrayRef<GenericInstance> instances;

  auto AsKey(GenericInstanceId id) const -> Key {
    const auto& instance = instances[id.index];
    return {.generic_id = instance.generic_id, .args_id = instance.args_id};
  }
  static auto AsKey(Key key) -> Key { return key; }

  template <typename KeyT>
  auto HashKey(KeyT key, uint64_t seed) const -> HashCode {
    return HashValue(AsKey(key), seed);
  }

  template <typename LHSKeyT, typename RHSKeyT>
  auto KeyEq(const LHSKeyT& lhs_key, const RHSKeyT& rhs_key) const -> bool {
    return AsKey(lhs_key) == AsKey(rhs_key);
  }
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
          KeyContext{.instances = generic_instances_.array_ref()})
      .key();
}

}  // namespace Carbon::SemIR
