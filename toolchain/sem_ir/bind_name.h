// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_
#define CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_

#include "common/hashing.h"
#include "common/set.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct BindNameInfo : public Printable<BindNameInfo> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", parent_scope: " << parent_scope_id
        << ", index: " << bind_index << "}";
  }

  friend auto CarbonHashtableEq(const BindNameInfo& lhs,
                                const BindNameInfo& rhs) -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(BindNameInfo)) == 0;
  }

  // The name.
  NameId name_id;
  // The parent scope.
  NameScopeId parent_scope_id;
  // The index for a compile-time binding. Invalid for a runtime binding.
  CompileTimeBindIndex bind_index;
};

// Hashing for BindNameInfo. See common/hashing.h.
inline auto CarbonHashValue(const BindNameInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(value);
  return static_cast<HashCode>(hasher);
}

// Value store for BindNameInfo. In addition to the regular ValueStore
// functionality, this can provide optional canonical IDs for BindNameInfos.
struct BindNameStore : public ValueStore<BindNameId> {
 public:
  // Convert an ID to a canonical ID. All calls to this with equivalent
  // `BindNameInfo`s will return the same `BindNameId`.
  auto MakeCanonical(BindNameId id) -> BindNameId;

 private:
  class KeyContext;

  Set<BindNameId, /*SmallSize=*/0, KeyContext> canonical_ids_;
};

class BindNameStore::KeyContext : public TranslatingKeyContext<KeyContext> {
 public:
  explicit KeyContext(const BindNameStore* store) : store_(store) {}

  // Note that it is safe to return a `const` reference here as the underlying
  // object's lifetime is provided by the `store_`.
  auto TranslateKey(BindNameId id) const -> const BindNameInfo& {
    return store_->Get(id);
  }

 private:
  const BindNameStore* store_;
};

inline auto BindNameStore::MakeCanonical(BindNameId id) -> BindNameId {
  return canonical_ids_.Insert(id, KeyContext(this)).key();
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_
