// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_
#define CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_

#include "common/hashing.h"
#include "common/set.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct EntityName : public Printable<EntityName> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", parent_scope: " << parent_scope_id
        << ", index: " << bind_index << "}";
  }

  friend auto CarbonHashtableEq(const EntityName& lhs, const EntityName& rhs)
      -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(EntityName)) == 0;
  }

  // The name.
  NameId name_id;
  // The parent scope.
  NameScopeId parent_scope_id;
  // The index for a compile-time binding. Invalid for a runtime binding, or
  // for a symbolic binding, like `.Self`, that does not correspond to a generic
  // parameter (and therefore has no index).
  CompileTimeBindIndex bind_index;
};

// Hashing for EntityName. See common/hashing.h.
inline auto CarbonHashValue(const EntityName& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(value);
  return static_cast<HashCode>(hasher);
}

// Value store for EntityName. In addition to the regular ValueStore
// functionality, this can provide optional canonical IDs for EntityNames.
struct EntityNameStore : public ValueStore<EntityNameId> {
 public:
  // Convert an ID to a canonical ID. All calls to this with equivalent
  // `EntityName`s will return the same `EntityNameId`.
  auto MakeCanonical(EntityNameId id) -> EntityNameId;

 private:
  class KeyContext;

  Set<EntityNameId, /*SmallSize=*/0, KeyContext> canonical_ids_;
};

class EntityNameStore::KeyContext : public TranslatingKeyContext<KeyContext> {
 public:
  explicit KeyContext(const EntityNameStore* store) : store_(store) {}

  // Note that it is safe to return a `const` reference here as the underlying
  // object's lifetime is provided by the `store_`.
  auto TranslateKey(EntityNameId id) const -> const EntityName& {
    return store_->Get(id);
  }

 private:
  const EntityNameStore* store_;
};

inline auto EntityNameStore::MakeCanonical(EntityNameId id) -> EntityNameId {
  return canonical_ids_.Insert(id, KeyContext(this)).key();
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_
