// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_
#define CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_

#include "common/hashing.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct EntityName : public Printable<EntityName> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", parent_scope: " << parent_scope_id
        << ", index: " << bind_index << "}";
  }

  // The name.
  NameId name_id;
  // The parent scope.
  NameScopeId parent_scope_id;
  // The index for a compile-time binding. Invalid for a runtime binding.
  CompileTimeBindIndex bind_index;
};

// Hashing for EntityName. See common/hashing.h.
inline auto CarbonHashValue(const EntityName& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.Hash(value);
  return static_cast<HashCode>(hasher);
}

// DenseMapInfo for EntityName.
struct EntityNameDenseMapInfo {
  static auto getEmptyKey() -> EntityName {
    return EntityName{.name_id = NameId::Invalid,
                      .parent_scope_id = NameScopeId::Invalid,
                      .bind_index = CompileTimeBindIndex(
                          CompileTimeBindIndex::InvalidIndex - 1)};
  }
  static auto getTombstoneKey() -> EntityName {
    return EntityName{.name_id = NameId::Invalid,
                      .parent_scope_id = NameScopeId::Invalid,
                      .bind_index = CompileTimeBindIndex(
                          CompileTimeBindIndex::InvalidIndex - 2)};
  }
  static auto getHashValue(const EntityName& val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(const EntityName& lhs, const EntityName& rhs) -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(EntityName)) == 0;
  }
};

// Value store for EntityName. In addition to the regular ValueStore
// functionality, this can provide optional canonical IDs for EntityNames.
struct EntityNameStore : public ValueStore<EntityNameId> {
 public:
  // Convert an ID to a canonical ID. All calls to this with equivalent
  // `EntityName`s will return the same `BindNameId`.
  auto MakeCanonical(EntityNameId id) -> EntityNameId {
    return canonical_ids_.insert({Get(id), id}).first->second;
  }

 private:
  llvm::DenseMap<EntityName, EntityNameId, EntityNameDenseMapInfo>
      canonical_ids_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_
