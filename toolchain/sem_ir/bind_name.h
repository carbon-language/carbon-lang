// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_
#define CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_

#include "common/hashing.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct BindNameInfo : public Printable<BindNameInfo> {
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

// Hashing for BindNameInfo. See common/hashing.h.
inline auto CarbonHashValue(const BindNameInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.Hash(value);
  return static_cast<HashCode>(hasher);
}

// DenseMapInfo for BindNameInfo.
struct BindNameInfoDenseMapInfo {
  static auto getEmptyKey() -> BindNameInfo {
    return BindNameInfo{.name_id = NameId::Invalid,
                        .parent_scope_id = NameScopeId::Invalid,
                        .bind_index = CompileTimeBindIndex(
                            CompileTimeBindIndex::InvalidIndex - 1)};
  }
  static auto getTombstoneKey() -> BindNameInfo {
    return BindNameInfo{.name_id = NameId::Invalid,
                        .parent_scope_id = NameScopeId::Invalid,
                        .bind_index = CompileTimeBindIndex(
                            CompileTimeBindIndex::InvalidIndex - 2)};
  }
  static auto getHashValue(const BindNameInfo& val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(const BindNameInfo& lhs, const BindNameInfo& rhs)
      -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(BindNameInfo)) == 0;
  }
};

// Value store for BindNameInfo. In addition to the regular ValueStore
// functionality, this can provide optional canonical IDs for BindNameInfos.
struct BindNameStore : public ValueStore<BindNameId> {
 public:
  // Convert an ID to a canonical ID. All calls to this with equivalent
  // `BindNameInfo`s will return the same `BindNameId`.
  auto MakeCanonical(BindNameId id) -> BindNameId {
    return canonical_ids_.insert({Get(id), id}).first->second;
  }

 private:
  llvm::DenseMap<BindNameInfo, BindNameId, BindNameInfoDenseMapInfo>
      canonical_ids_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BIND_NAME_H_
