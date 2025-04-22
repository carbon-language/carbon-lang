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
        << ", index: " << bind_index_value << ", is_template: " << is_template
        << "}";
  }

  friend auto CarbonHashtableEq(const EntityName& lhs, const EntityName& rhs)
      -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(EntityName)) == 0;
  }

  // The index of the binding, if this is the name of a symbolic binding, or
  // `None` otherwise. This is also `None` for a `.Self` symbolic binding,
  // because such a binding is not assigned an index.
  auto bind_index() const -> CompileTimeBindIndex {
    return CompileTimeBindIndex(bind_index_value);
  }

  // The name.
  NameId name_id;
  // The parent scope.
  NameScopeId parent_scope_id;

  // TODO: The following two fields are only meaningful for a symbolic binding.
  // Consider splitting them off into a separate type so that we don't store
  // them for other kinds of `EntityName`.

  // The bind_index() value, unwrapped so it can be stored in a bit-field.
  int32_t bind_index_value : 31 = CompileTimeBindIndex::None.index;
  // Whether this binding is a template parameter.
  bool is_template : 1 = false;
};

// Hashing for EntityName. See common/hashing.h.
inline auto CarbonHashValue(const EntityName& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(value);
  return static_cast<HashCode>(hasher);
}

using EntityNameStore = CanonicalValueStore<EntityNameId>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_
