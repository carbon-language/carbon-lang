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

// Value store for EntityName. In addition to the regular ValueStore
// functionality, this can provide optional canonical IDs for EntityNames.
struct EntityNameStore : public ValueStore<EntityNameId> {
 public:
  // Adds an entity name for a symbolic binding.
  auto AddSymbolicBindingName(NameId name_id, NameScopeId parent_scope_id,
                              CompileTimeBindIndex bind_index, bool is_template)
      -> EntityNameId {
    return Add({.name_id = name_id,
                .parent_scope_id = parent_scope_id,
                .bind_index_value = bind_index.index,
                .is_template = is_template});
  }

  // Convert an `EntityName` to a canonical ID. All calls to this with
  // equivalent `EntityName`s will return the same `EntityNameId`. Same as
  // `MakeCanonical(Add(name))` except that no new `EntityName` is added if we
  // already have a canonical `EntityNameId` for that name.
  auto AddCanonical(EntityName name) -> EntityNameId;

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

inline auto EntityNameStore::AddCanonical(EntityName name) -> EntityNameId {
  return canonical_ids_
      .Insert(
          name, [&] { return Add(name); }, KeyContext(this))
      .key();
}

inline auto EntityNameStore::MakeCanonical(EntityNameId id) -> EntityNameId {
  return canonical_ids_.Insert(id, KeyContext(this)).key();
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_
