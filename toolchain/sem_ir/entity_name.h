// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_
#define CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_

#include "common/hashing.h"
#include "common/set.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct EntityName : public Printable<EntityName> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", parent_scope: " << parent_scope_id
        << ", index: " << bind_index_value << ", is_template: " << is_template
        << ", is_unused: " << is_unused;
    if (name_id == SemIR::NameId::PeriodSelf) {
      out << ", is_frozen_period_self: " << is_frozen_period_self << "}";
    }
    out << ", form: " << form_id << ", type: " << type_inst_id << "}";
  }

  // Returns a copy of this name with the fields that are not part of its
  // identity cleared. Two `EntityName`s that differ only in how the declared
  // type of the binding was written describe the same name.
  auto IdentityKey() const -> EntityName {
    EntityName key = *this;
    key.type_inst_id = TypeInstId::None;
    return key;
  }

  friend auto CarbonHashtableEq(const EntityName& lhs, const EntityName& rhs)
      -> bool {
    // This requires that there are no padding bits in the type. This is upheld
    // since it holds values all of the same size: each is 32 bits, with one
    // split into 29, 1, 1, and 1 bits.
    EntityName lhs_key = lhs.IdentityKey();
    EntityName rhs_key = rhs.IdentityKey();
    return std::memcmp(&lhs_key, &rhs_key, sizeof(EntityName)) == 0;
  }

  // Hashing for EntityName. See common/hashing.h.
  friend auto CarbonHashValue(const EntityName& value, uint64_t seed)
      -> HashCode {
    Hasher hasher(seed);
    hasher.HashRaw(value.IdentityKey());
    return static_cast<HashCode>(hasher);
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
  int32_t bind_index_value : 29 = CompileTimeBindIndex::None.index;
  // Whether this binding is a template parameter.
  bool is_template : 1 = false;
  // Whether this binding is marked unused.
  bool is_unused : 1 = false;
  // Whether this binding is a `.Self` symbolic binding, within the scope of the
  // `.Self` name during facet type under construction. Such bindings cannot be
  // replaced during identify.
  bool is_frozen_period_self : 1 = false;

  // The declared form of the binding. This is guaranteed to be set for
  // `:?` bindings, and may be set for other binding kinds as well.
  //
  // TODO: Unify this with the previous three fields, which also represent form
  // information.
  InstId form_id = InstId::None;

  // The declared type of the binding, as written. This describes the same type
  // as the binding's `type_id`, but may retain type sugar that `type_id` loses,
  // such as the use of an alias to name the type. For a `:?` binding, this is
  // the type component of `form_id`. This is `None` if the binding has no
  // declared type, or if it was not written in the source, for example because
  // the binding was synthesized or imported.
  TypeInstId type_inst_id = TypeInstId::None;
};

// Value store for EntityName. In addition to the regular ValueStore
// functionality, this can provide optional canonical IDs for EntityNames.
struct EntityNameStore
    : public ValueStore<EntityNameId, EntityName, Tag<CheckIRId>> {
 public:
  using ValueStore::ValueStore;

  // Adds an entity name for a symbolic binding.
  auto AddSymbolicBindingName(NameId name_id, NameScopeId parent_scope_id,
                              CompileTimeBindIndex bind_index, bool is_template,
                              bool is_unused, bool is_frozen_period_self)
      -> EntityNameId {
    EntityName name = {.name_id = name_id,
                       .parent_scope_id = parent_scope_id,
                       .bind_index_value = bind_index.index,
                       .is_template = is_template,
                       .is_unused = is_unused,
                       .is_frozen_period_self = is_frozen_period_self};
    CARBON_CHECK(name.bind_index_value == bind_index.index,
                 "Bind index out of range for bit-field: {0}",
                 bind_index.index);
    return Add(name);
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

namespace Carbon {
extern template class ValueStore<SemIR::EntityNameId, SemIR::EntityName,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_ENTITY_NAME_H_
