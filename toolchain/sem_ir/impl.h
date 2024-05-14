// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
#define CARBON_TOOLCHAIN_SEM_IR_IMPL_H_

#include "llvm/ADT/DenseMap.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// An implementation of a constraint.
struct Impl : public Printable<Impl> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{self: " << self_id << ", constraint: " << constraint_id << "}";
  }

  // Determines whether this impl has been fully defined. This is false until we
  // reach the `}` of the impl definition.
  auto is_defined() const -> bool { return witness_id.is_valid(); }

  // Determines whether this impl's definition has begun but not yet ended.
  auto is_being_defined() const -> bool {
    return definition_id.is_valid() && !is_defined();
  }

  // The following members always have values, and do not change throughout the
  // lifetime of the interface.

  // TODO: Track the generic parameters for `impl forall`.
  // The type for which the impl is implementing a constraint.
  TypeId self_id;
  // The constraint that the impl implements.
  TypeId constraint_id;

  // The following members are set at the `{` of the impl definition.

  // The definition of the impl. This is an ImplDecl.
  InstId definition_id = InstId::Invalid;
  // The impl scope.
  NameScopeId scope_id = NameScopeId::Invalid;
  // The first block of the impl body.
  // TODO: Handle control flow in the impl body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::Invalid;

  // The following members are set at the `}` of the impl definition.

  // The witness for the impl. This can be `BuiltinError`.
  InstId witness_id = InstId::Invalid;
};

// A collection of `Impl`s, which can be accessed by the self type and
// constraint implemented.
class ImplStore {
 public:
  // Looks up the impl with this self type and constraint, or creates a new
  // `Impl` if none exists.
  // TODO: Handle parameters.
  auto LookupOrAdd(TypeId self_id, TypeId constraint_id) -> ImplId {
    auto [it, added] =
        lookup_.insert({{self_id, constraint_id}, ImplId::Invalid});
    if (added) {
      it->second =
          values_.Add({.self_id = self_id, .constraint_id = constraint_id});
    }
    return it->second;
  }

  // Returns a mutable value for an ID.
  auto Get(ImplId id) -> Impl& { return values_.Get(id); }

  // Returns the value for an ID.
  auto Get(ImplId id) const -> const Impl& { return values_.Get(id); }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  auto array_ref() const -> llvm::ArrayRef<Impl> { return values_.array_ref(); }
  auto size() const -> size_t { return values_.size(); }

 private:
  ValueStore<ImplId> values_;
  llvm::DenseMap<std::pair<TypeId, TypeId>, ImplId> lookup_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
