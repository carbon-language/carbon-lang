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
  auto is_defined() const -> bool { return defined; }

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
  bool defined = false;
};

// A collection of `Impl`s, which can be accessed by the self type and
// constraint implemented.
class ImplStore : private ValueStore<ImplId> {
 public:
  // Looks up the impl with this self type and constraint, or creates a new
  // `Impl` if none exists.
  // TODO: Handle parameters.
  auto LookupOrAdd(TypeId self_id, TypeId constraint_id) -> ImplId {
    auto [it, added] =
        values_.insert({{self_id, constraint_id}, ImplId::Invalid});
    if (added) {
      it->second =
          ValueStore::Add({.self_id = self_id, .constraint_id = constraint_id});
    }
    return it->second;
  }

  using ValueStore::array_ref;
  using ValueStore::Get;
  using ValueStore::OutputYaml;
  using ValueStore::Reserve;
  using ValueStore::size;

 private:
  llvm::DenseMap<std::pair<TypeId, TypeId>, ImplId> values_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
