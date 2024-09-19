// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
#define CARBON_TOOLCHAIN_SEM_IR_IMPL_H_

#include "common/map.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct ImplFields {
  // The following members always have values, and do not change throughout the
  // lifetime of the interface.

  // The type for which the impl is implementing a constraint.
  TypeId self_id;
  // The constraint that the impl implements.
  TypeId constraint_id;

  // The following members are set at the `{` of the impl definition.

  // The impl scope.
  NameScopeId scope_id = NameScopeId::Invalid;
  // The first block of the impl body.
  // TODO: Handle control flow in the impl body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::Invalid;

  // The following members are set at the `}` of the impl definition.

  // The witness for the impl. This can be `BuiltinError`.
  InstId witness_id = InstId::Invalid;
};

// An implementation of a constraint. See EntityWithParamsBase regarding the
// inheritance here.
struct Impl : public EntityWithParamsBase,
              public ImplFields,
              public Printable<Impl> {
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
};

// A collection of `Impl`s, which can be accessed by the self type and
// constraint implemented.
class ImplStore {
 public:
  // TODO: Switch to something like TinyPtrVector. We expect it to be rare for
  // there to be more than one ImplId per bucket.
  using IdVector = llvm::SmallVector<ImplId, 1>;

  // Looks up the list of impls with this self type and constraint. This only
  // includes impls from the current file and its API file.
  auto LookupBucket(TypeId self_id, TypeId constraint_id) -> IdVector& {
    return lookup_
        .Insert(std::pair{self_id, constraint_id}, [] { return IdVector(); })
        .value();
  }

  // Adds the specified impl to the store. Does not add it to impl lookup.
  auto Add(Impl impl) -> ImplId {
    return values_.Add(impl);
  }

  // Returns a mutable value for an ID.
  auto Get(ImplId id) -> Impl& { return values_.Get(id); }

  // Returns the value for an ID.
  auto Get(ImplId id) const -> const Impl& { return values_.Get(id); }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    mem_usage.Add(MemUsage::ConcatLabel(label, "lookup_"), lookup_);
  }

  auto array_ref() const -> llvm::ArrayRef<Impl> { return values_.array_ref(); }
  auto size() const -> size_t { return values_.size(); }

 private:
  ValueStore<ImplId> values_;
  Map<std::pair<TypeId, TypeId>, IdVector> lookup_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
