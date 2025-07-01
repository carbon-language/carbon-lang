// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/generic.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

class SpecificStore::KeyContext : public TranslatingKeyContext<KeyContext> {
 public:
  // A lookup key for a specific.
  struct Key {
    GenericId generic_id;
    InstBlockId args_id;

    friend auto operator==(const Key&, const Key&) -> bool = default;
  };

  explicit KeyContext(const ValueStore<SpecificId, Specific>* specifics)
      : specifics_(specifics) {}

  auto TranslateKey(SpecificId id) const -> Key {
    const auto& specific = specifics_->Get(id);
    return {.generic_id = specific.generic_id, .args_id = specific.args_id};
  }

 private:
  const ValueStore<SpecificId, Specific>* specifics_;
};

auto SpecificStore::GetOrAdd(GenericId generic_id, InstBlockId args_id)
    -> SpecificId {
  CARBON_CHECK(generic_id.has_value());
  return lookup_table_
      .Insert(
          KeyContext::Key{.generic_id = generic_id, .args_id = args_id},
          [&] {
            return specifics_.Add(
                {.generic_id = generic_id, .args_id = args_id});
          },
          KeyContext(&specifics_))
      .key();
}

auto SpecificStore::CollectMemUsage(MemUsage& mem_usage,
                                    llvm::StringRef label) const -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "specifics_"), specifics_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "lookup_table_"),
                    lookup_table_, KeyContext(&specifics_));
}

static auto GetConstantInSpecific(const File& specific_ir,
                                  SpecificId specific_id, const File& const_ir,
                                  ConstantId const_id)
    -> std::pair<const File*, ConstantId> {
  if (!const_id.is_symbolic()) {
    // The constant does not depend on a generic parameter.
    return {&const_ir, const_id};
  }

  const auto& symbolic =
      const_ir.constant_values().GetSymbolicConstant(const_id);
  if (!symbolic.generic_id.has_value()) {
    // The constant is an unattached symbolic constant, not associated with some
    // particular generic.
    return {&const_ir, const_id};
  }

  if (!specific_id.has_value()) {
    // We have a generic constant but no specific. We treat this as a request
    // for the value that should be used within the generic itself, which is the
    // unattached constant.
    return {&const_ir, const_ir.constant_values().Get(symbolic.inst_id)};
  }

  const auto& specific = specific_ir.specifics().Get(specific_id);
  // TODO: Enforce this check even if the generic and specific are in different
  // IRs.
  CARBON_CHECK(
      &specific_ir != &const_ir || specific.generic_id == symbolic.generic_id,
      "Given a specific for the wrong generic");

  auto value_block_id = specific.GetValueBlock(symbolic.index.region());
  if (!value_block_id.has_value()) {
    // For the self specific, we can see queries before the definition is
    // resolved. Return the unattached constant value.
    CARBON_CHECK(
        specific_ir.generics().GetSelfSpecific(
            specific_ir.specifics().Get(specific_id).generic_id) == specific_id,
        "Queried {0} in {1} for {2} before it was resolved.", symbolic.index,
        specific_id,
        specific_ir.insts().Get(
            specific_ir.generics().Get(specific.generic_id).decl_id));
    // TODO: Make sure this is the same value that we put in the self specific
    // when it's resolved. Consider not building value blocks for a self
    // specific.
    return {&const_ir, const_ir.constant_values().Get(symbolic.inst_id)};
  }
  return {&specific_ir,
          specific_ir.constant_values().Get(specific_ir.inst_blocks().Get(
              value_block_id)[symbolic.index.index()])};
}

auto GetConstantValueInSpecific(const File& sem_ir, SpecificId specific_id,
                                InstId inst_id) -> ConstantId {
  return GetConstantInSpecific(sem_ir, specific_id, sem_ir,
                               sem_ir.constant_values().GetAttached(inst_id))
      .second;
}

auto GetConstantValueInSpecific(const File& specific_ir, SpecificId specific_id,
                                const File& inst_ir, InstId inst_id)
    -> std::pair<const File*, ConstantId> {
  return GetConstantInSpecific(specific_ir, specific_id, inst_ir,
                               inst_ir.constant_values().GetAttached(inst_id));
}

auto GetTypeOfInstInSpecific(const File& sem_ir, SpecificId specific_id,
                             InstId inst_id) -> TypeId {
  auto type_id = sem_ir.insts().GetAttachedType(inst_id);
  auto const_id = sem_ir.types().GetConstantId(type_id);
  auto [_, specific_const_id] =
      GetConstantInSpecific(sem_ir, specific_id, sem_ir, const_id);
  return TypeId::ForTypeConstant(specific_const_id);
}

auto GetTypeOfInstInSpecific(const File& specific_ir, SpecificId specific_id,
                             const File& inst_ir, InstId inst_id)
    -> std::pair<const File*, TypeId> {
  auto type_id = inst_ir.insts().GetAttachedType(inst_id);
  auto const_id = inst_ir.types().GetConstantId(type_id);
  auto [result_ir, result_const_id] =
      GetConstantInSpecific(specific_ir, specific_id, inst_ir, const_id);
  return {result_ir, TypeId::ForTypeConstant(result_const_id)};
}

}  // namespace Carbon::SemIR
