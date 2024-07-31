// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/generic.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

class SpecificStore::KeyContext : public TranslatingKeyContext<KeyContext> {
 public:
  // A lookup key for a specific.
  struct Key {
    GenericId generic_id;
    InstBlockId args_id;

    friend auto operator==(const Key&, const Key&) -> bool = default;
  };

  explicit KeyContext(llvm::ArrayRef<Specific> specifics)
      : specifics_(specifics) {}

  auto TranslateKey(SpecificId id) const -> Key {
    const auto& specific = specifics_[id.index];
    return {.generic_id = specific.generic_id, .args_id = specific.args_id};
  }

 private:
  llvm::ArrayRef<Specific> specifics_;
};

auto SpecificStore::GetOrAdd(GenericId generic_id, InstBlockId args_id)
    -> SpecificId {
  CARBON_CHECK(generic_id.is_valid());
  return lookup_table_
      .Insert(
          KeyContext::Key{.generic_id = generic_id, .args_id = args_id},
          [&] {
            return specifics_.Add(
                {.generic_id = generic_id, .args_id = args_id});
          },
          KeyContext(specifics_.array_ref()))
      .key();
}

auto SpecificStore::CollectMemUsage(MemUsage& mem_usage,
                                    llvm::StringRef label) const -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "specifics_"), specifics_);
  mem_usage.Add(MemUsage::ConcatLabel(label, "lookup_table_"), lookup_table_,
                KeyContext(specifics_.array_ref()));
}

auto GetConstantInSpecific(const File& sem_ir, SpecificId specific_id,
                           ConstantId const_id) -> ConstantId {
  if (!const_id.is_symbolic()) {
    // Type does not depend on a generic parameter.
    return const_id;
  }

  const auto& symbolic = sem_ir.constant_values().GetSymbolicConstant(const_id);
  if (!symbolic.generic_id.is_valid()) {
    // Constant is an abstract symbolic constant, not associated with some
    // particular generic.
    return const_id;
  }

  if (!specific_id.is_valid()) {
    // TODO: We have a generic constant but no specific. Investigate whether we
    // can CHECK-fail here. For now, produce the canonical value of the
    // constant.
    return sem_ir.constant_values().Get(symbolic.inst_id);
  }

  const auto& specific = sem_ir.specifics().Get(specific_id);
  if (specific.generic_id != symbolic.generic_id) {
    // TODO: Given an specific for the wrong generic. If the symbolic constant
    // is from an enclosing generic, take the value from the corresponding
    // specific. Otherwise, CHECK-fail.
    return sem_ir.constant_values().Get(symbolic.inst_id);
  }

  auto value_block_id = specific.GetValueBlock(symbolic.index.region());
  CARBON_CHECK(value_block_id.is_valid())
      << "Queried region of " << specific_id << " before it was resolved.";
  return sem_ir.constant_values().Get(
      sem_ir.inst_blocks().Get(value_block_id)[symbolic.index.index()]);
}

auto GetConstantValueInSpecific(const File& sem_ir, SpecificId specific_id,
                                InstId inst_id) -> ConstantId {
  return GetConstantInSpecific(sem_ir, specific_id,
                               sem_ir.constant_values().Get(inst_id));
}

auto GetTypeInSpecific(const File& sem_ir, SpecificId specific_id,
                       TypeId type_id) -> TypeId {
  return TypeId::ForTypeConstant(GetConstantInSpecific(
      sem_ir, specific_id, sem_ir.types().GetConstantId(type_id)));
}

}  // namespace Carbon::SemIR
