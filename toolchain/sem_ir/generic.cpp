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
  CARBON_CHECK(generic_id.has_value());
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
  mem_usage.Collect(MemUsage::ConcatLabel(label, "lookup_table_"),
                    lookup_table_, KeyContext(specifics_.array_ref()));
}

auto GetConstantInSpecific(const File& sem_ir, SpecificId specific_id,
                           ConstantId const_id) -> ConstantId {
  if (!const_id.is_symbolic()) {
    // Type does not depend on a generic parameter.
    return const_id;
  }

  const auto& symbolic = sem_ir.constant_values().GetSymbolicConstant(const_id);
  if (!symbolic.generic_id.has_value()) {
    // Constant is an abstract symbolic constant, not associated with some
    // particular generic.
    return const_id;
  }

  if (!specific_id.has_value()) {
    // We have a generic constant but no specific. We treat as a request for the
    // canonical value of the constant.
    return sem_ir.constant_values().Get(symbolic.inst_id);
  }

  const auto& specific = sem_ir.specifics().Get(specific_id);
  CARBON_CHECK(specific.generic_id == symbolic.generic_id,
               "Given a specific for the wrong generic");

  auto value_block_id = specific.GetValueBlock(symbolic.index.region());
  CARBON_CHECK(
      value_block_id.has_value(),
      "Queried {0} in {1} for {2} before it was resolved.", symbolic.index,
      specific_id,
      sem_ir.insts().Get(sem_ir.generics().Get(specific.generic_id).decl_id));
  return sem_ir.constant_values().Get(
      sem_ir.inst_blocks().Get(value_block_id)[symbolic.index.index()]);
}

auto GetConstantValueInSpecific(const File& sem_ir, SpecificId specific_id,
                                InstId inst_id) -> ConstantId {
  return GetConstantInSpecific(sem_ir, specific_id,
                               sem_ir.constant_values().Get(inst_id));
}

auto GetTypeOfInstInSpecific(const File& sem_ir, SpecificId specific_id,
                             InstId inst_id) -> TypeId {
  auto type_id = sem_ir.insts().Get(inst_id).type_id();
  auto const_id = sem_ir.types().GetConstantId(type_id);
  auto specific_const_id = GetConstantInSpecific(sem_ir, specific_id, const_id);
  return TypeId::ForTypeConstant(specific_const_id);
}

}  // namespace Carbon::SemIR
