// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/generic.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

class GenericInstanceStore::KeyContext
    : public TranslatingKeyContext<KeyContext> {
 public:
  // A lookup key for a generic instance.
  struct Key {
    GenericId generic_id;
    InstBlockId args_id;

    friend auto operator==(const Key&, const Key&) -> bool = default;
  };

  explicit KeyContext(llvm::ArrayRef<GenericInstance> instances)
      : instances_(instances) {}

  auto TranslateKey(GenericInstanceId id) const -> Key {
    const auto& instance = instances_[id.index];
    return {.generic_id = instance.generic_id, .args_id = instance.args_id};
  }

 private:
  llvm::ArrayRef<GenericInstance> instances_;
};

auto GenericInstanceStore::GetOrAdd(GenericId generic_id, InstBlockId args_id)
    -> GenericInstanceId {
  return lookup_table_
      .Insert(
          KeyContext::Key{.generic_id = generic_id, .args_id = args_id},
          [&] {
            return generic_instances_.Add(
                {.generic_id = generic_id, .args_id = args_id});
          },
          KeyContext(generic_instances_.array_ref()))
      .key();
}

auto GenericInstanceStore::CollectMemUsage(MemUsage& mem_usage,
                                           llvm::StringRef label) const
    -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "generic_instances_"),
                    generic_instances_);
  mem_usage.Add(MemUsage::ConcatLabel(label, "lookup_table_"), lookup_table_,
                KeyContext(generic_instances_.array_ref()));
}

auto GetConstantInInstance(const File& sem_ir, GenericInstanceId instance_id,
                           ConstantId const_id) -> ConstantId {
  if (!const_id.is_symbolic()) {
    // Type does not depend on a generic parameter.
    return const_id;
  }

  const auto& symbolic = sem_ir.constant_values().GetSymbolicConstant(const_id);
  if (!symbolic.generic_id.is_valid()) {
    // Constant is an abstract symbolic constant, not an instance-specific one.
    return const_id;
  }

  if (!instance_id.is_valid()) {
    // TODO: We have a generic constant but no instance. Investigate whether we
    // can CHECK-fail here. For now, produce the canonical value of the
    // constant.
    return sem_ir.constant_values().Get(symbolic.inst_id);
  }

  const auto& specific = sem_ir.generic_instances().Get(instance_id);
  if (specific.generic_id != symbolic.generic_id) {
    // TODO: Given an instance for the wrong generic. If the symbolic constant
    // is from an enclosing generic, take the value from the corresponding
    // instance. Otherwise, CHECK-fail.
    return sem_ir.constant_values().Get(symbolic.inst_id);
  }

  auto value_block_id = specific.GetValueBlock(symbolic.index.region());
  CARBON_CHECK(value_block_id.is_valid())
      << "Queried region of " << instance_id << " before it was resolved.";
  return sem_ir.constant_values().Get(
      sem_ir.inst_blocks().Get(value_block_id)[symbolic.index.index()]);
}

auto GetConstantValueInInstance(const File& sem_ir,
                                GenericInstanceId instance_id, InstId inst_id)
    -> ConstantId {
  return GetConstantInInstance(sem_ir, instance_id,
                               sem_ir.constant_values().Get(inst_id));
}

auto GetTypeInInstance(const File& sem_ir, GenericInstanceId instance_id,
                       TypeId type_id) -> TypeId {
  return TypeId::ForTypeConstant(GetConstantInInstance(
      sem_ir, instance_id, sem_ir.types().GetConstantId(type_id)));
}

}  // namespace Carbon::SemIR
