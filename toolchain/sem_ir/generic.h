// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_
#define CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// Information for a generic entity, such as a generic class, a generic
// interface, or generic function.
//
// Note that this includes both checked generics and template generics.
struct Generic : public Printable<Generic> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{decl: " << decl_id << ", bindings: " << bindings_id << "}";
  }

  // The following members always have values, and do not change throughout the
  // lifetime of the generic.

  // The first declaration of the generic entity.
  InstId decl_id;
  // A block containing the IDs of compile time bindings in this generic scope.
  // The index in this block will match the `bind_index` in the name binding
  // instruction's `BindNameInfo`.
  InstBlockId bindings_id;
};

// An instance of a generic entity, such as an instance of a generic function.
// For each construct that depends on a compile-time parameter in the generic
// entity, this contains the corresponding non-generic value. This includes
// values for the compile-time parameters themselves.
struct GenericInstance : Printable<GenericInstance> {
  // Values corresponding to a region of a generic.
  struct Region {
    InstBlockId symbolic_constant_values_id = InstBlockId::Invalid;
  };

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{generic: " << generic_id << ", args: " << args_id << "}";
  }

  // The generic that this is an instance of.
  GenericId generic_id;
  // Argument values, corresponding to the bindings in `Generic::bindings_id`.
  InstBlockId args_id;
};

// Provides storage for deduplicated instances of generics.
class GenericInstanceStore {
 public:
  explicit GenericInstanceStore(InstBlockStore& inst_block_store,
                                llvm::BumpPtrAllocator& allocator)
      : allocator_(&allocator),
        inst_block_store_(&inst_block_store),
        lookup_table_(this) {}

  // Adds a new generic instance, or gets the existing generic instance for a
  // specified generic and argument list. Returns the ID of the generic
  // instance.
  //
  // This allocates a new InstBlock for the arguments if the instance is new.
  auto GetOrAdd(GenericId generic_id,
                llvm::ArrayRef<ConstantId> arg_ids) -> GenericInstanceId;

 private:
  // A deduplicated generic instance node in our folding set.
  struct Node : llvm::FoldingSetNode {
    GenericInstanceId generic_instance_id;

    auto Profile(llvm::FoldingSetNodeID& id,
                 GenericInstanceStore* store) -> void;
  };

  ValueStore<GenericInstanceId> generic_instances_;
  llvm::BumpPtrAllocator* allocator_;
  InstBlockStore* inst_block_store_;
  llvm::ContextualFoldingSet<Node, GenericInstanceStore*> lookup_table_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_
