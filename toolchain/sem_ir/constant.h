// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_
#define CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_

#include "common/map.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// Information about a symbolic constant value. These are indexed by
// `ConstantId`s for which `is_symbolic` is true.
struct SymbolicConstant : Printable<SymbolicConstant> {
  // The constant instruction that defines the value of this symbolic constant.
  InstId inst_id;
  // The enclosing generic. If this is invalid, then this is an abstract
  // symbolic constant, such as a constant instruction in the constants block,
  // rather than one associated with a particular generic.
  GenericId generic_id;
  // The index of this symbolic constant within the generic's list of symbolic
  // constants, or invalid if `generic_id` is invalid.
  GenericInstIndex index;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{inst: " << inst_id << ", generic: " << generic_id
        << ", index: " << index << "}";
  }
};

// Provides a ValueStore wrapper for tracking the constant values of
// instructions.
class ConstantValueStore {
 public:
  explicit ConstantValueStore(ConstantId default_value)
      : default_(default_value) {}

  // Returns the constant value of the requested instruction, which is default_
  // if unallocated.
  auto Get(InstId inst_id) const -> ConstantId {
    CARBON_DCHECK(inst_id.index >= 0);
    return static_cast<size_t>(inst_id.index) >= values_.size()
               ? default_
               : values_[inst_id.index];
  }

  // Sets the constant value of the given instruction, or sets that it is known
  // to not be a constant.
  auto Set(InstId inst_id, ConstantId const_id) -> void {
    CARBON_DCHECK(inst_id.index >= 0);
    if (static_cast<size_t>(inst_id.index) >= values_.size()) {
      values_.resize(inst_id.index + 1, default_);
    }
    values_[inst_id.index] = const_id;
  }

  // Gets the instruction ID that defines the value of the given constant.
  // Returns Invalid if the constant ID is non-constant. Requires is_valid.
  auto GetInstId(ConstantId const_id) const -> InstId {
    if (const_id.is_template()) {
      return const_id.template_inst_id();
    }
    if (const_id.is_symbolic()) {
      return GetSymbolicConstant(const_id).inst_id;
    }
    return InstId::Invalid;
  }

  // Gets the instruction ID that defines the value of the given constant.
  // Returns Invalid if the constant ID is non-constant or invalid.
  auto GetInstIdIfValid(ConstantId const_id) const -> InstId {
    return const_id.is_valid() ? GetInstId(const_id) : InstId::Invalid;
  }

  // Given an instruction, returns the unique constant instruction that is
  // equivalent to it. Returns Invalid for a non-constant instruction.
  auto GetConstantInstId(InstId inst_id) const -> InstId {
    return GetInstId(Get(inst_id));
  }

  // Returns whether two constant IDs represent the same constant value. This
  // includes the case where they might be in different generics and thus might
  // have different ConstantIds, but are still symbolically equal.
  auto AreEqualAcrossDeclarations(ConstantId a, ConstantId b) const -> bool {
    return GetInstId(a) == GetInstId(b);
  }

  auto AddSymbolicConstant(SymbolicConstant constant) -> ConstantId {
    symbolic_constants_.push_back(constant);
    return ConstantId::ForSymbolicConstantIndex(symbolic_constants_.size() - 1);
  }

  auto GetSymbolicConstant(ConstantId const_id) -> SymbolicConstant& {
    return symbolic_constants_[const_id.symbolic_index()];
  }

  auto GetSymbolicConstant(ConstantId const_id) const
      -> const SymbolicConstant& {
    return symbolic_constants_[const_id.symbolic_index()];
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(MemUsage::ConcatLabel(label, "values_"), values_);
    mem_usage.Add(MemUsage::ConcatLabel(label, "symbolic_constants_"),
                  symbolic_constants_);
  }

  // Returns the constant values mapping as an ArrayRef whose keys are
  // instruction indexes. Some of the elements in this mapping may be Invalid or
  // NotConstant.
  auto array_ref() const -> llvm::ArrayRef<ConstantId> { return values_; }

  // Returns the symbolic constants mapping as an ArrayRef whose keys are
  // symbolic indexes of constants.
  auto symbolic_constants() const -> llvm::ArrayRef<SymbolicConstant> {
    return symbolic_constants_;
  }

 private:
  const ConstantId default_;

  // A mapping from `InstId::index` to the corresponding constant value. This is
  // expected to be sparse, and may be smaller than the list of instructions if
  // there are trailing non-constant instructions.
  //
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<ConstantId, 0> values_;

  // A mapping from a symbolic constant ID index to information about the
  // symbolic constant. For a template constant, the only information that we
  // track is the instruction ID, which is stored directly within the
  // `ConstantId`. For a symbolic constant, we also track information about
  // where the constant was used, which is stored here.
  llvm::SmallVector<SymbolicConstant, 0> symbolic_constants_;
};

// Provides storage for instructions representing deduplicated global constants.
class ConstantStore {
 public:
  explicit ConstantStore(File& sem_ir) : sem_ir_(sem_ir) {}

  // Adds a new constant instruction, or gets the existing constant with this
  // value. Returns the ID of the constant.
  //
  // This updates `sem_ir.insts()` and `sem_ir.constant_values()` if the
  // constant is new.
  auto GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(MemUsage::ConcatLabel(label, "map_"), map_);
    mem_usage.Add(MemUsage::ConcatLabel(label, "constants_"), constants_);
  }

  // Returns a copy of the constant IDs as a vector, in an arbitrary but
  // stable order. This should not be used anywhere performance-sensitive.
  auto array_ref() const -> llvm::ArrayRef<InstId> { return constants_; }

  auto size() const -> int { return constants_.size(); }

 private:
  File& sem_ir_;
  Map<Inst, ConstantId> map_;
  llvm::SmallVector<InstId, 0> constants_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_
