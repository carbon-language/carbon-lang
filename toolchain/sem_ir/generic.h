// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_
#define CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_

#include "common/set.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Information for a generic entity, such as a generic class, a generic
// interface, or generic function.
//
// Note that this includes both checked generics and template generics.
struct Generic : public Printable<Generic> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{decl: " << decl_id << ", bindings: " << bindings_id << "}";
  }

  // Returns the eval block for the specified region of the generic. This is a
  // block of instructions that should be evaluated to compute the values and
  // instructions needed by that region of the generic.
  auto GetEvalBlock(GenericInstIndex::Region region) const -> InstBlockId {
    return region == GenericInstIndex::Region::Declaration
               ? decl_block_id
               : definition_block_id;
  }

  // The following members always have values, and do not change throughout the
  // lifetime of the generic.

  // The first declaration of the generic entity.
  InstId decl_id;
  // A block containing the IDs of compile time bindings in this generic scope.
  // The index in this block will match the `bind_index` in the name binding
  // instruction's `EntityName`.
  InstBlockId bindings_id;
  // The self specific of this generic, which is a specific where every generic
  // parameter's argument is that same parameter. For example, the self specific
  // of `Vector(T:! type)` is `Vector(T)`.
  SpecificId self_specific_id;

  // The following members are set at the end of the corresponding region of the
  // generic.

  // The eval block for the declaration region of the generic.
  InstBlockId decl_block_id = InstBlockId::Invalid;
  // The eval block for the definition region of the generic.
  InstBlockId definition_block_id = InstBlockId::Invalid;
};

// Provides storage for generics.
class GenericStore : public ValueStore<GenericId> {
 public:
  // Get the self specific for a generic, or an invalid specific for an invalid
  // generic ID.
  auto GetSelfSpecific(GenericId id) -> SpecificId {
    return id.is_valid() ? Get(id).self_specific_id : SpecificId::Invalid;
  }
};

// A specific, which is the combination of a generic and specified generic
// arguments. For each construct that depends on a compile-time parameter in the
// generic entity, this contains the corresponding specific value. This includes
// values for the compile-time parameters themselves.
struct Specific : Printable<Specific> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{generic: " << generic_id << ", args: " << args_id << "}";
  }

  // Returns the value block for this region of the specific. This is a block
  // containing values and instructions produced by evaluating the corresponding
  // eval block of the generic within the context of this specific. These are
  // the constant values and types and the instantiated template-dependent
  // instructions that are used in this region of the specific.
  auto GetValueBlock(GenericInstIndex::Region region) const -> InstBlockId {
    return region == GenericInstIndex::Region::Declaration
               ? decl_block_id
               : definition_block_id;
  }

  // The generic that this is a specific version of.
  GenericId generic_id;
  // Argument values, corresponding to the bindings in `Generic::bindings_id`.
  InstBlockId args_id;

  // The following members are set when the corresponding region of the specific
  // is resolved.

  // The value block for the declaration region of the specific.
  InstBlockId decl_block_id = InstBlockId::Invalid;
  // The value block for the definition region of the specific.
  InstBlockId definition_block_id = InstBlockId::Invalid;
};

// Provides storage for deduplicated specifics, which represent generics plus
// their associated generic argument list.
class SpecificStore : public Yaml::Printable<SpecificStore> {
 public:
  // Adds a new specific, or gets the existing specific for a specified generic
  // and argument list. Returns the ID of the specific. The argument IDs must be
  // for instructions in the constant block, and must be a canonical instruction
  // block ID.
  auto GetOrAdd(GenericId generic_id, InstBlockId args_id) -> SpecificId;

  // Gets the specific with the given ID.
  auto Get(SpecificId specific_id) const -> const Specific& {
    return specifics_.Get(specific_id);
  }

  // Gets the specific with the given ID.
  auto Get(SpecificId specific_id) -> Specific& {
    return specifics_.Get(specific_id);
  }

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return specifics_.OutputYaml();
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  auto array_ref() const -> llvm::ArrayRef<Specific> {
    return specifics_.array_ref();
  }
  auto size() const -> size_t { return specifics_.size(); }

 private:
  // Context for hashing keys.
  class KeyContext;

  ValueStore<SpecificId> specifics_;
  Carbon::Set<SpecificId, 0, KeyContext> lookup_table_;
};

// Gets the substituted value of a potentially generic constant within a
// specific. Note that this does not perform substitution, and will return
// `Invalid` if the substituted constant value is not yet known.
auto GetConstantInSpecific(const File& sem_ir, SpecificId specific_id,
                           ConstantId const_id) -> ConstantId;

// Gets the substituted constant value of a potentially generic instruction
// within a specific. Note that this does not perform substitution, and will
// return `Invalid` if the substituted constant value is not yet known.
auto GetConstantValueInSpecific(const File& sem_ir, SpecificId specific_id,
                                InstId inst_id) -> ConstantId;

// Gets the substituted value of a potentially generic type within a specific.
// Note that this does not perform substitution, and will return `Invalid` if
// the substituted type is not yet known.
auto GetTypeInSpecific(const File& sem_ir, SpecificId specific_id,
                       TypeId type_id) -> TypeId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_
