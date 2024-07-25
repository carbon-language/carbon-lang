// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_

#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// A value that describes whether the function uses a return slot.
enum class ReturnSlot : int8_t {
  // The function is known to not use a return slot.
  Absent,
  // The function has a return slot, and a call to the function is expected to
  // have an additional final argument corresponding to the return slot.
  Present,
  // Computing whether the function should have a return slot failed because
  // the return type was incomplete.
  Incomplete,
};

// Information about how a function returns its return value.
struct ReturnTypeInfo {
  // Builds return information for a given declared return type.
  static auto ForType(const File& file, TypeId type_id) -> ReturnTypeInfo;

  // Returns whether the return information could be fully computed.
  auto is_valid() const -> bool {
    return return_slot != ReturnSlot::Incomplete;
  }

  // Returns whether the function has a return slot. Can only be called for
  // valid return info.
  auto has_return_slot() const -> bool {
    CARBON_CHECK(is_valid());
    return return_slot == ReturnSlot::Present;
  }

  // The return type. Invalid if no return type was specified.
  TypeId type_id;
  // The return slot usage for this function.
  ReturnSlot return_slot;
};

// Function-specific fields.
struct FunctionFields {
  // The following members always have values, and do not change throughout the
  // lifetime of the function.

  // The storage for the return value, which is a reference expression whose
  // type is the return type of the function. This may or may not be used by the
  // function, depending on whether the return type needs a return slot, but is
  // always present if the function has a declared return type.
  InstId return_storage_id;
  // Whether the declaration is extern.
  bool is_extern;

  // The following member is set on the first call to the function, or at the
  // point where the function is defined.

  // The following members are set at the end of a builtin function definition.

  // If this is a builtin function, the corresponding builtin kind.
  BuiltinFunctionKind builtin_function_kind = BuiltinFunctionKind::None;

  // The following members are accumulated throughout the function definition.

  // A list of the statically reachable code blocks in the body of the
  // function, in lexical order. The first block is the entry block. This will
  // be empty for declarations that don't have a visible definition.
  llvm::SmallVector<InstBlockId> body_block_ids = {};
};

// A function. See EntityWithParamsBase regarding the inheritance here.
struct Function : public EntityWithParamsBase,
                  public FunctionFields,
                  public Printable<Function> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{";
    PrintBaseFields(out);
    if (return_storage_id.is_valid()) {
      out << ", return_storage: " << return_storage_id;
    }
    if (!body_block_ids.empty()) {
      out << llvm::formatv(
          ", body: [{0}]",
          llvm::make_range(body_block_ids.begin(), body_block_ids.end()));
    }
    out << "}";
  }

  // Given a parameter reference instruction from `param_refs_id` or
  // `implicit_param_refs_id`, returns the corresponding `Param` instruction
  // and its ID.
  static auto GetParamFromParamRefId(const File& sem_ir, InstId param_ref_id)
      -> std::pair<InstId, Param>;

  // Gets the declared return type for a specific version of this function, or
  // the canonical return type for the original declaration no specific is
  // specified.  Returns `Invalid` if no return type was specified, in which
  // case the effective return type is an empty tuple.
  auto GetDeclaredReturnType(const File& file,
                             SpecificId specific_id = SpecificId::Invalid) const
      -> TypeId;

  // Returns information about how the function returns its return value.
  auto GetReturnTypeInfo(const File& file,
                     SpecificId specific_id = SpecificId::Invalid) const
      -> ReturnTypeInfo {
    return ReturnTypeInfo::ForType(file, GetDeclaredReturnType(file, specific_id));
  }
};

class File;

struct CalleeFunction {
  // The function. Invalid if not a function.
  SemIR::FunctionId function_id;
  // The specific that contains the function.
  SemIR::SpecificId specific_id;
  // The bound `self` parameter. Invalid if not a method.
  SemIR::InstId self_id;
  // True if an error instruction was found.
  bool is_error;
};

// Returns information for the function corresponding to callee_id.
auto GetCalleeFunction(const File& sem_ir, InstId callee_id) -> CalleeFunction;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
