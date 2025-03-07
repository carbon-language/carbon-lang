// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_

#include "clang/AST/Decl.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Function-specific fields.
struct FunctionFields {
  // Kinds of virtual modifiers that can apply to functions.
  enum class VirtualModifier : uint8_t { None, Virtual, Abstract, Impl };

  // The following members always have values, and do not change throughout the
  // lifetime of the function.

  // A reference to the instruction in the entity's pattern block that depends
  // on all other pattern insts pertaining to the return slot pattern. This may
  // or may not be used by the function, depending on whether the return type
  // needs a return slot, but is always present if the function has a declared
  // return type.
  InstId return_slot_pattern_id;

  // Which, if any, virtual modifier (virtual, abstract, or impl) is applied to
  // this function.
  VirtualModifier virtual_modifier;

  // The implicit self parameter, if any, in implicit_param_patterns_id from
  // EntityWithParamsBase.
  InstId self_param_id = SemIR::InstId::None;

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

  // If the function is imported from C++, points to the Clang declaration in
  // the AST. Used for mangling. The AST is owned by `CompileSubcommand` so we
  // expect it to be live from `Function` creation to mangling.
  // TODO: #4666 Ensure we can easily serialize/deserialize this. Consider decl
  // ID to point into the AST.
  const clang::NamedDecl* cpp_decl = nullptr;
};

// A function. See EntityWithParamsBase regarding the inheritance here.
struct Function : public EntityWithParamsBase,
                  public FunctionFields,
                  public Printable<Function> {
  struct ParamPatternInfo {
    InstId inst_id;
    AnyParamPattern inst;
    EntityNameId entity_name_id;
  };

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{";
    PrintBaseFields(out);
    if (return_slot_pattern_id.has_value()) {
      out << ", return_slot_pattern: " << return_slot_pattern_id;
    }
    if (!body_block_ids.empty()) {
      out << llvm::formatv(
          ", body: [{0}]",
          llvm::make_range(body_block_ids.begin(), body_block_ids.end()));
    }
    out << "}";
  }

  // Given the ID of an instruction from `param_patterns_id` or
  // `implicit_param_patterns_id`, returns a `ParamPatternInfo` value with the
  // corresponding `Call` parameter pattern, its ID, and the entity_name_id of
  // the underlying binding pattern, or std::nullopt if there is no
  // corresponding `Call` parameter.
  // TODO: Remove this, by exposing `Call` parameter patterns instead of `Call`
  // parameters in EntityWithParams.
  static auto GetParamPatternInfoFromPatternId(const File& sem_ir,
                                               InstId param_pattern_id)
      -> std::optional<ParamPatternInfo>;

  // Gets the name from the name binding instruction, or `None` if this pattern
  // has been replaced with BuiltinErrorInst.
  static auto GetNameFromPatternId(const File& sem_ir, InstId param_pattern_id)
      -> SemIR::NameId;

  // Gets the declared return type for a specific version of this function, or
  // the canonical return type for the original declaration no specific is
  // specified.  Returns `None` if no return type was specified, in which
  // case the effective return type is an empty tuple.
  auto GetDeclaredReturnType(const File& file,
                             SpecificId specific_id = SpecificId::None) const
      -> TypeId;
};

class File;

struct CalleeFunction {
  // The function. `None` if not a function.
  SemIR::FunctionId function_id;
  // The specific that contains the function.
  SemIR::SpecificId enclosing_specific_id;
  // The specific for the callee itself, in a resolved call.
  SemIR::SpecificId resolved_specific_id;
  // The bound `Self` type. `None` if not a bound interface member.
  SemIR::InstId self_type_id;
  // The bound `self` parameter. `None` if not a method.
  SemIR::InstId self_id;
  // True if an error instruction was found.
  bool is_error;
};

// Returns information for the function corresponding to callee_id.
auto GetCalleeFunction(const File& sem_ir, InstId callee_id,
                       SpecificId specific_id = SpecificId::None)
    -> CalleeFunction;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
