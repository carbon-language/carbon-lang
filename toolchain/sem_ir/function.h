// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Function-specific fields.
struct FunctionFields {
  // Kinds of special functions.
  enum class SpecialFunctionKind : uint8_t {
    None,
    Builtin,
    Thunk,
    HasCppThunk,
  };

  // Kinds of virtual modifiers that can apply to functions.
  enum class VirtualModifier : uint8_t { None, Virtual, Abstract, Override };

  // The following members always have values, and do not change throughout the
  // lifetime of the function.

  // This block consists of references to the `AnyParam` insts that represent
  // the function's `Call` parameters. The "`Call` parameters" are the
  // parameters corresponding to the arguments that are passed to a `Call`
  // inst, so they do not include compile-time parameters, but they do include
  // the return slot.
  //
  // The parameters appear in declaration order: `self` (if present), then the
  // explicit runtime parameters, then the return slot (which is "declared" by
  // the function's return type declaration). This is not populated on imported
  // functions, because it is relevant only for a function definition.
  InstBlockId call_params_id;

  // A reference to the instruction in the entity's pattern block that depends
  // on all other pattern insts pertaining to the return slot pattern. This may
  // or may not be used by the function, depending on whether the return type
  // needs a return slot, but is always present if the function has a declared
  // return type.
  InstId return_slot_pattern_id;

  // Which kind of special function this is, if any. This is used in cases where
  // a special function would otherwise be indistinguishable from a normal
  // function.
  SpecialFunctionKind special_function_kind = SpecialFunctionKind::None;

  // Which, if any, virtual modifier (virtual, abstract, or impl) is applied to
  // this function.
  VirtualModifier virtual_modifier;

  // The index of the vtable slot for this virtual function. -1 if the function
  // is not virtual (ie: (virtual_modifier == None) == (virtual_index == -1)).
  int32_t virtual_index = -1;

  // The implicit self parameter pattern, if any, in
  // implicit_param_patterns_id from EntityWithParamsBase.
  InstId self_param_id = InstId::None;

  // Data that is specific to the special function kind. Use
  // `builtin_function_kind()`, `thunk_decl_id()` or `cpp_thunk_decl_id()` to
  // access this.
  AnyRawId special_function_kind_data = AnyRawId(AnyRawId::NoneIndex);

  // The following members are accumulated throughout the function definition.

  // A list of the statically reachable code blocks in the body of the
  // function, in lexical order. The first block is the entry block. This will
  // be empty for declarations that don't have a visible definition.
  llvm::SmallVector<InstBlockId> body_block_ids = {};

  // If the function is imported from C++, the Clang function declaration. Used
  // for mangling and inline function definition code generation. The AST is
  // owned by `CompileSubcommand` so we expect it to be live from `Function`
  // creation to mangling.
  ClangDeclId clang_decl_id = ClangDeclId::None;
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
    if (call_params_id.has_value()) {
      out << ", call_params_id: " << call_params_id;
    }
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

  // Returns the builtin function kind for this function, or None if this is not
  // a builtin function.
  auto builtin_function_kind() const -> BuiltinFunctionKind {
    return special_function_kind == SpecialFunctionKind::Builtin
               ? BuiltinFunctionKind::FromInt(special_function_kind_data.index)
               : BuiltinFunctionKind::None;
  }

  // Returns the declaration that this is a non C++ thunk for, or None if this
  // function is not a thunk.
  auto thunk_decl_id() const -> InstId {
    return special_function_kind == SpecialFunctionKind::Thunk
               ? InstId(special_function_kind_data.index)
               : InstId::None;
  }

  // Returns the declaration of the thunk that should be called to call this
  // function, or None if this function is not a C++ function that requires
  // calling a thunk.
  auto cpp_thunk_decl_id() const -> InstId {
    return special_function_kind == SpecialFunctionKind::HasCppThunk
               ? InstId(special_function_kind_data.index)
               : InstId::None;
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

  // Gets the declared return type for a specific version of this function, or
  // the canonical return type for the original declaration no specific is
  // specified.  Returns `None` if no return type was specified, in which
  // case the effective return type is an empty tuple.
  auto GetDeclaredReturnType(const File& file,
                             SpecificId specific_id = SpecificId::None) const
      -> TypeId;

  // Sets that this function is a builtin function.
  auto SetBuiltinFunction(BuiltinFunctionKind kind) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::Builtin;
    special_function_kind_data = AnyRawId(kind.AsInt());
  }

  // Sets that this function is a thunk.
  auto SetThunk(InstId decl_id) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::Thunk;
    special_function_kind_data = AnyRawId(decl_id.index);
  }

  // Sets that this function is a C++ function that should be called using a C++
  // thunk.
  auto SetHasCppThunk(InstId decl_id) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::HasCppThunk;
    special_function_kind_data = AnyRawId(decl_id.index);
  }
};

using FunctionStore = ValueStore<FunctionId, Function>;

class File;

// Information about a callee that's a C++ overload set.
struct CalleeCppOverloadSet {
  // The overload set.
  CppOverloadSetId cpp_overload_set_id;
  // The bound `self` parameter. `None` if not a method.
  InstId self_id;
};

// Information about a callee that's `ErrorInst`.
struct CalleeError {};

// Information about a callee that's a function.
struct CalleeFunction {
  // The function.
  FunctionId function_id;
  // The specific that contains the function.
  SpecificId enclosing_specific_id;
  // The specific for the callee itself, in a resolved call.
  SpecificId resolved_specific_id;
  // The bound `Self` type or facet value. `None` if not a bound interface
  // member.
  InstId self_type_id;
  // The bound `self` parameter. `None` if not a method.
  InstId self_id;
};

// Information about a callee that may be a generic type, or could be an
// invalid callee.
struct CalleeNonFunction {};

// A variant combining the callee forms.
using Callee = std::variant<CalleeCppOverloadSet, CalleeError, CalleeFunction,
                            CalleeNonFunction>;

// Returns information for the function corresponding to callee_id.
auto GetCallee(const File& sem_ir, InstId callee_id,
               SpecificId specific_id = SpecificId::None) -> Callee;

// Like `GetCallee`, but restricts to the `Function` callee kind.
auto GetCalleeAsFunction(const File& sem_ir, InstId callee_id,
                         SpecificId specific_id = SpecificId::None)
    -> CalleeFunction;

struct DecomposedVirtualFunction {
  // The canonical instruction from the `fn_decl_const_id`.
  InstId fn_decl_id;
  // The constant for the underlying instruction.
  ConstantId fn_decl_const_id;
  // The function.
  FunctionId function_id;
  // The specific for the function.
  SpecificId specific_id;
};

// Returns information for the virtual function table entry instruction.
auto DecomposeVirtualFunction(const File& sem_ir, InstId fn_decl_id,
                              SpecificId base_class_specific_id)
    -> DecomposedVirtualFunction;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
