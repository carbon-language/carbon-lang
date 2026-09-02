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
  // Kinds of special functions. See `Function::Set*` for details on each; these
  // shouldn't be assigned directly (but are used for reads/switches).
  enum class SpecialFunctionKind : uint8_t {
    // A regular function.
    None,
    // A builtin function. `special_function_kind_data` is the corresponding
    // `BuiltinFunctionKind`.
    Builtin,
    // A synthesized function generated for a `Core` witness.
    // `special_function_kind_data` is the corresponding `BuiltinFunctionKind`,
    // or may be `BuiltinFunctionKind::None` if a function body is synthesized
    // too. During name mangling, extra information is included for this
    // function to avoid collisions.
    CoreWitness,
    // A thunk that adapts a function with one signature to have another
    // signature, by forwarding the arguments and return value, with implicit
    // conversions applied as necessary. `special_function_kind_data` is the
    // corresponding `ThunkId`. A call to this function can generally be
    // rewritten as a call to its target function, after performing the
    // intermediate parameter conversions.
    Thunk,
    // A thunk for calling to or from C++, with an intentionally-simple ABI so
    // that it can be called across the language barrier. `Context::clang_decls`
    // can be used to find the corresponding C++ function, which will have the
    // same mangled name.
    //
    // `special_function_kind_data` is the `InstId` of the wrapped function. If
    // the wrapped function is in Carbon, the Carbon version of the thunk will
    // be a definition. If the wrapped function is in C++, the C++ version of
    // the thunk will be a definition.
    CppThunk,
    // A function that was imported from C++, for which we generated a
    // `CppThunk`. `special_function_kind_data` is the `InstId` of that thunk.
    HasCppThunk,
  };

  // Kinds of virtual modifiers that can apply to functions.
  enum class VirtualModifier : uint8_t { None, Virtual, Abstract, Override };

  // Kinds of evaluation modifiers that can apply to functions.
  enum class EvaluationMode : uint8_t { None, Eval, MustEval };

  // Kinds of interface modifiers that can apply to functions.
  // TODO: turn into a CARBON_ENUM for more convenient checks.
  enum class InterfaceModifier : uint8_t { None, Default, Final };

  // The following members always have values, and do not change throughout the
  // lifetime of the function.

  // This block consists of references to the `*ParamPattern` insts that
  // represent the function's `Call` parameters. The "`Call` parameters" are the
  // parameters corresponding to the arguments that are passed to a `Call` inst,
  // so they do not include compile-time parameters, but they do include the
  // return slot.
  //
  // The parameters appear in declaration order: `self` (if present), then the
  // explicit runtime parameters, then the return parameters (which are
  // "declared" by the function's return type declaration).
  InstBlockId call_param_patterns_id;

  // This block consists of references to the `AnyParam` insts that correspond
  // to call_param_patterns_id. This is not populated on imported functions,
  // because it is relevant only for a function definition.
  InstBlockId call_params_id;

  // Instructions representing the canonical default values for parameters.
  InstBlockId call_param_default_values_id;

  // The index ranges within the `Call` parameters that correspond to the
  // implicit parameters, explicit parameters, and return.
  //
  // Those sub-ranges are represented in terms of their end indices, but for
  // convenience and clarity it provides `begin`, `end`, and `size` accessors
  // for all three ranges, which should be preferred over directly accessing the
  // fields. The accessors follow STL conventions but with indices rather than
  // iterators (because they can index into `call_params`,
  // `call_param_patterns`, and the argument list): all indices in a range are
  // greater than or equal to `begin`, and less than `end`.
  class CallParamIndexRanges {
   public:
    // A CallParamIndexRanges representing an entity with no `Call` parameters.
    static const CallParamIndexRanges Empty;
    constexpr CallParamIndexRanges()
        : implicit_end_(CallParamIndex(0)),
          explicit_end_(CallParamIndex(0)),
          return_end_(CallParamIndex(0)) {}

    // Constructs a CallParamIndexRanges with the given end indices. None
    // of the arguments can be CallParamIndex::None.
    constexpr CallParamIndexRanges(CallParamIndex implicit_end,
                                   CallParamIndex explicit_end,
                                   CallParamIndex return_end)
        : implicit_end_(implicit_end),
          explicit_end_(explicit_end),
          return_end_(return_end) {
      CARBON_CHECK(implicit_end_.has_value() && explicit_end_.has_value() &&
                   return_end_.has_value());
    }

    auto implicit_size() const -> int { return implicit_end_.index; }
    auto implicit_begin() const -> CallParamIndex { return CallParamIndex(0); }
    auto implicit_end() const -> CallParamIndex { return implicit_end_; }

    auto explicit_size() const -> int {
      return explicit_end_.index - implicit_end_.index;
    }
    auto explicit_begin() const -> CallParamIndex { return implicit_end_; }
    auto explicit_end() const -> CallParamIndex { return explicit_end_; }

    auto return_size() const -> int {
      return return_end_.index - explicit_end_.index;
    }
    auto return_begin() const -> CallParamIndex { return explicit_end_; }
    auto return_end() const -> CallParamIndex { return return_end_; }

   private:
    CallParamIndex implicit_end_;
    CallParamIndex explicit_end_;
    CallParamIndex return_end_;
  };

  CallParamIndexRanges call_param_ranges;

  // The inst representing the type component of return_form_inst_id.
  // TODO: remove this in favor of return_form_inst_id.
  TypeInstId return_type_inst_id;

  // The inst representing the function's explicitly declared return form, if
  // any.
  InstId return_form_inst_id;

  // The parameter pattern inst that is declared by the function's return
  // declaration. This will be a ReturnSlotPattern, or None if the function
  // doesn't have a return declaration. It may or may not be used, depending on
  // whether the type has an in-place initializing representation.
  //
  // TODO: Extend this to support composite return forms.
  InstId return_pattern_id;

  // This block consists of `ObserveId`s declared inside the function body.
  ObserveBlockId observe_block_id = ObserveBlockId::None;

  // Which kind of special function this is, if any. This is used in cases where
  // a special function would otherwise be indistinguishable from a normal
  // function.
  SpecialFunctionKind special_function_kind = SpecialFunctionKind::None;

  // Which, if any, virtual modifier (virtual, abstract, or impl) is applied to
  // this function.
  VirtualModifier virtual_modifier = VirtualModifier::None;

  // The index of the vtable slot for this virtual function. -1 if the function
  // is not in the vtable. A function with `virtual_modifier != None` may still
  // have `virtual_index == -1` if the corresponding vtable entry is a thunk.
  int32_t virtual_index = -1;

  // Which, if any, evaluation modifier (eval or musteval) is applied to this
  // function.
  EvaluationMode evaluation_mode = EvaluationMode::None;

  // Which, if any, interface modifier is applied to this function.
  InterfaceModifier interface_modifier = InterfaceModifier::None;

  // The `self` parameter pattern, if any. This is the first pattern in
  // `param_patterns_id` (from EntityWithParamsBase).
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

  friend auto operator<<(llvm::raw_ostream& out, InterfaceModifier modifier)
      -> llvm::raw_ostream& {
    using enum InterfaceModifier;
    switch (modifier) {
      case None:
        return out << "none";
      case Default:
        return out << "default";
      case Final:
        return out << "final";
    }

    CARBON_FATAL("unhandled `ImplModifier` during printing");
  }
};

inline constexpr FunctionFields::CallParamIndexRanges
    FunctionFields::CallParamIndexRanges::Empty;

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
    if (call_param_patterns_id.has_value()) {
      out << ", call_param_patterns_id: " << call_param_patterns_id;
    }
    if (call_params_id.has_value()) {
      out << ", call_params_id: " << call_params_id;
    }
    if (call_param_default_values_id.has_value()) {
      out << ", call_param_default_values_id: " << call_param_default_values_id;
    }
    if (return_type_inst_id.has_value()) {
      out << ", return_type_inst_id: " << return_type_inst_id;
    }
    if (return_type_inst_id.has_value()) {
      out << ", return_form_inst_id: " << return_form_inst_id;
    }
    if (return_pattern_id.has_value()) {
      out << ", return_pattern_id: " << return_pattern_id;
    }
    if (auto builtin_kind = builtin_function_kind();
        builtin_kind != BuiltinFunctionKind::None) {
      out << ", builtin: " << builtin_kind;
    }
    if (auto thunk_id_val = thunk_id(); thunk_id_val.has_value()) {
      out << ", thunk: " << thunk_id_val;
    }
    if (auto cpp_thunk_decl_id_val = cpp_thunk_decl_id();
        cpp_thunk_decl_id_val.has_value()) {
      out << ", cpp_thunk_decl: " << cpp_thunk_decl_id_val;
    }
    if (auto cpp_thunk_callee_val = cpp_thunk_callee();
        cpp_thunk_callee_val.has_value()) {
      out << ", cpp_thunk_callee: " << cpp_thunk_callee_val;
    }
    if (interface_modifier != InterfaceModifier::None) {
      out << ", interface_modifier: " << interface_modifier;
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
    return (special_function_kind == SpecialFunctionKind::Builtin ||
            special_function_kind == SpecialFunctionKind::CoreWitness)
               ? BuiltinFunctionKind::FromInt(special_function_kind_data.index)
               : BuiltinFunctionKind::None;
  }

  // Returns the ThunkId for this thunk function, or None if it's not a thunk.
  auto thunk_id() const -> ThunkId {
    return special_function_kind == SpecialFunctionKind::Thunk
               ? ThunkId(special_function_kind_data.index)
               : ThunkId::None;
  }

  // Returns the declaration of the thunk that should be called to call this
  // function, or None if this function is not a C++ function that requires
  // calling a thunk.
  auto cpp_thunk_decl_id() const -> InstId {
    return special_function_kind == SpecialFunctionKind::HasCppThunk
               ? InstId(special_function_kind_data.index)
               : InstId::None;
  }

  // Gets the `InstId` of the C++ function called by this thunk.
  auto cpp_thunk_callee() const -> InstId {
    return special_function_kind == SpecialFunctionKind::CppThunk
               ? InstId(special_function_kind_data.index)
               : InstId::None;
  }

  // Gets the declared return type for a specific version of this function, or
  // the canonical return type for the original declaration no specific is
  // specified.  Returns `None` if no return type was specified, in which
  // case the effective return type is an empty tuple.
  auto GetDeclaredReturnType(const File& file,
                             SpecificId specific_id = SpecificId::None) const
      -> TypeId;

  // Gets the canonical declared return form for a specific version of this
  // function, or for the original declaration if no specific is specified.
  // Returns `None` if the function was declared without a return form, in which
  // case the effective return form is an empty tuple initializing expression.
  auto GetDeclaredReturnForm(const File& file,
                             SpecificId specific_id = SpecificId::None) const
      -> InstId;

  // When merging a declaration and definition, prefer things which would point
  // at the definition for diagnostics. Note that merging parameter default
  // values needs more context, so doesn't happen here.
  auto MergeDefinition(const Function& definition) -> void {
    EntityWithParamsBase::MergeBaseDefinition(definition);
    call_param_patterns_id = definition.call_param_patterns_id;
    call_params_id = definition.call_params_id;
    return_type_inst_id = definition.return_type_inst_id;
    return_form_inst_id = definition.return_form_inst_id;
    return_pattern_id = definition.return_pattern_id;
    self_param_id = definition.self_param_id;
  }

  // Sets that this function is a builtin function.
  auto SetBuiltinFunction(BuiltinFunctionKind kind) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::Builtin;
    special_function_kind_data = AnyRawId(kind.AsInt());
  }

  // Sets that this function is generated for a `Core` witness. These will
  // typically have a custom implementation for a `None` kind, but may use
  // builtin functions, most often `NoOp`. We still track them differently in
  // order to support mangling.
  auto SetCoreWitness(BuiltinFunctionKind kind) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::CoreWitness;
    special_function_kind_data = AnyRawId(kind.AsInt());
  }

  // Sets that this function is a thunk.
  auto SetThunk(ThunkId thunk_id) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::Thunk;
    special_function_kind_data = AnyRawId(thunk_id.index);
  }

  // Sets that this function is a C++ thunk.
  auto SetCppThunk(InstId decl_id) -> void {
    CARBON_CHECK(special_function_kind == SpecialFunctionKind::None);
    special_function_kind = SpecialFunctionKind::CppThunk;
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

using FunctionStore = ValueStore<FunctionId, Function, Tag<CheckIRId>>;

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

// Given a callee expression in a function call, attempt to convert the callee
// to a `BoundMethod`, minimally unwrapping it while doing so.
auto TryGetCalleeAsBoundMethod(const File& sem_ir, InstId callee_id,
                               SpecificId caller_specific_id)
    -> std::optional<BoundMethod>;

// Returns information for the function corresponding to callee_id in
// caller_specific_id.
auto GetCallee(const File& sem_ir, InstId callee_id,
               SpecificId caller_specific_id = SpecificId::None) -> Callee;

// Like `GetCallee`, but restricts to the `Function` callee kind.
//
// It is invalid to call this with a callee that has an error inside it.
auto GetCalleeAsFunction(const File& sem_ir, InstId callee_id,
                         SpecificId caller_specific_id = SpecificId::None)
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

namespace Carbon {
extern template class ValueStore<SemIR::FunctionId, SemIR::Function,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_FUNCTION_H_
