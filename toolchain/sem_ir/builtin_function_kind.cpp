// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/builtin_function_kind.h"

#include <utility>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

namespace BuiltinFunctionInfo {

// A function that validates that a builtin was declared properly.
using ValidateFn = auto(const File& sem_ir, llvm::ArrayRef<TypeId> arg_types,
                        TypeId return_type) -> bool;

namespace {
// Information about a builtin function.
struct BuiltinInfo {
  llvm::StringLiteral name;
  ValidateFn* validate;
};

// The maximum number of type parameters any builtin needs.
constexpr int MaxTypeParams = 1;

// State used when validating a builtin signature that persists between
// individual checks.
struct ValidateState {
  // The type values of type parameters in the builtin signature. Invalid if
  // either no value has been deduced yet or the parameter is not used.
  TypeId type_params[MaxTypeParams] = {TypeId::Invalid};
};

// Helper declaring a generic parameter with a constraint on its type. See
// ValidateSignature for details.
template <typename ParamT, typename TypeConstraint>
struct Param {
  static auto Check(const File& sem_ir, ValidateState& state) -> bool {
    return TypeConstraint::Check(sem_ir, state, ParamT::Get(state));
  }
};

// Constraint that a type is generic type parameter `I` of the builtin. See
// ValidateSignature for details.
template <int I>
struct TypeParam {
  static_assert(I >= 0 && I < MaxTypeParams);

  static auto Check(const File& /*sem_ir*/, ValidateState& state,
                    TypeId type_id) -> bool {
    if (state.type_params[I].is_valid() && type_id != state.type_params[I]) {
      return false;
    }
    state.type_params[I] = type_id;
    return true;
  }

  static auto Get(ValidateState& state) -> TypeId {
    return state.type_params[I];
  }
};

// Convenience name for the first type parameter of a builtin.
using T = TypeParam<0>;

// Constraint that requires the type to be an integer type. See
// ValidateSignature for details.
struct AnyInt {
  static auto Check(const File& sem_ir, ValidateState& /*state*/,
                    TypeId type_id) -> bool {
    if (sem_ir.types().GetInstId(type_id) == InstId::BuiltinIntType) {
      return true;
    }
    // TODO: Support iN for all N, and the Core.BigInt type we use to implement
    // for integer literals.
    return false;
  }
};
}  // namespace

// Validates that this builtin has a signature matching the specified signature.
//
// `SignatureFnType` is a C++ function type that describes the signature that is
// expected for this builtin. For example, `auto (T, AnyInt) -> T` specifies
// that the builtin returns the same type as its first parameter, and that its
// second parameter is an integer. Types used within the signature should
// provide a `Check` function that validates that the Carbon type is expected:
//
//   auto Check(const File&, ValidateState&, TypeId) -> bool;
//
// `Params` describes any generic parameters that appear in the signature, and
// each listed type should be a `Param` specialization. For example,
// `Param<T, AnyInt>` means `T:! AnyInt`; continuing the above example, this
// constrains that the first parameter type and the return type are the same
// integer type.
template <typename SignatureFnType, typename... Params>
static auto ValidateSignature(const File& sem_ir,
                              llvm::ArrayRef<TypeId> arg_types,
                              TypeId return_type) -> bool {
  using SignatureTraits = llvm::function_traits<SignatureFnType*>;
  ValidateState state;

  // Must have expected number of arguments.
  if (arg_types.size() != SignatureTraits::num_args) {
    return false;
  }

  // Argument types must match.
  if (![&]<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
        return ((SignatureTraits::template arg_t<Indexes>::Check(
                    sem_ir, state, arg_types[Indexes])) &&
                ...);
      }(std::make_index_sequence<SignatureTraits::num_args>())) {
    return false;
  }

  // Result type must match.
  if (!SignatureTraits::result_t::Check(sem_ir, state, return_type)) {
    return false;
  }

  // Generic parameters must have the right types.
  if (!((Params::Check(sem_ir, state)) && ...)) {
    return false;
  }
  return true;
}

// Descriptions of builtin functions follow. For each builtin, a corresponding
// `BuiltinInfo` constant is declared describing properties of that builtin.

// Not a builtin function.
constexpr BuiltinInfo None = {"", nullptr};

// "int.add": integer addition.
constexpr BuiltinInfo IntAdd = {
    "int.add", ValidateSignature<auto(T, T)->T, Param<T, AnyInt>>};

}  // namespace BuiltinFunctionInfo

CARBON_DEFINE_ENUM_CLASS_NAMES(BuiltinFunctionKind) = {
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  BuiltinFunctionInfo::Name.name,
#include "toolchain/sem_ir/builtin_function_kind.def"
};

// Returns the builtin function kind with the given name, or None if the name
// is unknown.
auto BuiltinFunctionKind::ForBuiltinName(llvm::StringRef name)
    -> BuiltinFunctionKind {
  return llvm::StringSwitch<BuiltinFunctionKind>(name)
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  .Case(BuiltinFunctionInfo::Name.name, BuiltinFunctionKind::Name)
#include "toolchain/sem_ir/builtin_function_kind.def"
      .Default(BuiltinFunctionKind::None);
}

// Returns the builtin function kind corresponding to the given function
// callee, or None if the callee is not known to be a builtin.
auto BuiltinFunctionKind::ForCallee(const File& sem_ir, InstId callee_id)
    -> BuiltinFunctionKind {
  if (auto bound_method =
          sem_ir.insts().TryGetAs<SemIR::BoundMethod>(callee_id)) {
    callee_id = bound_method->function_id;
  }
  callee_id = sem_ir.constant_values().Get(callee_id).inst_id();
  if (!callee_id.is_valid()) {
    return SemIR::BuiltinFunctionKind::None;
  }
  if (auto callee = sem_ir.insts().TryGetAs<SemIR::FunctionDecl>(callee_id)) {
    const auto& function = sem_ir.functions().Get(callee->function_id);
    return function.builtin_kind;
  }
  return SemIR::BuiltinFunctionKind::None;
}

auto BuiltinFunctionKind::IsValidType(const File& sem_ir,
                                      llvm::ArrayRef<TypeId> arg_types,
                                      TypeId return_type) const -> bool {
  static constexpr BuiltinFunctionInfo::ValidateFn* ValidateFns[] = {
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  BuiltinFunctionInfo::Name.validate,
#include "toolchain/sem_ir/builtin_function_kind.def"
  };
  return ValidateFns[AsInt()](sem_ir, arg_types, return_type);
}

}  // namespace Carbon::SemIR
