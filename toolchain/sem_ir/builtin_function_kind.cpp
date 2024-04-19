// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/builtin_function_kind.h"

#include <utility>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

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
constexpr int MaxTypeParams = 2;

// State used when validating a builtin signature that persists between
// individual checks.
struct ValidateState {
  // The type values of type parameters in the builtin signature. Invalid if
  // either no value has been deduced yet or the parameter is not used.
  TypeId type_params[MaxTypeParams] = {TypeId::Invalid, TypeId::Invalid};
};

// Constraint that a type is generic type parameter `I` of the builtin,
// satisfying `TypeConstraint`. See ValidateSignature for details.
template <int I, typename TypeConstraint>
struct TypeParam {
  static_assert(I >= 0 && I < MaxTypeParams);

  static auto Check(const File& sem_ir, ValidateState& state, TypeId type_id)
      -> bool {
    if (state.type_params[I].is_valid() && type_id != state.type_params[I]) {
      return false;
    }
    state.type_params[I] = type_id;
    return TypeConstraint::Check(sem_ir, state, type_id);
  }
};

// Constraint that a type is a specific builtin. See ValidateSignature for
// details.
template <const InstId& BuiltinId>
struct BuiltinType {
  static auto Check(const File& sem_ir, ValidateState& /*state*/,
                    TypeId type_id) -> bool {
    return sem_ir.types().GetInstId(type_id) == BuiltinId;
  }
};

// Constraint that a type is `bool`.
using Bool = BuiltinType<InstId::BuiltinBoolType>;

// Constraint that requires the type to be an integer type.
struct AnyInt {
  static auto Check(const File& sem_ir, ValidateState& state, TypeId type_id)
      -> bool {
    // TODO: Support Core.BigInt once it exists.
    if (BuiltinType<InstId::BuiltinIntType>::Check(sem_ir, state, type_id)) {
      return true;
    }
    return sem_ir.types().Is<IntType>(type_id);
  }
};

// Constraint that requires the type to be the type type.
using Type = BuiltinType<InstId::BuiltinTypeType>;

}  // namespace

// Validates that this builtin has a signature matching the specified signature.
//
// `SignatureFnType` is a C++ function type that describes the signature that is
// expected for this builtin. For example, `auto (AnyInt, AnyInt) -> AnyInt`
// specifies that the builtin takes values of two integer types and returns a
// value of a third integer type. Types used within the signature should provide
// a `Check` function that validates that the Carbon type is expected:
//
//   auto Check(const File&, ValidateState&, TypeId) -> bool;
//
// To constrain that the same type is used in multiple places in the signature,
// `TypeParam<I, T>` can be used. For example:
//
//   auto (TypeParam<0, AnyInt>, AnyInt) -> TypeParam<0, AnyInt>
//
// describes a builtin that takes two integers, and whose return type matches
// its first parameter type. For convenience, typedefs for `TypeParam<I, T>`
// are used in the descriptions of the builtins.
template <typename SignatureFnType>
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

  return true;
}

// Descriptions of builtin functions follow. For each builtin, a corresponding
// `BuiltinInfo` constant is declared describing properties of that builtin.
namespace BuiltinFunctionInfo {

// Convenience name used in the builtin type signatures below for a first
// generic type parameter that is constrained to be an integer type.
using IntT = TypeParam<0, AnyInt>;

// Convenience name used in the builtin type signatures below for a second
// generic type parameter that is constrained to be an integer type.
using IntU = TypeParam<1, AnyInt>;

// Not a builtin function.
constexpr BuiltinInfo None = {"", nullptr};

// Returns the `i32` type. Doesn't take a bit size because we need an integer
// type as a basis for that.
constexpr BuiltinInfo IntMakeType32 = {"int.make_type_32",
                                       ValidateSignature<auto()->Type>};

// Returns the `iN` type.
// TODO: Should we use a more specific type as the type of the bit width?
constexpr BuiltinInfo IntMakeTypeSigned = {
    "int.make_type_signed", ValidateSignature<auto(AnyInt)->Type>};

// Returns the `uN` type.
constexpr BuiltinInfo IntMakeTypeUnsigned = {
    "int.make_type_unsigned", ValidateSignature<auto(AnyInt)->Type>};

// Returns float types, such as `f64`. Currently only supports `f64`.
constexpr BuiltinInfo FloatMakeType = {"float.make_type",
                                       ValidateSignature<auto(AnyInt)->Type>};

// Returns the `bool` type.
constexpr BuiltinInfo BoolMakeType = {"bool.make_type",
                                      ValidateSignature<auto()->Type>};

// "int.snegate": integer negation.
constexpr BuiltinInfo IntSNegate = {"int.snegate",
                                    ValidateSignature<auto(IntT)->IntT>};

// "int.sadd": integer addition.
constexpr BuiltinInfo IntSAdd = {"int.sadd",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.ssub": integer subtraction.
constexpr BuiltinInfo IntSSub = {"int.ssub",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.smul": integer multiplication.
constexpr BuiltinInfo IntSMul = {"int.smul",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.sdiv": integer division.
constexpr BuiltinInfo IntSDiv = {"int.sdiv",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.smod": integer modulo.
constexpr BuiltinInfo IntSMod = {"int.smod",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.unegate": unsigned integer negation.
constexpr BuiltinInfo IntUNegate = {"int.unegate",
                                    ValidateSignature<auto(IntT)->IntT>};

// "int.uadd": unsigned integer addition.
constexpr BuiltinInfo IntUAdd = {"int.uadd",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.usub": unsigned integer subtraction.
constexpr BuiltinInfo IntUSub = {"int.usub",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.umul": unsigned integer multiplication.
constexpr BuiltinInfo IntUMul = {"int.umul",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.udiv": unsigned integer division.
constexpr BuiltinInfo IntUDiv = {"int.udiv",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.mod": integer modulo.
constexpr BuiltinInfo IntUMod = {"int.umod",
                                 ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.complement": integer bitwise complement.
constexpr BuiltinInfo IntComplement = {"int.complement",
                                       ValidateSignature<auto(IntT)->IntT>};

// "int.and": integer bitwise and.
constexpr BuiltinInfo IntAnd = {"int.and",
                                ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.or": integer bitwise or.
constexpr BuiltinInfo IntOr = {"int.or",
                               ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.xor": integer bitwise xor.
constexpr BuiltinInfo IntXor = {"int.xor",
                                ValidateSignature<auto(IntT, IntT)->IntT>};

// "int.left_shift": integer left shift.
constexpr BuiltinInfo IntLeftShift = {
    "int.left_shift", ValidateSignature<auto(IntT, IntU)->IntT>};

// "int.left_shift": integer right shift.
constexpr BuiltinInfo IntRightShift = {
    "int.right_shift", ValidateSignature<auto(IntT, IntU)->IntT>};

// "int.eq": integer equality comparison.
constexpr BuiltinInfo IntEq = {"int.eq",
                               ValidateSignature<auto(IntT, IntT)->Bool>};

// "int.neq": integer non-equality comparison.
constexpr BuiltinInfo IntNeq = {"int.neq",
                                ValidateSignature<auto(IntT, IntT)->Bool>};

// "int.less": integer less than comparison.
constexpr BuiltinInfo IntLess = {"int.less",
                                 ValidateSignature<auto(IntT, IntT)->Bool>};

// "int.less_eq": integer less than or equal comparison.
constexpr BuiltinInfo IntLessEq = {"int.less_eq",
                                   ValidateSignature<auto(IntT, IntT)->Bool>};

// "int.greater": integer greater than comparison.
constexpr BuiltinInfo IntGreater = {"int.greater",
                                    ValidateSignature<auto(IntT, IntT)->Bool>};

// "int.greater_eq": integer greater than or equal comparison.
constexpr BuiltinInfo IntGreaterEq = {
    "int.greater_eq", ValidateSignature<auto(IntT, IntT)->Bool>};

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
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  if (name == BuiltinFunctionInfo::Name.name) {   \
    return BuiltinFunctionKind::Name;             \
  }
#include "toolchain/sem_ir/builtin_function_kind.def"
  return BuiltinFunctionKind::None;
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
    return sem_ir.functions().Get(callee->function_id).builtin_kind;
  }
  return SemIR::BuiltinFunctionKind::None;
}

auto BuiltinFunctionKind::IsValidType(const File& sem_ir,
                                      llvm::ArrayRef<TypeId> arg_types,
                                      TypeId return_type) const -> bool {
  static constexpr ValidateFn* ValidateFns[] = {
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  BuiltinFunctionInfo::Name.validate,
#include "toolchain/sem_ir/builtin_function_kind.def"
  };
  return ValidateFns[AsInt()](sem_ir, arg_types, return_type);
}

}  // namespace Carbon::SemIR
