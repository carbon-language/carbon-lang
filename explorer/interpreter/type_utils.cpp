// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_utils.h"

#include "explorer/ast/value.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

auto IsNonDeduceableType(Nonnull<const Value*> value) -> bool {
  return IsType(value) && !TypeIsDeduceable(value);
}

auto IsType(Nonnull<const Value*> value) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LocationValue:
    case Value::Kind::ReferenceExpressionValue:
    case Value::Kind::BoolValue:
    case Value::Kind::TupleValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::PointerType:
    case Value::Kind::FunctionType:
    case Value::Kind::StructType:
    case Value::Kind::TupleType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ChoiceType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
      return true;
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfNamespaceName:
      // These aren't first-class types, but they are still types.
      return true;
    case Value::Kind::AssociatedConstant: {
      // An associated type is an associated constant whose type is a
      // type-of-type.
      const auto& assoc = cast<AssociatedConstant>(*value);
      // TODO: Should we substitute in the arguments? Given
      //   interface I(T:! type) { let V:! T; }
      // ... is T.(I(type).V) considered to be a type?
      return IsTypeOfType(&assoc.constant().static_type());
    }
    case Value::Kind::MixinPseudoType:
      // Mixin type is a second-class type that cannot be used
      // within a type annotation expression.
      return false;
  }
}

auto TypeIsDeduceable(Nonnull<const Value*> type) -> bool {
  CARBON_CHECK(IsType(type)) << "expected a type, but found " << *type;

  switch (type->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LocationValue:
    case Value::Kind::ReferenceExpressionValue:
    case Value::Kind::BoolValue:
    case Value::Kind::TupleValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::MixinPseudoType:
      CARBON_FATAL() << "non-type value";
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::TypeOfNamespaceName:
      // These types do not contain other types.
      return false;
    case Value::Kind::FunctionType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ChoiceType:
    case Value::Kind::AssociatedConstant:
      // These types can contain other types, but those types can't involve
      // `auto`.
      return false;
    case Value::Kind::AutoType:
      return true;
    case Value::Kind::StructType:
      return llvm::any_of(
          llvm::map_range(cast<StructType>(type)->fields(),
                          [](const NamedValue& v) { return v.value; }),
          TypeIsDeduceable);
    case Value::Kind::TupleType:
      return llvm::any_of(cast<TupleType>(type)->elements(), TypeIsDeduceable);
    case Value::Kind::PointerType:
      return TypeIsDeduceable(&cast<PointerType>(type)->pointee_type());
    case Value::Kind::StaticArrayType:
      const auto* array_type = cast<StaticArrayType>(type);
      return !array_type->has_size() ||
             TypeIsDeduceable(&array_type->element_type());
  }
}

auto GetSize(Nonnull<const Value*> from) -> size_t {
  switch (from->kind()) {
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue: {
      const auto& from_tup = cast<TupleValueBase>(*from);
      return from_tup.elements().size();
    }
    case Value::Kind::StaticArrayType: {
      const auto& from_arr = cast<StaticArrayType>(*from);
      CARBON_CHECK(from_arr.has_size());
      return from_arr.size();
    }
    default:
      return 0;
  }
}

auto IsTypeOfType(Nonnull<const Value*> value) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LocationValue:
    case Value::Kind::ReferenceExpressionValue:
    case Value::Kind::BoolValue:
    case Value::Kind::TupleValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
      // These are values, not types.
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::MixinPseudoType:
    case Value::Kind::ChoiceType:
    case Value::Kind::StringType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::TupleType:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::TypeOfNamespaceName:
      // These are types whose values are not types.
      return false;
    case Value::Kind::AutoType:
    case Value::Kind::VariableType:
    case Value::Kind::AssociatedConstant:
      // A value of one of these types could be a type, but isn't known to be.
      return false;
    case Value::Kind::TypeType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
      // A value of one of these types is itself always a type.
      return true;
  }
}

auto DeducePatternType(Nonnull<const Value*> type,
                       Nonnull<const Value*> expected, Nonnull<Arena*> arena)
    -> Nonnull<const Value*> {
  if (type->kind() == Value::Kind::StaticArrayType) {
    const auto& arr = cast<StaticArrayType>(*type);
    const size_t size = arr.has_size() ? arr.size() : GetSize(expected);
    if (!IsNonDeduceableType(&arr.element_type())) {
      CARBON_CHECK(expected->kind() == Value::Kind::StaticArrayType ||
                   expected->kind() == Value::Kind::TupleType);

      Nonnull<const Value*> expected_elem_type =
          expected->kind() == Value::Kind::StaticArrayType
              ? &cast<StaticArrayType>(expected)->element_type()
              : cast<TupleType>(expected)->elements()[0];
      return arena->New<StaticArrayType>(
          DeducePatternType(&arr.element_type(), expected_elem_type, arena),
          size);
    } else {
      return arena->New<StaticArrayType>(&arr.element_type(), size);
    }
  }

  return expected;
}
}  // namespace Carbon
