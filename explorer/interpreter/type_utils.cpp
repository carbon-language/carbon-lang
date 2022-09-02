// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_utils.h"

namespace Carbon {

using ::llvm::cast;

auto IsTypeOfType(Nonnull<const Value*> value) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::SymbolicWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      // These are values, not types.
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::TupleValue:
      // These are types whose values are not types.
      return false;
    case Value::Kind::AutoType:
    case Value::Kind::VariableType:
    case Value::Kind::AssociatedConstant:
      // A value of one of these types could be a type, but isn't known to be.
      return false;
    case Value::Kind::TypeType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ConstraintType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
      // A value of one of these types is itself always a type.
      return true;
  }
}

auto IsType(Nonnull<const Value*> value, bool concrete) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::SymbolicWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
      return false;
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      // Names aren't first-class values, and their types aren't first-class
      // types.
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::StaticArrayType:
      return true;
    case Value::Kind::AutoType:
      // `auto` isn't a concrete type, it's a pattern that matches types.
      return !concrete;
    case Value::Kind::TupleValue: {
      for (Nonnull<const Value*> field : cast<TupleValue>(*value).elements()) {
        if (!IsType(field, concrete)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::PointerType: {
      return IsType(&cast<PointerType>(*value).type(), concrete);
    }
    case Value::Kind::AssociatedConstant: {
      // An associated type is an associated constant whose type is a
      // type-of-type.
      const auto& assoc = cast<AssociatedConstant>(*value);
      // TODO: Should we substitute in the arguments? Given
      //   interface I(T:! Type) { let V:! T; }
      // ... is T.(I(Type).V) considered to be a type?
      return IsTypeOfType(&assoc.constant().static_type());
    }
  }
}

auto IsConcreteType(Nonnull<const Value*> value) -> bool {
  return IsType(value, /*concrete=*/true);
}

auto IsSameType(Nonnull<const Value*> type1, Nonnull<const Value*> type2,
                const ImplScope& impl_scope) -> bool {
  SingleStepEqualityContext equality_ctx(this, &impl_scope);
  return TypeEqual(type1, type2, &equality_ctx);
}

auto ExpectType(SourceLocation source_loc, const std::string& context,
                Nonnull<const Value*> expected, Nonnull<const Value*> actual,
                const ImplScope& impl_scope) -> ErrorOr<Success> {
  if (!IsImplicitlyConvertible(actual, expected, impl_scope,
                               /*allow_user_defined_conversions=*/true)) {
    return CompilationError(source_loc)
           << "type error in " << context << ": "
           << "'" << *actual << "' is not implicitly convertible to '"
           << *expected << "'";
  } else {
    return Success();
  }
}

auto ExpectExactType(SourceLocation source_loc, const std::string& context,
                     Nonnull<const Value*> expected,
                     Nonnull<const Value*> actual, const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (!IsSameType(expected, actual, impl_scope)) {
    return CompilationError(source_loc) << "type error in " << context << "\n"
                                        << "expected: " << *expected << "\n"
                                        << "actual: " << *actual;
  }
  return Success();
}

}  // namespace Carbon
