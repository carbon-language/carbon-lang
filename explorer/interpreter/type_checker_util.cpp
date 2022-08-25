// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_checker_util.h"

#include "explorer/interpreter/value.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

void SetValue(Nonnull<Pattern*> pattern, Nonnull<const Value*> value) {
  // TODO: find some way to CHECK that `value` is identical to pattern->value(),
  // if it's already set. Unclear if `ValueEqual` is suitable, because it
  // currently focuses more on "real" values, and disallows the pseudo-values
  // like `BindingPlaceholderValue` that we get in pattern evaluation.
  if (!pattern->has_value()) {
    pattern->set_value(value);
  }
}

auto ExpectPointerType(SourceLocation source_loc, const std::string& context,
                       Nonnull<const Value*> actual) -> ErrorOr<Success> {
  // TODO: Try to resolve in equality context.
  if (actual->kind() != Value::Kind::PointerType) {
    return CompilationError(source_loc) << "type error in " << context << "\n"
                                        << "expected a pointer type\n"
                                        << "actual: " << *actual;
  }
  return Success();
}

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

auto IsType(Nonnull<const Value*> value, bool concrete /*= false*/) -> bool {
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

auto FindField(llvm::ArrayRef<NamedValue> fields, const std::string& field_name)
    -> std::optional<NamedValue> {
  auto it = std::find_if(
      fields.begin(), fields.end(),
      [&](const NamedValue& field) { return field.name == field_name; });
  if (it == fields.end()) {
    return std::nullopt;
  }
  return *it;
}

auto LookupInConstraint(SourceLocation source_loc, std::string_view lookup_kind,
                        Nonnull<const Value*> type,
                        std::string_view member_name)
    -> ErrorOr<ConstraintLookupResult> {
  // Find the set of lookup contexts.
  llvm::ArrayRef<ConstraintType::LookupContext> lookup_contexts;
  ConstraintType::LookupContext interface_context[1];
  if (const auto* iface_type = dyn_cast<InterfaceType>(type)) {
    // For an interface, look into that interface alone.
    // TODO: Also look into any interfaces extended by it.
    interface_context[0].context = iface_type;
    lookup_contexts = interface_context;
  } else if (const auto* constraint_type = dyn_cast<ConstraintType>(type)) {
    // For a constraint, look in all of its lookup contexts.
    lookup_contexts = constraint_type->lookup_contexts();
  } else {
    // Other kinds of constraint, such as TypeType, have no lookup contexts.
  }

  std::optional<ConstraintLookupResult> found;
  for (ConstraintType::LookupContext lookup : lookup_contexts) {
    if (!isa<InterfaceType>(lookup.context)) {
      // TODO: Support other kinds of lookup context, notably named
      // constraints.
      continue;
    }
    const InterfaceType& iface_type = cast<InterfaceType>(*lookup.context);
    if (std::optional<Nonnull<const Declaration*>> member =
            FindMember(member_name, iface_type.declaration().members());
        member.has_value()) {
      if (found.has_value()) {
        if (ValueEqual(found->interface, &iface_type, std::nullopt)) {
          continue;
        }
        // TODO: If we resolve to the same member either way, this
        // is not ambiguous.
        return CompilationError(source_loc)
               << "ambiguous " << lookup_kind << ", " << member_name
               << " found in " << *found->interface << " and " << iface_type;
      }
      found = {.interface = &iface_type, .member = member.value()};
    }
  }

  if (!found) {
    if (isa<TypeType>(type)) {
      return CompilationError(source_loc)
             << lookup_kind << " in unconstrained type";
    }
    return CompilationError(source_loc)
           << lookup_kind << ", " << member_name << " not in " << *type;
  }
  return found.value();
}

auto IsExhaustive(const Match& match) -> bool {
  for (const Match::Clause& clause : match.clauses()) {
    // A pattern consisting of a single variable binding is guaranteed to match.
    if (clause.pattern().kind() == PatternKind::BindingPattern) {
      return true;
    }
  }
  return false;
}

auto IsValidTypeForAliasTarget(Nonnull<const Value*> type) -> bool {
  switch (type->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::TupleValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::SymbolicWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
      CARBON_FATAL() << "type of alias target is not a type";

    case Value::Kind::AutoType:
    case Value::Kind::VariableType:
      CARBON_FATAL() << "pattern type in alias target";

    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::PointerType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::AssociatedConstant:
      return false;

    case Value::Kind::FunctionType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ConstraintType:
    case Value::Kind::TypeType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      return true;
  }
}

}  // namespace Carbon
