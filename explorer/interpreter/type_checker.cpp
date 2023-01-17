// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_checker.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "common/check.h"
#include "common/error.h"
#include "common/ostream.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/common/arena.h"
#include "explorer/common/error_builders.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/interpreter.h"
#include "explorer/interpreter/pattern_analysis.h"
#include "explorer/interpreter/value.h"
#include "explorer/interpreter/value_transform.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SaveAndRestore.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

auto TypeChecker::IsSameType(Nonnull<const Value*> type1,
                             Nonnull<const Value*> type2,
                             const ImplScope& /*impl_scope*/) const -> bool {
  return TypeEqual(type1, type2, std::nullopt);
}

auto TypeChecker::ExpectExactType(SourceLocation source_loc,
                                  std::string_view context,
                                  Nonnull<const Value*> expected,
                                  Nonnull<const Value*> actual,
                                  const ImplScope& impl_scope) const
    -> ErrorOr<Success> {
  if (!IsSameType(expected, actual, impl_scope)) {
    return ProgramError(source_loc) << "type error in " << context << "\n"
                                    << "expected: " << *expected << "\n"
                                    << "actual: " << *actual;
  }
  return Success();
}

static auto ExpectPointerType(SourceLocation source_loc,
                              std::string_view context,
                              Nonnull<const Value*> actual)
    -> ErrorOr<Success> {
  // TODO: Try to resolve in equality context.
  if (actual->kind() != Value::Kind::PointerType) {
    return ProgramError(source_loc) << "type error in " << context << "\n"
                                    << "expected a pointer type\n"
                                    << "actual: " << *actual;
  }
  return Success();
}

// Returns whether the value is a type whose values are themselves known to be
// types.
static auto IsTypeOfType(Nonnull<const Value*> value) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::TupleValue:
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
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::TupleType:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
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

// Returns whether the value is a type value, such as might be a valid type for
// a syntactic pattern. This includes types involving `auto`. Use
// `TypeContainsAuto` to determine if a type involves `auto`.
static auto IsType(Nonnull<const Value*> value) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::TupleValue:
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
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
      return true;
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::TypeOfMixinPseudoType:
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

// Expect that a type is complete. Issue a diagnostic if not.
static auto ExpectCompleteType(SourceLocation source_loc,
                               std::string_view context,
                               Nonnull<const Value*> type) -> ErrorOr<Success> {
  CARBON_CHECK(IsType(type));

  switch (type->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::TupleValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::MixinPseudoType:
      CARBON_FATAL() << "should not see non-type values";

    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::StringType:
    case Value::Kind::PointerType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::StructType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::AssociatedConstant:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::TypeOfMixinPseudoType: {
      // These types are always complete.
      return Success();
    }

    case Value::Kind::StaticArrayType:
      // TODO: This should probably be complete only if the element type is
      // complete.
      return Success();

    case Value::Kind::TupleType: {
      // TODO: Tuple types should be complete only if all element types are
      // complete.
      return Success();
    }

    // TODO: Once we support forward-declarations, make sure we have an actual
    // definition in these cases.
    case Value::Kind::NominalClassType: {
      if (cast<NominalClassType>(type)->declaration().is_declared()) {
        return Success();
      }
      break;
    }
    case Value::Kind::NamedConstraintType: {
      if (cast<NamedConstraintType>(type)->declaration().is_declared()) {
        return Success();
      }
      break;
    }
    case Value::Kind::InterfaceType: {
      if (cast<InterfaceType>(type)->declaration().is_declared()) {
        return Success();
      }
      break;
    }
    case Value::Kind::ChoiceType: {
      if (cast<ChoiceType>(type)->declaration().is_declared()) {
        return Success();
      }
      break;
    }

    case Value::Kind::AutoType: {
      // Undeduced `auto` is considered incomplete.
      break;
    }
  }

  return ProgramError(source_loc)
         << "incomplete type `" << *type << "` used in " << context;
}

// Returns whether *value represents the type of a Carbon value, as
// opposed to a type pattern or a non-type value.
static auto TypeContainsAuto(Nonnull<const Value*> type) -> bool {
  CARBON_CHECK(IsType(type)) << "expected a type, but found " << *type;

  switch (type->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::TupleValue:
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
      // These types do not contain other types.
      return false;
    case Value::Kind::FunctionType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
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
          TypeContainsAuto);
    case Value::Kind::TupleType:
      return llvm::any_of(cast<TupleType>(type)->elements(), TypeContainsAuto);
    case Value::Kind::PointerType:
      return TypeContainsAuto(&cast<PointerType>(type)->pointee_type());
    case Value::Kind::StaticArrayType:
      return TypeContainsAuto(&cast<StaticArrayType>(type)->element_type());
  }
}

// Returns whether `type` is a placeholder type, which is a second-class type
// that cannot be the type of a binding but can be the type of an expression.
static auto IsPlaceholderType(Nonnull<const Value*> type) -> bool {
  CARBON_CHECK(IsType(type)) << "expected a type, but found " << *type;
  return isa<TypeOfParameterizedEntityName, TypeOfMemberName,
             TypeOfMixinPseudoType>(type);
}

// Returns whether `value` is a concrete type, which would be valid as the
// static type of an expression. This is currently any type other than `auto`.
static auto IsConcreteType(Nonnull<const Value*> value) -> bool {
  return IsType(value) && !TypeContainsAuto(value);
}

// Returns the named field, or None if not found.
static auto FindField(llvm::ArrayRef<NamedValue> fields,
                      const std::string& field_name)
    -> std::optional<NamedValue> {
  const auto* it = std::find_if(
      fields.begin(), fields.end(),
      [&](const NamedValue& field) { return field.name == field_name; });
  if (it == fields.end()) {
    return std::nullopt;
  }
  return *it;
}

auto TypeChecker::FieldTypesImplicitlyConvertible(
    llvm::ArrayRef<NamedValue> source_fields,
    llvm::ArrayRef<NamedValue> destination_fields,
    const ImplScope& impl_scope) const -> bool {
  // TODO: If default fields are implemented, the
  // code must be adapted to skip them.
  // Ensure every field name exists in the destination.
  for (const auto& dest_field : destination_fields) {
    if (!FindField(source_fields, dest_field.name)) {
      return false;
    }
  }
  for (const auto& source_field : source_fields) {
    std::optional<NamedValue> destination_field =
        FindField(destination_fields, source_field.name);
    if (!destination_field.has_value() ||
        !IsImplicitlyConvertible(source_field.value,
                                 destination_field.value().value, impl_scope,
                                 // TODO: We don't have a way to perform
                                 // user-defined conversions of a struct field
                                 // yet, because we can't write a suitable impl
                                 // for ImplicitAs.
                                 /*allow_user_defined_conversions=*/false)) {
      return false;
    }
  }
  return true;
}

auto TypeChecker::FieldTypes(const NominalClassType& class_type) const
    -> std::vector<NamedValue> {
  std::vector<NamedValue> field_types;
  for (Nonnull<Declaration*> m : class_type.declaration().members()) {
    switch (m->kind()) {
      case DeclarationKind::VariableDeclaration: {
        const auto& var = cast<VariableDeclaration>(*m);
        Nonnull<const Value*> field_type =
            Substitute(class_type.bindings(), &var.binding().static_type());
        field_types.push_back({var.binding().name(), field_type});
        break;
      }
      default:
        break;
    }
  }
  return field_types;
}

auto TypeChecker::FieldTypesWithBase(const NominalClassType& class_type) const
    -> std::vector<NamedValue> {
  auto fields = FieldTypes(class_type);
  if (const auto base_type = class_type.base()) {
    auto base_fields = FieldTypesWithBase(*base_type.value());
    fields.emplace_back(NamedValue{std::string(NominalClassValue::BaseField),
                                   base_type.value()});
  }
  return fields;
}

auto TypeChecker::IsImplicitlyConvertible(
    Nonnull<const Value*> source, Nonnull<const Value*> destination,
    const ImplScope& impl_scope, bool allow_user_defined_conversions) const
    -> bool {
  // Check for an exact match or for an implicit conversion.
  // TODO: `impl`s of `ImplicitAs` should be provided to cover these
  // conversions.
  CARBON_CHECK(IsConcreteType(source));
  CARBON_CHECK(IsConcreteType(destination));
  if (IsSameType(source, destination, impl_scope)) {
    return true;
  }

  switch (source->kind()) {
    case Value::Kind::StructType:
      switch (destination->kind()) {
        case Value::Kind::StructType:
          if (FieldTypesImplicitlyConvertible(
                  cast<StructType>(*source).fields(),
                  cast<StructType>(*destination).fields(), impl_scope)) {
            return true;
          }
          break;
        case Value::Kind::NominalClassType:
          if (IsImplicitlyConvertible(
                  source,
                  arena_->New<StructType>(
                      FieldTypesWithBase(cast<NominalClassType>(*destination))),
                  impl_scope, allow_user_defined_conversions)) {
            return true;
          }
          break;
        case Value::Kind::TypeType:
        case Value::Kind::InterfaceType:
        case Value::Kind::NamedConstraintType:
        case Value::Kind::ConstraintType:
          // A value of empty struct type implicitly converts to a type.
          if (cast<StructType>(*source).fields().empty()) {
            return true;
          }
          break;
        default:
          break;
      }
      break;
    case Value::Kind::TupleType: {
      const auto& source_tuple = cast<TupleType>(*source);
      switch (destination->kind()) {
        case Value::Kind::TupleType: {
          const auto& destination_tuple = cast<TupleType>(*destination);
          if (source_tuple.elements().size() !=
              destination_tuple.elements().size()) {
            break;
          }
          bool all_ok = true;
          for (size_t i = 0; i < source_tuple.elements().size(); ++i) {
            if (!IsImplicitlyConvertible(
                    source_tuple.elements()[i], destination_tuple.elements()[i],
                    impl_scope, /*allow_user_defined_conversions=*/false)) {
              all_ok = false;
              break;
            }
          }
          if (all_ok) {
            return true;
          }
          break;
        }
        case Value::Kind::StaticArrayType: {
          const auto& destination_array = cast<StaticArrayType>(*destination);
          if (destination_array.size() != source_tuple.elements().size()) {
            break;
          }
          bool all_ok = true;
          for (Nonnull<const Value*> source_element : source_tuple.elements()) {
            if (!IsImplicitlyConvertible(
                    source_element, &destination_array.element_type(),
                    impl_scope, /*allow_user_defined_conversions=*/false)) {
              all_ok = false;
              break;
            }
          }
          if (all_ok) {
            return true;
          }
          break;
        }
        case Value::Kind::TypeType:
        case Value::Kind::InterfaceType:
        case Value::Kind::NamedConstraintType:
        case Value::Kind::ConstraintType: {
          // A tuple value converts to a type if all of its fields do.
          bool all_types = true;
          for (Nonnull<const Value*> source_element : source_tuple.elements()) {
            if (!IsImplicitlyConvertible(
                    source_element, destination, impl_scope,
                    /*allow_user_defined_conversions=*/false)) {
              all_types = false;
              break;
            }
          }
          if (all_types) {
            return true;
          }
          break;
        }
        default:
          break;
      }
      break;
    }
    case Value::Kind::TypeType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
      // TODO: We can't tell whether the conversion to this type-of-type would
      // work, because that depends on the source value, and we only have its
      // type.
      return IsTypeOfType(destination);
    case Value::Kind::PointerType: {
      if (destination->kind() != Value::Kind::PointerType) {
        break;
      }
      const auto* src_ptr = cast<PointerType>(source);
      const auto* dest_ptr = cast<PointerType>(destination);
      if (src_ptr->pointee_type().kind() != Value::Kind::NominalClassType ||
          dest_ptr->pointee_type().kind() != Value::Kind::NominalClassType) {
        break;
      }
      const auto& src_class = cast<NominalClassType>(src_ptr->pointee_type());
      if (src_class.InheritsClass(&dest_ptr->pointee_type())) {
        return true;
      }
      break;
    }
    default:
      break;
  }

  // If we're not supposed to look for a user-defined conversion, we're done.
  if (!allow_user_defined_conversions) {
    return false;
  }

  // We didn't find a builtin implicit conversion. Try a user-defined one.
  SourceLocation source_loc = SourceLocation::DiagnosticsIgnored();
  ErrorOr<Nonnull<const InterfaceType*>> iface_type = GetBuiltinInterfaceType(
      source_loc, BuiltinInterfaceName{Builtins::ImplicitAs, destination});
  return iface_type.ok() &&
         impl_scope.Resolve(*iface_type, source, source_loc, *this).ok();
}

auto TypeChecker::BuildSubtypeConversion(Nonnull<Expression*> source,
                                         Nonnull<const PointerType*> src_ptr,
                                         Nonnull<const PointerType*> dest_ptr)
    -> ErrorOr<Nonnull<const Expression*>> {
  const auto* src_class = dyn_cast<NominalClassType>(&src_ptr->pointee_type());
  const auto* dest_class =
      dyn_cast<NominalClassType>(&dest_ptr->pointee_type());
  const auto dest = dest_class->declaration().name();
  CARBON_CHECK(src_class && dest_class)
      << "Invalid source or destination pointee";
  Nonnull<Expression*> last_expr = source;
  const auto* cur_class = src_class;
  while (!TypeEqual(cur_class, dest_class, std::nullopt)) {
    const auto src = src_class->declaration().name();
    const auto base_class = cur_class->base();
    CARBON_CHECK(base_class) << "Invalid subtyping conversion";
    auto* base_expr = arena_->New<BaseAccessExpression>(
        source->source_loc(), last_expr,
        arena_->New<BaseElement>(arena_->New<PointerType>(*base_class)));
    last_expr = base_expr;
    cur_class = *base_class;
  }
  CARBON_CHECK(last_expr) << "Error, no conversion was needed";
  return last_expr;
}

auto TypeChecker::ImplicitlyConvert(std::string_view context,
                                    const ImplScope& impl_scope,
                                    Nonnull<Expression*> source,
                                    Nonnull<const Value*> destination)
    -> ErrorOr<Nonnull<Expression*>> {
  Nonnull<const Value*> source_type = &source->static_type();

  CARBON_RETURN_IF_ERROR(
      ExpectNonPlaceholderType(source->source_loc(), &source->static_type()));

  if (TypeEqual(&source->static_type(), destination, std::nullopt)) {
    // No conversions are required.
    return source;
  }

  // TODO: This doesn't work for cases of combined built-in and user-defined
  // conversion, such as converting a struct element via an `ImplicitAs` impl.
  if (IsImplicitlyConvertible(source_type, destination, impl_scope,
                              /*allow_user_defined_conversions=*/false)) {
    // A type only implicitly converts to a constraint if there is an impl of
    // that constraint for that type in scope.
    // TODO: Instead of excluding the special case where the destination is
    // `type`, we should check if the source type has a subset of the
    // constraints of the destination type. In that case, the source should not
    // be required to be constant.
    if (IsTypeOfType(destination) && !isa<TypeType>(destination)) {
      // First convert the source expression to type `type`.
      CARBON_ASSIGN_OR_RETURN(Nonnull<Expression*> source_as_type,
                              ImplicitlyConvert(context, impl_scope, source,
                                                arena_->New<TypeType>()));
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> converted_value,
                              InterpExp(source_as_type, arena_, trace_stream_));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const ConstraintType*> destination_constraint,
          ConvertToConstraintType(source->source_loc(), "implicit conversion",
                                  destination));
      destination = destination_constraint;
      if (trace_stream_) {
        **trace_stream_ << "converting type " << *converted_value
                        << " to constraint " << *destination_constraint
                        << " for " << context << " in scope " << impl_scope
                        << "\n";
      }
      // Note, we discard the witness. We don't actually need it in order to
      // perform the conversion, but we do want to know it exists.
      // TODO: A value of constraint type should carry both the type and the
      // witness.
      CARBON_RETURN_IF_ERROR(impl_scope.Resolve(destination_constraint,
                                                converted_value,
                                                source->source_loc(), *this));
      return arena_->New<ValueLiteral>(source->source_loc(), converted_value,
                                       destination_constraint,
                                       ValueCategory::Let);
    }

    if (IsTypeOfType(source_type) && IsTypeOfType(destination)) {
      // No conversion is required.
      return source;
    }

    // Perform the builtin conversion.
    auto* convert_expr =
        arena_->New<BuiltinConvertExpression>(source, destination);

    // For subtyping, rewrite into successive `.base` accesses.
    if (isa<PointerType>(source_type) && isa<PointerType>(destination) &&
        cast<PointerType>(destination)->pointee_type().kind() ==
            Value::Kind::NominalClassType) {
      CARBON_ASSIGN_OR_RETURN(
          const auto* rewrite,
          BuildSubtypeConversion(source, cast<PointerType>(source_type),
                                 cast<PointerType>(destination)))
      convert_expr->set_rewritten_form(rewrite);
    }

    return convert_expr;
  }

  ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
      impl_scope, source,
      BuiltinInterfaceName{Builtins::ImplicitAs, destination},
      BuiltinMethodCall{"Convert"});
  if (!converted.ok()) {
    // We couldn't find a matching `impl`.
    return ProgramError(source->source_loc())
           << "type error in " << context << ": "
           << "'" << *source_type << "' is not implicitly convertible to '"
           << *destination << "'";
  }
  return *converted;
}

auto TypeChecker::GetBuiltinInterfaceType(SourceLocation source_loc,
                                          BuiltinInterfaceName interface) const
    -> ErrorOr<Nonnull<const InterfaceType*>> {
  auto bad_builtin = [&]() -> Error {
    return ProgramError(source_loc)
           << "unsupported declaration for builtin `"
           << Builtins::GetName(interface.builtin) << "`";
  };

  // Find the builtin interface declaration.
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Declaration*> builtin_decl,
                          builtins_.Get(source_loc, interface.builtin));
  const auto* iface_decl = dyn_cast<InterfaceDeclaration>(builtin_decl);
  if (!iface_decl || !iface_decl->constant_value()) {
    return bad_builtin();
  }

  // Match the interface arguments up with the parameters and build the
  // interface type.
  bool has_parameters = iface_decl->params().has_value();
  bool has_arguments = !interface.arguments.empty();
  if (has_parameters != has_arguments) {
    return bad_builtin();
  }
  BindingMap binding_args;
  if (has_arguments) {
    TupleValue args(interface.arguments);
    if (!PatternMatch(&iface_decl->params().value()->value(), &args, source_loc,
                      std::nullopt, binding_args, trace_stream_,
                      this->arena_)) {
      return bad_builtin();
    }
  }
  Nonnull<const Bindings*> bindings =
      arena_->New<Bindings>(std::move(binding_args), Bindings::NoWitnesses);
  return arena_->New<InterfaceType>(iface_decl, bindings);
}

auto TypeChecker::BuildBuiltinMethodCall(const ImplScope& impl_scope,
                                         Nonnull<Expression*> source,
                                         BuiltinInterfaceName interface,
                                         BuiltinMethodCall method)
    -> ErrorOr<Nonnull<Expression*>> {
  const SourceLocation source_loc = source->source_loc();
  CARBON_ASSIGN_OR_RETURN(Nonnull<const InterfaceType*> iface_type,
                          GetBuiltinInterfaceType(source_loc, interface));

  // Build an expression to perform the call `source.(interface.method)(args)`.
  Nonnull<Expression*> iface_expr = arena_->New<ValueLiteral>(
      source_loc, iface_type, arena_->New<TypeType>(), ValueCategory::Let);
  Nonnull<Expression*> iface_member = arena_->New<SimpleMemberAccessExpression>(
      source_loc, iface_expr, method.name);
  Nonnull<Expression*> method_access =
      arena_->New<CompoundMemberAccessExpression>(source_loc, source,
                                                  iface_member);
  Nonnull<Expression*> call_args =
      arena_->New<TupleLiteral>(source_loc, method.arguments);
  Nonnull<Expression*> call =
      arena_->New<CallExpression>(source_loc, method_access, call_args);
  CARBON_RETURN_IF_ERROR(TypeCheckExp(call, impl_scope));
  return {call};
}

// Checks that the given type is not a placeholder type. Diagnoses otherwise.
auto TypeChecker::ExpectNonPlaceholderType(SourceLocation source_loc,
                                           Nonnull<const Value*> type)
    -> ErrorOr<Success> {
  if (!IsPlaceholderType(type)) {
    return Success();
  }
  if (auto* member_name = dyn_cast<TypeOfMemberName>(type)) {
    return ProgramError(source_loc)
           << *member_name << " can only be used in a member access or alias";
  }
  if (auto* param_entity = dyn_cast<TypeOfParameterizedEntityName>(type)) {
    return ProgramError(source_loc)
           << "'" << param_entity->name() << "' must be given an argument list";
  }
  if (auto* mixin_type = dyn_cast<TypeOfMixinPseudoType>(type)) {
    return ProgramError(source_loc)
           << "invalid use of mixin "
           << mixin_type->mixin_type().declaration().name();
  }
  CARBON_FATAL() << "unknown kind of placeholder type " << *type;
}

auto TypeChecker::ExpectType(SourceLocation source_loc,
                             std::string_view context,
                             Nonnull<const Value*> expected,
                             Nonnull<const Value*> actual,
                             const ImplScope& impl_scope) const
    -> ErrorOr<Success> {
  if (!IsImplicitlyConvertible(actual, expected, impl_scope,
                               /*allow_user_defined_conversions=*/true)) {
    return ProgramError(source_loc)
           << "type error in " << context << ": "
           << "'" << *actual << "' is not implicitly convertible to '"
           << *expected << "'";
  } else {
    return Success();
  }
}

// Argument deduction matches two values and attempts to find a set of
// substitutions into deduced bindings in one of them that would result in the
// other.
class TypeChecker::ArgumentDeduction {
 public:
  ArgumentDeduction(
      SourceLocation source_loc, std::string_view context,
      llvm::ArrayRef<Nonnull<const GenericBinding*>> bindings_to_deduce,
      std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
      : source_loc_(source_loc),
        context_(context),
        deduced_bindings_in_order_(bindings_to_deduce),
        trace_stream_(trace_stream) {
    if (trace_stream_) {
      **trace_stream_ << "performing argument deduction for bindings: ";
      llvm::ListSeparator sep;
      for (const auto* binding : bindings_to_deduce) {
        **trace_stream_ << sep << *binding;
      }
      **trace_stream_ << "\n";
    }
    for (const auto* binding : bindings_to_deduce) {
      deduced_values_.insert({binding, {}});
    }
  }

  // Deduces the values of deduced bindings in `param` from the corresponding
  // values in `arg`. `allow_implicit_conversion` specifies whether implicit
  // conversions are permitted from the argument to the parameter type.
  auto Deduce(Nonnull<const Value*> param, Nonnull<const Value*> arg,
              bool allow_implicit_conversion) -> ErrorOr<Success>;

  // Finds a binding to deduce that has not been deduced, if any exist.
  auto FindUndeducedBinding() const
      -> std::optional<Nonnull<const GenericBinding*>> {
    for (const auto* binding : deduced_bindings_in_order_) {
      llvm::ArrayRef<Nonnull<const Value*>> values =
          deduced_values_.find(binding)->second;
      if (values.empty()) {
        return binding;
      }
    }
    return std::nullopt;
  }

  // Adds a value for a binding that is not deduced but still participates in
  // substitution. For example, the `T` parameter in `fn F(T:! type, x: T)`.
  void AddNonDeducedBindingValue(Nonnull<const GenericBinding*> binding,
                                 Nonnull<Expression*> argument) {
    non_deduced_values_.push_back({binding, argument});
  }

  // Finishes deduction and forms a set of substitutions that transform `param`
  // into `arg`.
  auto Finish(TypeChecker& type_checker, const ImplScope& impl_scope) const
      -> ErrorOr<Bindings>;

 private:
  SourceLocation source_loc_;
  std::string_view context_;
  llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings_in_order_;
  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream_;

  // Values for deduced bindings.
  std::map<Nonnull<const GenericBinding*>,
           llvm::TinyPtrVector<Nonnull<const Value*>>>
      deduced_values_;
  // Values for non-deduced bindings, such as parameters with corresponding
  // argument expressions.
  std::vector<std::pair<Nonnull<const GenericBinding*>, Nonnull<Expression*>>>
      non_deduced_values_;

  // Non-deduced mismatches that we deferred until we could perform
  // substitutions into them.
  struct NonDeducedMismatch {
    Nonnull<const Value*> param;
    Nonnull<const Value*> arg;
    bool allow_implicit_conversion;
  };
  std::vector<NonDeducedMismatch> non_deduced_mismatches_;
};

auto TypeChecker::ArgumentDeduction::Deduce(Nonnull<const Value*> param,
                                            Nonnull<const Value*> arg,
                                            bool allow_implicit_conversion)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "deducing " << *param << " from " << *arg << "\n";
  }

  // Handle the case where we can't perform deduction, either because the
  // parameter is a primitive type or because the parameter and argument have
  // different forms. In this case, we require an implicit conversion to exist,
  // or for an exact type match if implicit conversions are not permitted.
  auto handle_non_deduced_type = [&]() -> ErrorOr<Success> {
    if (ValueEqual(param, arg, std::nullopt)) {
      return Success();
    }

    // Defer checking until we can substitute into the parameter and see if it
    // actually matches.
    non_deduced_mismatches_.push_back(
        {.param = param,
         .arg = arg,
         .allow_implicit_conversion = allow_implicit_conversion});
    return Success();
  };

  switch (param->kind()) {
    case Value::Kind::VariableType: {
      const auto& binding = cast<VariableType>(*param).binding();
      if (auto it = deduced_values_.find(&binding);
          it != deduced_values_.end()) {
        it->second.push_back(arg);
      } else {
        return handle_non_deduced_type();
      }
      return Success();
    }
    case Value::Kind::TupleType: {
      if (arg->kind() != Value::Kind::TupleType) {
        return handle_non_deduced_type();
      }
      const auto& param_tup = cast<TupleType>(*param);
      const auto& arg_tup = cast<TupleType>(*arg);
      if (param_tup.elements().size() != arg_tup.elements().size()) {
        return ProgramError(source_loc_)
               << "mismatch in tuple sizes, expected "
               << param_tup.elements().size() << " but got "
               << arg_tup.elements().size();
      }
      for (size_t i = 0; i < param_tup.elements().size(); ++i) {
        CARBON_RETURN_IF_ERROR(Deduce(param_tup.elements()[i],
                                      arg_tup.elements()[i],
                                      allow_implicit_conversion));
      }
      return Success();
    }
    case Value::Kind::StructType: {
      if (arg->kind() != Value::Kind::StructType) {
        return handle_non_deduced_type();
      }
      const auto& param_struct = cast<StructType>(*param);
      const auto& arg_struct = cast<StructType>(*arg);
      auto diagnose_missing_field = [&](const StructType& struct_type,
                                        const NamedValue& field,
                                        bool missing_from_source) -> Error {
        static constexpr const char* SourceOrDestination[2] = {"source",
                                                               "destination"};
        return ProgramError(source_loc_)
               << "mismatch in field names, "
               << SourceOrDestination[missing_from_source ? 1 : 0] << " field `"
               << field.name << "` not in "
               << SourceOrDestination[missing_from_source ? 0 : 1] << " type `"
               << struct_type << "`";
      };
      for (size_t i = 0; i < param_struct.fields().size(); ++i) {
        NamedValue param_field = param_struct.fields()[i];
        std::optional<NamedValue> arg_field;
        if (allow_implicit_conversion) {
          if (std::optional<NamedValue> maybe_arg_field =
                  FindField(arg_struct.fields(), param_field.name)) {
            arg_field = maybe_arg_field;
          } else {
            return diagnose_missing_field(arg_struct, param_field, true);
          }
        } else {
          if (i >= arg_struct.fields().size()) {
            return diagnose_missing_field(arg_struct, param_field, true);
          }
          arg_field = arg_struct.fields()[i];
          if (param_field.name != arg_field->name) {
            return ProgramError(source_loc_)
                   << "mismatch in field names, `" << param_field.name
                   << "` != `" << arg_field->name << "`";
          }
        }
        CARBON_RETURN_IF_ERROR(Deduce(param_field.value, arg_field->value,
                                      allow_implicit_conversion));
      }
      if (param_struct.fields().size() != arg_struct.fields().size()) {
        CARBON_CHECK(allow_implicit_conversion)
            << "should have caught this earlier";
        for (const NamedValue& arg_field : arg_struct.fields()) {
          if (!FindField(param_struct.fields(), arg_field.name).has_value()) {
            return diagnose_missing_field(param_struct, arg_field, false);
          }
        }
        CARBON_FATAL() << "field count mismatch but no missing field; "
                       << "duplicate field name?";
      }
      return Success();
    }
    case Value::Kind::FunctionType: {
      if (arg->kind() != Value::Kind::FunctionType) {
        return handle_non_deduced_type();
      }
      const auto& param_fn = cast<FunctionType>(*param);
      const auto& arg_fn = cast<FunctionType>(*arg);
      // TODO: handle situation when arg has deduced parameters.
      CARBON_RETURN_IF_ERROR(Deduce(&param_fn.parameters(),
                                    &arg_fn.parameters(),
                                    /*allow_implicit_conversion=*/false));
      CARBON_RETURN_IF_ERROR(Deduce(&param_fn.return_type(),
                                    &arg_fn.return_type(),
                                    /*allow_implicit_conversion=*/false));
      return Success();
    }
    case Value::Kind::PointerType: {
      if (arg->kind() != Value::Kind::PointerType) {
        return handle_non_deduced_type();
      }
      const auto& param_pointee = cast<PointerType>(param)->pointee_type();
      const auto& arg_pointee = cast<PointerType>(arg)->pointee_type();
      if (allow_implicit_conversion) {
        // TODO: Change based on whether we want to allow
        // deduce-from-base-class, for parametrized base class. See
        // https://github.com/carbon-language/carbon-lang/issues/2464.
        if (const auto* arg_class = dyn_cast<NominalClassType>(&arg_pointee);
            arg_class && arg_class->InheritsClass(&param_pointee)) {
          return Success();
        }
      }
      return Deduce(&param_pointee, &arg_pointee,
                    /*allow_implicit_conversion=*/false);
    }
    // Nothing to do in the case for `auto`.
    case Value::Kind::AutoType: {
      return Success();
    }
    case Value::Kind::NominalClassType: {
      const auto& param_class_type = cast<NominalClassType>(*param);
      if (arg->kind() != Value::Kind::NominalClassType) {
        // TODO: We could determine the parameters of the class from field
        // types in a struct argument.
        return handle_non_deduced_type();
      }
      const auto& arg_class_type = cast<NominalClassType>(*arg);
      if (param_class_type.declaration().name() !=
          arg_class_type.declaration().name()) {
        return handle_non_deduced_type();
      }
      for (const auto& [ty, param_ty] : param_class_type.type_args()) {
        CARBON_RETURN_IF_ERROR(Deduce(param_ty,
                                      arg_class_type.type_args().at(ty),
                                      /*allow_implicit_conversion=*/false));
      }
      return Success();
    }
    case Value::Kind::InterfaceType: {
      const auto& param_iface_type = cast<InterfaceType>(*param);
      if (arg->kind() != Value::Kind::InterfaceType) {
        return handle_non_deduced_type();
      }
      const auto& arg_iface_type = cast<InterfaceType>(*arg);
      if (param_iface_type.declaration().name() !=
          arg_iface_type.declaration().name()) {
        return handle_non_deduced_type();
      }
      for (const auto& [ty, param_ty] : param_iface_type.args()) {
        CARBON_RETURN_IF_ERROR(Deduce(param_ty, arg_iface_type.args().at(ty),
                                      /*allow_implicit_conversion=*/false));
      }
      return Success();
    }
    case Value::Kind::NamedConstraintType: {
      const auto& param_constraint_type = cast<NamedConstraintType>(*param);
      if (arg->kind() != Value::Kind::NamedConstraintType) {
        return handle_non_deduced_type();
      }
      const auto& arg_constraint_type = cast<NamedConstraintType>(*arg);
      if (param_constraint_type.declaration().name() !=
          arg_constraint_type.declaration().name()) {
        return handle_non_deduced_type();
      }
      for (const auto& [ty, param_ty] :
           param_constraint_type.bindings().args()) {
        CARBON_RETURN_IF_ERROR(
            Deduce(param_ty, arg_constraint_type.bindings().args().at(ty),
                   /*allow_implicit_conversion=*/false));
      }
      return Success();
    }
    // For the following cases, we check the type matches.
    case Value::Kind::StaticArrayType:
      // TODO: We could deduce the array type from an array or tuple argument.
    case Value::Kind::ContinuationType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ConstraintType:
    case Value::Kind::AssociatedConstant:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName: {
      return handle_non_deduced_type();
    }
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::StructValue:
    case Value::Kind::TupleValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue: {
      // Argument deduction within the parameters of a parameterized class type
      // or interface type can compare values, rather than types.
      // TODO: Deduce within the values where possible.
      // TODO: Consider in-scope value equalities here.
      if (!ValueEqual(param, arg, std::nullopt)) {
        return ProgramError(source_loc_) << "mismatch in non-type values, `"
                                         << *arg << "` != `" << *param << "`";
      }
      return Success();
    }
    case Value::Kind::MixinPseudoType:
    case Value::Kind::TypeOfMixinPseudoType:
      CARBON_CHECK(false) << "Type expression must not contain Mixin types";
  }
}

auto TypeChecker::ArgumentDeduction::Finish(TypeChecker& type_checker,
                                            const ImplScope& impl_scope) const
    -> ErrorOr<Bindings> {
  // Check deduced values and build our resulting `Bindings` set. We do this in
  // declaration order so that any bindings used in the type of a later binding
  // have known values before we check that binding.
  Bindings bindings;
  for (const auto* binding : deduced_bindings_in_order_) {
    llvm::ArrayRef<Nonnull<const Value*>> values =
        deduced_values_.find(binding)->second;
    if (values.empty()) {
      return ProgramError(source_loc_)
             << "could not deduce type argument for type parameter "
             << binding->name() << " in " << context_;
    }

    const Value* binding_type =
        type_checker.Substitute(bindings, &binding->static_type());
    const Value* substituted_type =
        type_checker.Substitute(bindings, binding_type);
    const auto* first_value = values[0];
    for (const auto* value : values) {
      // TODO: It's not clear that conversions are or should be possible here.
      // If they are permitted, we should allow user-defined conversions, and
      // actually perform the conversion.
      if (!IsTypeOfType(substituted_type) &&
          !type_checker.IsImplicitlyConvertible(value, substituted_type,
                                                impl_scope, false)) {
        return ProgramError(source_loc_)
               << "cannot convert deduced value " << *value << " for "
               << binding->name() << " to parameter type " << *substituted_type;
      }

      // All deductions are required to produce the same value. Note that we
      // intentionally don't consider equality constraints here; we need the
      // same symbolic type, otherwise it would be ambiguous which spelling
      // should be used, and we'd need to check all pairs of types for equality
      // because our notion of equality is non-transitive.
      if (!ValueEqual(first_value, value, std::nullopt)) {
        return ProgramError(source_loc_)
               << "deduced multiple different values for " << *binding
               << ":\n  " << *first_value << "\n  " << *value;
      }
    }

    // Find a witness for the binding if needed.
    std::optional<Nonnull<const Witness*>> witness;
    if (binding->impl_binding()) {
      CARBON_ASSIGN_OR_RETURN(
          witness, impl_scope.Resolve(binding_type, first_value, source_loc_,
                                      type_checker, bindings));
    }

    bindings.Add(binding, first_value, witness);
  }

  // Evaluate and add non-deduced values. These are assumed to lexically follow
  // the deduced bindings, so any bindings the type might reference are now
  // known.
  // TODO: This is not the case for `fn F(T:! type, u: (V:! ImplicitAs(T)))`.
  // However, we intend to disallow that.
  for (auto [binding, arg] : non_deduced_values_) {
    // Form the binding's resolved type and convert the argument expression to
    // it.
    const Value* binding_type = &binding->static_type();
    const Value* substituted_type =
        type_checker.Substitute(bindings, binding_type);
    CARBON_ASSIGN_OR_RETURN(
        arg, type_checker.ImplicitlyConvert(context_, impl_scope, arg,
                                            substituted_type));

    // Evaluate the argument to get the value.
    CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> value,
                            InterpExp(arg, type_checker.arena_, trace_stream_));
    if (trace_stream_) {
      **trace_stream_ << "evaluated generic parameter " << *binding << " as "
                      << *value << "\n";
    }

    // Find a witness for the binding if needed.
    std::optional<Nonnull<const Witness*>> witness;
    if (binding->impl_binding()) {
      CARBON_ASSIGN_OR_RETURN(
          witness, impl_scope.Resolve(binding_type, value, source_loc_,
                                      type_checker, bindings));
    }

    bindings.Add(binding, value, witness);
  }

  // Check non-deduced potential mismatches now we can substitute into them.
  for (const auto& mismatch : non_deduced_mismatches_) {
    const Value* subst_param_type =
        type_checker.Substitute(bindings, mismatch.param);
    CARBON_RETURN_IF_ERROR(
        mismatch.allow_implicit_conversion
            ? type_checker.ExpectType(source_loc_, context_, subst_param_type,
                                      mismatch.arg, impl_scope)
            : type_checker.ExpectExactType(source_loc_, context_,
                                           subst_param_type, mismatch.arg,
                                           impl_scope));
  }

  if (trace_stream_) {
    **trace_stream_ << "deduction succeeded with results: {";
    llvm::ListSeparator sep;
    for (const auto& [binding, val] : bindings.args()) {
      **trace_stream_ << sep << *binding << " = " << *val;
    }
    for (const auto& [binding, val] : bindings.witnesses()) {
      **trace_stream_ << sep << *binding << " = " << *val;
    }
    **trace_stream_ << "}\n";
  }

  return std::move(bindings);
}

// Look for a rewrite to use when naming the given interface member in a type
// that has the given list of rewrites.
static auto LookupRewrite(llvm::ArrayRef<RewriteConstraint> rewrites,
                          Nonnull<const InterfaceType*> interface,
                          Nonnull<const Declaration*> member)
    -> std::optional<const RewriteConstraint*> {
  if (!isa<AssociatedConstantDeclaration>(member)) {
    return std::nullopt;
  }

  for (auto& rewrite : rewrites) {
    if (ValueEqual(interface, &rewrite.constant->interface(), std::nullopt) &&
        member == &rewrite.constant->constant()) {
      // A ConstraintType can only have one rewrite per (interface, member)
      // pair, so we don't need to check the rest.
      return &rewrite;
    }
  }

  return std::nullopt;
}

// Look for a rewrite to use when naming the given interface member in a type
// declared with the given type-of-type.
static auto LookupRewrite(Nonnull<const Value*> type_of_type,
                          Nonnull<const InterfaceType*> interface,
                          Nonnull<const Declaration*> member)
    -> std::optional<const RewriteConstraint*> {
  // Find the set of rewrites. Only ConstraintTypes have rewrites.
  // TODO: If we can ever see an InterfaceType here, we should convert it to a
  // constraint type.
  llvm::ArrayRef<RewriteConstraint> rewrites;
  if (const auto* constraint_type = dyn_cast<ConstraintType>(type_of_type)) {
    rewrites = constraint_type->rewrite_constraints();
  }

  return LookupRewrite(rewrites, interface, member);
}

// Builder for constraint types.
//
// This type supports incrementally building a constraint type by adding
// constraints one at a time, and will deduplicate the constraints as it goes.
//
// TODO: The deduplication here is very inefficient. We should use value
// canonicalization or hashing or similar to speed this up.
class TypeChecker::ConstraintTypeBuilder {
 public:
  // Information about a rewrite constraint that is currently being rewritten.
  struct RewriteInfo {
    Nonnull<const RewriteConstraint*> rewrite;
    // Whether the rewrite has been found to refer to itself. If so, the
    // self-reference will not be expanded. `Resolve` uses this to detect
    // rewrites that cannot be resolved due to cycles.
    bool rewrite_references_itself = false;
  };

  ConstraintTypeBuilder(Nonnull<Arena*> arena, SourceLocation source_loc)
      : ConstraintTypeBuilder(arena, MakeSelfBinding(arena, source_loc)) {}
  ConstraintTypeBuilder(Nonnull<Arena*> arena,
                        Nonnull<GenericBinding*> self_binding)
      : arena_(arena),
        self_binding_(self_binding),
        impl_binding_(AddImplBinding(arena, self_binding_)) {}
  ConstraintTypeBuilder(Nonnull<Arena*> arena,
                        Nonnull<GenericBinding*> self_binding,
                        Nonnull<ImplBinding*> impl_binding)
      : arena_(arena),
        self_binding_(self_binding),
        impl_binding_(impl_binding) {}

  // Returns the self binding for this builder.
  auto self_binding() const -> Nonnull<const GenericBinding*> {
    return self_binding_;
  }

  // Returns the current set of rewrite constraints for this builder.
  auto rewrite_constraints() const -> llvm::ArrayRef<RewriteConstraint> {
    return rewrite_constraints_;
  }

  auto current_rewrite_info() -> std::optional<Nonnull<RewriteInfo*>> {
    return current_rewrite_info_;
  }

  // Produces a type that refers to the `.Self` type of the constraint.
  auto GetSelfType() const -> Nonnull<const Value*> {
    return *self_binding_->symbolic_identity();
  }

  // Gets a witness that `.Self` implements the eventual constraint type built
  // by this builder.
  auto GetSelfWitness() const -> Nonnull<const Witness*> {
    return cast<Witness>(*impl_binding_->symbolic_identity());
  }

  // Adds an `impl` constraint -- `T is C` if not already present.
  // Returns the index of the impl constraint within the self witness.
  auto AddImplConstraint(ImplConstraint impl) -> int {
    for (int i = 0; i != static_cast<int>(impl_constraints_.size()); ++i) {
      ImplConstraint& existing = impl_constraints_[i];
      if (TypeEqual(existing.type, impl.type, std::nullopt) &&
          TypeEqual(existing.interface, impl.interface, std::nullopt)) {
        return i;
      }
    }
    impl_constraints_.push_back(impl);
    return impl_constraints_.size() - 1;
  }

  // Adds an equality constraint -- `A == B`.
  void AddEqualityConstraint(EqualityConstraint equal) {
    if (equal.values.size() < 2) {
      // There's no need to track degenerate equality constraints. These can be
      // formed by rewrites.
      return;
    }

    // TODO: Check to see if this constraint is already present and deduplicate
    // if so. We could also look for a superset / subset and keep the larger
    // one. We could in theory detect `A == B and B == C and C == A` and merge
    // into a single `A == B == C` constraint, but that's more work than it's
    // worth doing here.
    equality_constraints_.push_back(std::move(equal));
  }

  void AddRewriteConstraint(RewriteConstraint rewrite) {
    rewrite_constraints_.push_back(rewrite);
  }

  // Add a context for qualified name lookup, if not already present.
  void AddLookupContext(LookupContext context) {
    for (LookupContext existing : lookup_contexts_) {
      if (ValueEqual(existing.context, context.context, std::nullopt)) {
        return;
      }
    }
    lookup_contexts_.push_back(context);
  }

  // Adds all the constraints from another constraint type. The given value
  // `self` is substituted for `.Self`, typically specified in terms of this
  // constraint's self binding. The `self_witness` is the witness for the
  // resulting constraint, and can be `GetSelfWitness()`. The `bindings`
  // parameter specifies any additional substitutions to perform.
  void AddAndSubstitute(const TypeChecker& type_checker,
                        Nonnull<const ConstraintType*> constraint,
                        Nonnull<const Value*> self,
                        Nonnull<const Witness*> self_witness,
                        const Bindings& bindings, bool add_lookup_contexts) {
    if (type_checker.trace_stream_) {
      **type_checker.trace_stream_
          << "merging " << *constraint << " into constraint with "
          << *constraint->self_binding() << " ~> " << *self << "\n";
    }

    // First substitute into the impl bindings to form the full witness for
    // the constraint type.
    std::vector<Nonnull<const Witness*>> witnesses;
    for (const auto& impl_constraint : constraint->impl_constraints()) {
      Bindings local_bindings = bindings;
      local_bindings.Add(constraint->self_binding(), self,
                         type_checker.MakeConstraintWitness(witnesses));
      int index = AddImplConstraint(
          {.type =
               type_checker.Substitute(local_bindings, impl_constraint.type),
           .interface = cast<InterfaceType>(type_checker.Substitute(
               local_bindings, impl_constraint.interface))});
      witnesses.push_back(
          type_checker.MakeConstraintWitnessAccess(self_witness, index));
    }

    // Now form a complete witness and substitute it into the rest of the
    // constraint.
    Bindings local_bindings = bindings;
    local_bindings.Add(
        constraint->self_binding(), self,
        type_checker.MakeConstraintWitness(std::move(witnesses)));

    // If lookups into the resulting constraint should look into this added
    // constraint, then rewrites for this added constraint become rewrites for
    // the resulting constraint. Otherwise, discard the rewrites and keep only
    // their corresponding equality constraints.
    for (const auto& rewrite_constraint : constraint->rewrite_constraints()) {
      const auto* interface = cast<InterfaceType>(type_checker.Substitute(
          local_bindings, &rewrite_constraint.constant->interface()));
      Nonnull<const Value*> converted_value = type_checker.Substitute(
          local_bindings, rewrite_constraint.converted_replacement);

      // Form a symbolic value naming the non-rewritten associated constant.
      // The impl constraint will always already exist.
      int index = AddImplConstraint({.type = self, .interface = interface});
      const auto* witness =
          type_checker.MakeConstraintWitnessAccess(self_witness, index);
      const auto* constant_value = arena_->New<AssociatedConstant>(
          self, interface, &rewrite_constraint.constant->constant(), witness);

      if (add_lookup_contexts) {
        // Add the constraint `.(I.C) = V`, tracking the value and type prior
        // to conversion for use in rewrites.
        Nonnull<const Value*> value = type_checker.Substitute(
            local_bindings, rewrite_constraint.unconverted_replacement);
        Nonnull<const Value*> type = type_checker.Substitute(
            local_bindings, rewrite_constraint.unconverted_replacement_type);
        AddRewriteConstraint({.constant = constant_value,
                              .unconverted_replacement = value,
                              .unconverted_replacement_type = type,
                              .converted_replacement = converted_value});
      } else {
        // Add the constraint `Self.(I.C) == V`.
        AddEqualityConstraint({.values = {constant_value, converted_value}});
      }
    }

    for (const auto& equality_constraint : constraint->equality_constraints()) {
      std::vector<Nonnull<const Value*>> values;
      for (const Value* value : equality_constraint.values) {
        // Ensure we don't create any duplicates through substitution.
        if (std::find_if(values.begin(), values.end(), [&](const Value* v) {
              return ValueEqual(v, value, std::nullopt);
            }) == values.end()) {
          values.push_back(type_checker.Substitute(local_bindings, value));
        }
      }
      AddEqualityConstraint({.values = std::move(values)});
    }

    if (add_lookup_contexts) {
      for (const auto& lookup_context : constraint->lookup_contexts()) {
        AddLookupContext({.context = type_checker.Substitute(
                              local_bindings, lookup_context.context)});
      }
    }
  }

  class ConstraintsInScopeTracker {
    friend class ConstraintTypeBuilder;

   private:
    int num_impls_added = 0;
    int num_equals_added = 0;
  };

  // Brings all the constraints accumulated so far into the given impl scope,
  // as if we built the constraint type and then added it into the scope. If
  // this will be called more than once, an ImplsInScopeTracker can be provided
  // to avoid adding the same impls more than once.
  void BringConstraintsIntoScope(const TypeChecker& type_checker,
                                 Nonnull<ImplScope*> impl_scope,
                                 Nonnull<ConstraintsInScopeTracker*> tracker) {
    // Figure out which constraints we're going to add.
    int first_impl_to_add =
        std::exchange(tracker->num_impls_added, impl_constraints_.size());
    int first_equal_to_add =
        std::exchange(tracker->num_equals_added, equality_constraints_.size());
    auto new_impl_constraints =
        llvm::ArrayRef<ImplConstraint>(impl_constraints_)
            .drop_front(first_impl_to_add);
    auto new_equality_constraints =
        llvm::ArrayRef<EqualityConstraint>(equality_constraints_)
            .drop_front(first_equal_to_add);

    // Add all of the new constraints.
    impl_scope->Add(new_impl_constraints, std::nullopt, std::nullopt,
                    GetSelfWitness(), type_checker);
    for (auto& equal : new_equality_constraints) {
      impl_scope->AddEqualityConstraint(arena_->New<EqualityConstraint>(equal));
    }
  }

  // Resolve this set of constraints. Form values for rewrite constraints,
  // apply the rewrites within the other constraints, and produce a
  // self-consistent set of constraints or diagnose if that is not possible.
  //
  // This should be done when attaching constraints to a declared name, not
  // when simply forming a constraint type for later use in the type of a
  // declared name.
  auto Resolve(TypeChecker& type_checker, SourceLocation source_loc,
               const ImplScope& /*impl_scope*/) -> ErrorOr<Success> {
    CARBON_RETURN_IF_ERROR(DeduplicateRewrites(source_loc));
    CARBON_RETURN_IF_ERROR(ApplyRewritesToRewrites(type_checker, source_loc));
    ApplyRewritesToConstraints(type_checker);
    return Success();
  }

  // Converts the builder into a ConstraintType. Note that this consumes the
  // builder.
  auto Build() && -> Nonnull<const ConstraintType*> {
    // Create the new type.
    auto* result = arena_->New<ConstraintType>(
        self_binding_, std::move(impl_constraints_),
        std::move(equality_constraints_), std::move(rewrite_constraints_),
        std::move(lookup_contexts_));
    // Update the impl binding to denote the constraint type itself.
    impl_binding_->set_interface(result);
    return result;
  }

  // Sets up a `.Self` binding to act as the self type of a constraint.
  static void PrepareSelfBinding(Nonnull<Arena*> arena,
                                 Nonnull<GenericBinding*> self_binding) {
    Nonnull<const Value*> self = arena->New<VariableType>(self_binding);
    self_binding->set_symbolic_identity(self);
    self_binding->set_value(self);
  }

 private:
  // Makes a generic binding to serve as the `.Self` of a constraint type.
  static auto MakeSelfBinding(Nonnull<Arena*> arena, SourceLocation source_loc)
      -> Nonnull<GenericBinding*> {
    // Note, the type-of-type here is a placeholder and isn't really
    // meaningful.
    auto* result = arena->New<GenericBinding>(
        source_loc, ".Self", arena->New<TypeTypeLiteral>(source_loc));
    PrepareSelfBinding(arena, result);
    return result;
  }

  // Adds an impl binding to the given self binding.
  static auto AddImplBinding(Nonnull<Arena*> arena,
                             Nonnull<GenericBinding*> self_binding)
      -> Nonnull<ImplBinding*> {
    // The `.Self` binding for a constraint should always have an
    // `ImplBinding`. The interface type will be set by `Build`.
    Nonnull<ImplBinding*> impl_binding = arena->New<ImplBinding>(
        self_binding->source_loc(), self_binding, std::nullopt);
    impl_binding->set_symbolic_identity(
        arena->New<BindingWitness>(impl_binding));
    self_binding->set_impl_binding(impl_binding);
    return impl_binding;
  }

  // Check for conflicting rewrites and deduplicate.
  auto DeduplicateRewrites(SourceLocation source_loc) -> ErrorOr<Success> {
    std::vector<RewriteConstraint> new_rewrite_constraints;
    for (auto& rewrite_a : rewrite_constraints_) {
      if (auto existing_rewrite = LookupRewrite(
              new_rewrite_constraints, &rewrite_a.constant->interface(),
              &rewrite_a.constant->constant())) {
        auto& rewrite_b = **existing_rewrite;
        if (ValueEqual(rewrite_a.unconverted_replacement,
                       rewrite_b.unconverted_replacement, std::nullopt) &&
            TypeEqual(rewrite_a.unconverted_replacement_type,
                      rewrite_b.unconverted_replacement_type, std::nullopt)) {
          // This is a duplicate, ignore it.
          continue;
        }
        return ProgramError(source_loc)
               << "multiple different rewrites for `" << *rewrite_a.constant
               << "`:\n"
               << "  " << *rewrite_b.unconverted_replacement << "\n"
               << "  " << *rewrite_a.unconverted_replacement;
      }
      new_rewrite_constraints.push_back(rewrite_a);
    }
    rewrite_constraints_ = std::move(new_rewrite_constraints);
    return Success();
  }

  // Apply rewrites to each other and find a fixed point, or diagnose if there
  // is a cycle.
  auto ApplyRewritesToRewrites(TypeChecker& type_checker,
                               SourceLocation source_loc) -> ErrorOr<Success> {
    // Add this builder to the type checker's scope so that it considers our
    // rewrites.
    type_checker.partial_constraint_types_.push_back(this);
    auto pop_partial_constraint_type = llvm::make_scope_exit(
        [&] { type_checker.partial_constraint_types_.pop_back(); });

    std::deque<Nonnull<RewriteConstraint*>> rewrite_queue;
    for (auto& rewrite : rewrite_constraints_) {
      if (type_checker.trace_stream_) {
        **type_checker.trace_stream_ << "initial rewrite of "
                                     << *rewrite.constant << " is "
                                     << *rewrite.converted_replacement << "\n";
      }
      rewrite_queue.push_back(&rewrite);
    }

    int rewrite_iterations = 0;
    while (!rewrite_queue.empty()) {
      auto* rewrite = rewrite_queue.front();
      rewrite_queue.pop_front();

      // This iteration limit exists only to prevent large fuzzer-generated
      // examples from leading to long compile times. If the limit is hit in a
      // real example, it should be increased.
      if (rewrite_iterations > 1000) {
        return ProgramError(source_loc)
               << "reached iteration limit resolving rewrite constraints";
      }
      ++rewrite_iterations;

      // Rebuild the rewrite and see if it changed. Also track whether it
      // attempted to reference itself recursively.
      RewriteInfo info = {.rewrite = rewrite};
      current_rewrite_info_ = &info;
      Nonnull<const Value*> rebuilt =
          type_checker.RebuildValue(rewrite->converted_replacement);
      current_rewrite_info_ = std::nullopt;

      if (info.rewrite_references_itself) {
        // This would result in an infinite loop.
        return ProgramError(source_loc)
               << "rewrite of " << *rewrite->constant
               << " applies within its own resolved expansion of " << *rebuilt;
      }

      if (!ValueEqual(rebuilt, rewrite->converted_replacement, std::nullopt)) {
        if (type_checker.trace_stream_) {
          **type_checker.trace_stream_ << "rewrote rewrite of "
                                       << *rewrite->constant << " to "
                                       << *rebuilt << "\n";
        }
        rewrite->converted_replacement = rebuilt;
        // Now we've rewritten this rewrite, we might find more rewrites apply
        // to the portion we rewrote.
        rewrite_queue.push_back(rewrite);
      } else {
        if (type_checker.trace_stream_) {
          **type_checker.trace_stream_ << "rewrite of " << *rewrite->constant
                                       << " converged to " << *rebuilt << "\n";
        }
      }
    }

    return Success();
  }

  // Apply the rewrite constraints throughout our constraints.
  auto ApplyRewritesToConstraints(TypeChecker& type_checker) -> void {
    // Add this builder to the type checker's scope so that it considers our
    // rewrites.
    type_checker.partial_constraint_types_.push_back(this);
    auto pop_partial_constraint_type = llvm::make_scope_exit(
        [&] { type_checker.partial_constraint_types_.pop_back(); });

    // Apply rewrites through the rewrite constraints. We assume that the
    // converted replacements have already been rewritten fully.
    for (auto& rewrite : rewrite_constraints_) {
      rewrite.unconverted_replacement =
          type_checker.RebuildValue(rewrite.unconverted_replacement);
      rewrite.unconverted_replacement_type =
          type_checker.RebuildValue(rewrite.unconverted_replacement_type);
    }

    // Apply rewrites throughout impl constraints.
    for (auto& impl_constraint : impl_constraints_) {
      impl_constraint.type = type_checker.RebuildValue(impl_constraint.type);
      impl_constraint.interface = cast<InterfaceType>(
          type_checker.RebuildValue(impl_constraint.interface));
    }

    // Apply rewrites throughout equality constraints.
    for (auto& equality_constraint : equality_constraints_) {
      for (auto*& value : equality_constraint.values) {
        value = type_checker.RebuildValue(value);
      }
    }

    // Apply rewrites throughout lookup contexts.
    for (auto& lookup_context : lookup_contexts_) {
      lookup_context.context =
          type_checker.RebuildValue(lookup_context.context);
    }
  }

  Nonnull<Arena*> arena_;
  Nonnull<GenericBinding*> self_binding_;
  Nonnull<ImplBinding*> impl_binding_;
  std::vector<ImplConstraint> impl_constraints_;
  std::vector<EqualityConstraint> equality_constraints_;
  std::vector<RewriteConstraint> rewrite_constraints_;
  std::vector<LookupContext> lookup_contexts_;
  std::optional<RewriteInfo*> current_rewrite_info_;
};

// A collection of substituted `GenericBinding`s and `ImplBinding`s.
class TypeChecker::SubstitutedGenericBindings {
 public:
  SubstitutedGenericBindings(Nonnull<const TypeChecker*> type_checker,
                             Bindings bindings)
      : type_checker_(type_checker), bindings_(std::move(bindings)) {}

  // Makes a new impl binding for a generic binding if needed, and returns its
  // witness.
  auto MakeImplBinding(Nonnull<GenericBinding*> new_binding,
                       Nonnull<const GenericBinding*> old_binding)
      -> std::optional<Nonnull<const Witness*>> {
    if (!old_binding->impl_binding()) {
      return std::nullopt;
    }
    Nonnull<ImplBinding*> impl_binding =
        type_checker_->arena_->New<ImplBinding>(new_binding->source_loc(),
                                                new_binding,
                                                &new_binding->static_type());
    impl_binding->set_original(old_binding->impl_binding().value());
    auto* witness = type_checker_->arena_->New<BindingWitness>(impl_binding);
    impl_binding->set_symbolic_identity(witness);
    new_binding->set_impl_binding(impl_binding);
    impl_bindings_.push_back(impl_binding);
    return witness;
  }

  // Substitutes into a generic binding and adds it to the bindings map.
  auto SubstituteIntoGenericBinding(Nonnull<const GenericBinding*> old_binding)
      -> Nonnull<GenericBinding*> {
    Nonnull<const Value*> new_type =
        type_checker_->Substitute(bindings_, &old_binding->static_type());
    Nonnull<GenericBinding*> new_binding =
        type_checker_->arena_->New<GenericBinding>(
            old_binding->source_loc(), old_binding->name(),
            const_cast<Expression*>(&old_binding->type()));
    new_binding->set_original(old_binding->original());
    new_binding->set_static_type(new_type);
    bindings_.Add(old_binding,
                  type_checker_->arena_->New<VariableType>(new_binding),
                  MakeImplBinding(new_binding, old_binding));
    return new_binding;
  }

  // Gets the current set of bindings, including any remappings for substituted
  // generic bindings and impl bindings.
  auto bindings() const -> const Bindings& { return bindings_; }

  // Returns ownership of the collection of created `ImplBinding`s.
  auto TakeImplBindings() && -> std::vector<Nonnull<const ImplBinding*>> {
    return std::move(impl_bindings_);
  }

 private:
  Nonnull<const TypeChecker*> type_checker_;
  Bindings bindings_;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings_;
};

auto TypeChecker::Substitute(const Bindings& bindings,
                             Nonnull<const Value*> type) const
    -> Nonnull<const Value*> {
  // Don't waste time recursively rebuilding a type if we have nothing to
  // substitute.
  if (bindings.empty()) {
    return type;
  }

  auto* result = SubstituteImpl(bindings, type);

  if (trace_stream_) {
    **trace_stream_ << "substitution of {";
    llvm::ListSeparator sep;
    for (const auto& [name, value] : bindings.args()) {
      **trace_stream_ << sep << *name << " -> " << *value;
    }
    for (const auto& [name, value] : bindings.witnesses()) {
      **trace_stream_ << sep << *name << " -> " << *value;
    }
    **trace_stream_ << "}\n  old: " << *type << "\n  new: " << *result << "\n";
  }
  return result;
}

auto TypeChecker::RebuildValue(Nonnull<const Value*> value) const
    -> Nonnull<const Value*> {
  return SubstituteImpl(Bindings(), value);
}

class TypeChecker::SubstituteTransform
    : public ValueTransform<SubstituteTransform> {
 public:
  SubstituteTransform(Nonnull<const TypeChecker*> type_checker,
                      const Bindings& bindings)
      : ValueTransform(type_checker->arena_),
        type_checker_(type_checker),
        bindings_(bindings) {}

  using ValueTransform::operator();

  // Replace a `VariableType` with its binding value if available.
  auto operator()(Nonnull<const VariableType*> var_type)
      -> Nonnull<const Value*> {
    auto it = bindings_.args().find(&var_type->binding());
    if (it == bindings_.args().end()) {
      if (auto& trace_stream = type_checker_->trace_stream_) {
        **trace_stream << "substitution: no value for binding " << *var_type
                       << ", leaving alone\n";
      }
      return var_type;
    } else {
      return it->second;
    }
  }

  // Replace a `BindingWitness` with its binding value if available.
  auto operator()(Nonnull<const BindingWitness*> witness)
      -> Nonnull<const Value*> {
    auto it = bindings_.witnesses().find(witness->binding());
    if (it == bindings_.witnesses().end()) {
      if (auto& trace_stream = type_checker_->trace_stream_) {
        **trace_stream << "substitution: no value for binding " << *witness
                       << ", leaving alone\n";
      }
      return witness;
    } else {
      return it->second;
    }
  }

  // For an associated constant, look for a rewrite.
  auto operator()(Nonnull<const AssociatedConstant*> assoc)
      -> Nonnull<const Value*> {
    Nonnull<const Value*> base = Transform(&assoc->base());
    Nonnull<const InterfaceType*> interface = Transform(&assoc->interface());
    // If we're substituting into an associated constant, we may now be able
    // to rewrite it to a concrete value.
    if (auto rewritten_value = type_checker_->LookupRewriteInTypeOf(
            base, interface, &assoc->constant())) {
      return (*rewritten_value)->converted_replacement;
    }
    const auto* witness = cast<Witness>(Transform(&assoc->witness()));
    witness = type_checker_->RefineWitness(witness, base, interface);
    if (auto rewritten_value = type_checker_->LookupRewriteInWitness(
            witness, interface, &assoc->constant())) {
      return (*rewritten_value)->converted_replacement;
    }
    return type_checker_->arena_->New<AssociatedConstant>(
        base, interface, &assoc->constant(), witness);
  }

  // Rebuilding a function type needs special handling to build new bindings.
  // TODO: This is probably not specific to substitution, and would apply to
  // other transforms too.
  auto operator()(Nonnull<const FunctionType*> fn_type)
      -> Nonnull<const FunctionType*> {
    SubstitutedGenericBindings subst_bindings(type_checker_, bindings_);

    // Apply substitution to into generic parameters and deduced bindings.
    std::vector<FunctionType::GenericParameter> generic_parameters;
    for (const FunctionType::GenericParameter& gp :
         fn_type->generic_parameters()) {
      generic_parameters.push_back(
          {.index = gp.index,
           .binding = subst_bindings.SubstituteIntoGenericBinding(gp.binding)});
    }
    std::vector<Nonnull<const GenericBinding*>> deduced_bindings;
    for (Nonnull<const GenericBinding*> gb : fn_type->deduced_bindings()) {
      deduced_bindings.push_back(
          subst_bindings.SubstituteIntoGenericBinding(gb));
    }

    // Apply substitution to parameter and return types and create the new
    // function type.
    const auto* param = type_checker_->SubstituteImpl(subst_bindings.bindings(),
                                                      &fn_type->parameters());
    const auto* ret = type_checker_->SubstituteImpl(subst_bindings.bindings(),
                                                    &fn_type->return_type());
    return type_checker_->arena_->New<FunctionType>(
        param, std::move(generic_parameters), ret, std::move(deduced_bindings),
        std::move(subst_bindings).TakeImplBindings());
  }

  // Substituting into a `ConstraintType` needs special handling if we replace
  // its self type.
  auto operator()(Nonnull<const ConstraintType*> constraint)
      -> Nonnull<const Value*> {
    if (auto it = bindings_.args().find(constraint->self_binding());
        it != bindings_.args().end()) {
      // This happens when we substitute into the parameter type of a
      // function that takes a `T:! Constraint` parameter. In this case we
      // produce the new type-of-type of the replacement type.
      Nonnull<const Value*> type_of_type;
      if (const auto* var_type = dyn_cast<VariableType>(it->second)) {
        type_of_type = &var_type->binding().static_type();
      } else if (const auto* assoc_type =
                     dyn_cast<AssociatedConstant>(it->second)) {
        type_of_type = type_checker_->GetTypeForAssociatedConstant(assoc_type);
      } else {
        type_of_type = type_checker_->arena_->New<TypeType>();
      }
      if (auto& trace_stream = type_checker_->trace_stream_) {
        **trace_stream << "substitution: self of constraint " << *constraint
                       << " is substituted, new type of type is "
                       << *type_of_type << "\n";
      }
      // TODO: Should we keep any part of the old constraint -- rewrites,
      // equality constraints, etc?
      return type_of_type;
    }
    ConstraintTypeBuilder builder(type_checker_->arena_,
                                  constraint->self_binding()->source_loc());
    builder.AddAndSubstitute(*type_checker_, constraint, builder.GetSelfType(),
                             builder.GetSelfWitness(), bindings_,
                             /*add_lookup_contexts=*/true);
    Nonnull<const ConstraintType*> new_constraint = std::move(builder).Build();
    if (auto& trace_stream = type_checker_->trace_stream_) {
      **trace_stream << "substitution: " << *constraint << " => "
                     << *new_constraint << "\n";
    }
    return new_constraint;
  }

 private:
  Nonnull<const TypeChecker*> type_checker_;
  const Bindings& bindings_;
};

auto TypeChecker::SubstituteImpl(const Bindings& bindings,
                                 Nonnull<const Value*> type) const
    -> Nonnull<const Value*> {
  return SubstituteTransform(this, bindings).Transform(type);
}

auto TypeChecker::RefineWitness(Nonnull<const Witness*> witness,
                                Nonnull<const Value*> type,
                                Nonnull<const Value*> constraint) const
    -> Nonnull<const Witness*> {
  if (!top_level_impl_scope_) {
    return witness;
  }

  // See if this is already resolved as some number of layers of
  // ConstraintImplWitness applied to an ImplWitness.
  Nonnull<const Witness*> inner_witness = witness;
  while (auto* inner_constraint_impl_witness =
             dyn_cast<ConstraintImplWitness>(inner_witness)) {
    inner_witness = inner_constraint_impl_witness->constraint_witness();
  }
  if (isa<ImplWitness>(inner_witness)) {
    return witness;
  }

  // Attempt to look for an impl witness in the top-level impl scope.
  if (auto refined_witness =
          (*top_level_impl_scope_)
              ->Resolve(constraint, type, SourceLocation::DiagnosticsIgnored(),
                        *this);
      refined_witness.ok()) {
    return *refined_witness;
  } else {
    if (trace_stream_) {
      **trace_stream_ << "could not refine " << *witness << ": "
                      << refined_witness.error().message() << "\n";
    }
    return witness;
  }
}

auto TypeChecker::MatchImpl(const InterfaceType& iface,
                            Nonnull<const Value*> impl_type,
                            const ImplScope::Impl& impl,
                            const ImplScope& impl_scope,
                            SourceLocation source_loc) const
    -> std::optional<Nonnull<const Witness*>> {
  // Avoid cluttering the trace output with matches that could obviously never
  // have worked.
  // TODO: Eventually, ImplScope should filter by type structure before calling
  // into here.
  if (impl.interface->declaration().name() != iface.declaration().name()) {
    return std::nullopt;
  }

  if (trace_stream_) {
    **trace_stream_ << "MatchImpl: looking for " << *impl_type << " as "
                    << iface << "\n";
    **trace_stream_ << "checking " << *impl.type << " as "
                    << *impl.interface << "\n";
  }

  ArgumentDeduction deduction(source_loc, "match", impl.deduced, trace_stream_);
  if (ErrorOr<Success> e =
          deduction.Deduce(impl.type, impl_type,
                           /*allow_implicit_conversion=*/false);
      !e.ok()) {
    if (trace_stream_) {
      **trace_stream_ << "type does not match: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (ErrorOr<Success> e = deduction.Deduce(
          impl.interface, &iface, /*allow_implicit_conversion=*/false);
      !e.ok()) {
    if (trace_stream_) {
      **trace_stream_ << "interface does not match: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (ErrorOr<Bindings> bindings_or_error =
          deduction.Finish(const_cast<TypeChecker&>(*this), impl_scope);
      !bindings_or_error.ok()) {
    if (trace_stream_) {
      **trace_stream_ << "impl does not match: " << bindings_or_error.error()
                      << "\n";
    }
    return std::nullopt;
  } else {
    if (trace_stream_) {
      **trace_stream_ << "matched with " << *impl.type << " as "
                      << *impl.interface << "\n\n";
    }
    return cast<Witness>(Substitute(*bindings_or_error, impl.witness));
  }
}

auto TypeChecker::MakeConstraintWitness(
    std::vector<Nonnull<const Witness*>> impl_constraint_witnesses) const
    -> Nonnull<const Witness*> {
  return arena_->New<ConstraintWitness>(std::move(impl_constraint_witnesses));
}

auto TypeChecker::MakeConstraintWitnessAccess(Nonnull<const Witness*> witness,
                                              int impl_offset) const
    -> Nonnull<const Witness*> {
  return ConstraintImplWitness::Make(arena_, witness, impl_offset);
}

auto TypeChecker::ConvertToConstraintType(
    SourceLocation source_loc, std::string_view context,
    Nonnull<const Value*> constraint) const
    -> ErrorOr<Nonnull<const ConstraintType*>> {
  if (const auto* constraint_type = dyn_cast<ConstraintType>(constraint)) {
    return constraint_type;
  }
  if (const auto* iface_type = dyn_cast<InterfaceType>(constraint)) {
    CARBON_RETURN_IF_ERROR(
        ExpectCompleteType(source_loc, "constraint", iface_type));
    return cast<ConstraintType>(Substitute(
        iface_type->bindings(), *iface_type->declaration().constraint_type()));
  }
  if (const auto* constraint_type = dyn_cast<NamedConstraintType>(constraint)) {
    CARBON_RETURN_IF_ERROR(
        ExpectCompleteType(source_loc, "constraint", constraint_type));
    return cast<ConstraintType>(
        Substitute(constraint_type->bindings(),
                   *constraint_type->declaration().constraint_type()));
  }
  if (isa<TypeType>(constraint)) {
    // TODO: Should we build this once and cache it?
    ConstraintTypeBuilder builder(arena_, source_loc);
    return std::move(builder).Build();
  }

  return ProgramError(source_loc)
         << "expected a constraint in " << context << ", found " << *constraint;
}

auto TypeChecker::CombineConstraints(
    SourceLocation source_loc,
    llvm::ArrayRef<Nonnull<const ConstraintType*>> constraints)
    -> ErrorOr<Nonnull<const ConstraintType*>> {
  ConstraintTypeBuilder builder(arena_, source_loc);
  for (Nonnull<const ConstraintType*> constraint : constraints) {
    builder.AddAndSubstitute(*this, constraint, builder.GetSelfType(),
                             builder.GetSelfWitness(), Bindings(),
                             /*add_lookup_contexts=*/true);
  }
  return std::move(builder).Build();
}

auto TypeChecker::DeduceCallBindings(
    CallExpression& call, Nonnull<const Value*> params_type,
    llvm::ArrayRef<FunctionType::GenericParameter> generic_params,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
  llvm::ArrayRef<Nonnull<const Value*>> params =
      cast<TupleType>(*params_type).elements();
  llvm::ArrayRef<Nonnull<Expression*>> args =
      cast<TupleLiteral>(call.argument()).fields();
  if (params.size() != args.size()) {
    return ProgramError(call.source_loc())
           << "wrong number of arguments in function call, expected "
           << params.size() << " but got " << args.size();
  }

  // Deductions performed for deduced parameters and generic parameters.
  ArgumentDeduction deduction(call.source_loc(), "call", deduced_bindings,
                              trace_stream_);

  // Deduce and/or convert each argument to the corresponding
  // parameter.
  for (size_t i = 0; i < params.size(); ++i) {
    const Value* param = params[i];
    Expression* arg = args[i];
    if (!generic_params.empty() && generic_params.front().index == i) {
      // The parameter is a `:!` binding. Collect its argument so we can
      // evaluate it when we're done with deduction.
      deduction.AddNonDeducedBindingValue(generic_params.front().binding, arg);
      generic_params = generic_params.drop_front();
    } else {
      // Otherwise deduce its type from the corresponding argument.
      CARBON_RETURN_IF_ERROR(
          deduction.Deduce(param, &arg->static_type(),
                           /*allow_implicit_conversion=*/true));
    }
  }
  CARBON_CHECK(generic_params.empty())
      << "did not find all generic parameters in parameter list";

  CARBON_ASSIGN_OR_RETURN(Bindings bindings,
                          deduction.Finish(*this, impl_scope));
  call.set_bindings(std::move(bindings));

  // Convert the arguments to the deduced and substituted parameter type.
  Nonnull<const Value*> param_type = Substitute(call.bindings(), params_type);
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<Expression*> converted_argument,
      ImplicitlyConvert("call", impl_scope, &call.argument(), param_type));
  call.set_argument(converted_argument);

  return Success();
}

auto TypeChecker::LookupInConstraint(SourceLocation source_loc,
                                     std::string_view lookup_kind,
                                     Nonnull<const Value*> type,
                                     std::string_view member_name)
    -> ErrorOr<ConstraintLookupResult> {
  // Find the set of lookup contexts.
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<const ConstraintType*> constraint_type,
      ConvertToConstraintType(source_loc, lookup_kind, type));
  llvm::ArrayRef<LookupContext> lookup_contexts =
      constraint_type->lookup_contexts();

  std::optional<ConstraintLookupResult> found;
  for (LookupContext lookup : lookup_contexts) {
    if (!isa<InterfaceType>(lookup.context)) {
      // TODO: Support other kinds of lookup context, notably named
      // constraints.
      continue;
    }
    const auto& iface_type = cast<InterfaceType>(*lookup.context);
    if (std::optional<Nonnull<const Declaration*>> member =
            FindMember(member_name, iface_type.declaration().members());
        member.has_value()) {
      if (found.has_value()) {
        if (ValueEqual(found->interface, &iface_type, std::nullopt)) {
          continue;
        }
        // TODO: If we resolve to the same member either way, this
        // is not ambiguous.
        return ProgramError(source_loc)
               << "ambiguous " << lookup_kind << ", " << member_name
               << " found in " << *found->interface << " and " << iface_type;
      }
      found = {.interface = &iface_type, .member = member.value()};
    }
  }

  if (!found) {
    if (isa<TypeType>(type)) {
      return ProgramError(source_loc)
             << lookup_kind << " in unconstrained type";
    }
    return ProgramError(source_loc)
           << lookup_kind << ", " << member_name << " not in " << *type;
  }

  return found.value();
}

auto TypeChecker::GetTypeForAssociatedConstant(
    Nonnull<const AssociatedConstant*> assoc) const -> Nonnull<const Value*> {
  const auto* assoc_type = &assoc->constant().static_type();
  Bindings bindings = assoc->interface().bindings();
  bindings.Add(assoc->interface().declaration().self(), &assoc->base(),
               &assoc->witness());
  return Substitute(bindings, assoc_type);
}

auto TypeChecker::LookupRewriteInTypeOf(
    Nonnull<const Value*> type, Nonnull<const InterfaceType*> interface,
    Nonnull<const Declaration*> member) const
    -> std::optional<const RewriteConstraint*> {
  // If the type is the self type of an incomplete `where` expression or a
  // constraint type we're in the process of resolving, find its set of
  // rewrites. These rewrites may not be complete -- earlier rewrites will have
  // been applied to later ones, but not vice versa -- but those are the
  // intended semantics in this case.
  for (auto* builder : partial_constraint_types_) {
    if (ValueEqual(type, builder->GetSelfType(), std::nullopt)) {
      if (auto result = LookupRewrite(builder->rewrite_constraints(), interface,
                                      member)) {
        // If we're in the middle of rewriting this rewrite, let the constraint
        // type builder know it applies within itself, and don't expand it
        // within itself.
        if (auto current_rewrite_info = builder->current_rewrite_info();
            current_rewrite_info &&
            (*current_rewrite_info)->rewrite == *result) {
          (*current_rewrite_info)->rewrite_references_itself = true;
          return std::nullopt;
        }
        return result;
      }
    }
  }

  // Given `(T:! C).Y`, look in `C` for rewrites.
  if (const auto* var_type = dyn_cast<VariableType>(type)) {
    if (!var_type->binding().has_static_type()) {
      // We looked for a rewrite before we finished type-checking the generic
      // binding. This happens when forming the type of a generic binding. Just
      // say there are no rewrites yet; any rewrites will be applied when the
      // constraint on the binding's type is resolved.
      return std::nullopt;
    }
    return LookupRewrite(&var_type->binding().static_type(), interface, member);
  }

  // Given `(T.U).Y` for an associated type `U`, substitute into the type of
  // `U` to find rewrites.
  if (const auto* assoc_const = dyn_cast<AssociatedConstant>(type)) {
    if (!assoc_const->constant().has_static_type()) {
      // We looked for a rewrite before we finished type-checking the
      // associated constant. This happens when forming the type of the
      // associated constant, if `.Self` is used to access an associated
      // constant. Just say that there are not rewrites yet; any rewrites will
      // be applied when the constraint on the binding's type is resolved.
      return std::nullopt;
    }
    // The following is an expanded version of
    //  return LookupRewrite(GetTypeForAssociatedConstant(assoc_const),
    //                       interface, member);
    // where we substitute as little as possible to try to avoid infinite
    // recursion.
    if (auto* constraint =
            dyn_cast<ConstraintType>(&assoc_const->constant().static_type())) {
      for (auto rewrite : constraint->rewrite_constraints()) {
        if (&rewrite.constant->constant() != &assoc_const->constant()) {
          continue;
        }
        Bindings bindings = assoc_const->interface().bindings();
        bindings.Add(assoc_const->interface().declaration().self(),
                     &assoc_const->base(), &assoc_const->witness());
        if (ValueEqual(interface,
                       Substitute(bindings, &rewrite.constant->interface()),
                       std::nullopt)) {
          // TODO: These substitutions can lead to infinite recursion.
          RewriteConstraint substituted = {
              // Not substituted, but our callers don't need it.
              .constant = rewrite.constant,
              .unconverted_replacement =
                  Substitute(bindings, rewrite.unconverted_replacement),
              .unconverted_replacement_type =
                  Substitute(bindings, rewrite.unconverted_replacement_type),
              .converted_replacement =
                  Substitute(bindings, rewrite.converted_replacement)};
          return arena_->New<RewriteConstraint>(substituted);
        }
      }
    }
  }

  return std::nullopt;
}

auto TypeChecker::LookupRewriteInWitness(
    Nonnull<const Witness*> witness, Nonnull<const InterfaceType*> interface,
    Nonnull<const Declaration*> member) const
    -> std::optional<const RewriteConstraint*> {
  if (const auto* impl_witness = dyn_cast<ImplWitness>(witness)) {
    Nonnull<const Value*> constraint =
        Substitute(impl_witness->bindings(),
                   impl_witness->declaration().constraint_type());
    return LookupRewrite(constraint, interface, member);
  }
  return std::nullopt;
}

// Rewrites a member access expression to produce the given constant value.
static void RewriteMemberAccess(Nonnull<MemberAccessExpression*> access,
                                Nonnull<const RewriteConstraint*> value) {
  access->set_value_category(ValueCategory::Let);
  access->set_static_type(value->unconverted_replacement_type);
  access->set_constant_value(value->unconverted_replacement);
}

// Determine whether the given member declaration declares an instance member.
static auto IsInstanceMember(Nonnull<const Element*> element) {
  switch (element->kind()) {
    case ElementKind::BaseElement:
    case ElementKind::PositionalElement:
      return true;
    case ElementKind::NamedElement:
      const auto nom_element = cast<NamedElement>(element);
      if (!nom_element->declaration()) {
        // This is a struct field.
        return true;
      }
      Nonnull<const Declaration*> declaration = *nom_element->declaration();
      switch (declaration->kind()) {
        case DeclarationKind::FunctionDeclaration:
          return cast<FunctionDeclaration>(declaration)->is_method();
        case DeclarationKind::VariableDeclaration:
          return true;
        default:
          return false;
      }
  }
}

auto TypeChecker::CheckAddrMeAccess(
    Nonnull<MemberAccessExpression*> access,
    Nonnull<const FunctionDeclaration*> func_decl, const Bindings& bindings,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
  if (func_decl->is_method() &&
      func_decl->self_pattern().kind() == PatternKind::AddrPattern) {
    access->set_is_addr_me_method();
    Nonnull<const Value*> me_type =
        Substitute(bindings, &func_decl->self_pattern().static_type());
    CARBON_RETURN_IF_ERROR(
        ExpectExactType(access->source_loc(), "method access, receiver type",
                        me_type, &access->object().static_type(), impl_scope));
    if (access->object().value_category() != ValueCategory::Var) {
      return ProgramError(access->source_loc())
             << "method " << *access
             << " requires its receiver to be an lvalue";
    }
  }
  return Success();
}

auto TypeChecker::TypeCheckExp(Nonnull<Expression*> e,
                               const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << ExpressionKindName(e->kind()) << " "
                    << *e;
    **trace_stream_ << "\n";
  }
  if (e->is_type_checked()) {
    if (trace_stream_) {
      **trace_stream_ << "expression has already been type-checked\n";
    }
    return Success();
  }
  switch (e->kind()) {
    case ExpressionKind::ValueLiteral:
    case ExpressionKind::BuiltinConvertExpression:
    case ExpressionKind::BaseAccessExpression:
      CARBON_FATAL() << "attempting to type check node " << *e
                     << " generated during type checking";
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&index.object(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&index.offset(), impl_scope));
      const Value& object_type = index.object().static_type();
      switch (object_type.kind()) {
        case Value::Kind::TupleType: {
          const auto& tuple_type = cast<TupleType>(object_type);
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(index.offset().source_loc(), "tuple index",
                              arena_->New<IntType>(),
                              &index.offset().static_type(), impl_scope));
          CARBON_ASSIGN_OR_RETURN(
              auto offset_value,
              InterpExp(&index.offset(), arena_, trace_stream_));
          int i = cast<IntValue>(*offset_value).value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            return ProgramError(e->source_loc())
                   << "index " << i << " is out of range for type "
                   << tuple_type;
          }
          index.set_static_type(tuple_type.elements()[i]);
          index.set_value_category(index.object().value_category());
          return Success();
        }
        case Value::Kind::StaticArrayType: {
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(index.offset().source_loc(), "array index",
                              arena_->New<IntType>(),
                              &index.offset().static_type(), impl_scope));
          index.set_static_type(
              &cast<StaticArrayType>(object_type).element_type());
          index.set_value_category(index.object().value_category());
          return Success();
        }
        default:
          return ProgramError(e->source_loc())
                 << "only arrays and tuples can be indexed, found "
                 << object_type;
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      for (auto* arg : cast<TupleLiteral>(*e).fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(arg, impl_scope));
        CARBON_RETURN_IF_ERROR(
            ExpectNonPlaceholderType(arg->source_loc(), &arg->static_type()));
        arg_types.push_back(&arg->static_type());
      }
      e->set_static_type(arena_->New<TupleType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        CARBON_RETURN_IF_ERROR(ExpectNonPlaceholderType(
            arg.expression().source_loc(), &arg.expression().static_type()));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      e->set_static_type(arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      std::vector<NamedValue> fields;
      for (auto& arg : struct_type.fields()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> type,
            TypeCheckTypeExp(&arg.expression(), impl_scope));
        fields.push_back({arg.name(), type});
      }
      struct_type.set_static_type(arena_->New<TypeType>());
      struct_type.set_value_category(ValueCategory::Let);
      struct_type.set_constant_value(
          arena_->New<StructType>(std::move(fields)));
      return Success();
    }
    case ExpressionKind::SimpleMemberAccessExpression: {
      auto& access = cast<SimpleMemberAccessExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.object(), impl_scope));
      const Value& object_type = access.object().static_type();
      CARBON_RETURN_IF_ERROR(ExpectCompleteType(access.source_loc(),
                                                "member access", &object_type));
      switch (object_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(object_type);
          for (const auto& field : struct_type.fields()) {
            if (access.member_name() == field.name) {
              access.set_member(arena_->New<NamedElement>(&field));
              access.set_static_type(field.value);
              access.set_value_category(access.object().value_category());
              return Success();
            }
          }
          return ProgramError(access.source_loc())
                 << "struct " << struct_type << " does not have a field named "
                 << access.member_name();
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(object_type);
          CARBON_ASSIGN_OR_RETURN(
              const auto res,
              FindMemberWithParents(access.member_name(), &t_class));
          if (res.has_value()) {
            auto [member_type, member, member_t_class] = res.value();
            Nonnull<const Value*> field_type =
                Substitute(member_t_class->bindings(), member_type);
            access.set_member(arena_->New<NamedElement>(member));
            access.set_static_type(field_type);
            access.set_is_type_access(!IsInstanceMember(&access.member()));
            switch (member->kind()) {
              case DeclarationKind::VariableDeclaration:
                access.set_value_category(access.object().value_category());
                break;
              case DeclarationKind::FunctionDeclaration: {
                const auto* func_decl = cast<FunctionDeclaration>(member);
                CARBON_RETURN_IF_ERROR(CheckAddrMeAccess(
                    &access, func_decl, t_class.bindings(), impl_scope));
                access.set_value_category(ValueCategory::Let);
                break;
              }
              default:
                CARBON_FATAL() << "member " << access.member_name()
                               << " is not a field or method";
                break;
            }
            return Success();
          } else {
            return ProgramError(e->source_loc())
                   << "class " << t_class.declaration().name()
                   << " does not have a field named " << access.member_name();
          }
        }
        case Value::Kind::VariableType:
        case Value::Kind::AssociatedConstant: {
          // This case handles access to a method on a receiver whose type is a
          // type variable or associated constant. For example, `x.foo` where
          // the type of `x` is `T` and `T` implements an interface that
          // includes `foo`, or `x.y().foo` where the type of `x` is `T` and
          // the return type of `y()` is an associated constant from `T`'s
          // constraint.
          Nonnull<const Value*> constraint;
          if (const auto* var_type = dyn_cast<VariableType>(&object_type)) {
            constraint = &var_type->binding().static_type();
          } else {
            constraint = GetTypeForAssociatedConstant(
                cast<AssociatedConstant>(&object_type));
          }
          CARBON_ASSIGN_OR_RETURN(
              ConstraintLookupResult result,
              LookupInConstraint(e->source_loc(), "member access", constraint,
                                 access.member_name()));
          if (auto replacement =
                  LookupRewrite(constraint, result.interface, result.member)) {
            RewriteMemberAccess(&access, *replacement);
            return Success();
          }
          // Compute a witness that the variable type implements this
          // interface. This will typically be either a reference to its
          // `ImplBinding` or, for a constraint, to a witness for an impl
          // constraint within it.
          // TODO: We should only need to look at the impl binding for this
          // variable or witness for this associated constant, not everything in
          // the impl scope, to find the witness.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const ConstraintType*> iface_constraint,
              ConvertToConstraintType(access.source_loc(), "member access",
                                      result.interface));
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Witness*> witness,
              impl_scope.Resolve(iface_constraint, &object_type,
                                 e->source_loc(), *this));

          Bindings bindings = result.interface->bindings();
          bindings.Add(result.interface->declaration().self(), &object_type,
                       witness);

          const Value& member_type = result.member->static_type();
          Nonnull<const Value*> inst_member_type =
              Substitute(bindings, &member_type);
          access.set_member(arena_->New<NamedElement>(result.member));
          access.set_found_in_interface(result.interface);
          access.set_is_type_access(!IsInstanceMember(&access.member()));
          access.set_static_type(inst_member_type);

          if (auto* func_decl = dyn_cast<FunctionDeclaration>(result.member)) {
            CARBON_RETURN_IF_ERROR(
                CheckAddrMeAccess(&access, func_decl, bindings, impl_scope));
          }

          // TODO: This is just a ConstraintImplWitness into the
          // iface_constraint. If we can compute the right index, we can avoid
          // re-resolving it.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Witness*> impl,
              impl_scope.Resolve(result.interface, &object_type,
                                 e->source_loc(), *this));
          access.set_impl(impl);
          return Success();
        }
        case Value::Kind::InterfaceType:
        case Value::Kind::NamedConstraintType:
        case Value::Kind::ConstraintType: {
          // This case handles access to a class function from a constrained
          // type variable. If `T` is a type variable and `foo` is a class
          // function in an interface implemented by `T`, then `T.foo` accesses
          // the `foo` class function of `T`.
          //
          // TODO: Per the language rules, we are supposed to also perform
          // lookup into `type` and report an ambiguity if the name is found in
          // both places.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              InterpExp(&access.object(), arena_, trace_stream_));
          CARBON_ASSIGN_OR_RETURN(
              ConstraintLookupResult result,
              LookupInConstraint(e->source_loc(), "member access", &object_type,
                                 access.member_name()));
          if (auto replacement = LookupRewrite(&object_type, result.interface,
                                               result.member)) {
            RewriteMemberAccess(&access, *replacement);
            return Success();
          }
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const ConstraintType*> iface_constraint,
              ConvertToConstraintType(access.source_loc(), "member access",
                                      result.interface));
          CARBON_ASSIGN_OR_RETURN(Nonnull<const Witness*> witness,
                                  impl_scope.Resolve(iface_constraint, type,
                                                     e->source_loc(), *this));
          CARBON_ASSIGN_OR_RETURN(Nonnull<const Witness*> impl,
                                  impl_scope.Resolve(result.interface, type,
                                                     e->source_loc(), *this));
          access.set_member(arena_->New<NamedElement>(result.member));
          access.set_impl(impl);
          access.set_found_in_interface(result.interface);

          if (IsInstanceMember(&access.member())) {
            // This is a member name denoting an instance member.
            // TODO: Consider setting the static type of all instance member
            // declarations to be member name types, rather than special-casing
            // member accesses that name them.
            access.set_static_type(
                arena_->New<TypeOfMemberName>(NamedElement(result.member)));
            access.set_value_category(ValueCategory::Let);
          } else {
            // This is a non-instance member whose value is found directly via
            // the witness table, such as a non-method function or an
            // associated constant.
            const Value& member_type = result.member->static_type();
            Bindings bindings = result.interface->bindings();
            bindings.Add(result.interface->declaration().self(), type, witness);
            Nonnull<const Value*> inst_member_type =
                Substitute(bindings, &member_type);
            access.set_static_type(inst_member_type);
            access.set_value_category(ValueCategory::Let);
          }
          return Success();
        }
        case Value::Kind::TypeType: {
          // This is member access into an unconstrained type. Evaluate it and
          // perform lookup in the result.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              InterpExp(&access.object(), arena_, trace_stream_));
          CARBON_RETURN_IF_ERROR(
              ExpectCompleteType(access.source_loc(), "member access", type));
          switch (type->kind()) {
            case Value::Kind::StructType: {
              for (const auto& field : cast<StructType>(type)->fields()) {
                if (access.member_name() == field.name) {
                  access.set_member(arena_->New<NamedElement>(&field));
                  access.set_static_type(
                      arena_->New<TypeOfMemberName>(NamedElement(&field)));
                  access.set_value_category(ValueCategory::Let);
                  return Success();
                }
              }
              return ProgramError(access.source_loc())
                     << "struct " << *type << " does not have a field named "
                     << " does not have a field named " << access.member_name();
            }
            case Value::Kind::ChoiceType: {
              const auto& choice = cast<ChoiceType>(*type);
              std::optional<Nonnull<const Value*>> parameter_types =
                  choice.FindAlternative(access.member_name());
              if (!parameter_types.has_value()) {
                return ProgramError(e->source_loc())
                       << "choice " << choice.name()
                       << " does not have an alternative named "
                       << access.member_name();
              }
              Nonnull<const Value*> substituted_parameter_type =
                  *parameter_types;
              if (choice.IsParameterized()) {
                substituted_parameter_type =
                    Substitute(choice.bindings(), *parameter_types);
              }
              Nonnull<const Value*> type = arena_->New<FunctionType>(
                  substituted_parameter_type, &choice);
              // TODO: Should there be a Declaration corresponding to each
              // choice type alternative?
              access.set_member(
                  arena_->New<NamedElement>(arena_->New<NamedValue>(
                      NamedValue{access.member_name(), type})));
              access.set_static_type(type);
              access.set_value_category(ValueCategory::Let);
              return Success();
            }
            case Value::Kind::NominalClassType: {
              const auto& class_type = cast<NominalClassType>(*type);
              CARBON_ASSIGN_OR_RETURN(
                  auto type_member,
                  FindMixedMemberAndType(access.member_name(),
                                         class_type.declaration().members(),
                                         &class_type));
              if (type_member.has_value()) {
                auto [member_type, member] = type_member.value();
                access.set_member(arena_->New<NamedElement>(member));
                switch (member->kind()) {
                  case DeclarationKind::FunctionDeclaration: {
                    const auto& func = cast<FunctionDeclaration>(*member);
                    if (func.is_method()) {
                      break;
                    }
                    Nonnull<const Value*> field_type = Substitute(
                        class_type.bindings(), &member->static_type());
                    access.set_static_type(field_type);
                    access.set_value_category(ValueCategory::Let);
                    return Success();
                  }
                  default:
                    break;
                }
                access.set_static_type(
                    arena_->New<TypeOfMemberName>(NamedElement(member)));
                access.set_value_category(ValueCategory::Let);
                return Success();
              } else {
                return ProgramError(access.source_loc())
                       << class_type << " does not have a member named "
                       << access.member_name();
              }
            }
            case Value::Kind::InterfaceType:
            case Value::Kind::NamedConstraintType:
            case Value::Kind::ConstraintType: {
              CARBON_ASSIGN_OR_RETURN(
                  ConstraintLookupResult result,
                  LookupInConstraint(e->source_loc(), "member access", type,
                                     access.member_name()));
              access.set_member(arena_->New<NamedElement>(result.member));
              access.set_found_in_interface(result.interface);
              access.set_static_type(
                  arena_->New<TypeOfMemberName>(NamedElement(result.member)));
              access.set_value_category(ValueCategory::Let);
              return Success();
            }
            default:
              // TODO: We should handle VariableType and AssociatedConstant
              // here.
              return ProgramError(access.source_loc())
                     << "unsupported member access into type " << *type;
          }
        }
        default:
          return ProgramError(e->source_loc())
                 << "member access, unexpected " << object_type << " in " << *e;
      }
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      auto& access = cast<CompoundMemberAccessExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.object(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.path(), impl_scope));
      if (!isa<TypeOfMemberName>(access.path().static_type())) {
        return ProgramError(e->source_loc())
               << "expected name of instance member or interface member in "
                  "compound member access, found "
               << access.path().static_type();
      }

      // Evaluate the member name expression to determine which member we're
      // accessing.
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> member_name_value,
                              InterpExp(&access.path(), arena_, trace_stream_));
      const auto& member_name = cast<MemberName>(*member_name_value);
      access.set_member(&member_name);
      bool is_instance_member = IsInstanceMember(&member_name.member());

      bool has_instance = true;
      std::optional<Nonnull<const Value*>> base_type = member_name.base_type();
      if (!base_type.has_value()) {
        if (IsTypeOfType(&access.object().static_type())) {
          // This is `Type.(member_name)`, where `member_name` doesn't specify
          // a type. This access doesn't perform instance binding.
          CARBON_ASSIGN_OR_RETURN(
              base_type, InterpExp(&access.object(), arena_, trace_stream_));
          has_instance = false;
        } else {
          // This is `value.(member_name)`, where `member_name` doesn't specify
          // a type. The member will be found in the type of `value`, or in a
          // corresponding `impl` if `member_name` is an interface member.
          base_type = &access.object().static_type();
        }
      } else {
        // This is `value.(member_name)`, where `member_name` specifies a type.
        // `value` is implicitly converted to that type.
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_object,
            ImplicitlyConvert("compound member access", impl_scope,
                              &access.object(), *base_type));
        access.set_object(converted_object);
      }
      access.set_is_type_access(has_instance && !is_instance_member);

      // Perform associated constant rewriting and impl selection if necessary.
      std::optional<Nonnull<const Witness*>> witness;
      if (std::optional<Nonnull<const InterfaceType*>> iface =
              member_name.interface()) {
        // If we're naming an associated constant, we might have a rewrite for
        // it that we can apply immediately.
        if (auto replacement = LookupRewriteInTypeOf(
                *base_type, *iface, *member_name.member().declaration())) {
          RewriteMemberAccess(&access, *replacement);
          return Success();
        }

        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> iface_constraint,
            ConvertToConstraintType(access.source_loc(),
                                    "compound member access", *iface));
        CARBON_ASSIGN_OR_RETURN(witness,
                                impl_scope.Resolve(iface_constraint, *base_type,
                                                   e->source_loc(), *this));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Witness*> impl,
            impl_scope.Resolve(*iface, *base_type, e->source_loc(), *this));
        if (auto replacement = LookupRewriteInWitness(
                impl, *iface, *member_name.member().declaration())) {
          RewriteMemberAccess(&access, *replacement);
          return Success();
        }
        access.set_impl(impl);
      }

      auto bindings_for_member = [&]() -> Bindings {
        if (member_name.interface()) {
          Nonnull<const InterfaceType*> iface_type = *member_name.interface();
          Bindings bindings = iface_type->bindings();
          bindings.Add(iface_type->declaration().self(), *base_type, witness);
          return bindings;
        }
        if (const auto* class_type =
                dyn_cast<NominalClassType>(base_type.value())) {
          return class_type->bindings();
        }
        return Bindings();
      };

      auto substitute_into_member_type = [&]() {
        Nonnull<const Value*> member_type = &member_name.member().type();
        return Substitute(bindings_for_member(), member_type);
      };

      switch (std::optional<Nonnull<const Declaration*>> decl =
                  member_name.member().declaration();
              decl ? decl.value()->kind()
                   : DeclarationKind::VariableDeclaration) {
        case DeclarationKind::VariableDeclaration:
          if (has_instance) {
            access.set_static_type(substitute_into_member_type());
            access.set_value_category(access.object().value_category());
            return Success();
          }
          break;
        case DeclarationKind::FunctionDeclaration: {
          if (has_instance || !is_instance_member) {
            // This should not be possible: the name of a static member
            // function should have function type not member name type.
            CARBON_CHECK(!has_instance || is_instance_member ||
                         !member_name.base_type().has_value())
                << "vacuous compound member access";
            access.set_static_type(substitute_into_member_type());
            access.set_value_category(ValueCategory::Let);
            CARBON_RETURN_IF_ERROR(
                CheckAddrMeAccess(&access, cast<FunctionDeclaration>(*decl),
                                  bindings_for_member(), impl_scope));
            return Success();
          }
          break;
        }
        case DeclarationKind::AssociatedConstantDeclaration:
          access.set_static_type(substitute_into_member_type());
          access.set_value_category(access.object().value_category());
          return Success();
        default:
          CARBON_FATAL() << "member " << member_name
                         << " is not a field or method";
          break;
      }

      access.set_static_type(
          arena_->New<TypeOfMemberName>(member_name.member()));
      access.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      if (ident.value_node().base().kind() ==
          AstNodeKind::FunctionDeclaration) {
        const auto& function =
            cast<FunctionDeclaration>(ident.value_node().base());
        if (!function.has_static_type()) {
          CARBON_CHECK(function.return_term().is_auto());
          return ProgramError(ident.source_loc())
                 << "Function calls itself, but has a deduced return type";
        }
      }
      ident.set_static_type(&ident.value_node().static_type());
      ident.set_value_category(ident.value_node().value_category());
      return Success();
    }
    case ExpressionKind::DotSelfExpression: {
      auto& dot_self = cast<DotSelfExpression>(*e);
      if (dot_self.self_binding().is_type_checked()) {
        dot_self.set_static_type(&dot_self.self_binding().static_type());
      } else {
        dot_self.set_static_type(arena_->New<TypeType>());
        dot_self.self_binding().set_named_as_type_via_dot_self();
      }
      dot_self.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::IntLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<IntType>());
      return Success();
    case ExpressionKind::BoolLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<BoolType>());
      return Success();
    case ExpressionKind::OperatorExpression: {
      auto& op = cast<OperatorExpression>(*e);
      std::vector<Nonnull<const Value*>> ts;
      for (Nonnull<Expression*> argument : op.arguments()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(argument, impl_scope));
        ts.push_back(&argument->static_type());
      }

      auto handle_unary_operator =
          [&](Builtins::Builtin builtin) -> ErrorOr<Success> {
        ErrorOr<Nonnull<Expression*>> result = BuildBuiltinMethodCall(
            impl_scope, op.arguments()[0], BuiltinInterfaceName{builtin},
            BuiltinMethodCall{"Op"});
        if (!result.ok()) {
          // We couldn't find a matching `impl`.
          return ProgramError(e->source_loc())
                 << "type error in `" << ToString(op.op()) << "`:\n"
                 << result.error().message();
        }
        op.set_rewritten_form(*result);
        return Success();
      };

      auto handle_binary_operator =
          [&](Builtins::Builtin builtin) -> ErrorOr<Success> {
        ErrorOr<Nonnull<Expression*>> result = BuildBuiltinMethodCall(
            impl_scope, op.arguments()[0], BuiltinInterfaceName{builtin, ts[1]},
            BuiltinMethodCall{"Op", {op.arguments()[1]}});
        if (!result.ok()) {
          // We couldn't find a matching `impl`.
          return ProgramError(e->source_loc())
                 << "type error in `" << ToString(op.op()) << "`:\n"
                 << result.error().message();
        }
        op.set_rewritten_form(*result);
        return Success();
      };

      auto handle_binary_arithmetic =
          [&](Builtins::Builtin builtin) -> ErrorOr<Success> {
        // Handle a built-in operator first.
        // TODO: Replace this with an intrinsic.
        if (isa<IntType>(ts[0]) && isa<IntType>(ts[1]) &&
            IsSameType(ts[0], ts[1], impl_scope)) {
          op.set_static_type(ts[0]);
          op.set_value_category(ValueCategory::Let);
          return Success();
        }

        // Now try an overloaded operator.
        return handle_binary_operator(builtin);
      };

      auto handle_compare =
          [&](Builtins::Builtin builtin, const std::string& method_name,
              const std::string_view& operator_desc) -> ErrorOr<Success> {
        ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
            impl_scope, op.arguments()[0], BuiltinInterfaceName{builtin, ts[1]},
            BuiltinMethodCall{method_name, op.arguments()[1]});
        if (!converted.ok()) {
          // We couldn't find a matching `impl`.
          return ProgramError(e->source_loc())
                 << *ts[0] << " is not " << operator_desc << " comparable with "
                 << *ts[1] << " (" << converted.error().message() << ")";
        }
        op.set_rewritten_form(*converted);
        return Success();
      };

      switch (op.op()) {
        case Operator::Neg: {
          // Handle a built-in negation first.
          // TODO: Replace this with an intrinsic.
          if (isa<IntType>(ts[0])) {
            op.set_static_type(arena_->New<IntType>());
            op.set_value_category(ValueCategory::Let);
            return Success();
          }
          // Now try an overloaded negation.
          return handle_unary_operator(Builtins::Negate);
        }
        case Operator::Add:
          return handle_binary_arithmetic(Builtins::AddWith);
        case Operator::Sub:
          return handle_binary_arithmetic(Builtins::SubWith);
        case Operator::Mul:
          return handle_binary_arithmetic(Builtins::MulWith);
        case Operator::Div:
          return handle_binary_arithmetic(Builtins::DivWith);
        case Operator::Mod:
          return handle_binary_arithmetic(Builtins::ModWith);
        case Operator::BitwiseAnd:
          // `&` between type-of-types performs constraint combination.
          // TODO: Should this be done via an intrinsic?
          if (IsTypeOfType(ts[0]) && IsTypeOfType(ts[1])) {
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> lhs,
                InterpExp(op.arguments()[0], arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> rhs,
                InterpExp(op.arguments()[1], arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const ConstraintType*> lhs_constraint,
                ConvertToConstraintType(op.arguments()[0]->source_loc(),
                                        "first operand of `&`", lhs));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const ConstraintType*> rhs_constraint,
                ConvertToConstraintType(op.arguments()[1]->source_loc(),
                                        "second operand of `&`", rhs));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const ConstraintType*> result,
                CombineConstraints(e->source_loc(),
                                   {lhs_constraint, rhs_constraint}));
            op.set_rewritten_form(arena_->New<ValueLiteral>(
                op.source_loc(), result, arena_->New<TypeType>(),
                ValueCategory::Let));
            return Success();
          }
          return handle_binary_operator(Builtins::BitAndWith);
        case Operator::BitwiseOr:
          return handle_binary_operator(Builtins::BitOrWith);
        case Operator::BitwiseXor:
          return handle_binary_operator(Builtins::BitXorWith);
        case Operator::BitShiftLeft:
          return handle_binary_operator(Builtins::LeftShiftWith);
        case Operator::BitShiftRight:
          return handle_binary_operator(Builtins::RightShiftWith);
        case Operator::Complement:
          return handle_unary_operator(Builtins::BitComplement);
        case Operator::And:
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(1)",
                                                 arena_->New<BoolType>(), ts[0],
                                                 impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(2)",
                                                 arena_->New<BoolType>(), ts[1],
                                                 impl_scope));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Or:
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(1)",
                                                 arena_->New<BoolType>(), ts[0],
                                                 impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(2)",
                                                 arena_->New<BoolType>(), ts[1],
                                                 impl_scope));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Not:
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "!",
                                                 arena_->New<BoolType>(), ts[0],
                                                 impl_scope));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Eq:
          return handle_compare(Builtins::EqWith, "Equal", "equality");
        case Operator::NotEq:
          return handle_compare(Builtins::EqWith, "NotEqual", "equality");
        case Operator::Less:
          return handle_compare(Builtins::LessWith, "Less", "less");
        case Operator::LessEq:
          return handle_compare(Builtins::LessEqWith, "LessEq", "less equal");
        case Operator::GreaterEq:
          return handle_compare(Builtins::GreaterEqWith, "GreaterEq",
                                "greater equal");
        case Operator::Greater:
          return handle_compare(Builtins::GreaterWith, "Greater", "greater");
        case Operator::Deref:
          CARBON_RETURN_IF_ERROR(
              ExpectPointerType(e->source_loc(), "*", ts[0]));
          op.set_static_type(&cast<PointerType>(*ts[0]).pointee_type());
          op.set_value_category(ValueCategory::Var);
          return Success();
        case Operator::Ptr:
          CARBON_RETURN_IF_ERROR(ExpectType(e->source_loc(), "*",
                                            arena_->New<TypeType>(), ts[0],
                                            impl_scope));
          op.set_static_type(arena_->New<TypeType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::AddressOf:
          if (op.arguments()[0]->value_category() != ValueCategory::Var) {
            return ProgramError(op.arguments()[0]->source_loc())
                   << "Argument to " << ToString(op.op())
                   << " should be an lvalue.";
          }
          op.set_static_type(arena_->New<PointerType>(ts[0]));
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::As: {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              TypeCheckTypeExp(op.arguments()[1], impl_scope));
          ErrorOr<Nonnull<Expression*>> converted =
              BuildBuiltinMethodCall(impl_scope, op.arguments()[0],
                                     BuiltinInterfaceName{Builtins::As, type},
                                     BuiltinMethodCall{"Convert"});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return ProgramError(e->source_loc())
                   << "type error in `as`: `" << *ts[0]
                   << "` is not explicitly convertible to `" << *type << "`:\n"
                   << converted.error().message();
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&call.function(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&call.argument(), impl_scope));
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          if (trace_stream_) {
            **trace_stream_
                << "checking call to function of type " << fun_t
                << "\nwith arguments of type: " << call.argument().static_type()
                << "\n";
          }
          CARBON_RETURN_IF_ERROR(DeduceCallBindings(
              call, &fun_t.parameters(), fun_t.generic_parameters(),
              fun_t.deduced_bindings(), impl_scope));

          // Substitute into the return type to determine the type of the call
          // expression.
          Nonnull<const Value*> return_type =
              Substitute(call.bindings(), &fun_t.return_type());
          call.set_static_type(return_type);
          call.set_value_category(ValueCategory::Let);
          return Success();
        }
        case Value::Kind::TypeOfParameterizedEntityName: {
          // This case handles the application of a parameterized class or
          // interface to a set of arguments, such as Point(i32) or
          // AddWith(i32).
          const ParameterizedEntityName& param_name =
              cast<TypeOfParameterizedEntityName>(call.function().static_type())
                  .name();

          // Collect the top-level generic parameters and their constraints.
          std::vector<FunctionType::GenericParameter> generic_parameters;
          llvm::ArrayRef<Nonnull<const Pattern*>> params =
              param_name.params().fields();
          for (size_t i = 0; i != params.size(); ++i) {
            // TODO: Should we disallow all other kinds of top-level params?
            if (const auto* binding = dyn_cast<GenericBinding>(params[i])) {
              generic_parameters.push_back({i, binding});
            }
          }

          CARBON_RETURN_IF_ERROR(DeduceCallBindings(
              call, &param_name.params().static_type(), generic_parameters,
              /*deduced_bindings=*/std::nullopt, impl_scope));

          // Currently the only kinds of parameterized entities we support are
          // types.
          CARBON_CHECK(
              isa<ClassDeclaration, InterfaceDeclaration, ConstraintDeclaration,
                  ChoiceDeclaration>(param_name.declaration()))
              << "unknown type of ParameterizedEntityName for " << param_name;
          call.set_static_type(arena_->New<TypeType>());
          call.set_value_category(ValueCategory::Let);
          return Success();
        }
        default: {
          return ProgramError(e->source_loc())
                 << "in call `" << *e
                 << "`, expected callee to be a function, found `"
                 << call.function().static_type() << "`";
        }
      }
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fn = cast<FunctionTypeLiteral>(*e);
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> param,
                              TypeCheckTypeExp(&fn.parameter(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> ret,
                              TypeCheckTypeExp(&fn.return_type(), impl_scope));
      fn.set_static_type(arena_->New<TypeType>());
      fn.set_value_category(ValueCategory::Let);
      fn.set_constant_value(arena_->New<FunctionType>(param, ret));
      return Success();
    }
    case ExpressionKind::StringLiteral:
      e->set_static_type(arena_->New<StringType>());
      e->set_value_category(ValueCategory::Let);
      return Success();
    case ExpressionKind::IntrinsicExpression: {
      auto& intrinsic_exp = cast<IntrinsicExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&intrinsic_exp.args(), impl_scope));
      const auto& args = intrinsic_exp.args().fields();
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          // TODO: Remove Print special casing once we have variadics or
          // overloads. Here, that's the name Print instead of __intrinsic_print
          // in errors.
          if (args.empty() || args.size() > 2) {
            return ProgramError(e->source_loc())
                   << "Print takes 1 or 2 arguments, received " << args.size();
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "Print argument 0", arena_->New<StringType>(),
              &args[0]->static_type(), impl_scope));
          if (args.size() >= 2) {
            CARBON_RETURN_IF_ERROR(ExpectExactType(
                e->source_loc(), "Print argument 1", arena_->New<IntType>(),
                &args[1]->static_type(), impl_scope));
          }
          e->set_static_type(TupleType::Empty());
          e->set_value_category(ValueCategory::Let);
          return Success();
        case IntrinsicExpression::Intrinsic::Assert: {
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_assert takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectType(
              e->source_loc(), "__intrinsic_assert argument 0",
              arena_->New<BoolType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectType(
              e->source_loc(), "__intrinsic_assert argument 1",
              arena_->New<StringType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(TupleType::Empty());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Alloc: {
          if (args.size() != 1) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_new takes 1 argument";
          }
          const auto* arg_type = &args[0]->static_type();
          e->set_static_type(arena_->New<PointerType>(arg_type));
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Dealloc: {
          if (args.size() != 1) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_new takes 1 argument";
          }
          const auto* arg_type = &args[0]->static_type();
          CARBON_RETURN_IF_ERROR(
              ExpectPointerType(e->source_loc(), "*", arg_type));
          e->set_static_type(TupleType::Empty());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Rand: {
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << "Rand takes 2 arguments, received " << args.size();
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "Rand argument 0", arena_->New<IntType>(),
              &args[0]->static_type(), impl_scope));

          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "Rand argument 1", arena_->New<IntType>(),
              &args[1]->static_type(), impl_scope));

          e->set_static_type(arena_->New<IntType>());

          return Success();
        }
        case IntrinsicExpression::Intrinsic::IntEq: {
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_int_eq takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_eq argument 1",
              arena_->New<IntType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_eq argument 2",
              arena_->New<IntType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<BoolType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::IntCompare: {
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_int_compare takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_compare argument 1",
              arena_->New<IntType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_compare argument 2",
              arena_->New<IntType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::StrEq: {
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_str_eq takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_eq argument 1",
              arena_->New<StringType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_eq argument 2",
              arena_->New<StringType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<BoolType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::StrCompare: {
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_str_compare takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_compare argument 1",
              arena_->New<StringType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_compare argument 2",
              arena_->New<StringType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::IntBitComplement:
          if (args.size() != 1) {
            return ProgramError(e->source_loc())
                   << intrinsic_exp.name() << " takes 1 argument";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "complement argument", arena_->New<IntType>(),
              &args[0]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        case IntrinsicExpression::Intrinsic::IntBitAnd:
        case IntrinsicExpression::Intrinsic::IntBitOr:
        case IntrinsicExpression::Intrinsic::IntBitXor:
        case IntrinsicExpression::Intrinsic::IntLeftShift:
        case IntrinsicExpression::Intrinsic::IntRightShift:
          if (args.size() != 2) {
            return ProgramError(e->source_loc())
                   << intrinsic_exp.name() << " takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "argument 1", arena_->New<IntType>(),
              &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "argument 2", arena_->New<IntType>(),
              &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
      }
    }
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<TypeType>());
      return Success();
    case ExpressionKind::IfExpression: {
      auto& if_expr = cast<IfExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&if_expr.condition(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_condition,
          ImplicitlyConvert("condition of `if`", impl_scope,
                            &if_expr.condition(), arena_->New<BoolType>()));
      if_expr.set_condition(converted_condition);

      // TODO: Compute the common type and convert both operands to it.
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&if_expr.then_expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&if_expr.else_expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(ExpectExactType(
          e->source_loc(), "expression of `if` expression",
          &if_expr.then_expression().static_type(),
          &if_expr.else_expression().static_type(), impl_scope));
      e->set_static_type(&if_expr.then_expression().static_type());
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::WhereExpression: {
      auto& where = cast<WhereExpression>(*e);
      ImplScope inner_impl_scope;
      inner_impl_scope.AddParent(&impl_scope);

      auto& self = where.self_binding();

      // If there's some enclosing `.Self` value, our self is symbolically
      // equal to that. Otherwise it's a new type variable.
      if (auto enclosing_dot_self = where.enclosing_dot_self()) {
        // TODO: We need to also enforce that our `.Self` does end up being the
        // same as the enclosing type.
        self.set_symbolic_identity(*(*enclosing_dot_self)->symbolic_identity());
        self.set_value(&(*enclosing_dot_self)->value());
      } else {
        ConstraintTypeBuilder::PrepareSelfBinding(arena_, &self);
      }

      ConstraintTypeBuilder builder(arena_, &self);
      ConstraintTypeBuilder::ConstraintsInScopeTracker constraint_tracker;

      // Keep track of the builder so that we can look up its rewrites while
      // processing later constraints.
      partial_constraint_types_.push_back(&builder);
      auto pop_partial_constraint_type =
          llvm::make_scope_exit([&] { partial_constraint_types_.pop_back(); });

      // Note, we don't want to call `TypeCheckPattern` here. Most of the setup
      // for the self binding is instead done by the `ConstraintTypeBuilder`.
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> base_type,
                              TypeCheckTypeExp(&self.type(), impl_scope));
      self.set_static_type(base_type);

      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const ConstraintType*> base,
          ConvertToConstraintType(where.source_loc(),
                                  "first operand of `where` expression",
                                  base_type));

      // Start with the given constraint.
      builder.AddAndSubstitute(*this, base, builder.GetSelfType(),
                               builder.GetSelfWitness(), Bindings(),
                               /*add_lookup_contexts=*/true);

      // Type-check and apply the `where` clauses.
      for (Nonnull<WhereClause*> clause : where.clauses()) {
        // Constraints from the LHS of `where` are in scope in the RHS, and
        // constraints from earlier `where` clauses are in scope in later
        // clauses.
        builder.BringConstraintsIntoScope(*this, &inner_impl_scope,
                                          &constraint_tracker);

        CARBON_RETURN_IF_ERROR(TypeCheckWhereClause(clause, inner_impl_scope));

        switch (clause->kind()) {
          case WhereClauseKind::IsWhereClause: {
            const auto& is_clause = cast<IsWhereClause>(*clause);
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> type,
                InterpExp(&is_clause.type(), arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> constraint,
                InterpExp(&is_clause.constraint(), arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const ConstraintType*> constraint_type,
                ConvertToConstraintType(is_clause.source_loc(),
                                        "expression after `is`", constraint));
            // Transform `where .B is (C where .D is E)` into `where .B is C
            // and .B.D is E` then add all the resulting constraints.
            builder.AddAndSubstitute(*this, constraint_type, type,
                                     builder.GetSelfWitness(), Bindings(),
                                     /*add_lookup_contexts=*/false);
            break;
          }
          case WhereClauseKind::EqualsWhereClause: {
            const auto& equals_clause = cast<EqualsWhereClause>(*clause);
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> lhs,
                InterpExp(&equals_clause.lhs(), arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> rhs,
                InterpExp(&equals_clause.rhs(), arena_, trace_stream_));
            if (!ValueEqual(lhs, rhs, std::nullopt)) {
              builder.AddEqualityConstraint({.values = {lhs, rhs}});
            }
            break;
          }
          case WhereClauseKind::RewriteWhereClause: {
            const auto& rewrite_clause = cast<RewriteWhereClause>(*clause);
            CARBON_ASSIGN_OR_RETURN(
                ConstraintLookupResult result,
                LookupInConstraint(clause->source_loc(),
                                   "rewrite constraint lookup", base_type,
                                   rewrite_clause.member_name()));
            const auto* constant =
                dyn_cast<AssociatedConstantDeclaration>(result.member);
            if (!constant) {
              return ProgramError(clause->source_loc())
                     << "in rewrite constraint lookup, `"
                     << rewrite_clause.member_name()
                     << "` does not name an associated constant";
            }

            // Find (or add) `.Self is I`, and form a symbolic value naming the
            // associated constant.
            // TODO: Reject if the impl constraint didn't already exist.
            int index = builder.AddImplConstraint(
                {.type = builder.GetSelfType(), .interface = result.interface});
            const auto* witness =
                MakeConstraintWitnessAccess(builder.GetSelfWitness(), index);
            auto* constant_value = arena_->New<AssociatedConstant>(
                builder.GetSelfType(), result.interface, constant, witness);

            // Find the replacement value prior to conversion to the constant's
            // type. This is the value we'll rewrite to when type-checking a
            // member access.
            CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> replacement_value,
                                    InterpExp(&rewrite_clause.replacement(),
                                              arena_, trace_stream_));
            Nonnull<const Value*> replacement_type =
                &rewrite_clause.replacement().static_type();

            auto* replacement_literal = arena_->New<ValueLiteral>(
                rewrite_clause.source_loc(), replacement_value,
                replacement_type, ValueCategory::Let);

            // Convert the replacement value to the type of the associated
            // constant and find the converted value. This is the value that
            // we'll produce during evaluation and substitution.
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<Expression*> converted_expression,
                ImplicitlyConvert(
                    "rewrite constraint", inner_impl_scope, replacement_literal,
                    GetTypeForAssociatedConstant(constant_value)));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> converted_value,
                InterpExp(converted_expression, arena_, trace_stream_));

            // Add the rewrite constraint.
            builder.AddRewriteConstraint(
                {.constant = constant_value,
                 .unconverted_replacement = replacement_value,
                 .unconverted_replacement_type = replacement_type,
                 .converted_replacement = converted_value});
            break;
          }
        }
      }

      where.set_rewritten_form(arena_->New<ValueLiteral>(
          where.source_loc(), std::move(builder).Build(),
          arena_->New<TypeType>(), ValueCategory::Let));
      return Success();
    }
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << *e;
    case ExpressionKind::ArrayTypeLiteral: {
      auto& array_literal = cast<ArrayTypeLiteral>(*e);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> element_type,
          TypeCheckTypeExp(&array_literal.element_type_expression(),
                           impl_scope));
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&array_literal.size_expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(ExpectExactType(
          array_literal.size_expression().source_loc(), "array size",
          arena_->New<IntType>(),
          &array_literal.size_expression().static_type(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> size_value,
          InterpExp(&array_literal.size_expression(), arena_, trace_stream_));
      if (cast<IntValue>(size_value)->value() < 0) {
        return ProgramError(array_literal.size_expression().source_loc())
               << "Array size cannot be negative";
      }
      array_literal.set_static_type(arena_->New<TypeType>());
      array_literal.set_value_category(ValueCategory::Let);
      array_literal.set_constant_value(arena_->New<StaticArrayType>(
          element_type, cast<IntValue>(size_value)->value()));
      return Success();
    }
  }
}

void TypeChecker::CollectGenericBindingsInPattern(
    Nonnull<const Pattern*> p,
    std::vector<Nonnull<const GenericBinding*>>& generic_bindings) {
  VisitNestedPatterns(*p, [&](const Pattern& pattern) {
    if (const auto* binding = dyn_cast<GenericBinding>(&pattern)) {
      generic_bindings.push_back(binding);
    }
    return true;
  });
}

void TypeChecker::CollectImplBindingsInPattern(
    Nonnull<const Pattern*> p,
    std::vector<Nonnull<const ImplBinding*>>& impl_bindings) {
  VisitNestedPatterns(*p, [&](const Pattern& pattern) {
    if (const auto* binding = dyn_cast<GenericBinding>(&pattern)) {
      if (binding->impl_binding().has_value()) {
        impl_bindings.push_back(binding->impl_binding().value());
      }
    }
    return true;
  });
}

void TypeChecker::BringPatternImplsIntoScope(Nonnull<const Pattern*> p,
                                             ImplScope& impl_scope) {
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;
  CollectImplBindingsInPattern(p, impl_bindings);
  BringImplsIntoScope(impl_bindings, impl_scope);
}

void TypeChecker::BringImplsIntoScope(
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
    ImplScope& impl_scope) {
  for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
    BringImplIntoScope(impl_binding, impl_scope);
  }
}

void TypeChecker::BringImplIntoScope(Nonnull<const ImplBinding*> impl_binding,
                                     ImplScope& impl_scope) {
  CARBON_CHECK(impl_binding->type_var()->symbolic_identity().has_value() &&
               impl_binding->symbolic_identity().has_value());
  impl_scope.Add(impl_binding->interface(),
                 *impl_binding->type_var()->symbolic_identity(),
                 cast<Witness>(*impl_binding->symbolic_identity()), *this);
}

auto TypeChecker::TypeCheckTypeExp(Nonnull<Expression*> type_expression,
                                   const ImplScope& impl_scope, bool concrete)
    -> ErrorOr<Nonnull<const Value*>> {
  CARBON_RETURN_IF_ERROR(TypeCheckExp(type_expression, impl_scope));
  CARBON_ASSIGN_OR_RETURN(
      type_expression,
      ImplicitlyConvert("type expression", impl_scope, type_expression,
                        arena_->New<TypeType>()));
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                          InterpExp(type_expression, arena_, trace_stream_));
  CARBON_CHECK(IsType(type))
      << "type expression did not produce a type, got " << *type;
  if (concrete) {
    if (TypeContainsAuto(type)) {
      return ProgramError(type_expression->source_loc())
             << "`auto` is not permitted in this context";
    }
    CARBON_CHECK(IsConcreteType(type))
        << "unknown kind of non-concrete type " << *type;
  }
  CARBON_CHECK(!IsPlaceholderType(type))
      << "should be no way to write a placeholder type";
  return type;
}

auto TypeChecker::TypeCheckWhereClause(Nonnull<WhereClause*> clause,
                                       const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  switch (clause->kind()) {
    case WhereClauseKind::IsWhereClause: {
      auto& is_clause = cast<IsWhereClause>(*clause);
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&is_clause.type(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&is_clause.constraint(), impl_scope));
      if (!isa<TypeType>(is_clause.constraint().static_type())) {
        return ProgramError(is_clause.constraint().source_loc())
               << "expression after `is` does not resolve to a constraint, "
               << "found " << is_clause.constraint().static_type();
      }
      return Success();
    }
    case WhereClauseKind::EqualsWhereClause: {
      auto& equals_clause = cast<EqualsWhereClause>(*clause);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&equals_clause.lhs(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&equals_clause.rhs(), impl_scope));

      // TODO: It's not clear what level of type compatibility is required
      // between the operands. For now we require a builtin no-op implicit
      // conversion.
      Nonnull<const Value*> lhs_type = &equals_clause.lhs().static_type();
      Nonnull<const Value*> rhs_type = &equals_clause.rhs().static_type();
      if (!IsImplicitlyConvertible(lhs_type, rhs_type, impl_scope,
                                   /*allow_user_defined_conversions=*/false) &&
          !IsImplicitlyConvertible(rhs_type, lhs_type, impl_scope,
                                   /*allow_user_defined_conversions=*/false)) {
        return ProgramError(clause->source_loc())
               << "type mismatch between values in `where LHS == RHS`\n"
               << "  LHS type: " << *lhs_type << "\n"
               << "  RHS type: " << *rhs_type;
      }
      return Success();
    }
    case WhereClauseKind::RewriteWhereClause: {
      auto& rewrite_clause = cast<RewriteWhereClause>(*clause);
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&rewrite_clause.replacement(), impl_scope));
      return Success();
    }
  }
}

auto TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, std::optional<Nonnull<const Value*>> expected,
    ImplScope& impl_scope, ValueCategory enclosing_value_category)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << PatternKindName(p->kind()) << " " << *p;
    if (expected) {
      **trace_stream_ << ", expecting " << **expected;
    }
    **trace_stream_ << "\n";
  }
  switch (p->kind()) {
    case PatternKind::AutoPattern: {
      p->set_static_type(arena_->New<TypeType>());
      p->set_value(arena_->New<AutoType>());
      return Success();
    }
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(*p);
      if (!VisitNestedPatterns(binding.type(), [](const Pattern& pattern) {
            return !isa<BindingPattern>(pattern);
          })) {
        return ProgramError(binding.type().source_loc())
               << "the type of a binding pattern cannot contain bindings";
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(
          &binding.type(), std::nullopt, impl_scope, enclosing_value_category));
      Nonnull<const Value*> type = &binding.type().value();
      // Convert to a type.
      // TODO: Convert the pattern before interpreting it rather than doing
      // this as a separate step.
      if (!isa<TypeType>(binding.type().static_type())) {
        auto* literal = arena_->New<ValueLiteral>(binding.source_loc(), type,
                                                  &binding.type().static_type(),
                                                  ValueCategory::Let);
        CARBON_ASSIGN_OR_RETURN(
            auto* converted,
            ImplicitlyConvert("type of name binding", impl_scope, literal,
                              arena_->New<TypeType>()));
        CARBON_ASSIGN_OR_RETURN(type,
                                InterpExp(converted, arena_, trace_stream_));
      }
      CARBON_CHECK(IsType(type))
          << "conversion to type succeeded but didn't produce a type, got "
          << *type;
      if (expected) {
        if (IsConcreteType(type)) {
          CARBON_RETURN_IF_ERROR(ExpectType(p->source_loc(), "name binding",
                                            type, *expected, impl_scope));
        } else {
          BindingMap generic_args;
          if (!PatternMatch(type, *expected, binding.type().source_loc(),
                            std::nullopt, generic_args, trace_stream_,
                            this->arena_)) {
            return ProgramError(binding.type().source_loc())
                   << "type pattern '" << *type
                   << "' does not match actual type '" << **expected << "'";
          }
          type = *expected;
        }
      } else if (TypeContainsAuto(type)) {
        return ProgramError(binding.source_loc())
               << "cannot deduce `auto` type for " << binding;
      }
      CARBON_CHECK(IsConcreteType(type)) << "did not resolve " << binding
                                         << " to concrete type, got " << *type;
      CARBON_CHECK(!IsPlaceholderType(type))
          << "should be no way to write a placeholder type";
      binding.set_static_type(type);
      binding.set_value(binding.name() != AnonymousName
                            ? arena_->New<BindingPlaceholderValue>(&binding)
                            : arena_->New<BindingPlaceholderValue>());

      if (!binding.has_value_category()) {
        binding.set_value_category(enclosing_value_category);
      }
      return Success();
    }
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(*p);
      if (expected) {
        return ProgramError(binding.type().source_loc())
               << "generic binding may not occur in pattern with expected "
                  "type "
               << binding;
      }

      return TypeCheckGenericBinding(binding, "generic binding", impl_scope);
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      std::vector<Nonnull<const Value*>> field_types;
      std::vector<Nonnull<const Value*>> field_patterns;
      if (expected && (*expected)->kind() != Value::Kind::TupleType) {
        return ProgramError(p->source_loc()) << "didn't expect a tuple";
      }
      if (expected && tuple.fields().size() !=
                          cast<TupleType>(**expected).elements().size()) {
        return ProgramError(tuple.source_loc()) << "tuples of different length";
      }
      for (size_t i = 0; i < tuple.fields().size(); ++i) {
        Nonnull<Pattern*> field = tuple.fields()[i];
        std::optional<Nonnull<const Value*>> expected_field_type;
        if (expected) {
          expected_field_type = cast<TupleType>(**expected).elements()[i];
        }
        CARBON_RETURN_IF_ERROR(TypeCheckPattern(
            field, expected_field_type, impl_scope, enclosing_value_category));
        if (trace_stream_) {
          **trace_stream_ << "finished checking tuple pattern field " << *field
                          << "\n";
        }
        field_types.push_back(&field->static_type());
        field_patterns.push_back(&field->value());
      }
      tuple.set_static_type(arena_->New<TupleType>(std::move(field_types)));
      tuple.set_value(arena_->New<TupleValue>(std::move(field_patterns)));
      return Success();
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&alternative.choice_type(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> type,
          InterpExp(&alternative.choice_type(), arena_, trace_stream_));
      if (!isa<ChoiceType>(type)) {
        return ProgramError(alternative.source_loc())
               << "alternative pattern does not name a choice type.";
      }
      const auto& choice_type = cast<ChoiceType>(*type);
      if (expected) {
        CARBON_RETURN_IF_ERROR(ExpectType(alternative.source_loc(),
                                          "alternative pattern", &choice_type,
                                          *expected, impl_scope));
      }
      std::optional<Nonnull<const Value*>> parameter_types =
          choice_type.FindAlternative(alternative.alternative_name());
      if (parameter_types == std::nullopt) {
        return ProgramError(alternative.source_loc())
               << "'" << alternative.alternative_name()
               << "' is not an alternative of " << choice_type;
      }

      Nonnull<const Value*> substituted_parameter_type = *parameter_types;
      if (choice_type.IsParameterized()) {
        substituted_parameter_type =
            Substitute(choice_type.bindings(), *parameter_types);
      }
      CARBON_RETURN_IF_ERROR(
          TypeCheckPattern(&alternative.arguments(), substituted_parameter_type,
                           impl_scope, enclosing_value_category));
      alternative.set_static_type(&choice_type);
      alternative.set_value(arena_->New<AlternativeValue>(
          alternative.alternative_name(), choice_type.name(),
          &alternative.arguments().value()));
      return Success();
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&expression, impl_scope));
      p->set_static_type(&expression.static_type());
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> expr_value,
                              InterpExp(&expression, arena_, trace_stream_));
      p->set_value(expr_value);
      return Success();
    }
    case PatternKind::VarPattern: {
      auto& var_pattern = cast<VarPattern>(*p);

      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var_pattern.pattern(), expected,
                                              impl_scope,
                                              var_pattern.value_category()));
      var_pattern.set_static_type(&var_pattern.pattern().static_type());
      var_pattern.set_value(&var_pattern.pattern().value());
      return Success();
    }
    case PatternKind::AddrPattern: {
      std::optional<Nonnull<const Value*>> expected_ptr;
      auto& addr_pattern = cast<AddrPattern>(*p);
      if (expected) {
        expected_ptr = arena_->New<PointerType>(expected.value());
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&addr_pattern.binding(),
                                              expected_ptr, impl_scope,
                                              enclosing_value_category));

      if (const auto* inner_binding_type =
              dyn_cast<PointerType>(&addr_pattern.binding().static_type())) {
        addr_pattern.set_static_type(&inner_binding_type->pointee_type());
      } else {
        return ProgramError(addr_pattern.source_loc())
               << "Type associated with addr must be a pointer type.";
      }
      addr_pattern.set_value(
          arena_->New<AddrValue>(&addr_pattern.binding().value()));
      return Success();
    }
  }
}

auto TypeChecker::TypeCheckGenericBinding(GenericBinding& binding,
                                          std::string_view context,
                                          ImplScope& impl_scope)
    -> ErrorOr<Success> {
  // The binding can be referred to in its own type via `.Self`, so set up
  // its symbolic identity before we type-check and interpret the type.
  auto* symbolic_value = arena_->New<VariableType>(&binding);
  binding.set_symbolic_identity(symbolic_value);
  binding.set_value(symbolic_value);

  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                          TypeCheckTypeExp(&binding.type(), impl_scope));
  if (binding.named_as_type_via_dot_self() && !IsTypeOfType(type)) {
    return ProgramError(binding.type().source_loc())
           << "`.Self` used in type of non-type " << context << " `"
           << binding.name() << "`";
  }

  // Create an impl binding if we have a constraint.
  if (IsTypeOfType(type) && !isa<TypeType>(type)) {
    CARBON_ASSIGN_OR_RETURN(
        Nonnull<const ConstraintType*> constraint,
        ConvertToConstraintType(binding.source_loc(), context, type));
    Nonnull<ImplBinding*> impl_binding =
        arena_->New<ImplBinding>(binding.source_loc(), &binding, std::nullopt);
    auto* witness = arena_->New<BindingWitness>(impl_binding);
    impl_binding->set_symbolic_identity(witness);
    binding.set_impl_binding(impl_binding);

    // Substitute the VariableType as `.Self` of the constraint to form the
    // resolved type of the binding. Eg, `T:! X where .Self is Y` resolves
    // to `T:! <constraint T is X and T is Y>`.
    ConstraintTypeBuilder builder(arena_, &binding, impl_binding);
    builder.AddAndSubstitute(*this, constraint, symbolic_value, witness,
                             Bindings(), /*add_lookup_contexts=*/true);
    if (trace_stream_) {
      **trace_stream_ << "resolving constraint type for " << binding << " from "
                      << *constraint << "\n";
    }
    CARBON_RETURN_IF_ERROR(
        builder.Resolve(*this, binding.type().source_loc(), impl_scope));
    type = std::move(builder).Build();
    if (trace_stream_) {
      **trace_stream_ << "resolved constraint type is " << *type << "\n";
    }

    BringImplIntoScope(impl_binding, impl_scope);
  }

  binding.set_static_type(type);
  return Success();
}

auto TypeChecker::TypeCheckStmt(Nonnull<Statement*> s,
                                const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << StatementKindName(s->kind()) << " " << *s
                    << "\n";
  }
  switch (s->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&match.expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(ExpectNonPlaceholderType(
          match.expression().source_loc(), &match.expression().static_type()));
      std::vector<Match::Clause> new_clauses;
      std::optional<Nonnull<const Value*>> expected_type;
      PatternMatrix patterns;
      for (auto& clause : match.clauses()) {
        ImplScope clause_scope;
        clause_scope.AddParent(&impl_scope);
        // TODO: Should user-defined conversions be permitted in `match`
        // statements? When would we run them? See #1283.
        CARBON_RETURN_IF_ERROR(TypeCheckPattern(
            &clause.pattern(), &match.expression().static_type(), clause_scope,
            ValueCategory::Let));
        if (expected_type.has_value()) {
          // TODO: For now, we require all patterns to have the same type. If
          // that's not the same type as the scrutinee, we will convert the
          // scrutinee. We might want to instead allow a different conversion
          // to be performed for each pattern.
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(clause.pattern().source_loc(),
                              "`match` pattern type", expected_type.value(),
                              &clause.pattern().static_type(), impl_scope));
        } else {
          expected_type = &clause.pattern().static_type();
        }
        if (patterns.IsRedundant({&clause.pattern()})) {
          return ProgramError(clause.pattern().source_loc())
                 << "unreachable case: all values matched by this case "
                 << "are matched by earlier cases";
        }
        patterns.Add({&clause.pattern()});
        CARBON_RETURN_IF_ERROR(
            TypeCheckStmt(&clause.statement(), clause_scope));
      }
      if (expected_type.has_value()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_expression,
            ImplicitlyConvert("`match` expression", impl_scope,
                              &match.expression(), expected_type.value()));
        match.set_expression(converted_expression);
      }
      return Success();
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&while_stmt.condition(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_condition,
          ImplicitlyConvert("condition of `while`", impl_scope,
                            &while_stmt.condition(), arena_->New<BoolType>()));
      while_stmt.set_condition(converted_condition);
      CARBON_RETURN_IF_ERROR(TypeCheckStmt(&while_stmt.body(), impl_scope));
      return Success();
    }
    case StatementKind::For: {
      auto& for_stmt = cast<For>(*s);
      ImplScope inner_impl_scope;
      inner_impl_scope.AddParent(&impl_scope);

      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&for_stmt.loop_target(), inner_impl_scope));

      const Value& rhs = for_stmt.loop_target().static_type();
      if (rhs.kind() == Value::Kind::StaticArrayType) {
        CARBON_RETURN_IF_ERROR(
            TypeCheckPattern(&for_stmt.variable_declaration(),
                             &cast<StaticArrayType>(rhs).element_type(),
                             inner_impl_scope, ValueCategory::Var));

      } else {
        return ProgramError(for_stmt.source_loc())
               << "expected array type after in, found value of type " << rhs;
      }

      CARBON_RETURN_IF_ERROR(TypeCheckStmt(&for_stmt.body(), inner_impl_scope));
      return Success();
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return Success();
    case StatementKind::Block: {
      auto& block = cast<Block>(*s);
      for (auto* block_statement : block.statements()) {
        CARBON_RETURN_IF_ERROR(TypeCheckStmt(block_statement, impl_scope));
      }
      return Success();
    }
    case StatementKind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);

      // TODO: If the pattern contains a binding that implies a new impl is
      // available, should that remain in scope for as long as its binding?
      // ```
      // var a: (T:! Widget) = ...;
      // // Is the `impl T as Widget` in scope here?
      // a.(Widget.F)();
      // ```
      ImplScope var_scope;
      var_scope.AddParent(&impl_scope);
      std::optional<Nonnull<const Value*>> init_type;

      // Type-check the initializer before we inspect the type of the variable
      // so we can use its type to deduce parts of the type of the binding.
      if (var.has_init()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&var.init(), impl_scope));
        CARBON_RETURN_IF_ERROR(ExpectNonPlaceholderType(
            var.init().source_loc(), &var.init().static_type()));
        init_type = &var.init().static_type();
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var.pattern(), init_type,
                                              var_scope, var.value_category()));
      CARBON_RETURN_IF_ERROR(ExpectCompleteType(
          var.source_loc(), "type of variable", &var.pattern().static_type()));

      if (var.has_init()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_init,
            ImplicitlyConvert("initializer of variable", impl_scope,
                              &var.init(), &var.pattern().static_type()));
        var.set_init(converted_init);
      }
      return Success();
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&assign.rhs(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&assign.lhs(), impl_scope));
      if (assign.lhs().value_category() != ValueCategory::Var) {
        return ProgramError(assign.source_loc())
               << "Cannot assign to rvalue '" << assign.lhs() << "'";
      }
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_rhs,
          ImplicitlyConvert("assignment", impl_scope, &assign.rhs(),
                            &assign.lhs().static_type()));
      assign.set_rhs(converted_rhs);
      return Success();
    }
    case StatementKind::ExpressionStatement: {
      CARBON_RETURN_IF_ERROR(TypeCheckExp(
          &cast<ExpressionStatement>(*s).expression(), impl_scope));
      return Success();
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&if_stmt.condition(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_condition,
          ImplicitlyConvert("condition of `if`", impl_scope,
                            &if_stmt.condition(), arena_->New<BoolType>()));
      if_stmt.set_condition(converted_condition);
      CARBON_RETURN_IF_ERROR(TypeCheckStmt(&if_stmt.then_block(), impl_scope));
      if (if_stmt.else_block()) {
        CARBON_RETURN_IF_ERROR(
            TypeCheckStmt(*if_stmt.else_block(), impl_scope));
      }
      return Success();
    }
    case StatementKind::ReturnVar: {
      auto& ret = cast<ReturnVar>(*s);
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        return_term.set_static_type(&ret.value_node().static_type());
      } else {
        // TODO: Consider using `ExpectExactType` here.
        CARBON_CHECK(IsConcreteType(&return_term.static_type()));
        CARBON_CHECK(IsConcreteType(&ret.value_node().static_type()));
        if (!IsSameType(&return_term.static_type(),
                        &ret.value_node().static_type(), impl_scope)) {
          return ProgramError(ret.value_node().base().source_loc())
                 << "type of returned var `" << ret.value_node().static_type()
                 << "` does not match return type `"
                 << return_term.static_type() << "`";
        }
      }
      return Success();
    }
    case StatementKind::ReturnExpression: {
      auto& ret = cast<ReturnExpression>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&ret.expression(), impl_scope));
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        CARBON_RETURN_IF_ERROR(ExpectNonPlaceholderType(
            ret.source_loc(), &ret.expression().static_type()));
        return_term.set_static_type(&ret.expression().static_type());
      } else {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_ret_val,
            ImplicitlyConvert("return value", impl_scope, &ret.expression(),
                              &return_term.static_type()));
        ret.set_expression(converted_ret_val);
      }
      return Success();
    }
    case StatementKind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckStmt(&cont.body(), impl_scope));
      cont.set_static_type(arena_->New<ContinuationType>());
      return Success();
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&run.argument(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_argument,
          ImplicitlyConvert("argument of `run`", impl_scope, &run.argument(),
                            arena_->New<ContinuationType>()));
      run.set_argument(converted_argument);
      return Success();
    }
    case StatementKind::Await: {
      // Nothing to do here.
      return Success();
    }
  }
}

// Returns true if we can statically verify that `match` is exhaustive, meaning
// that one of its clauses will be executed for any possible operand value.
static auto IsExhaustive(const Match& match) -> bool {
  PatternMatrix matrix;
  for (const Match::Clause& clause : match.clauses()) {
    matrix.Add({&clause.pattern()});
  }
  return matrix.IsRedundant({AbstractPattern::MakeWildcard()});
}

auto TypeChecker::ExpectReturnOnAllPaths(
    std::optional<Nonnull<Statement*>> opt_stmt, SourceLocation source_loc)
    -> ErrorOr<Success> {
  if (!opt_stmt) {
    return ProgramError(source_loc)
           << "control-flow reaches end of function that provides a `->` "
              "return type without reaching a return statement";
  }
  Nonnull<Statement*> stmt = *opt_stmt;
  switch (stmt->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*stmt);
      if (!IsExhaustive(match)) {
        return ProgramError(source_loc)
               << "non-exhaustive match may allow control-flow to reach the "
                  "end "
                  "of a function that provides a `->` return type";
      }
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        CARBON_RETURN_IF_ERROR(
            ExpectReturnOnAllPaths(&clause.statement(), stmt->source_loc()));
      }
      return Success();
    }
    case StatementKind::Block: {
      auto& block = cast<Block>(*stmt);
      if (block.statements().empty()) {
        return ProgramError(stmt->source_loc())
               << "control-flow reaches end of function that provides a `->` "
                  "return type without reaching a return statement";
      }
      CARBON_RETURN_IF_ERROR(ExpectReturnOnAllPaths(
          block.statements()[block.statements().size() - 1],
          block.source_loc()));
      return Success();
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*stmt);
      CARBON_RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(&if_stmt.then_block(), stmt->source_loc()));
      CARBON_RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(if_stmt.else_block(), stmt->source_loc()));
      return Success();
    }
    case StatementKind::ReturnVar:
    case StatementKind::ReturnExpression:
      return Success();
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
    case StatementKind::Assign:
    case StatementKind::ExpressionStatement:
    case StatementKind::While:
    case StatementKind::For:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::VariableDefinition:
      return ProgramError(stmt->source_loc())
             << "control-flow reaches end of function that provides a `->` "
                "return type without reaching a return statement";
  }
}

// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
auto TypeChecker::DeclareCallableDeclaration(Nonnull<CallableDeclaration*> f,
                                             const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** declaring function " << f->name() << "\n";
  }
  ImplScope function_scope;
  function_scope.AddParent(scope_info.innermost_scope);
  std::vector<Nonnull<const GenericBinding*>> deduced_bindings;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;
  // Bring the deduced parameters into scope.
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(
        deduced, std::nullopt, function_scope, ValueCategory::Let));
    CollectGenericBindingsInPattern(deduced, deduced_bindings);
    CollectImplBindingsInPattern(deduced, impl_bindings);
  }
  // Type check the receiver pattern.
  if (f->is_method()) {
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(
        &f->self_pattern(), std::nullopt, function_scope, ValueCategory::Let));
    CollectGenericBindingsInPattern(&f->self_pattern(), deduced_bindings);
    CollectImplBindingsInPattern(&f->self_pattern(), impl_bindings);
  }
  // Type check the parameter pattern.
  CARBON_RETURN_IF_ERROR(TypeCheckPattern(&f->param_pattern(), std::nullopt,
                                          function_scope, ValueCategory::Let));
  CollectImplBindingsInPattern(&f->param_pattern(), impl_bindings);

  // Keep track of any generic parameters and nested generic bindings in the
  // parameter pattern.
  std::vector<FunctionType::GenericParameter> generic_parameters;
  for (size_t i = 0; i != f->param_pattern().fields().size(); ++i) {
    const Pattern* param_pattern = f->param_pattern().fields()[i];
    if (const auto* binding = dyn_cast<GenericBinding>(param_pattern)) {
      generic_parameters.push_back({i, binding});
    } else {
      CollectGenericBindingsInPattern(param_pattern, deduced_bindings);
    }
  }

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    CARBON_ASSIGN_OR_RETURN(
        Nonnull<const Value*> ret_type,
        TypeCheckTypeExp(*return_expression, function_scope));
    // TODO: This is setting the constant value of the return type. It would
    // make more sense if this were called `set_constant_value` rather than
    // `set_static_type`.
    f->return_term().set_static_type(ret_type);
  } else if (f->return_term().is_omitted()) {
    f->return_term().set_static_type(TupleType::Empty());
  } else {
    // We have to type-check the body in order to determine the return type.
    if (!f->body().has_value()) {
      return ProgramError(f->return_term().source_loc())
             << "Function declaration has deduced return type but no body";
    }
    CARBON_RETURN_IF_ERROR(TypeCheckStmt(*f->body(), function_scope));
    if (!f->return_term().is_omitted()) {
      CARBON_RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(f->body(), f->source_loc()));
    }
  }
  CARBON_CHECK(IsConcreteType(&f->return_term().static_type()));

  f->set_static_type(arena_->New<FunctionType>(
      &f->param_pattern().static_type(), std::move(generic_parameters),
      &f->return_term().static_type(), std::move(deduced_bindings),
      std::move(impl_bindings)));
  switch (f->kind()) {
    case DeclarationKind::FunctionDeclaration:
      f->set_constant_value(
          arena_->New<FunctionValue>(cast<FunctionDeclaration>(f)));
      break;
    case DeclarationKind::DestructorDeclaration:
      f->set_constant_value(
          arena_->New<DestructorValue>(cast<DestructorDeclaration>(f)));
      break;
    default:
      CARBON_FATAL() << "f is not a callable declaration";
  }

  if (f->name() == "Main") {
    if (!f->return_term().type_expression().has_value()) {
      return ProgramError(f->return_term().source_loc())
             << "`Main` must have an explicit return type";
    }
    CARBON_RETURN_IF_ERROR(
        ExpectExactType(f->return_term().source_loc(), "return type of `Main`",
                        arena_->New<IntType>(), &f->return_term().static_type(),
                        function_scope));
    if (!f->param_pattern().fields().empty()) {
      return ProgramError(f->source_loc())
             << "`Main` must not take any parameters";
    }
  }

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring function " << f->name()
                    << " of type " << f->static_type() << "\n";
  }
  return Success();
}

auto TypeChecker::TypeCheckCallableDeclaration(Nonnull<CallableDeclaration*> f,
                                               const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** checking function " << f->name() << "\n";
  }
  // If f->return_term().is_auto(), the function body was already
  // type checked in DeclareFunctionDeclaration.
  if (f->body().has_value() && !f->return_term().is_auto()) {
    // Bring the impls into scope.
    ImplScope function_scope;
    function_scope.AddParent(&impl_scope);
    BringImplsIntoScope(cast<FunctionType>(f->static_type()).impl_bindings(),
                        function_scope);
    if (trace_stream_) {
      **trace_stream_ << function_scope;
    }
    CARBON_RETURN_IF_ERROR(TypeCheckStmt(*f->body(), function_scope));
    if (!f->return_term().is_omitted()) {
      CARBON_RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(f->body(), f->source_loc()));
    }
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking function " << f->name() << "\n";
  }
  return Success();
}

auto TypeChecker::DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                          const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** declaring class " << class_decl->name() << "\n";
  }
  Nonnull<SelfDeclaration*> self = class_decl->self();

  ImplScope class_scope;
  class_scope.AddParent(scope_info.innermost_scope);

  std::optional<Nonnull<const NominalClassType*>> base_class;
  if (class_decl->base_expr().has_value()) {
    Nonnull<Expression*> base_class_expr = *class_decl->base_expr();
    CARBON_ASSIGN_OR_RETURN(const auto base_type,
                            TypeCheckTypeExp(base_class_expr, class_scope));
    if (base_type->kind() != Value::Kind::NominalClassType) {
      return ProgramError(class_decl->source_loc())
             << "Unsupported base class type for class `" << class_decl->name()
             << "`. Only simple classes are currently supported as base "
                "class.";
    }

    base_class = cast<NominalClassType>(base_type);
    if (base_class.value()->declaration().extensibility() ==
        ClassExtensibility::None) {
      return ProgramError(class_decl->source_loc())
             << "Base class `" << base_class.value()->declaration().name()
             << "` is `final` and cannot be inherited. Add the `base` or "
                "`abstract` class prefix to `"
             << base_class.value()->declaration().name()
             << "` to allow it to be inherited";
    }
    class_decl->set_base_type(base_class);
  }

  std::vector<Nonnull<const GenericBinding*>> bindings = scope_info.bindings;
  if (class_decl->type_params().has_value()) {
    Nonnull<TuplePattern*> type_params = *class_decl->type_params();
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(type_params, std::nullopt,
                                            class_scope, ValueCategory::Let));
    CollectGenericBindingsInPattern(type_params, bindings);
    if (trace_stream_) {
      **trace_stream_ << class_scope;
    }
  }

  // Generate a vtable for the type if necessary.
  VTable class_vtable = base_class ? (*base_class)->vtable() : VTable();
  const int class_level = base_class ? (*base_class)->hierarchy_level() + 1 : 0;
  for (const auto* m : class_decl->members()) {
    const auto* fun = dyn_cast<FunctionDeclaration>(m);
    if (!fun) {
      continue;
    }
    if (fun->virt_override() != VirtualOverride::None && !fun->is_method()) {
      return ProgramError(fun->source_loc())
             << "Error declaring `" << fun->name() << "`"
             << ": class functions cannot be virtual.";
    }
    bool has_vtable_entry =
        class_vtable.find(fun->name()) != class_vtable.end();
    // TODO: Implement complete declaration logic from
    // `/docs/design/classes.md#virtual-methods`.
    switch (fun->virt_override()) {
      case VirtualOverride::Abstract:
        // Not supported yet.
        return ProgramError(fun->source_loc())
               << "Error declaring `" << fun->name() << "`"
               << ": `abstract` methods are not yet supported.";
      case VirtualOverride::None:
      case VirtualOverride::Virtual:
        if (has_vtable_entry) {
          return ProgramError(fun->source_loc())
                 << "Error declaring `" << fun->name() << "`"
                 << ": method is declared virtual in base class, use `impl` "
                    "to override it.";
        }
        // TODO: Error if declaring virtual method shadowing non-virtual method.
        // See https://github.com/carbon-language/carbon-lang/issues/2355.
        if (fun->virt_override() == VirtualOverride::None) {
          // Not added to the vtable.
          continue;
        }
        break;
      case VirtualOverride::Impl:
        if (!has_vtable_entry) {
          return ProgramError(fun->source_loc())
                 << "Error declaring `" << fun->name() << "`"
                 << ": cannot override a method that is not declared "
                    "`abstract` or `virtual` in base class.";
        }
        break;
    }
    class_vtable[fun->name()] = {fun, class_level};
  }

  // For class declaration `class MyType(T:! type, U:! AnInterface)`, `Self`
  // should have the value `MyType(T, U)`.
  Nonnull<NominalClassType*> self_type = arena_->New<NominalClassType>(
      class_decl, Bindings::SymbolicIdentity(arena_, bindings), base_class,
      std::move(class_vtable));
  self->set_static_type(arena_->New<TypeType>());
  self->set_constant_value(self_type);

  // The declarations of the members may refer to the class, so we must set the
  // constant value of the class and its static type before we start processing
  // the members.
  if (class_decl->type_params().has_value()) {
    // TODO: The `enclosing_bindings` should be tracked in the parameterized
    // entity name so that they can be included in the eventual type.
    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(class_decl,
                                             *class_decl->type_params());
    class_decl->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
    class_decl->set_constant_value(param_name);
  } else {
    class_decl->set_static_type(&self->static_type());
    class_decl->set_constant_value(self_type);
  }

  ScopeInfo class_scope_info =
      ScopeInfo::ForClassScope(scope_info, &class_scope, std::move(bindings));
  for (Nonnull<Declaration*> m : class_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, class_scope_info));
  }

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring class " << class_decl->name()
                    << "\n";
  }
  return Success();
}

auto TypeChecker::TypeCheckClassDeclaration(
    Nonnull<ClassDeclaration*> class_decl, const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** checking class " << class_decl->name() << "\n";
  }
  ImplScope class_scope;
  class_scope.AddParent(&impl_scope);
  if (class_decl->type_params().has_value()) {
    BringPatternImplsIntoScope(*class_decl->type_params(), class_scope);
  }
  if (trace_stream_) {
    **trace_stream_ << class_scope;
  }
  auto [it, inserted] =
      collected_members_.insert({class_decl, CollectedMembersMap()});
  CARBON_CHECK(inserted) << "Adding class " << class_decl->name()
                         << " to collected_members_ must not fail";
  for (Nonnull<Declaration*> m : class_decl->members()) {
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, class_scope, class_decl));
    CARBON_RETURN_IF_ERROR(CollectMember(class_decl, m));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking class " << class_decl->name()
                    << "\n";
  }
  return Success();
}

// EXPERIMENTAL MIXIN FEATURE
auto TypeChecker::DeclareMixinDeclaration(Nonnull<MixinDeclaration*> mixin_decl,
                                          const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** declaring mixin " << mixin_decl->name() << "\n";
  }
  ImplScope mixin_scope;
  mixin_scope.AddParent(scope_info.innermost_scope);

  if (mixin_decl->params().has_value()) {
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(*mixin_decl->params(), std::nullopt,
                                            mixin_scope, ValueCategory::Let));
    if (trace_stream_) {
      **trace_stream_ << mixin_scope;
    }

    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(mixin_decl, *mixin_decl->params());
    mixin_decl->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
    mixin_decl->set_constant_value(param_name);
  } else {
    Nonnull<MixinPseudoType*> mixin_type =
        arena_->New<MixinPseudoType>(mixin_decl);
    mixin_decl->set_static_type(arena_->New<TypeOfMixinPseudoType>(mixin_type));
    mixin_decl->set_constant_value(mixin_type);
  }

  // Process the Self parameter.
  CARBON_RETURN_IF_ERROR(TypeCheckPattern(mixin_decl->self(), std::nullopt,
                                          mixin_scope, ValueCategory::Let));

  ScopeInfo mixin_scope_info = ScopeInfo::ForNonClassScope(&mixin_scope);
  for (Nonnull<Declaration*> m : mixin_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, mixin_scope_info));
  }

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring mixin " << mixin_decl->name()
                    << "\n";
  }
  return Success();
}

// EXPERIMENTAL MIXIN FEATURE
// Checks to see if mixin_decl is already within collected_members_. If it is,
// then the mixin has already been type checked before either while type
// checking a previous mix declaration or while type checking the original mixin
// declaration. If not, then every member declaration is type checked and then
// added to collected_members_ under the mixin_decl key.
auto TypeChecker::TypeCheckMixinDeclaration(
    Nonnull<const MixinDeclaration*> mixin_decl, const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  auto [it, inserted] =
      collected_members_.insert({mixin_decl, CollectedMembersMap()});
  if (!inserted) {
    // This declaration has already been type checked before
    if (trace_stream_) {
      **trace_stream_ << "** skipped checking mixin " << mixin_decl->name()
                      << "\n";
    }
    return Success();
  }
  if (trace_stream_) {
    **trace_stream_ << "** checking mixin " << mixin_decl->name() << "\n";
  }
  ImplScope mixin_scope;
  mixin_scope.AddParent(&impl_scope);
  if (mixin_decl->params().has_value()) {
    BringPatternImplsIntoScope(*mixin_decl->params(), mixin_scope);
  }
  if (trace_stream_) {
    **trace_stream_ << mixin_scope;
  }
  for (Nonnull<Declaration*> m : mixin_decl->members()) {
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, mixin_scope, mixin_decl));
    CARBON_RETURN_IF_ERROR(CollectMember(mixin_decl, m));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking mixin " << mixin_decl->name()
                    << "\n";
  }
  return Success();
}

// EXPERIMENTAL MIXIN FEATURE
// Type checks the mixin mentioned in the mix declaration.
// TypeCheckMixinDeclaration ensures that the members of that mixin are
// available in collected_members_. The mixin members are then collected as
// members of the enclosing class or mixin declaration.
auto TypeChecker::TypeCheckMixDeclaration(
    Nonnull<MixDeclaration*> mix_decl, const ImplScope& impl_scope,
    std::optional<Nonnull<const Declaration*>> enclosing_decl)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** checking " << *mix_decl << "\n";
  }
  // TODO(darshal): Check if the imports (interface mentioned in the 'for'
  // clause) of the mixin being mixed are being impl'd in the enclosed
  // class/mixin declaration This raises the question of how to handle impl
  // declarations in mixin declarations

  CARBON_CHECK(enclosing_decl.has_value());
  Nonnull<const Declaration*> encl_decl = enclosing_decl.value();
  const auto& mixin_decl = mix_decl->mixin_value().declaration();
  CARBON_RETURN_IF_ERROR(TypeCheckMixinDeclaration(&mixin_decl, impl_scope));
  CollectedMembersMap& mix_members = FindCollectedMembers(&mixin_decl);

  // Merge members collected in the enclosing declaration with the members
  // collected for the mixin declaration associated with the mix declaration
  for (auto [mix_member_name, mix_member] : mix_members) {
    CARBON_RETURN_IF_ERROR(CollectMember(encl_decl, mix_member));
  }

  if (trace_stream_) {
    **trace_stream_ << "** finished checking " << *mix_decl << "\n";
  }

  return Success();
}

auto TypeChecker::DeclareConstraintTypeDeclaration(
    Nonnull<ConstraintTypeDeclaration*> constraint_decl,
    const ScopeInfo& scope_info) -> ErrorOr<Success> {
  CARBON_CHECK(
      isa<InterfaceDeclaration, ConstraintDeclaration>(constraint_decl))
      << "unexpected kind of constraint type declaration";
  bool is_interface = isa<InterfaceDeclaration>(constraint_decl);

  if (trace_stream_) {
    **trace_stream_ << "** declaring ";
    constraint_decl->PrintID(**trace_stream_);
    **trace_stream_ << "\n";
  }
  ImplScope constraint_scope;
  constraint_scope.AddParent(scope_info.innermost_scope);

  // Type-check the parameters and find the set of bindings that are in scope.
  std::vector<Nonnull<const GenericBinding*>> bindings = scope_info.bindings;
  if (constraint_decl->params().has_value()) {
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(*constraint_decl->params(),
                                            std::nullopt, constraint_scope,
                                            ValueCategory::Let));
    if (trace_stream_) {
      **trace_stream_ << constraint_scope;
    }
    CollectGenericBindingsInPattern(*constraint_decl->params(), bindings);
  }

  // Form the full symbolic type of the interface or named constraint. This is
  // used as part of the value of associated constants, if they're referenced
  // within their interface, and as the symbolic value of the declaration.
  Nonnull<const Value*> constraint_type;
  if (is_interface) {
    constraint_type = arena_->New<InterfaceType>(
        cast<InterfaceDeclaration>(constraint_decl),
        Bindings::SymbolicIdentity(arena_, bindings));
  } else {
    constraint_type = arena_->New<NamedConstraintType>(
        cast<ConstraintDeclaration>(constraint_decl),
        Bindings::SymbolicIdentity(arena_, bindings));
  }

  // Set up the meaning of the declaration when used as an identifier.
  if (constraint_decl->params().has_value()) {
    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(constraint_decl,
                                             *constraint_decl->params());
    constraint_decl->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
    constraint_decl->set_constant_value(param_name);
  } else {
    constraint_decl->set_static_type(arena_->New<TypeType>());
    constraint_decl->set_constant_value(constraint_type);
  }

  // Set the type of Self to be the instantiated constraint type.
  Nonnull<SelfDeclaration*> self_type = constraint_decl->self_type();
  self_type->set_static_type(arena_->New<TypeType>());
  self_type->set_constant_value(constraint_type);

  // Build a constraint corresponding to this constraint type.
  ConstraintTypeBuilder::PrepareSelfBinding(arena_, constraint_decl->self());
  ConstraintTypeBuilder builder(arena_, constraint_decl->self());
  ConstraintTypeBuilder::ConstraintsInScopeTracker constraint_tracker;
  constraint_decl->self()->set_static_type(constraint_type);

  // Lookups into this constraint type look in this declaration.
  builder.AddLookupContext({.context = constraint_type});

  // If this is an interface, this is a symbolic witness that Self implements
  // this interface.
  std::optional<Nonnull<const Witness*>> iface_impl_witness;
  if (is_interface) {
    // The impl constraint says only that the direct members of the interface
    // are available. For any indirect constraints, we need to add separate
    // entries to the constraint type. This ensures that all indirect
    // constraints are lifted to the top level so they can be accessed directly
    // and resolved independently if necessary.
    int index = builder.AddImplConstraint(
        {.type = builder.GetSelfType(),
         .interface = cast<InterfaceType>(constraint_type)});
    iface_impl_witness =
        MakeConstraintWitnessAccess(builder.GetSelfWitness(), index);
  }

  ScopeInfo constraint_scope_info =
      ScopeInfo::ForNonClassScope(&constraint_scope);
  for (Nonnull<Declaration*> m : constraint_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, constraint_scope_info));

    // TODO: This should probably live in `DeclareDeclaration`, but it needs
    // to update state that's not available from there.
    switch (m->kind()) {
      case DeclarationKind::InterfaceExtendsDeclaration: {
        // For an `extends C;` declaration, add `Self is C` to our constraint.
        auto* extends = cast<InterfaceExtendsDeclaration>(m);
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> base,
            TypeCheckTypeExp(extends->base(), constraint_scope));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> constraint_type,
            ConvertToConstraintType(m->source_loc(), "extends declaration",
                                    base));
        builder.AddAndSubstitute(*this, constraint_type, builder.GetSelfType(),
                                 builder.GetSelfWitness(), Bindings(),
                                 /*add_lookup_contexts=*/true);
        break;
      }

      case DeclarationKind::InterfaceImplDeclaration: {
        // For an `impl X as Y;` declaration, add `X is Y` to our constraint.
        auto* impl = cast<InterfaceImplDeclaration>(m);
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> impl_type,
            TypeCheckTypeExp(impl->impl_type(), constraint_scope));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> constraint,
            TypeCheckTypeExp(impl->constraint(), constraint_scope));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> constraint_type,
            ConvertToConstraintType(m->source_loc(), "impl as declaration",
                                    constraint));
        builder.AddAndSubstitute(*this, constraint_type, impl_type,
                                 builder.GetSelfWitness(), Bindings(),
                                 /*add_lookup_contexts=*/false);
        break;
      }

      case DeclarationKind::AssociatedConstantDeclaration: {
        auto* assoc = cast<AssociatedConstantDeclaration>(m);
        if (!is_interface) {
          // TODO: Template constraints can have associated constants.
          return ProgramError(assoc->source_loc())
                 << "associated constant not permitted in named constraint";
        }

        CARBON_RETURN_IF_ERROR(TypeCheckGenericBinding(
            assoc->binding(), "associated constant", constraint_scope));
        Nonnull<const Value*> constraint = &assoc->binding().static_type();
        assoc->set_static_type(constraint);

        // The constant value is used if the constant is named later in the
        // same constraint type. Note that this differs from the symbolic
        // identity of the binding, which was set in TypeCheckGenericBinding to
        // a VariableType naming the binding so that .Self resolves to the
        // binding itself.
        auto* assoc_value = arena_->New<AssociatedConstant>(
            &constraint_decl->self()->value(),
            cast<InterfaceType>(constraint_type), assoc, *iface_impl_witness);
        assoc->set_constant_value(assoc_value);

        // The type specified for the associated constant becomes a
        // constraint for the constraint type: `let X:! Interface` adds a
        // `Self.X is Interface` constraint that `impl`s must satisfy and users
        // of the constraint type can rely on.
        if (auto* constraint_type = dyn_cast<ConstraintType>(constraint)) {
          builder.AddAndSubstitute(*this, constraint_type, assoc_value,
                                   builder.GetSelfWitness(), Bindings(),
                                   /*add_lookup_contexts=*/false);
        }
        break;
      }

      case DeclarationKind::FunctionDeclaration: {
        if (!is_interface) {
          // TODO: Template constraints can have associated functions.
          return ProgramError(m->source_loc())
                 << "associated function not permitted in named constraint";
        }
        break;
      }

      default: {
        CARBON_FATAL()
            << "unexpected declaration in constraint type declaration:\n"
            << *m;
        break;
      }
    }

    // Add any new impl constraints to the scope.
    builder.BringConstraintsIntoScope(*this, &constraint_scope,
                                      &constraint_tracker);
  }

  constraint_decl->set_constraint_type(std::move(builder).Build());

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring ";
    constraint_decl->PrintID(**trace_stream_);
    **trace_stream_ << "\n";
  }
  return Success();
}

auto TypeChecker::TypeCheckConstraintTypeDeclaration(
    Nonnull<ConstraintTypeDeclaration*> constraint_decl,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** checking ";
    constraint_decl->PrintID(**trace_stream_);
    **trace_stream_ << "\n";
  }
  ImplScope constraint_scope;
  constraint_scope.AddParent(&impl_scope);
  if (constraint_decl->params().has_value()) {
    BringPatternImplsIntoScope(*constraint_decl->params(), constraint_scope);
  }
  if (trace_stream_) {
    **trace_stream_ << constraint_scope;
  }
  for (Nonnull<Declaration*> m : constraint_decl->members()) {
    CARBON_RETURN_IF_ERROR(
        TypeCheckDeclaration(m, constraint_scope, constraint_decl));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking ";
    constraint_decl->PrintID(**trace_stream_);
    **trace_stream_ << "\n";
  }
  return Success();
}

auto TypeChecker::CheckImplIsDeducible(
    SourceLocation source_loc, Nonnull<const Value*> impl_type,
    Nonnull<const InterfaceType*> impl_iface,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    const ImplScope& /*impl_scope*/) -> ErrorOr<Success> {
  ArgumentDeduction deduction(source_loc, "impl", deduced_bindings,
                              trace_stream_);
  CARBON_RETURN_IF_ERROR(deduction.Deduce(impl_type, impl_type,
                                          /*allow_implicit_conversion=*/false));
  CARBON_RETURN_IF_ERROR(deduction.Deduce(impl_iface, impl_iface,
                                          /*allow_implicit_conversion=*/false));
  if (auto not_deduced = deduction.FindUndeducedBinding()) {
    return ProgramError(source_loc)
           << "parameter `" << **not_deduced << "` is not deducible from `impl "
           << *impl_type << " as " << *impl_iface << "`";
  }
  return Success();
}

auto TypeChecker::CheckImplIsComplete(Nonnull<const InterfaceType*> iface_type,
                                      Nonnull<const ImplDeclaration*> impl_decl,
                                      Nonnull<const Value*> self_type,
                                      Nonnull<const Witness*> /*self_witness*/,
                                      Nonnull<const Witness*> iface_witness,
                                      const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  const auto& iface_decl = iface_type->declaration();
  for (Nonnull<Declaration*> m : iface_decl.members()) {
    if (auto* assoc = dyn_cast<AssociatedConstantDeclaration>(m)) {
      // An associated constant must be given a value.
      if (!LookupRewrite(impl_decl->constraint_type(), iface_type, assoc)) {
        return ProgramError(impl_decl->source_loc())
               << "implementation doesn't provide a concrete value for "
               << *iface_type << "." << assoc->binding().name();
      }
    } else if (isa<InterfaceImplDeclaration, InterfaceExtendsDeclaration>(m)) {
      // These get translated into constraints so there's nothing we need to
      // check here.
    } else {
      // Every member function must be declared.
      std::optional<std::string_view> mem_name = GetName(*m);
      CARBON_CHECK(mem_name.has_value()) << "unnamed interface member " << *m;

      std::optional<Nonnull<const Declaration*>> mem =
          FindMember(*mem_name, impl_decl->members());
      if (!mem.has_value()) {
        return ProgramError(impl_decl->source_loc())
               << "implementation missing " << *mem_name;
      }

      Bindings bindings = iface_type->bindings();
      bindings.Add(iface_decl.self(), self_type, iface_witness);
      Nonnull<const Value*> iface_mem_type =
          Substitute(bindings, &m->static_type());
      // TODO: How should the signature in the implementation be permitted
      // to differ from the signature in the interface?
      CARBON_RETURN_IF_ERROR(
          ExpectExactType((*mem)->source_loc(), "member of implementation",
                          iface_mem_type, &(*mem)->static_type(), impl_scope));
    }
  }
  return Success();
}

auto TypeChecker::CheckAndAddImplBindings(
    Nonnull<const ImplDeclaration*> impl_decl, Nonnull<const Value*> impl_type,
    Nonnull<const Witness*> self_witness, Nonnull<const Witness*> impl_witness,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    const ScopeInfo& scope_info) -> ErrorOr<Success> {
  // Each interface that is a lookup context is required to be implemented by
  // the impl members. Other constraints are required to be satisfied by
  // either those impls or impls available elsewhere.
  Nonnull<const ConstraintType*> constraint = impl_decl->constraint_type();
  for (auto lookup : constraint->lookup_contexts()) {
    if (const auto* iface_type = dyn_cast<InterfaceType>(lookup.context)) {
      CARBON_RETURN_IF_ERROR(ExpectCompleteType(
          impl_decl->source_loc(), "impl declaration", iface_type));
      CARBON_RETURN_IF_ERROR(
          CheckImplIsDeducible(impl_decl->source_loc(), impl_type, iface_type,
                               deduced_bindings, *scope_info.innermost_scope));

      // Bring the associated constant values for this interface into scope. We
      // know that if the methods of this interface are used, they will use
      // these values.
      ImplScope iface_scope;
      iface_scope.AddParent(scope_info.innermost_scope);
      BringAssociatedConstantsIntoScope(constraint, impl_type, iface_type,
                                        iface_scope);

      // Compute a witness that the implementing type implements this interface
      // by resolving the interface constraint in a context where this `impl`
      // is used for it. We don't actually want the whole `impl` to be in
      // scope, though, because it could be partially specialized.
      Nonnull<const Witness*> iface_witness;
      {
        ImplScope impl_scope;
        impl_scope.AddParent(&iface_scope);
        impl_scope.Add(impl_decl->constraint_type(), impl_type, impl_witness,
                       *this);
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> iface_constraint,
            ConvertToConstraintType(impl_decl->source_loc(), "impl declaration",
                                    iface_type));
        CARBON_ASSIGN_OR_RETURN(
            iface_witness, impl_scope.Resolve(iface_constraint, impl_type,
                                              impl_decl->source_loc(), *this));
      }

      CARBON_RETURN_IF_ERROR(CheckImplIsComplete(iface_type, impl_decl,
                                                 impl_type, self_witness,
                                                 iface_witness, iface_scope));

      // TODO: We should do this either before checking any interface or after
      // checking all of them, so that the order of lookup contexts doesn't
      // matter.
      scope_info.innermost_non_class_scope->Add(
          iface_type, deduced_bindings, impl_type, impl_decl->impl_bindings(),
          self_witness, *this);
    } else if (isa<NamedConstraintType>(lookup.context)) {
      // Nothing to check here, since a named constraint can't introduce any
      // associated entities.
    } else {
      // TODO: Add support for implementing `adapter`s.
      return ProgramError(impl_decl->source_loc())
             << "cannot implement a constraint whose lookup context includes "
             << *lookup.context;
    }
  }
  return Success();
}

auto TypeChecker::DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                         const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "declaring " << *impl_decl << "\n";
  }
  ImplScope impl_scope;
  impl_scope.AddParent(scope_info.innermost_scope);
  std::vector<Nonnull<const GenericBinding*>> generic_bindings =
      scope_info.bindings;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;

  // Bring the deduced parameters into scope.
  for (Nonnull<GenericBinding*> deduced : impl_decl->deduced_parameters()) {
    generic_bindings.push_back(deduced);
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(deduced, std::nullopt, impl_scope,
                                            ValueCategory::Let));
    CollectImplBindingsInPattern(deduced, impl_bindings);
  }
  impl_decl->set_impl_bindings(impl_bindings);

  // Check and interpret the impl_type
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> impl_type_value,
                          TypeCheckTypeExp(impl_decl->impl_type(), impl_scope));

  // Set `Self` to `impl_type`. We do this whether `Self` resolves to it or to
  // the `Self` from an enclosing scope. This needs to be done before
  // processing the interface, in case the interface expression uses `Self`.
  Nonnull<SelfDeclaration*> self = impl_decl->self();
  self->set_constant_value(impl_type_value);
  self->set_static_type(&impl_decl->impl_type()->static_type());

  // Check and interpret the interface.
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<const Value*> implemented_type,
      TypeCheckTypeExp(&impl_decl->interface(), impl_scope));
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<const ConstraintType*> implemented_constraint,
      ConvertToConstraintType(impl_decl->interface().source_loc(),
                              "impl declaration", implemented_type));

  // Substitute the given type for `.Self` to form the resolved constraint that
  // this `impl` implements.
  Nonnull<const ConstraintType*> constraint_type;
  {
    // TODO: Combine this with the SelfDeclaration.
    auto* self_binding = arena_->New<GenericBinding>(self->source_loc(), "Self",
                                                     impl_decl->impl_type());
    self_binding->set_symbolic_identity(impl_type_value);
    self_binding->set_value(impl_type_value);
    auto* impl_binding = arena_->New<ImplBinding>(self_binding->source_loc(),
                                                  self_binding, std::nullopt);
    impl_binding->set_symbolic_identity(
        arena_->New<BindingWitness>(impl_binding));
    self_binding->set_impl_binding(impl_binding);

    ConstraintTypeBuilder builder(arena_, self_binding, impl_binding);
    builder.AddAndSubstitute(*this, implemented_constraint, impl_type_value,
                             builder.GetSelfWitness(), Bindings(),
                             /*add_lookup_contexts=*/true);
    if (trace_stream_) {
      **trace_stream_ << "resolving impl constraint type for " << *impl_decl
                      << " from " << *implemented_constraint << "\n";
    }
    CARBON_RETURN_IF_ERROR(builder.Resolve(
        *this, impl_decl->interface().source_loc(), impl_scope));
    constraint_type = std::move(builder).Build();
    if (trace_stream_) {
      **trace_stream_ << "resolving impl constraint type as "
                      << *constraint_type << "\n";
    }
    impl_decl->set_constraint_type(constraint_type);
  }

  // Build the self witness. This is the witness used to demonstrate that
  // this impl implements its lookup contexts.
  auto* self_witness = arena_->New<ImplWitness>(
      impl_decl, Bindings::SymbolicIdentity(arena_, generic_bindings));

  // Compute a witness that the impl implements its constraint.
  Nonnull<const Witness*> impl_witness;
  {
    std::vector<EqualityConstraint> rewrite_constraints_as_equality_constraints;
    ImplScope self_impl_scope;
    self_impl_scope.AddParent(&impl_scope);

    // For each interface we're going to implement, this impl is the witness
    // that that interface is implemented.
    for (auto lookup : constraint_type->lookup_contexts()) {
      if (const auto* iface_type = dyn_cast<InterfaceType>(lookup.context)) {
        self_impl_scope.Add(iface_type, impl_type_value, self_witness, *this);
      }
    }

    // This impl also provides all of the equalities from its rewrite
    // constraints.
    for (const auto& rewrite : constraint_type->rewrite_constraints()) {
      rewrite_constraints_as_equality_constraints.push_back(
          {.values = {rewrite.constant, rewrite.converted_replacement}});
    }
    for (const auto& eq : rewrite_constraints_as_equality_constraints) {
      self_impl_scope.AddEqualityConstraint(&eq);
    }

    // Ensure that's enough for our interface to be satisfied.
    CARBON_ASSIGN_OR_RETURN(
        impl_witness, self_impl_scope.Resolve(constraint_type, impl_type_value,
                                              impl_decl->source_loc(), *this));
  }

  // Declare the impl members. An `impl` behaves like a class scope.
  ScopeInfo impl_scope_info =
      ScopeInfo::ForClassScope(scope_info, &impl_scope, generic_bindings);
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, impl_scope_info));
  }

  // Create the implied impl bindings.
  CARBON_RETURN_IF_ERROR(
      CheckAndAddImplBindings(impl_decl, impl_type_value, self_witness,
                              impl_witness, generic_bindings, impl_scope_info));

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring impl " << *impl_decl->impl_type()
                    << " as " << impl_decl->interface() << "\n";
  }
  return Success();
}

void TypeChecker::BringAssociatedConstantsIntoScope(
    Nonnull<const ConstraintType*> constraint, Nonnull<const Value*> self,
    Nonnull<const InterfaceType*> interface, ImplScope& scope) {
  std::set<Nonnull<const AssociatedConstantDeclaration*>> assocs_in_interface;
  for (Nonnull<Declaration*> m : interface->declaration().members()) {
    if (auto* assoc = dyn_cast<AssociatedConstantDeclaration>(m)) {
      assocs_in_interface.insert(assoc);
    }
  }

  for (const auto& eq : constraint->equality_constraints()) {
    for (Nonnull<const Value*> value : eq.values) {
      if (const auto* assoc = dyn_cast<AssociatedConstant>(value)) {
        if (assocs_in_interface.count(&assoc->constant()) &&
            ValueEqual(&assoc->base(), self, std::nullopt) &&
            ValueEqual(&assoc->interface(), interface, std::nullopt)) {
          // This equality constraint mentions an associated constant that is
          // part of interface. Bring it into scope.
          scope.AddEqualityConstraint(&eq);
          break;
        }
      }
    }
  }

  // TODO: Find a way to bring rewrite constraints into scope.
}

auto TypeChecker::TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                           const ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << *impl_decl << "\n";
  }

  Nonnull<const Value*> self = *impl_decl->self()->constant_value();
  Nonnull<const ConstraintType*> constraint = impl_decl->constraint_type();

  // Bring the impls from the parameters into scope.
  ImplScope impl_scope;
  impl_scope.AddParent(&enclosing_scope);
  BringImplsIntoScope(impl_decl->impl_bindings(), impl_scope);
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    CARBON_ASSIGN_OR_RETURN(
        ConstraintLookupResult result,
        LookupInConstraint(m->source_loc(), "member impl declaration",
                           constraint, GetName(*m).value()));

    // Bring the associated constant values for the interface that this method
    // implements part of into scope.
    ImplScope member_scope;
    member_scope.AddParent(&impl_scope);
    BringAssociatedConstantsIntoScope(constraint, self, result.interface,
                                      member_scope);

    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, member_scope, impl_decl));
  }
  if (trace_stream_) {
    **trace_stream_ << "finished checking impl\n";
  }
  return Success();
}

auto TypeChecker::DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                           const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  ImplScope choice_scope;
  choice_scope.AddParent(scope_info.innermost_scope);
  std::vector<Nonnull<const GenericBinding*>> bindings = scope_info.bindings;
  if (choice->type_params().has_value()) {
    Nonnull<TuplePattern*> type_params = *choice->type_params();
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(type_params, std::nullopt,
                                            choice_scope, ValueCategory::Let));
    CollectGenericBindingsInPattern(type_params, bindings);
    if (trace_stream_) {
      **trace_stream_ << choice_scope;
    }
  }

  std::vector<NamedValue> alternatives;
  for (Nonnull<AlternativeSignature*> alternative : choice->alternatives()) {
    CARBON_ASSIGN_OR_RETURN(auto signature,
                            TypeCheckTypeExp(&alternative->signature(),
                                             *scope_info.innermost_scope));
    alternatives.push_back({alternative->name(), signature});
  }
  choice->set_members(alternatives);
  if (choice->type_params().has_value()) {
    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(choice, *choice->type_params());
    choice->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
    choice->set_constant_value(param_name);
    return Success();
  }

  auto* ct = arena_->New<ChoiceType>(
      choice, Bindings::SymbolicIdentity(arena_, bindings));

  choice->set_static_type(arena_->New<TypeType>());
  choice->set_constant_value(ct);
  return Success();
}

auto TypeChecker::TypeCheckChoiceDeclaration(
    Nonnull<ChoiceDeclaration*> /*choice*/, const ImplScope& /*impl_scope*/)
    -> ErrorOr<Success> {
  // Nothing to do here, but perhaps that will change in the future?
  return Success();
}

static auto IsValidTypeForAliasTarget(Nonnull<const Value*> type) -> bool {
  switch (type->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::MixinPseudoType:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::AlternativeValue:
    case Value::Kind::TupleValue:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
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
    case Value::Kind::TupleType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::AssociatedConstant:
      return false;

    case Value::Kind::FunctionType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
    case Value::Kind::TypeType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      return true;
  }
}

auto TypeChecker::DeclareAliasDeclaration(Nonnull<AliasDeclaration*> alias,
                                          const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  CARBON_RETURN_IF_ERROR(
      TypeCheckExp(&alias->target(), *scope_info.innermost_scope));

  if (!IsValidTypeForAliasTarget(&alias->target().static_type())) {
    return ProgramError(alias->source_loc())
           << "invalid target for alias declaration";
  }

  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> target,
                          InterpExp(&alias->target(), arena_, trace_stream_));

  alias->set_static_type(&alias->target().static_type());
  alias->set_constant_value(target);
  return Success();
}

auto TypeChecker::TypeCheck(AST& ast) -> ErrorOr<Success> {
  ImplScope impl_scope;
  ScopeInfo top_level_scope_info = ScopeInfo::ForNonClassScope(&impl_scope);

  // Track that `impl_scope` is the top-level `ImplScope`.
  llvm::SaveAndRestore<decltype(top_level_impl_scope_)>
      set_top_level_impl_scope(top_level_impl_scope_, &impl_scope);

  for (Nonnull<Declaration*> declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(
        DeclareDeclaration(declaration, top_level_scope_info));
    CARBON_RETURN_IF_ERROR(
        TypeCheckDeclaration(declaration, impl_scope, std::nullopt));
    // Check to see if this declaration is a builtin.
    // TODO: Only do this when type-checking the prelude.
    builtins_.Register(declaration);
  }
  CARBON_RETURN_IF_ERROR(TypeCheckExp(*ast.main_call, impl_scope));
  return Success();
}

auto TypeChecker::TypeCheckDeclaration(
    Nonnull<Declaration*> d, const ImplScope& impl_scope,
    std::optional<Nonnull<const Declaration*>> enclosing_decl)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << DeclarationKindName(d->kind()) << "\n";
  }
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      CARBON_RETURN_IF_ERROR(TypeCheckConstraintTypeDeclaration(
          &cast<ConstraintTypeDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      CARBON_RETURN_IF_ERROR(
          TypeCheckImplDeclaration(&cast<ImplDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::MatchFirstDeclaration: {
      auto* match_first = cast<MatchFirstDeclaration>(d);
      for (auto* impl : match_first->impls()) {
        impl->set_match_first(match_first);
        CARBON_RETURN_IF_ERROR(TypeCheckImplDeclaration(impl, impl_scope));
      }
      break;
    }
    case DeclarationKind::DestructorDeclaration:
    case DeclarationKind::FunctionDeclaration:
      CARBON_RETURN_IF_ERROR(TypeCheckCallableDeclaration(
          &cast<CallableDeclaration>(*d), impl_scope));
      break;
    case DeclarationKind::ClassDeclaration:
      CARBON_RETURN_IF_ERROR(
          TypeCheckClassDeclaration(&cast<ClassDeclaration>(*d), impl_scope));
      break;
    case DeclarationKind::MixinDeclaration: {
      CARBON_RETURN_IF_ERROR(
          TypeCheckMixinDeclaration(&cast<MixinDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::MixDeclaration: {
      CARBON_RETURN_IF_ERROR(TypeCheckMixDeclaration(
          &cast<MixDeclaration>(*d), impl_scope, enclosing_decl));
      break;
    }
    case DeclarationKind::ChoiceDeclaration:
      CARBON_RETURN_IF_ERROR(
          TypeCheckChoiceDeclaration(&cast<ChoiceDeclaration>(*d), impl_scope));
      break;
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      if (var.has_initializer()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&var.initializer(), impl_scope));
      }
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        return ProgramError(var.source_loc())
               << "Type of a top-level variable must be an expression.";
      }
      if (var.has_initializer()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_initializer,
            ImplicitlyConvert("initializer of variable", impl_scope,
                              &var.initializer(), &var.static_type()));
        var.set_initializer(converted_initializer);
      }
      break;
    }
    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration: {
      // Checked in DeclareConstraintTypeDeclaration.
      break;
    }
    case DeclarationKind::SelfDeclaration: {
      CARBON_FATAL() << "Unreachable TypeChecker `Self` declaration";
    }
    case DeclarationKind::AliasDeclaration: {
      break;
    }
  }
  d->set_is_type_checked();
  return Success();
}

auto TypeChecker::DeclareDeclaration(Nonnull<Declaration*> d,
                                     const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      auto& iface_decl = cast<ConstraintTypeDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(
          DeclareConstraintTypeDeclaration(&iface_decl, scope_info));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl_decl = cast<ImplDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareImplDeclaration(&impl_decl, scope_info));
      break;
    }
    case DeclarationKind::MatchFirstDeclaration: {
      for (auto* impl : cast<MatchFirstDeclaration>(d)->impls()) {
        CARBON_RETURN_IF_ERROR(DeclareImplDeclaration(impl, scope_info));
      }
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      auto& func_def = cast<CallableDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareCallableDeclaration(&func_def, scope_info));
      break;
    }
    case DeclarationKind::DestructorDeclaration: {
      auto& destructor_def = cast<CallableDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(
          DeclareCallableDeclaration(&destructor_def, scope_info));
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareClassDeclaration(&class_decl, scope_info));
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      auto& mixin_decl = cast<MixinDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareMixinDeclaration(&mixin_decl, scope_info));
      break;
    }
    case DeclarationKind::MixDeclaration: {
      auto& mix_decl = cast<MixDeclaration>(*d);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> mixin,
          InterpExp(&mix_decl.mixin(), arena_, trace_stream_));
      mix_decl.set_mixin_value(cast<MixinPseudoType>(mixin));
      const auto& mixin_decl = mix_decl.mixin_value().declaration();
      if (!mixin_decl.is_declared()) {
        return ProgramError(mix_decl.source_loc())
               << "incomplete mixin `" << mixin_decl.name()
               << "` used in mix declaration";
      }
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareChoiceDeclaration(&choice, scope_info));
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      if (!llvm::isa<ExpressionPattern>(var.binding().type())) {
        return ProgramError(var.binding().type().source_loc())
               << "Expected expression for variable type";
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var.binding(), std::nullopt,
                                              *scope_info.innermost_scope,
                                              var.value_category()));
      CARBON_RETURN_IF_ERROR(ExpectCompleteType(
          var.source_loc(), "type of variable", &var.binding().static_type()));
      var.set_static_type(&var.binding().static_type());
      break;
    }

    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration: {
      // The semantic effects are handled by DeclareConstraintTypeDeclaration.
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      CARBON_FATAL() << "Unreachable TypeChecker declare `Self` declaration";
    }

    case DeclarationKind::AliasDeclaration: {
      auto& alias = cast<AliasDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareAliasDeclaration(&alias, scope_info));
      break;
    }
  }
  d->set_is_declared();
  return Success();
}

auto TypeChecker::FindMemberWithParents(
    std::string_view name, Nonnull<const NominalClassType*> enclosing_class)
    -> ErrorOr<std::optional<
        std::tuple<Nonnull<const Value*>, Nonnull<const Declaration*>,
                   Nonnull<const NominalClassType*>>>> {
  CARBON_ASSIGN_OR_RETURN(
      const auto res,
      FindMixedMemberAndType(name, enclosing_class->declaration().members(),
                             enclosing_class));
  if (res.has_value()) {
    return {std::make_tuple(res->first, res->second, enclosing_class)};
  }
  if (const auto base = enclosing_class->base(); base.has_value()) {
    return FindMemberWithParents(name, base.value());
  }
  return {std::nullopt};
}

auto TypeChecker::FindMixedMemberAndType(
    const std::string_view& name, llvm::ArrayRef<Nonnull<Declaration*>> members,
    const Nonnull<const Value*> enclosing_type)
    -> ErrorOr<std::optional<
        std::pair<Nonnull<const Value*>, Nonnull<const Declaration*>>>> {
  for (Nonnull<const Declaration*> member : members) {
    if (llvm::isa<MixDeclaration>(member)) {
      const auto& mix_decl = cast<MixDeclaration>(*member);
      Nonnull<const MixinPseudoType*> mixin = &mix_decl.mixin_value();
      CARBON_ASSIGN_OR_RETURN(
          const auto res,
          FindMixedMemberAndType(name, mixin->declaration().members(), mixin));
      if (res.has_value()) {
        if (isa<NominalClassType>(enclosing_type)) {
          Bindings temp_map;
          // TODO: What is the type of Self? Do we ever need a witness?
          temp_map.Add(mixin->declaration().self(), enclosing_type,
                       std::nullopt);
          const auto* const mix_member_type =
              Substitute(temp_map, res.value().first);
          return {std::make_pair(mix_member_type, res.value().second)};
        } else {
          return res;
        }
      }

    } else if (std::optional<std::string_view> mem_name = GetName(*member);
               mem_name.has_value()) {
      if (*mem_name == name) {
        return {std::make_pair(&member->static_type(), member)};
      }
    }
  }

  return {std::nullopt};
}

auto TypeChecker::CollectMember(Nonnull<const Declaration*> enclosing_decl,
                                Nonnull<const Declaration*> member_decl)
    -> ErrorOr<Success> {
  CARBON_CHECK(isa<MixinDeclaration>(enclosing_decl) ||
               isa<ClassDeclaration>(enclosing_decl))
      << "Can't collect members for " << *enclosing_decl;
  auto member_name = GetName(*member_decl);
  if (!member_name.has_value()) {
    // No need to collect members without a name
    return Success();
  }
  auto encl_decl_name = GetName(*enclosing_decl);
  CARBON_CHECK(encl_decl_name.has_value());
  auto enclosing_decl_name = encl_decl_name.value();
  auto enclosing_decl_loc = enclosing_decl->source_loc();
  CollectedMembersMap& encl_members = FindCollectedMembers(enclosing_decl);
  auto [it, inserted] = encl_members.insert({member_name.value(), member_decl});
  if (!inserted) {
    if (member_decl == it->second) {
      return ProgramError(enclosing_decl_loc)
             << "Member named " << member_name.value() << " (declared at "
             << member_decl->source_loc() << ")"
             << " is being mixed multiple times into " << enclosing_decl_name;
    } else {
      return ProgramError(enclosing_decl_loc)
             << "Member named " << member_name.value() << " (declared at "
             << member_decl->source_loc() << ") cannot be mixed into "
             << enclosing_decl_name
             << " because it clashes with an existing member"
             << " with the same name (declared at " << it->second->source_loc()
             << ")";
    }
  }
  return Success();
}

auto TypeChecker::FindCollectedMembers(Nonnull<const Declaration*> decl)
    -> CollectedMembersMap& {
  switch (decl->kind()) {
    case DeclarationKind::MixinDeclaration:
    case DeclarationKind::ClassDeclaration: {
      auto it = collected_members_.find(decl);
      CARBON_CHECK(it != collected_members_.end());
      return it->second;
    }
    default:
      CARBON_FATAL() << "Can't collect members for " << *decl;
  }
}

}  // namespace Carbon
