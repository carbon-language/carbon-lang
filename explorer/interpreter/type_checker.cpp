// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_checker.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "common/error.h"
#include "common/ostream.h"
#include "explorer/ast/declaration.h"
#include "explorer/common/arena.h"
#include "explorer/common/error_builders.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/interpreter.h"
#include "explorer/interpreter/pattern_analysis.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

namespace Carbon {

struct TypeChecker::SingleStepEqualityContext : public EqualityContext {
 public:
  SingleStepEqualityContext(Nonnull<const TypeChecker*> type_checker,
                            Nonnull<const ImplScope*> impl_scope)
      : type_checker_(type_checker), impl_scope_(impl_scope) {}

  // Visits the values that are equal to the given value and a single step away
  // according to an equality constraint that is either scope or within a final
  // impl corresponding to an associated constant. Stops and returns `false` if
  // the visitor returns `false`, otherwise returns `true`.
  auto VisitEqualValues(Nonnull<const Value*> value,
                        llvm::function_ref<bool(Nonnull<const Value*>)> visitor)
      const -> bool override {
    if (type_checker_->trace_stream_) {
      **type_checker_->trace_stream_ << "looking for values equal to " << *value
                                     << " in\n"
                                     << *impl_scope_;
    }

    if (!impl_scope_->VisitEqualValues(value, visitor)) {
      return false;
    }

    // Also look up and visit the corresponding impl if this is an associated
    // constant.
    if (auto* assoc = dyn_cast<AssociatedConstant>(value)) {
      // Perform an impl lookup to see if we can resolve this constant.
      // The source location doesn't matter, we're discarding the diagnostics.
      if (auto* impl_witness = dyn_cast<ImplWitness>(&assoc->witness())) {
        // Instantiate the impl to find the concrete constraint it implements.
        Nonnull<const ConstraintType*> constraint =
            impl_witness->declaration().constraint_type();
        constraint = cast<ConstraintType>(
            type_checker_->Substitute(impl_witness->bindings(), constraint));
        if (type_checker_->trace_stream_) {
          **type_checker_->trace_stream_ << "found constraint " << *constraint
                                         << " for associated constant "
                                         << *assoc << "\n";
        }

        // Look for the value of this constant within that constraint.
        if (!constraint->VisitEqualValues(value, visitor)) {
          return false;
        }
      } else {
        if (type_checker_->trace_stream_) {
          **type_checker_->trace_stream_
              << "Could not resolve associated constant " << *assoc << ": "
              << "witness " << assoc->witness()
              << " depends on a generic parameter\n";
        }
      }
    }

    return true;
  }

 private:
  Nonnull<const TypeChecker*> type_checker_;
  Nonnull<const ImplScope*> impl_scope_;
};

static void SetValue(Nonnull<Pattern*> pattern, Nonnull<const Value*> value) {
  // TODO: find some way to CHECK that `value` is identical to pattern->value(),
  // if it's already set. Unclear if `ValueEqual` is suitable, because it
  // currently focuses more on "real" values, and disallows the pseudo-values
  // like `BindingPlaceholderValue` that we get in pattern evaluation.
  if (!pattern->has_value()) {
    pattern->set_value(value);
  }
}

auto TypeChecker::IsSameType(Nonnull<const Value*> type1,
                             Nonnull<const Value*> type2,
                             const ImplScope& impl_scope) const -> bool {
  SingleStepEqualityContext equality_ctx(this, &impl_scope);
  return TypeEqual(type1, type2, &equality_ctx);
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
    case Value::Kind::MixinPseudoType:
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
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
      // A value of one of these types is itself always a type.
      return true;
  }
}

// Returns whether the value is a valid result from a type expression,
// as opposed to a non-type value.
// `auto` is not considered a type by the function if `concrete` is false.
static auto IsType(Nonnull<const Value*> value, bool concrete = false) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
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
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
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
    case Value::Kind::MixinPseudoType:
    case Value::Kind::TypeOfMixinPseudoType:
      // Mixin type is a second-class type that cannot be used
      // within a type annotation expression.
      return false;
  }
}

static auto ExpectIsType(SourceLocation source_loc, Nonnull<const Value*> value)
    -> ErrorOr<Success> {
  if (!IsType(value)) {
    return ProgramError(source_loc) << "Expected a type, but got " << *value;
  } else {
    return Success();
  }
}

// Expect that a type is complete. Issue a diagnostic if not.
static auto ExpectCompleteType(SourceLocation source_loc,
                               std::string_view context,
                               Nonnull<const Value*> type) -> ErrorOr<Success> {
  CARBON_RETURN_IF_ERROR(ExpectIsType(source_loc, type));

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
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::MixinPseudoType:
    case Value::Kind::TypeOfMixinPseudoType:
      CARBON_FATAL() << "should not see non-type values";

    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::StringType:
    case Value::Kind::PointerType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::StructType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::AssociatedConstant: {
      // These types are always complete.
      return Success();
    }

    case Value::Kind::StaticArrayType:
      // TODO: This should probably be complete only if the element type is
      // complete.
      return Success();

    case Value::Kind::TupleValue: {
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
static auto IsConcreteType(Nonnull<const Value*> value) -> bool {
  return IsType(value, /*concrete=*/true);
}

auto TypeChecker::ExpectIsConcreteType(SourceLocation source_loc,
                                       Nonnull<const Value*> value)
    -> ErrorOr<Success> {
  if (!IsConcreteType(value)) {
    return ProgramError(source_loc) << "Expected a type, but got " << *value;
  } else {
    return Success();
  }
}

// Returns the named field, or None if not found.
static auto FindField(llvm::ArrayRef<NamedValue> fields,
                      const std::string& field_name)
    -> std::optional<NamedValue> {
  auto it = std::find_if(
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
  if (source_fields.size() != destination_fields.size()) {
    return false;
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
        field_types.push_back(
            {.name = var.binding().name(), .value = field_type});
        break;
      }
      default:
        break;
    }
  }
  return field_types;
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
          if (FieldTypesImplicitlyConvertible(
                  cast<StructType>(*source).fields(),
                  FieldTypes(cast<NominalClassType>(*destination)),
                  impl_scope)) {
            return true;
          }
          break;
        default:
          break;
      }
      break;
    case Value::Kind::TupleValue: {
      const auto& source_tuple = cast<TupleValue>(*source);
      switch (destination->kind()) {
        case Value::Kind::TupleValue: {
          const auto& destination_tuple = cast<TupleValue>(*destination);
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
        case Value::Kind::TypeType: {
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
      // TODO: This seems suspicious. Shouldn't this require that the type
      // implements the interface?
      if (isa<InterfaceType, ConstraintType>(destination)) {
        return true;
      }
      break;
    case Value::Kind::InterfaceType:
    case Value::Kind::ConstraintType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
      // TODO: These types should presumably also convert to constraint types.
      if (isa<TypeType>(destination)) {
        return true;
      }
      break;
    default:
      break;
  }

  // If we're not supposed to look for a user-defined conversion, we're done.
  if (!allow_user_defined_conversions) {
    return false;
  }

  // We didn't find a builtin implicit conversion. Try a user-defined one.
  // The source location doesn't matter, we're discarding the diagnostics.
  SourceLocation source_loc("", 0);
  ErrorOr<Nonnull<const InterfaceType*>> iface_type = GetBuiltinInterfaceType(
      source_loc, BuiltinInterfaceName{Builtins::ImplicitAs, destination});
  return iface_type.ok() &&
         impl_scope.Resolve(*iface_type, source, source_loc, *this).ok();
}

auto TypeChecker::ImplicitlyConvert(std::string_view context,
                                    const ImplScope& impl_scope,
                                    Nonnull<Expression*> source,
                                    Nonnull<const Value*> destination)
    -> ErrorOr<Nonnull<Expression*>> {
  Nonnull<const Value*> source_type = &source->static_type();

  // TODO: If a builtin conversion works, for now we don't create any
  // expression to do the conversion and rely on the interpreter to know how to
  // do it.
  // TODO: This doesn't work for cases of combined built-in and user-defined
  // conversion, such as converting a struct element via an `ImplicitAs` impl.
  if (IsImplicitlyConvertible(source_type, destination, impl_scope,
                              /*allow_user_defined_conversions=*/false)) {
    return source;
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
  auto* iface_decl = dyn_cast<InterfaceDeclaration>(builtin_decl);
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
      source_loc, iface_type, arena_->New<TypeOfInterfaceType>(iface_type),
      ValueCategory::Let);
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
      for (auto* binding : bindings_to_deduce) {
        **trace_stream_ << sep << *binding;
      }
      **trace_stream_ << "\n";
    }
    for (auto* binding : bindings_to_deduce) {
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
    for (auto* binding : deduced_bindings_in_order_) {
      llvm::ArrayRef<Nonnull<const Value*>> values =
          deduced_values_.find(binding)->second;
      if (values.empty()) {
        return binding;
      }
    }
    return std::nullopt;
  }

  // Adds a value for a binding that is not deduced but still participates in
  // substitution. For example, the `T` parameter in `fn F(T:! Type, x: T)`.
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
    if (!IsConcreteType(param)) {
      // Parameter type contains a nested `auto` and argument type isn't the
      // same kind of type.
      // TODO: This seems like something we should be able to accept.
      return ProgramError(source_loc_) << "type error in " << context_ << "\n"
                                       << "expected: " << *param << "\n"
                                       << "actual: " << *arg;
    }

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
    case Value::Kind::TupleValue: {
      if (arg->kind() != Value::Kind::TupleValue) {
        return handle_non_deduced_type();
      }
      const auto& param_tup = cast<TupleValue>(*param);
      const auto& arg_tup = cast<TupleValue>(*arg);
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
        NamedValue arg_field;
        if (allow_implicit_conversion) {
          if (std::optional<NamedValue> maybe_arg_field =
                  FindField(arg_struct.fields(), param_field.name)) {
            arg_field = *maybe_arg_field;
          } else {
            return diagnose_missing_field(arg_struct, param_field, true);
          }
        } else {
          if (i >= arg_struct.fields().size()) {
            return diagnose_missing_field(arg_struct, param_field, true);
          }
          arg_field = arg_struct.fields()[i];
          if (param_field.name != arg_field.name) {
            return ProgramError(source_loc_)
                   << "mismatch in field names, `" << param_field.name
                   << "` != `" << arg_field.name << "`";
          }
        }
        CARBON_RETURN_IF_ERROR(Deduce(param_field.value, arg_field.value,
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
      return Deduce(&cast<PointerType>(*param).type(),
                    &cast<PointerType>(*arg).type(),
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
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
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
  for (auto* binding : deduced_bindings_in_order_) {
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
    auto* first_value = values[0];
    for (auto* value : values) {
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
  // TODO: This is not the case for `fn F(T:! Type, u: (V:! ImplicitAs(T)))`.
  // However, we intend to disallow that.
  for (auto [binding, arg] : non_deduced_values_) {
    // Form the binding's resolved type and convert the argument expression to
    // it.
    const Value* binding_type = &binding->static_type();
    const Value* substituted_type =
        type_checker.Substitute(bindings, binding_type);
    if (!IsTypeOfType(substituted_type)) {
      CARBON_ASSIGN_OR_RETURN(
          arg, type_checker.ImplicitlyConvert(context_, impl_scope, arg,
                                              substituted_type));
    }

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
  for (auto& mismatch : non_deduced_mismatches_) {
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

// Builder for constraint types.
//
// This type supports incrementally building a constraint type by adding
// constraints one at a time, and will deduplicate the constraints as it goes.
//
// TODO: The deduplication here is very inefficient. We should use value
// canonicalization or hashing or similar to speed this up.
class TypeChecker::ConstraintTypeBuilder {
 public:
  ConstraintTypeBuilder(Nonnull<Arena*> arena, SourceLocation source_loc)
      : ConstraintTypeBuilder(arena, MakeSelfBinding(arena, source_loc)) {}
  ConstraintTypeBuilder(Nonnull<Arena*> arena,
                        Nonnull<GenericBinding*> self_binding)
      : self_binding_(PrepareSelfBinding(arena, self_binding)),
        impl_binding_(AddImplBinding(arena, self_binding_)) {}
  ConstraintTypeBuilder(Nonnull<Arena*> arena,
                        Nonnull<GenericBinding*> self_binding,
                        Nonnull<ImplBinding*> impl_binding)
      : self_binding_(self_binding), impl_binding_(impl_binding) {}

  // Produces a type that refers to the `.Self` type of the constraint.
  auto GetSelfType() const -> Nonnull<const Value*> {
    return &self_binding_->value();
  }

  // Gets a witness that `.Self` implements the eventual constraint type built
  // by this builder.
  auto GetSelfWitness() const -> Nonnull<const Witness*> {
    return cast<Witness>(impl_binding_->symbolic_identity().value());
  }

  // Adds an `impl` constraint -- `T is C` if not already present.
  // Returns the index of the impl constraint within the self witness.
  auto AddImplConstraint(ConstraintType::ImplConstraint impl) -> int {
    for (int i = 0; i != static_cast<int>(impl_constraints_.size()); ++i) {
      ConstraintType::ImplConstraint& existing = impl_constraints_[i];
      if (TypeEqual(existing.type, impl.type, std::nullopt) &&
          TypeEqual(existing.interface, impl.interface, std::nullopt)) {
        return i;
      }
    }
    impl_constraints_.push_back(std::move(impl));
    return impl_constraints_.size() - 1;
  }

  // Adds an equality constraint -- `A == B`.
  void AddEqualityConstraint(ConstraintType::EqualityConstraint equal) {
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

  auto AddRewriteConstraint(SourceLocation source_loc,
                            ConstraintType::RewriteConstraint rewrite)
      -> ErrorOr<Success> {
    for (ConstraintType::RewriteConstraint existing : rewrite_constraints_) {
      if (ValueEqual(existing.interface, rewrite.interface, std::nullopt) &&
          // TODO: Want a "declares same entity" check.
          GetName(*existing.constant) == GetName(*rewrite.constant)) {
        if (ValueEqual(&existing.replacement->value(),
                       &rewrite.replacement->value(), std::nullopt) &&
            TypeEqual(&existing.replacement->static_type(),
                      &rewrite.replacement->static_type(), std::nullopt)) {
          return Success();
        }
        return ProgramError(source_loc)
               << "multiple different rewrites for `.("
               << *rewrite.interface << "." << *GetName(*rewrite.constant)
               << ")`:\n"
               << "  " << *existing.replacement << "\n"
               << "  " << *rewrite.replacement;
      }
    }
    rewrite_constraints_.push_back(std::move(rewrite));
    return Success();
  }

  // Add a context for qualified name lookup, if not already present.
  void AddLookupContext(ConstraintType::LookupContext context) {
    for (ConstraintType::LookupContext existing : lookup_contexts_) {
      if (ValueEqual(existing.context, context.context, std::nullopt)) {
        return;
      }
    }
    lookup_contexts_.push_back(std::move(context));
  }

  // Adds all the constraints from another constraint type. The given value
  // `self` is substituted for `.Self`, typically specified in terms of this
  // constraint's self binding. The `self_witness` is the witness for the
  // resulting constraint, and can be `GetSelfWitness()`. The `bindings`
  // parameter specifies any additional substitutions to perform.
  auto AddAndSubstitute(const TypeChecker& type_checker,
                        Nonnull<const ConstraintType*> constraint,
                        Nonnull<const Value*> self,
                        Nonnull<const Witness*> self_witness,
                        const Bindings& bindings, bool add_lookup_contexts)
      -> ErrorOr<Success> {
    // First substitute into the impl bindings to form the full witness for
    // the constraint type.
    std::vector<Nonnull<const Witness*>> witnesses;
    for (const auto& impl_constraint : constraint->impl_constraints()) {
      Bindings local_bindings = bindings;
      local_bindings.Add(constraint->self_binding(), self,
                         type_checker.MakeConstraintWitness(
                             *constraint, witnesses,
                             constraint->self_binding()->source_loc()));
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
    local_bindings.Add(constraint->self_binding(), self,
                       type_checker.MakeConstraintWitness(
                           *constraint, std::move(witnesses),
                           constraint->self_binding()->source_loc()));

    // TODO: What happens if these rewrites appear in the impl constraints?
    // TODO: What happens if these rewrites appear in each other?
    for (const auto& rewrite_constraint : constraint->rewrite_constraints()) {
      auto* interface = cast<InterfaceType>(type_checker.Substitute(
          local_bindings, rewrite_constraint.interface));
      Nonnull<const Value*> value = type_checker.Substitute(
          local_bindings, &rewrite_constraint.replacement->value());
      Nonnull<const Value*> type = type_checker.Substitute(
          local_bindings, &rewrite_constraint.replacement->static_type());
      auto* replacement = type_checker.arena_->New<ValueLiteral>(
          rewrite_constraint.replacement->source_loc(), value, type,
          ValueCategory::Let);
      CARBON_RETURN_IF_ERROR(AddRewriteConstraint(
          replacement->source_loc(), {.interface = interface,
                                      .constant = rewrite_constraint.constant,
                                      .replacement = replacement}));
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

    return Success();
  }

  class ImplsInScopeTracker {
    friend class ConstraintTypeBuilder;

   private:
    int num_added = 0;
  };

  // Brings all the `impl`s accumulated so far into the given impl scope.
  // If this will be called more than once, an ImplsInScopeTracker can be
  // provided to avoid adding the same impls more than once.
  void BringImplsIntoScope(
      const TypeChecker& type_checker, Nonnull<ImplScope*> impl_scope,
      std::optional<Nonnull<ImplsInScopeTracker*>> tracker = std::nullopt) {
    llvm::ArrayRef<ConstraintType::ImplConstraint> impl_constraints =
        impl_constraints_;
    if (tracker) {
      impl_constraints = impl_constraints.drop_front((*tracker)->num_added);
      (*tracker)->num_added = impl_constraints_.size();
    }
    impl_scope->Add(impl_constraints, llvm::None, llvm::None, GetSelfWitness(),
                    type_checker);
  }

  // Converts the builder into a ConstraintType. Note that this consumes the
  // builder.
  auto Build(Nonnull<Arena*> arena) && -> Nonnull<const ConstraintType*> {
    // Rewrite `Self.X is Y` to `Replacement is Y` if we have a rewrite for
    // `Self.X`.
    // TODO: Properly apply rewrites throughout all the constraints. Check for
    // cycles. This is just a very short-term hack.
    for (auto& impl_constraint : impl_constraints_) {
      bool performed_rewrite;
      do {
        performed_rewrite = false;
        if (auto* assoc = dyn_cast<AssociatedConstant>(impl_constraint.type);
            assoc && ValueEqual(&assoc->base(), GetSelfType(), std::nullopt)) {
          for (const auto& rewrite : rewrite_constraints_) {
            if (&assoc->constant() == rewrite.constant &&
                ValueEqual(&assoc->interface(), rewrite.interface,
                           std::nullopt)) {
              impl_constraint.type = &rewrite.replacement->value();
              performed_rewrite = true;
            }
          }
        }
      } while (performed_rewrite);
    }

    // Create the new type.
    auto* result = arena->New<ConstraintType>(
        self_binding_, std::move(impl_constraints_),
        std::move(equality_constraints_), std::move(rewrite_constraints_),
        std::move(lookup_contexts_));
    // Update the impl binding to denote the constraint type itself.
    impl_binding_->set_interface(result);
    return result;
  }

 private:
  // Makes a generic binding to serve as the `.Self` of this constraint type.
  static auto MakeSelfBinding(Nonnull<Arena*> arena, SourceLocation source_loc)
      -> Nonnull<GenericBinding*> {
    // Note, the type-of-type here is a placeholder and isn't really
    // meaningful.
    return arena->New<GenericBinding>(source_loc, ".Self",
                                      arena->New<TypeTypeLiteral>(source_loc));
  }

  // Sets up a `.Self` binding to act as the self type of this constraint.
  static auto PrepareSelfBinding(Nonnull<Arena*> arena,
                                 Nonnull<GenericBinding*> self_binding)
      -> Nonnull<GenericBinding*> {
    Nonnull<const Value*> self = arena->New<VariableType>(self_binding);
    // TODO: Do we really need both of these?
    self_binding->set_symbolic_identity(self);
    self_binding->set_value(self);
    return self_binding;
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

 private:
  Nonnull<GenericBinding*> self_binding_;
  Nonnull<ImplBinding*> impl_binding_;
  std::vector<ConstraintType::ImplConstraint> impl_constraints_;
  std::vector<ConstraintType::EqualityConstraint> equality_constraints_;
  std::vector<ConstraintType::RewriteConstraint> rewrite_constraints_;
  std::vector<ConstraintType::LookupContext> lookup_contexts_;
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

  auto SubstituteIntoBindings =
      [&](Nonnull<const Bindings*> inner_bindings) -> Nonnull<const Bindings*> {
    BindingMap values;
    for (const auto& [name, value] : inner_bindings->args()) {
      values[name] = Substitute(bindings, value);
    }
    ImplWitnessMap witnesses;
    for (const auto& [name, value] : inner_bindings->witnesses()) {
      witnesses[name] = Substitute(bindings, value);
    }
    if (values == inner_bindings->args() &&
        witnesses == inner_bindings->witnesses()) {
      return inner_bindings;
    }
    return arena_->New<Bindings>(std::move(values), std::move(witnesses));
  };

  switch (type->kind()) {
    case Value::Kind::VariableType: {
      auto it = bindings.args().find(&cast<VariableType>(*type).binding());
      if (it == bindings.args().end()) {
        if (trace_stream_) {
          **trace_stream_ << "substitution: no value for binding " << *type
                          << ", leaving alone\n";
        }
        return type;
      } else {
        return it->second;
      }
    }
    case Value::Kind::AssociatedConstant: {
      const auto& assoc = cast<AssociatedConstant>(*type);
      Nonnull<const Value*> base = Substitute(bindings, &assoc.base());
      const auto* interface =
          cast<InterfaceType>(Substitute(bindings, &assoc.interface()));
      // If we're substituting into an associated constant, we may now be able
      // to rewrite it to a concrete value.
      if (std::optional<const ValueLiteral*> rewritten_value =
              LookupRewriteInTypeOf(base, interface, &assoc.constant())) {
        return &rewritten_value.value()->value();
      }
      const auto* witness =
          cast<Witness>(Substitute(bindings, &assoc.witness()));
      if (std::optional<const ValueLiteral*> rewritten_value =
              LookupRewriteInWitness(witness, interface, &assoc.constant())) {
        return &rewritten_value.value()->value();
      }
      return arena_->New<AssociatedConstant>(base, interface, &assoc.constant(),
                                             witness);
    }
    case Value::Kind::TupleValue: {
      std::vector<Nonnull<const Value*>> elts;
      for (const auto& elt : cast<TupleValue>(*type).elements()) {
        elts.push_back(Substitute(bindings, elt));
      }
      return arena_->New<TupleValue>(elts);
    }
    case Value::Kind::StructType: {
      std::vector<NamedValue> fields;
      for (const auto& [name, value] : cast<StructType>(*type).fields()) {
        auto new_type = Substitute(bindings, value);
        fields.push_back({name, new_type});
      }
      return arena_->New<StructType>(std::move(fields));
    }
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*type);
      SubstitutedGenericBindings subst_bindings(this, bindings);

      // Apply substitution to into generic parameters and deduced bindings.
      std::vector<FunctionType::GenericParameter> generic_parameters;
      for (const FunctionType::GenericParameter& gp :
           fn_type.generic_parameters()) {
        generic_parameters.push_back(
            {.index = gp.index,
             .binding =
                 subst_bindings.SubstituteIntoGenericBinding(gp.binding)});
      }
      std::vector<Nonnull<const GenericBinding*>> deduced_bindings;
      for (Nonnull<const GenericBinding*> gb : fn_type.deduced_bindings()) {
        deduced_bindings.push_back(
            subst_bindings.SubstituteIntoGenericBinding(gb));
      }

      // Apply substitution to parameter and return types and create the new
      // function type.
      auto param = Substitute(subst_bindings.bindings(), &fn_type.parameters());
      auto ret = Substitute(subst_bindings.bindings(), &fn_type.return_type());
      return arena_->New<FunctionType>(
          param, std::move(generic_parameters), ret,
          std::move(deduced_bindings),
          std::move(subst_bindings).TakeImplBindings());
    }
    case Value::Kind::PointerType: {
      return arena_->New<PointerType>(
          Substitute(bindings, &cast<PointerType>(*type).type()));
    }
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*type);
      Nonnull<const NominalClassType*> new_class_type =
          arena_->New<NominalClassType>(
              &class_type.declaration(),
              SubstituteIntoBindings(&class_type.bindings()));
      return new_class_type;
    }
    case Value::Kind::InterfaceType: {
      const auto& iface_type = cast<InterfaceType>(*type);
      Nonnull<const InterfaceType*> new_iface_type = arena_->New<InterfaceType>(
          &iface_type.declaration(),
          SubstituteIntoBindings(&iface_type.bindings()));
      return new_iface_type;
    }
    case Value::Kind::ConstraintType: {
      const auto& constraint = cast<ConstraintType>(*type);
      if (auto it = bindings.args().find(constraint.self_binding());
          it != bindings.args().end()) {
        // This happens when we substitute into the parameter type of a
        // function that takes a `T:! Constraint` parameter. In this case we
        // produce the new type-of-type of the replacement type.
        Nonnull<const Value*> type_of_type;
        if (auto* var_type = dyn_cast<VariableType>(it->second)) {
          type_of_type = &var_type->binding().static_type();
        } else if (auto* assoc_type =
                       dyn_cast<AssociatedConstant>(it->second)) {
          type_of_type = GetTypeForAssociatedConstant(assoc_type);
        } else {
          type_of_type = arena_->New<TypeType>();
        }
        if (trace_stream_) {
          **trace_stream_ << "substitution: self of constraint " << constraint
                          << " is substituted, new type of type is "
                          << *type_of_type << "\n";
        }
        // TODO: Should we keep any part of the old constraint -- rewrites,
        // equality constraints, etc?
        return type_of_type;
      }
      ConstraintTypeBuilder builder(arena_,
                                    constraint.self_binding()->source_loc());
      ErrorOr<Success> result =
          builder.AddAndSubstitute(*this, &constraint, builder.GetSelfType(),
                                   builder.GetSelfWitness(), bindings,
                                   /*add_lookup_contexts=*/true);
      // TODO: This appears to theoretically be possible, and should be handled
      // better.
      CARBON_CHECK(result.ok()) << "substitution into " << constraint
                                << " failed: " << result.error();
      Nonnull<const ConstraintType*> new_constraint =
          std::move(builder).Build(arena_);
      if (trace_stream_) {
        **trace_stream_ << "substitution: " << constraint << " => "
                        << *new_constraint << "\n";
      }
      return new_constraint;
    }
    case Value::Kind::ImplWitness: {
      const auto& witness = cast<ImplWitness>(*type);
      return arena_->New<ImplWitness>(
          &witness.declaration(), SubstituteIntoBindings(&witness.bindings()));
    }
    case Value::Kind::BindingWitness: {
      auto it =
          bindings.witnesses().find(cast<BindingWitness>(*type).binding());
      if (it == bindings.witnesses().end()) {
        if (trace_stream_) {
          **trace_stream_ << "substitution: no value for binding " << *type
                          << ", leaving alone\n";
        }
        return type;
      } else {
        return it->second;
      }
    }
    case Value::Kind::ConstraintWitness: {
      const auto& witness = cast<ConstraintWitness>(*type);
      std::vector<Nonnull<const Witness*>> witnesses;
      witnesses.reserve(witness.witnesses().size());
      for (auto* witness : witness.witnesses()) {
        witnesses.push_back(cast<Witness>(Substitute(bindings, witness)));
      }
      return arena_->New<ConstraintWitness>(std::move(witnesses));
    }
    case Value::Kind::ConstraintImplWitness: {
      const auto& witness = cast<ConstraintImplWitness>(*type);
      return ConstraintImplWitness::Make(
          arena_,
          cast<Witness>(Substitute(bindings, witness.constraint_witness())),
          witness.index());
    }
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::MixinPseudoType:
      return type;
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      // TODO: We should substitute into the value and produce a new type of
      // type for it.
      return type;
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
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::UninitializedValue:
      // This can happen when substituting into the arguments of a class or
      // interface.
      // TODO: Implement substitution for these cases.
      return type;
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
    const ConstraintType& constraint,
    std::vector<Nonnull<const Witness*>> impl_constraint_witnesses,
    SourceLocation source_loc) const -> Nonnull<const Witness*> {
  return arena_->New<ConstraintWitness>(std::move(impl_constraint_witnesses));
}

auto TypeChecker::MakeConstraintWitnessAccess(Nonnull<const Witness*> witness,
                                              int impl_offset) const
    -> Nonnull<const Witness*> {
  return ConstraintImplWitness::Make(arena_, witness, impl_offset);
}

auto TypeChecker::MakeConstraintForInterface(
    SourceLocation source_loc, Nonnull<const InterfaceType*> iface_type)
    -> ErrorOr<Nonnull<const ConstraintType*>> {
  CARBON_RETURN_IF_ERROR(
      ExpectCompleteType(source_loc, "constraint", iface_type));

  auto constraint_type = iface_type->declaration().constraint_type();
  CARBON_CHECK(constraint_type)
      << "complete interface should have a constraint type";

  if (iface_type->bindings().empty()) {
    return *constraint_type;
  }

  ConstraintTypeBuilder builder(arena_, source_loc);
  CARBON_RETURN_IF_ERROR(
      builder.AddAndSubstitute(*this, *constraint_type, builder.GetSelfType(),
                               builder.GetSelfWitness(), iface_type->bindings(),
                               /*add_lookup_contexts=*/true));
  return std::move(builder).Build(arena_);
}

auto TypeChecker::ConvertToConstraintType(SourceLocation source_loc,
                                          std::string_view context,
                                          Nonnull<const Value*> constraint)
    -> ErrorOr<Nonnull<const ConstraintType*>> {
  if (const auto* constraint_type = dyn_cast<ConstraintType>(constraint)) {
    return constraint_type;
  }
  if (const auto* iface_type = dyn_cast<InterfaceType>(constraint)) {
    return MakeConstraintForInterface(source_loc, iface_type);
  }
  if (isa<TypeType>(constraint)) {
    // TODO: Should we build this once and cache it?
    ConstraintTypeBuilder builder(arena_, source_loc);
    return std::move(builder).Build(arena_);
  }
  // TODO: Should we convert `TypeOfXType` into the constraint
  //       `Type where .Self == X`?

  return ProgramError(source_loc)
         << "expected a constraint in " << context << ", found " << *constraint;
}

auto TypeChecker::CombineConstraints(
    SourceLocation source_loc,
    llvm::ArrayRef<Nonnull<const ConstraintType*>> constraints)
    -> ErrorOr<Nonnull<const ConstraintType*>> {
  ConstraintTypeBuilder builder(arena_, source_loc);
  for (Nonnull<const ConstraintType*> constraint : constraints) {
    CARBON_RETURN_IF_ERROR(
        builder.AddAndSubstitute(*this, constraint, builder.GetSelfType(),
                                 builder.GetSelfWitness(), Bindings(),
                                 /*add_lookup_contexts=*/true));
  }
  return std::move(builder).Build(arena_);
}

auto TypeChecker::DeduceCallBindings(
    CallExpression& call, Nonnull<const Value*> params_type,
    llvm::ArrayRef<FunctionType::GenericParameter> generic_params,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
  llvm::ArrayRef<Nonnull<const Value*>> params =
      cast<TupleValue>(*params_type).elements();
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
  llvm::ArrayRef<ConstraintType::LookupContext> lookup_contexts =
      constraint_type->lookup_contexts();

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

// Look for a rewrite to use when naming the given interface member in a type
// declared with the given type-of-type.
static auto LookupRewrite(Nonnull<const Value*> type_of_type,
                          Nonnull<const InterfaceType*> interface,
                          Nonnull<const Declaration*> member)
    -> std::optional<const ValueLiteral*> {
  if (!isa<AssociatedConstantDeclaration>(member)) {
    return std::nullopt;
  }

  // Find the set of rewrites. Only ConstraintTypes have rewrites.
  // TODO: If we can ever see an InterfaceType here, we should convert it to a
  // constraint type.
  llvm::ArrayRef<ConstraintType::RewriteConstraint> rewrites;
  if (const auto* constraint_type = dyn_cast<ConstraintType>(type_of_type)) {
    rewrites = constraint_type->rewrite_constraints();
  }

  for (ConstraintType::RewriteConstraint rewrite : rewrites) {
    if (ValueEqual(interface, rewrite.interface, std::nullopt) &&
        // TODO: Using name comparison here seems brittle.
        GetName(*member) == GetName(*rewrite.constant)) {
      // A ConstraintType can only have one rewrite per (interface, member)
      // pair, so we don't need to check the rest.
      return rewrite.replacement;
    }
  }

  return std::nullopt;
}

auto TypeChecker::GetTypeForAssociatedConstant(
    Nonnull<const AssociatedConstant*> assoc) const -> Nonnull<const Value*> {
  auto* assoc_type = &assoc->constant().static_type();
  Bindings bindings = assoc->interface().bindings();
  bindings.Add(assoc->interface().declaration().self(), &assoc->base(),
               &assoc->witness());
  return Substitute(bindings, assoc_type);
}

auto TypeChecker::LookupRewriteInTypeOf(
    Nonnull<const Value*> type, Nonnull<const InterfaceType*> interface,
    Nonnull<const Declaration*> member) const
    -> std::optional<const ValueLiteral*> {
  // Given `(T:! C).Y`, look in `C` for rewrites.
  if (auto* var_type = dyn_cast<VariableType>(type)) {
    if (!var_type->binding().has_static_type()) {
      // We looked for a rewrite before we finished type-checking the generic
      // binding. This happens when forming the type of a generic binding. Just
      // say there are no rewrites yet.
      return std::nullopt;
    }
    return LookupRewrite(&var_type->binding().static_type(), interface, member);
  }

  // Given `(T.U).Y` for an associated type `U`, substitute into the type of
  // `U` to find rewrites.
  // TODO: This substitution can lead to infinite recursion.
  if (auto* assoc_const = dyn_cast<AssociatedConstant>(type)) {
    return LookupRewrite(GetTypeForAssociatedConstant(assoc_const), interface,
                         member);
  }

  return std::nullopt;
}

auto TypeChecker::LookupRewriteInWitness(
    Nonnull<const Witness*> witness, Nonnull<const InterfaceType*> interface,
    Nonnull<const Declaration*> member) const
    -> std::optional<const ValueLiteral*> {
  if (auto* impl_witness = dyn_cast<ImplWitness>(witness)) {
    Nonnull<const Value*> constraint =
        Substitute(impl_witness->bindings(),
                   impl_witness->declaration().constraint_type());
    return LookupRewrite(constraint, interface, member);
  }
  return std::nullopt;
}

// Rewrites a member access expression to produce the given constant value.
static void RewriteMemberAccess(Nonnull<MemberAccessExpression*> access,
                                Nonnull<const ValueLiteral*> value) {
  access->set_static_type(&value->static_type());
  access->set_value_category(value->value_category());
  access->set_constant_value(&value->value());
}

// Determine whether the given member declaration declares an instance member.
static auto IsInstanceMember(Member member) {
  if (!member.declaration()) {
    // This is a struct field.
    return true;
  }
  Nonnull<const Declaration*> declaration = *member.declaration();
  switch (declaration->kind()) {
    case DeclarationKind::FunctionDeclaration:
      return cast<FunctionDeclaration>(declaration)->is_method();
    case DeclarationKind::VariableDeclaration:
      return true;
    default:
      return false;
  }
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
      CARBON_FATAL() << "attempting to type check node " << *e
                     << " generated during type checking";
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&index.object(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&index.offset(), impl_scope));
      const Value& object_type = index.object().static_type();
      switch (object_type.kind()) {
        case Value::Kind::TupleValue: {
          const auto& tuple_type = cast<TupleValue>(object_type);
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
          return ProgramError(e->source_loc()) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      for (auto* arg : cast<TupleLiteral>(*e).fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(arg, impl_scope));
        CARBON_RETURN_IF_ERROR(
            ExpectIsConcreteType(arg->source_loc(), &arg->static_type()));
        arg_types.push_back(&arg->static_type());
      }
      e->set_static_type(arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        CARBON_RETURN_IF_ERROR(ExpectIsConcreteType(
            arg.expression().source_loc(), &arg.expression().static_type()));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      e->set_static_type(arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      for (auto& arg : struct_type.fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&arg.expression(), impl_scope));
      }
      if (struct_type.fields().empty()) {
        // `{}` is the type of `{}`, just as `()` is the type of `()`.
        // This applies only if there are no fields, because (unlike with
        // tuples) non-empty struct types are syntactically disjoint
        // from non-empty struct values.
        struct_type.set_static_type(arena_->New<StructType>());
      } else {
        struct_type.set_static_type(arena_->New<TypeType>());
      }
      e->set_value_category(ValueCategory::Let);
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
              access.set_member(Member(&field));
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
              auto type_member, FindMixedMemberAndType(
                                    access.source_loc(), access.member_name(),
                                    t_class.declaration().members(), &t_class));
          if (type_member.has_value()) {
            auto [member_type, member] = type_member.value();
            Nonnull<const Value*> field_type =
                Substitute(t_class.bindings(), member_type);
            access.set_member(Member(member));
            access.set_static_type(field_type);
            access.set_is_type_access(!IsInstanceMember(access.member()));
            switch (member->kind()) {
              case DeclarationKind::VariableDeclaration:
                access.set_value_category(access.object().value_category());
                break;
              case DeclarationKind::FunctionDeclaration: {
                auto func_decl = cast<FunctionDeclaration>(member);
                if (func_decl->is_method() && func_decl->me_pattern().kind() ==
                                                  PatternKind::AddrPattern) {
                  access.set_is_field_addr_me_method();
                  Nonnull<const Value*> me_type =
                      Substitute(t_class.bindings(),
                                 &func_decl->me_pattern().static_type());
                  CARBON_RETURN_IF_ERROR(ExpectType(
                      e->source_loc(), "method access, receiver type", me_type,
                      &access.object().static_type(), impl_scope));
                  if (access.object().value_category() != ValueCategory::Var) {
                    return ProgramError(e->source_loc())
                           << "method " << access.member_name()
                           << " requires its receiver to be an lvalue";
                  }
                }
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
          if (auto* var_type = dyn_cast<VariableType>(&object_type)) {
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
              MakeConstraintForInterface(access.source_loc(),
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
          access.set_member(Member(result.member));
          access.set_found_in_interface(result.interface);
          access.set_is_type_access(!IsInstanceMember(access.member()));
          access.set_static_type(inst_member_type);

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
              MakeConstraintForInterface(access.source_loc(),
                                         result.interface));
          CARBON_ASSIGN_OR_RETURN(Nonnull<const Witness*> witness,
                                  impl_scope.Resolve(iface_constraint, type,
                                                     e->source_loc(), *this));
          CARBON_ASSIGN_OR_RETURN(Nonnull<const Witness*> impl,
                                  impl_scope.Resolve(result.interface, type,
                                                     e->source_loc(), *this));
          access.set_member(Member(result.member));
          access.set_impl(impl);
          access.set_found_in_interface(result.interface);

          if (IsInstanceMember(access.member())) {
            // This is a member name denoting an instance member.
            // TODO: Consider setting the static type of all instance member
            // declarations to be member name types, rather than special-casing
            // member accesses that name them.
            access.set_static_type(
                arena_->New<TypeOfMemberName>(Member(result.member)));
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
        case Value::Kind::TypeType:
        case Value::Kind::TypeOfChoiceType:
        case Value::Kind::TypeOfClassType:
        case Value::Kind::TypeOfConstraintType:
        case Value::Kind::TypeOfInterfaceType: {
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
                  access.set_member(Member(&field));
                  access.set_static_type(
                      arena_->New<TypeOfMemberName>(Member(&field)));
                  access.set_value_category(ValueCategory::Let);
                  return Success();
                }
              }
              return ProgramError(access.source_loc())
                     << "struct " << *type << " does not have a field named "
                     << " does not have a field named " << access.member_name();
            }
            case Value::Kind::ChoiceType: {
              const ChoiceType& choice = cast<ChoiceType>(*type);
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
              access.set_member(Member(arena_->New<NamedValue>(
                  NamedValue{access.member_name(), type})));
              access.set_static_type(type);
              access.set_value_category(ValueCategory::Let);
              return Success();
            }
            case Value::Kind::NominalClassType: {
              const NominalClassType& class_type =
                  cast<NominalClassType>(*type);
              CARBON_ASSIGN_OR_RETURN(
                  auto type_member,
                  FindMixedMemberAndType(
                      access.source_loc(), access.member_name(),
                      class_type.declaration().members(), &class_type));
              if (type_member.has_value()) {
                auto [member_type, member] = type_member.value();
                access.set_member(Member(member));
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
                    arena_->New<TypeOfMemberName>(Member(member)));
                access.set_value_category(ValueCategory::Let);
                return Success();
              } else {
                return ProgramError(access.source_loc())
                       << class_type << " does not have a member named "
                       << access.member_name();
              }
            }
            case Value::Kind::InterfaceType:
            case Value::Kind::ConstraintType: {
              CARBON_ASSIGN_OR_RETURN(
                  ConstraintLookupResult result,
                  LookupInConstraint(e->source_loc(), "member access", type,
                                     access.member_name()));
              access.set_member(Member(result.member));
              access.set_found_in_interface(result.interface);
              access.set_static_type(
                  arena_->New<TypeOfMemberName>(Member(result.member)));
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
      bool is_instance_member = IsInstanceMember(member_name.member());

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
            MakeConstraintForInterface(access.source_loc(), *iface));
        CARBON_ASSIGN_OR_RETURN(witness,
                                impl_scope.Resolve(iface_constraint, *base_type,
                                                   e->source_loc(), *this));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Witness*> impl,
            impl_scope.Resolve(*iface, *base_type, e->source_loc(), *this));
        if (std::optional<Nonnull<const ValueLiteral*>> replacement =
                LookupRewriteInWitness(impl, *iface,
                                       *member_name.member().declaration())) {
          RewriteMemberAccess(&access, *replacement);
          return Success();
        }
        access.set_impl(impl);
      }

      auto SubstituteIntoMemberType = [&]() {
        Nonnull<const Value*> member_type = &member_name.member().type();
        if (member_name.interface()) {
          Nonnull<const InterfaceType*> iface_type = *member_name.interface();
          Bindings bindings = iface_type->bindings();
          bindings.Add(iface_type->declaration().self(), *base_type, witness);
          return Substitute(bindings, member_type);
        }
        if (auto* class_type = dyn_cast<NominalClassType>(base_type.value())) {
          return Substitute(class_type->bindings(), member_type);
        }
        return member_type;
      };

      switch (std::optional<Nonnull<const Declaration*>> decl =
                  member_name.member().declaration();
              decl ? decl.value()->kind()
                   : DeclarationKind::VariableDeclaration) {
        case DeclarationKind::VariableDeclaration:
          if (has_instance) {
            access.set_static_type(SubstituteIntoMemberType());
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
            access.set_static_type(SubstituteIntoMemberType());
            access.set_value_category(ValueCategory::Let);
            return Success();
          }
          break;
        }
        case DeclarationKind::AssociatedConstantDeclaration:
          access.set_static_type(SubstituteIntoMemberType());
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
            std::optional<Nonnull<const ConstraintType*>> constraints[2];
            for (int i : {0, 1}) {
              // TODO: This should be done based on the values, not their
              // types.
              if (auto* iface_type_type =
                      dyn_cast<TypeOfInterfaceType>(ts[i])) {
                CARBON_ASSIGN_OR_RETURN(
                    constraints[i],
                    MakeConstraintForInterface(
                        e->source_loc(), &iface_type_type->interface_type()));
              } else if (auto* constraint_type_type =
                             dyn_cast<TypeOfConstraintType>(ts[i])) {
                constraints[i] = &constraint_type_type->constraint_type();
              } else {
                return ProgramError(op.arguments()[i]->source_loc())
                       << "argument to " << ToString(op.op())
                       << " should be a constraint, found `" << *ts[i] << "`";
              }
            }
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const ConstraintType*> result,
                CombineConstraints(e->source_loc(),
                                   {*constraints[0], *constraints[1]}));
            op.set_static_type(arena_->New<TypeOfConstraintType>(result));
            op.set_value_category(ValueCategory::Let);
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
          op.set_static_type(&cast<PointerType>(*ts[0]).type());
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
              InterpExp(op.arguments()[1], arena_, trace_stream_));
          CARBON_RETURN_IF_ERROR(
              ExpectIsConcreteType(op.arguments()[1]->source_loc(), type));
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
              fun_t.deduced_bindings(), fun_t.impl_bindings(), impl_scope));

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
          std::vector<Nonnull<const ImplBinding*>> impl_bindings;
          llvm::ArrayRef<Nonnull<const Pattern*>> params =
              param_name.params().fields();
          for (size_t i = 0; i != params.size(); ++i) {
            // TODO: Should we disallow all other kinds of top-level params?
            if (auto* binding = dyn_cast<GenericBinding>(params[i])) {
              generic_parameters.push_back({i, binding});
              if (binding->impl_binding().has_value()) {
                impl_bindings.push_back(*binding->impl_binding());
              }
            }
          }

          CARBON_RETURN_IF_ERROR(DeduceCallBindings(
              call, &param_name.params().static_type(), generic_parameters,
              /*deduced_bindings=*/llvm::None, impl_bindings, impl_scope));
          Nonnull<const Bindings*> bindings = &call.bindings();

          const Declaration& decl = param_name.declaration();
          switch (decl.kind()) {
            case DeclarationKind::ClassDeclaration: {
              Nonnull<NominalClassType*> inst_class_type =
                  arena_->New<NominalClassType>(&cast<ClassDeclaration>(decl),
                                                bindings);
              call.set_static_type(
                  arena_->New<TypeOfClassType>(inst_class_type));
              call.set_value_category(ValueCategory::Let);
              break;
            }
            case DeclarationKind::InterfaceDeclaration: {
              Nonnull<InterfaceType*> inst_iface_type =
                  arena_->New<InterfaceType>(&cast<InterfaceDeclaration>(decl),
                                             bindings);
              call.set_static_type(
                  arena_->New<TypeOfInterfaceType>(inst_iface_type));
              call.set_value_category(ValueCategory::Let);
              break;
            }
            case DeclarationKind::ChoiceDeclaration: {
              Nonnull<ChoiceType*> ct = arena_->New<ChoiceType>(
                  cast<ChoiceDeclaration>(&decl), bindings);
              Nonnull<TypeOfChoiceType*> inst_choice_type =
                  arena_->New<TypeOfChoiceType>(ct);
              call.set_static_type(inst_choice_type);
              call.set_value_category(ValueCategory::Let);
              break;
            }
            default:
              CARBON_FATAL()
                  << "unknown type of ParameterizedEntityName for " << decl;
          }
          return Success();
        }
        case Value::Kind::TypeOfChoiceType:
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
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&fn.parameter(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&fn.return_type(), impl_scope));
      fn.set_static_type(arena_->New<TypeType>());
      fn.set_value_category(ValueCategory::Let);
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
          if (args.size() < 1 || args.size() > 2) {
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
          e->set_static_type(TupleValue::Empty());
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
          e->set_static_type(TupleValue::Empty());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Alloc: {
          if (args.size() != 1) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_new takes 1 argument";
          }
          auto arg_type = &args[0]->static_type();
          e->set_static_type(arena_->New<PointerType>(arg_type));
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Dealloc: {
          if (args.size() != 1) {
            return ProgramError(e->source_loc())
                   << "__intrinsic_new takes 1 argument";
          }
          auto arg_type = &args[0]->static_type();
          CARBON_RETURN_IF_ERROR(
              ExpectPointerType(e->source_loc(), "*", arg_type));
          e->set_static_type(TupleValue::Empty());
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

      // Note, we don't want to call `TypeCheckPattern` here. Most of the setup
      // for the self binding is instead done by the `ConstraintTypeBuilder`.
      auto& self = where.self_binding();
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> base_type,
                              TypeCheckTypeExp(&self.type(), impl_scope));
      self.set_static_type(base_type);

      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const ConstraintType*> base,
          ConvertToConstraintType(where.source_loc(),
                                  "first operand of `where` expression",
                                  base_type));

      // Start with the given constraint.
      ConstraintTypeBuilder builder(arena_, &self);
      CARBON_RETURN_IF_ERROR(
          builder.AddAndSubstitute(*this, base, builder.GetSelfType(),
                                   builder.GetSelfWitness(), Bindings(),
                                   /*add_lookup_contexts=*/true));
      // Constraints from the LHS of `where` are in scope in the RHS. But
      // constraints from earlier `where` clauses are not in scope in later
      // clauses.
      builder.BringImplsIntoScope(*this, &inner_impl_scope);

      // Type-check and apply the `where` clauses.
      for (Nonnull<WhereClause*> clause : where.clauses()) {
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
            if (auto* interface = dyn_cast<InterfaceType>(constraint)) {
              // `where X is Y` produces an `impl` constraint.
              builder.AddImplConstraint({.type = type, .interface = interface});
            } else if (auto* constraint_type =
                           dyn_cast<ConstraintType>(constraint)) {
              // Transform `where .B is (C where .D is E)` into
              // `where .B is C and .B.D is E` then add all the resulting
              // constraints.
              CARBON_RETURN_IF_ERROR(
                  builder.AddAndSubstitute(*this, constraint_type, type,
                                           builder.GetSelfWitness(), Bindings(),
                                           /*add_lookup_contexts=*/false));
            } else {
              return ProgramError(is_clause.constraint().source_loc())
                     << "expression after `is` does not resolve to a "
                        "constraint, found value "
                     << *constraint << " of type "
                     << is_clause.constraint().static_type();
            }
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
            // TODO: Decide what type constraints we want to impose on the
            // replacement. Given
            //
            //   interface A {
            //     let N:! i32;
            //   }
            //   fn F[T:! A where .N = (i32, i32)]() {}
            //
            // ... no call can ever succeed. Should we reject? We want to
            // preserve the type of the replacement in the case where it is a
            // constraint type containing further rewrites.
            CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> replacement_value,
                                    InterpExp(&rewrite_clause.replacement(),
                                              arena_, trace_stream_));
            auto* replacement = arena_->New<ValueLiteral>(
                rewrite_clause.source_loc(), replacement_value,
                &rewrite_clause.replacement().static_type(),
                ValueCategory::Let);
            CARBON_RETURN_IF_ERROR(builder.AddRewriteConstraint(
                rewrite_clause.source_loc(), {.interface = result.interface,
                                              .constant = constant,
                                              .replacement = replacement}));
            break;
          }
        }
      }

      where.set_static_type(
          arena_->New<TypeOfConstraintType>(std::move(builder).Build(arena_)));
      where.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << *e;
    case ExpressionKind::ArrayTypeLiteral: {
      auto& array_literal = cast<ArrayTypeLiteral>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(
          &array_literal.element_type_expression(), impl_scope));

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
      return Success();
    }
  }
}

void TypeChecker::CollectGenericBindingsInPattern(
    Nonnull<const Pattern*> p,
    std::vector<Nonnull<const GenericBinding*>>& generic_bindings) {
  VisitNestedPatterns(*p, [&](const Pattern& pattern) {
    if (auto* binding = dyn_cast<GenericBinding>(&pattern)) {
      generic_bindings.push_back(binding);
    }
    return true;
  });
}

void TypeChecker::CollectImplBindingsInPattern(
    Nonnull<const Pattern*> p,
    std::vector<Nonnull<const ImplBinding*>>& impl_bindings) {
  VisitNestedPatterns(*p, [&](const Pattern& pattern) {
    if (auto* binding = dyn_cast<GenericBinding>(&pattern)) {
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
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                          InterpExp(type_expression, arena_, trace_stream_));
  CARBON_RETURN_IF_ERROR(
      concrete ? ExpectIsConcreteType(type_expression->source_loc(), type)
               : ExpectIsType(type_expression->source_loc(), type));
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
      if (!isa<TypeOfInterfaceType, TypeOfConstraintType, TypeType>(
              is_clause.constraint().static_type())) {
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
      return Success();
    }
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(*p);
      if (!VisitNestedPatterns(binding.type(), [](const Pattern& pattern) {
            return !isa<BindingPattern>(pattern);
          })) {
        return ProgramError(binding.type().source_loc())
               << "The type of a binding pattern cannot contain bindings.";
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(
          &binding.type(), std::nullopt, impl_scope, enclosing_value_category));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> type,
          InterpPattern(&binding.type(), arena_, trace_stream_));
      CARBON_RETURN_IF_ERROR(ExpectIsType(binding.source_loc(), type));
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
                   << "Type pattern '" << *type
                   << "' does not match actual type '" << **expected << "'";
          }
          type = *expected;
        }
      }
      CARBON_RETURN_IF_ERROR(ExpectIsConcreteType(binding.source_loc(), type));
      binding.set_static_type(type);
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> binding_value,
                              InterpPattern(&binding, arena_, trace_stream_));
      SetValue(&binding, binding_value);

      if (!binding.has_value_category()) {
        binding.set_value_category(enclosing_value_category);
      }
      return Success();
    }
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(*p);
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                              TypeCheckTypeExp(&binding.type(), impl_scope));
      if (expected) {
        return ProgramError(binding.type().source_loc())
               << "Generic binding may not occur in pattern with expected "
                  "type: "
               << binding;
      }
      if (binding.named_as_type_via_dot_self() && !IsTypeOfType(type)) {
        return ProgramError(binding.type().source_loc())
               << "`.Self` used in type of non-type binding `" << binding.name()
               << "`";
      }
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                              InterpPattern(&binding, arena_, trace_stream_));
      binding.set_symbolic_identity(val);
      SetValue(&binding, val);

      // Create an impl binding if we have a constraint.
      if (isa<ConstraintType, InterfaceType>(type)) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> constraint,
            ConvertToConstraintType(binding.source_loc(), "generic binding",
                                    type));
        Nonnull<ImplBinding*> impl_binding = arena_->New<ImplBinding>(
            binding.source_loc(), &binding, std::nullopt);
        auto* witness = arena_->New<BindingWitness>(impl_binding);
        impl_binding->set_symbolic_identity(witness);
        binding.set_impl_binding(impl_binding);

        // Substitute the VariableType as `.Self` of the constraint to form the
        // resolved type of the binding. Eg, `T:! X where .Self is Y` resolves
        // to `T:! <constraint T is X and T is Y>`.
        ConstraintTypeBuilder builder(arena_, &binding, impl_binding);
        CARBON_RETURN_IF_ERROR(builder.AddAndSubstitute(
            *this, constraint, val, witness, Bindings(),
            /*add_lookup_contexts=*/true));
        type = std::move(builder).Build(arena_);

        BringImplIntoScope(impl_binding, impl_scope);
      }

      binding.set_static_type(type);
      return Success();
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      std::vector<Nonnull<const Value*>> field_types;
      if (expected && (*expected)->kind() != Value::Kind::TupleValue) {
        return ProgramError(p->source_loc()) << "didn't expect a tuple";
      }
      if (expected && tuple.fields().size() !=
                          cast<TupleValue>(**expected).elements().size()) {
        return ProgramError(tuple.source_loc()) << "tuples of different length";
      }
      for (size_t i = 0; i < tuple.fields().size(); ++i) {
        Nonnull<Pattern*> field = tuple.fields()[i];
        std::optional<Nonnull<const Value*>> expected_field_type;
        if (expected) {
          expected_field_type = cast<TupleValue>(**expected).elements()[i];
        }
        CARBON_RETURN_IF_ERROR(TypeCheckPattern(
            field, expected_field_type, impl_scope, enclosing_value_category));
        if (trace_stream_)
          **trace_stream_ << "finished checking tuple pattern field " << *field
                          << "\n";
        field_types.push_back(&field->static_type());
      }
      tuple.set_static_type(arena_->New<TupleValue>(std::move(field_types)));
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> tuple_value,
                              InterpPattern(&tuple, arena_, trace_stream_));
      SetValue(&tuple, tuple_value);
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
      const ChoiceType& choice_type = cast<ChoiceType>(*type);
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
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> alternative_value,
          InterpPattern(&alternative, arena_, trace_stream_));
      SetValue(&alternative, alternative_value);
      return Success();
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&expression, impl_scope));
      p->set_static_type(&expression.static_type());
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> expr_value,
                              InterpPattern(p, arena_, trace_stream_));
      SetValue(p, expr_value);
      return Success();
    }
    case PatternKind::VarPattern: {
      auto& var_pattern = cast<VarPattern>(*p);

      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var_pattern.pattern(), expected,
                                              impl_scope,
                                              var_pattern.value_category()));
      var_pattern.set_static_type(&var_pattern.pattern().static_type());
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> pattern_value,
          InterpPattern(&var_pattern, arena_, trace_stream_));
      SetValue(&var_pattern, pattern_value);
      return Success();
    }
    case PatternKind::AddrPattern:
      std::optional<Nonnull<const Value*>> expected_ptr;
      auto& addr_pattern = cast<AddrPattern>(*p);
      if (expected) {
        expected_ptr = arena_->New<PointerType>(expected.value());
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&addr_pattern.binding(),
                                              expected_ptr, impl_scope,
                                              enclosing_value_category));

      if (auto* inner_binding_type =
              dyn_cast<PointerType>(&addr_pattern.binding().static_type())) {
        addr_pattern.set_static_type(&inner_binding_type->type());
      } else {
        return ProgramError(addr_pattern.source_loc())
               << "Type associated with addr must be a pointer type.";
      }
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> pattern_value,
          InterpPattern(&addr_pattern, arena_, trace_stream_));
      SetValue(&addr_pattern, pattern_value);
      return Success();
  }
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
        &f->me_pattern(), std::nullopt, function_scope, ValueCategory::Let));
    CollectGenericBindingsInPattern(&f->me_pattern(), deduced_bindings);
    CollectImplBindingsInPattern(&f->me_pattern(), impl_bindings);
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
    if (auto* binding = dyn_cast<GenericBinding>(param_pattern)) {
      generic_parameters.push_back({i, binding});
    } else {
      CollectGenericBindingsInPattern(param_pattern, deduced_bindings);
    }
  }

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> ret_type,
                            TypeCheckTypeExp(*return_expression, function_scope,
                                             /*concrete=*/false));
    // TODO: This is setting the constant value of the return type. It would
    // make more sense if this were called `set_constant_value` rather than
    // `set_static_type`.
    f->return_term().set_static_type(ret_type);
  } else if (f->return_term().is_omitted()) {
    f->return_term().set_static_type(TupleValue::Empty());
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

  CARBON_RETURN_IF_ERROR(
      ExpectIsConcreteType(f->source_loc(), &f->return_term().static_type()));
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
    // TODO: Check that main doesn't have any parameters.
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
    if (trace_stream_)
      **trace_stream_ << function_scope;
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

  if (class_decl->extensibility() != ClassExtensibility::None) {
    return ProgramError(class_decl->source_loc())
           << "Class prefixes `base` and `abstract` are not supported yet";
  }
  if (class_decl->extends()) {
    return ProgramError(class_decl->source_loc())
           << "Class extension with `extends` is not supported yet";
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

  // For class declaration `class MyType(T:! Type, U:! AnInterface)`, `Self`
  // should have the value `MyType(T, U)`.
  Nonnull<NominalClassType*> self_type = arena_->New<NominalClassType>(
      class_decl, Bindings::SymbolicIdentity(arena_, bindings));
  self->set_static_type(arena_->New<TypeOfClassType>(self_type));
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
/*
** Checks to see if mixin_decl is already within collected_members_. If it is,
** then the mixin has already been type checked before either while type
** checking a previous mix declaration or while type checking the original mixin
** declaration. If not, then every member declaration is type checked and then
** added to collected_members_ under the mixin_decl key.
*/
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
/*
** Type checks the mixin mentioned in the mix declaration.
** TypeCheckMixinDeclaration ensures that the members of that mixin are
** available in collected_members_. The mixin members are then collected as
** members of the enclosing class or mixin declaration.
*/
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
  auto& mixin_decl = mix_decl->mixin_value().declaration();
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

auto TypeChecker::DeclareInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** declaring interface " << iface_decl->name() << "\n";
  }
  ImplScope iface_scope;
  iface_scope.AddParent(scope_info.innermost_scope);

  Nonnull<InterfaceType*> iface_type;
  if (iface_decl->params().has_value()) {
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(*iface_decl->params(), std::nullopt,
                                            iface_scope, ValueCategory::Let));
    if (trace_stream_) {
      **trace_stream_ << iface_scope;
    }

    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(iface_decl, *iface_decl->params());
    iface_decl->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
    iface_decl->set_constant_value(param_name);

    // Form the full symbolic type of the interface. This is used as part of
    // the value of associated constants, if they're referenced within the
    // interface itself.
    std::vector<Nonnull<const GenericBinding*>> bindings = scope_info.bindings;
    CollectGenericBindingsInPattern(*iface_decl->params(), bindings);
    iface_type = arena_->New<InterfaceType>(
        iface_decl, Bindings::SymbolicIdentity(arena_, bindings));
  } else {
    iface_type = arena_->New<InterfaceType>(iface_decl);
    iface_decl->set_static_type(arena_->New<TypeOfInterfaceType>(iface_type));
    iface_decl->set_constant_value(iface_type);
  }

  // Set the type of Self to be the instantiated interface.
  Nonnull<SelfDeclaration*> self_type = iface_decl->self_type();
  self_type->set_static_type(arena_->New<TypeType>());
  self_type->set_constant_value(iface_type);

  // Build a constraint corresponding to this interface.
  ConstraintTypeBuilder builder(arena_, iface_decl->self());
  ConstraintTypeBuilder::ImplsInScopeTracker impl_tracker;
  iface_decl->self()->set_static_type(iface_type);

  // The impl constraint says only that the direct members of the interface are
  // available. For any indirect constraints, we need to add separate entries
  // to the constraint type. This ensures that all indirect constraints are
  // lifted to the top level so they can be accessed directly and resolved
  // independently if necessary.
  int index = builder.AddImplConstraint(
      {.type = builder.GetSelfType(), .interface = iface_type});
  builder.AddLookupContext({.context = iface_type});
  auto* impl_witness =
      MakeConstraintWitnessAccess(builder.GetSelfWitness(), index);

  ScopeInfo iface_scope_info = ScopeInfo::ForNonClassScope(&iface_scope);
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, iface_scope_info));

    // TODO: This should probably live in `DeclareDeclaration`, but it needs
    // to update state that's not available from there.
    switch (m->kind()) {
      case DeclarationKind::InterfaceExtendsDeclaration: {
        // For an `extends C;` declaration, add `Self is C` to our constraint.
        auto* extends = cast<InterfaceExtendsDeclaration>(m);
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> base,
                                TypeCheckTypeExp(extends->base(), iface_scope));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> constraint_type,
            ConvertToConstraintType(m->source_loc(), "extends declaration",
                                    base));
        CARBON_RETURN_IF_ERROR(builder.AddAndSubstitute(
            *this, constraint_type, builder.GetSelfType(),
            builder.GetSelfWitness(), Bindings(),
            /*add_lookup_contexts=*/true));
        break;
      }

      case DeclarationKind::InterfaceImplDeclaration: {
        // For an `impl X as Y;` declaration, add `X is Y` to our constraint.
        auto* impl = cast<InterfaceImplDeclaration>(m);
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> impl_type,
            TypeCheckTypeExp(impl->impl_type(), iface_scope));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> constraint,
            TypeCheckTypeExp(impl->constraint(), iface_scope));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const ConstraintType*> constraint_type,
            ConvertToConstraintType(m->source_loc(), "impl as declaration",
                                    constraint));
        CARBON_RETURN_IF_ERROR(
            builder.AddAndSubstitute(*this, constraint_type, impl_type,
                                     builder.GetSelfWitness(), Bindings(),
                                     /*add_lookup_contexts=*/false));
        break;
      }

      case DeclarationKind::AssociatedConstantDeclaration: {
        auto* assoc = cast<AssociatedConstantDeclaration>(m);
        auto* assoc_value = arena_->New<AssociatedConstant>(
            &iface_decl->self()->value(), iface_type, assoc, impl_witness);
        assoc->binding().set_symbolic_identity(assoc_value);

        // The type specified for the associated constant becomes a
        // constraint for the interface: `let X:! Interface` adds a `Self.X
        // is Interface` constraint that `impl`s must satisfy and users of
        // the interface can rely on.
        Nonnull<const Value*> constraint = &assoc->static_type();
        if (isa<ConstraintType, InterfaceType>(constraint)) {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const ConstraintType*> constraint_type,
              ConvertToConstraintType(assoc->source_loc(),
                                      "type of associated constant",
                                      constraint));
          CARBON_RETURN_IF_ERROR(
              builder.AddAndSubstitute(*this, constraint_type, assoc_value,
                                       builder.GetSelfWitness(), Bindings(),
                                       /*add_lookup_contexts=*/false));
        }
        break;
      }

      default: {
        break;
      }
    }

    // Add any new impl constraints to the scope.
    builder.BringImplsIntoScope(*this, &iface_scope, &impl_tracker);
  }

  iface_decl->set_constraint_type(std::move(builder).Build(arena_));

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring interface " << iface_decl->name()
                    << "\n";
  }
  return Success();
}

auto TypeChecker::TypeCheckInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** checking interface " << iface_decl->name() << "\n";
  }
  ImplScope iface_scope;
  iface_scope.AddParent(&impl_scope);
  if (iface_decl->params().has_value()) {
    BringPatternImplsIntoScope(*iface_decl->params(), iface_scope);
  }
  if (trace_stream_) {
    **trace_stream_ << iface_scope;
  }
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, iface_scope, iface_decl));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking interface " << iface_decl->name()
                    << "\n";
  }
  return Success();
}

auto TypeChecker::CheckImplIsDeducible(
    SourceLocation source_loc, Nonnull<const Value*> impl_type,
    Nonnull<const InterfaceType*> impl_iface,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
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
                                      Nonnull<const Witness*> self_witness,
                                      Nonnull<const Witness*> iface_witness,
                                      const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  const auto& iface_decl = iface_type->declaration();
  for (Nonnull<Declaration*> m : iface_decl.members()) {
    if (auto* assoc = dyn_cast<AssociatedConstantDeclaration>(m)) {
      // An associated constant must be given exactly one value.
      if (LookupRewrite(impl_decl->constraint_type(), iface_type, assoc)) {
        // OK, named by `=` constraint.
        continue;
      }

      // TODO: Remove the rest of this and just reject if there's no `=`.
      Nonnull<const Value*> expected = arena_->New<AssociatedConstant>(
          self_type, iface_type, assoc, self_witness);

      bool found_any = false;
      std::optional<Nonnull<const Value*>> found_value;
      std::optional<Nonnull<const Value*>> second_value;
      auto visitor = [&](Nonnull<const Value*> equal_value) {
        found_any = true;
        if (!isa<AssociatedConstant>(equal_value)) {
          if (!found_value ||
              ValueEqual(equal_value, *found_value, std::nullopt)) {
            found_value = equal_value;
          } else {
            second_value = equal_value;
            return false;
          }
        }
        return true;
      };
      impl_decl->constraint_type()->VisitEqualValues(expected, visitor);
      if (!found_any) {
        return ProgramError(impl_decl->source_loc())
               << "implementation missing " << *expected << "; have "
               << *impl_decl->constraint_type();
      } else if (!found_value) {
        // TODO: It's not clear what the right rule is here. Clearly
        //   impl T as HasX & HasY where .X == .Y {}
        // ... is insufficient to establish a value for either X or Y.
        // But perhaps we can allow
        //   impl forall [T:! HasX] T as HasY where .Y == .X {}
        return ProgramError(impl_decl->source_loc())
               << "implementation doesn't provide a concrete value for "
               << *expected;
      } else if (second_value) {
        return ProgramError(impl_decl->source_loc())
               << "implementation provides multiple values for " << *expected
               << ": " << **found_value << " and " << **second_value;
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
    if (auto* iface_type = dyn_cast<InterfaceType>(lookup.context)) {
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
            MakeConstraintForInterface(impl_decl->source_loc(), iface_type));
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
    ConstraintTypeBuilder builder(arena_, impl_decl->source_loc());
    CARBON_RETURN_IF_ERROR(
        builder.AddAndSubstitute(*this, implemented_constraint, impl_type_value,
                                 builder.GetSelfWitness(), Bindings(),
                                 /*add_lookup_contexts=*/true));
    constraint_type = std::move(builder).Build(arena_);
    impl_decl->set_constraint_type(constraint_type);
  }

  // Build the self witness. This is the witness used to demonstrate that
  // this impl implements its lookup contexts.
  auto* self_witness = arena_->New<ImplWitness>(
      impl_decl, Bindings::SymbolicIdentity(arena_, generic_bindings));

  // Compute a witness that the impl implements its constraint.
  Nonnull<const Witness*> impl_witness;
  {
    ImplScope self_impl_scope;
    self_impl_scope.AddParent(&impl_scope);
    // For each interface we're going to implement, this impl is the witness
    // that that interface is implemented.
    for (auto lookup : constraint_type->lookup_contexts()) {
      if (auto* iface_type = dyn_cast<InterfaceType>(lookup.context)) {
        self_impl_scope.Add(iface_type, impl_type_value, self_witness, *this);
      }
    }
    // Ensure that's enough for our interface to be satisfied.
    CARBON_ASSIGN_OR_RETURN(
        impl_witness, self_impl_scope.Resolve(constraint_type, impl_type_value,
                                              impl_decl->source_loc(), *this));
  }

  // Declare the impl members.
  ScopeInfo impl_scope_info = ScopeInfo::ForNonClassScope(&impl_scope);
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, impl_scope_info));
  }

  // Create the implied impl bindings.
  CARBON_RETURN_IF_ERROR(CheckAndAddImplBindings(impl_decl, impl_type_value,
                                                 self_witness, impl_witness,
                                                 generic_bindings, scope_info));

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
      if (auto* assoc = dyn_cast<AssociatedConstant>(value)) {
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
    alternatives.push_back({.name = alternative->name(), .value = signature});
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

  auto ct = arena_->New<ChoiceType>(
      choice, Bindings::SymbolicIdentity(arena_, bindings));

  choice->set_static_type(arena_->New<TypeOfChoiceType>(ct));
  choice->set_constant_value(ct);
  return Success();
}

auto TypeChecker::TypeCheckChoiceDeclaration(
    Nonnull<ChoiceDeclaration*> /*choice*/, const ImplScope& /*impl_scope*/)
    -> ErrorOr<Success> {
  // Nothing to do here, but perhaps that will change in the future?
  return Success();
}

static bool IsValidTypeForAliasTarget(Nonnull<const Value*> type) {
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
    case DeclarationKind::InterfaceDeclaration: {
      CARBON_RETURN_IF_ERROR(TypeCheckInterfaceDeclaration(
          &cast<InterfaceDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      CARBON_RETURN_IF_ERROR(
          TypeCheckImplDeclaration(&cast<ImplDeclaration>(*d), impl_scope));
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
    case DeclarationKind::InterfaceExtendsDeclaration: {
      // Checked in DeclareInterfaceDeclaration.
      break;
    }
    case DeclarationKind::InterfaceImplDeclaration: {
      // Checked in DeclareInterfaceDeclaration.
      break;
    }
    case DeclarationKind::AssociatedConstantDeclaration:
      break;
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
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface_decl = cast<InterfaceDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(
          DeclareInterfaceDeclaration(&iface_decl, scope_info));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl_decl = cast<ImplDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareImplDeclaration(&impl_decl, scope_info));
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
      auto& mixin_decl = mix_decl.mixin_value().declaration();
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
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var.binding(), std::nullopt,
                                              *scope_info.innermost_scope,
                                              var.value_category()));
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> declared_type,
                              InterpExp(&type, arena_, trace_stream_));
      CARBON_RETURN_IF_ERROR(ExpectCompleteType(
          var.source_loc(), "type of variable", declared_type));
      var.set_static_type(declared_type);
      break;
    }

    case DeclarationKind::InterfaceExtendsDeclaration: {
      // The semantic effects are handled by DeclareInterfaceDeclaration.
      break;
    }

    case DeclarationKind::InterfaceImplDeclaration: {
      // The semantic effects are handled by DeclareInterfaceDeclaration.
      break;
    }

    case DeclarationKind::AssociatedConstantDeclaration: {
      auto& let = cast<AssociatedConstantDeclaration>(*d);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> type,
          TypeCheckTypeExp(&let.binding().type(), *scope_info.innermost_scope));
      let.binding().set_static_type(type);
      let.set_static_type(type);
      // The symbolic identity is set by DeclareInterfaceDeclaration.
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

auto TypeChecker::FindMixedMemberAndType(
    SourceLocation source_loc, const std::string_view& name,
    llvm::ArrayRef<Nonnull<Declaration*>> members,
    const Nonnull<const Value*> enclosing_type)
    -> ErrorOr<std::optional<
        std::pair<Nonnull<const Value*>, Nonnull<const Declaration*>>>> {
  for (Nonnull<const Declaration*> member : members) {
    if (llvm::isa<MixDeclaration>(member)) {
      const auto& mix_decl = cast<MixDeclaration>(*member);
      Nonnull<const MixinPseudoType*> mixin = &mix_decl.mixin_value();
      CARBON_ASSIGN_OR_RETURN(
          const auto res,
          FindMixedMemberAndType(source_loc, name,
                                 mixin->declaration().members(), mixin));
      if (res.has_value()) {
        if (isa<NominalClassType>(enclosing_type)) {
          Bindings temp_map;
          // TODO: What is the type of Self? Do we ever need a witness?
          temp_map.Add(mixin->declaration().self(), enclosing_type,
                       std::nullopt);
          const auto mix_member_type = Substitute(temp_map, res.value().first);
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
