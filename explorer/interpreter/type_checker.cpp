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
#include "explorer/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

namespace Carbon {

static void SetValue(Nonnull<Pattern*> pattern, Nonnull<const Value*> value) {
  // TODO: find some way to CHECK that `value` is identical to pattern->value(),
  // if it's already set. Unclear if `ValueEqual` is suitable, because it
  // currently focuses more on "real" values, and disallows the pseudo-values
  // like `BindingPlaceholderValue` that we get in pattern evaluation.
  if (!pattern->has_value()) {
    pattern->set_value(value);
  }
}

static auto ExpectExactType(SourceLocation source_loc,
                            const std::string& context,
                            Nonnull<const Value*> expected,
                            Nonnull<const Value*> actual) -> ErrorOr<Success> {
  if (!TypeEqual(expected, actual)) {
    return CompilationError(source_loc) << "type error in " << context << "\n"
                                        << "expected: " << *expected << "\n"
                                        << "actual: " << *actual;
  }
  return Success();
}

static auto ExpectPointerType(SourceLocation source_loc,
                              const std::string& context,
                              Nonnull<const Value*> actual)
    -> ErrorOr<Success> {
  if (actual->kind() != Value::Kind::PointerType) {
    return CompilationError(source_loc) << "type error in " << context << "\n"
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
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::Witness:
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
      // A value of one of these types could be a type, but isn't known to be.
      return false;
    case Value::Kind::TypeType:
    case Value::Kind::InterfaceType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      // A value of one of these types is itself always a type.
      return true;
  }
}

// Returns whether the value is a valid result from a type expression,
// as opposed to a non-type value.
static auto IsType(Nonnull<const Value*> value) -> bool {
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
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::Witness:
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
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
      return true;
    case Value::Kind::TupleValue: {
      for (Nonnull<const Value*> field : cast<TupleValue>(*value).elements()) {
        if (!IsType(field)) {
          return false;
        }
      }
      return true;
    }
  }
}

auto TypeChecker::ExpectIsType(SourceLocation source_loc,
                               Nonnull<const Value*> value)
    -> ErrorOr<Success> {
  if (!IsType(value)) {
    return CompilationError(source_loc)
           << "Expected a type, but got " << *value;
  } else {
    return Success();
  }
}

// Returns whether *value represents the type of a Carbon value, as
// opposed to a type pattern or a non-type value.
static auto IsConcreteType(Nonnull<const Value*> value) -> bool {
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
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
    case Value::Kind::Witness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::StaticArrayType:
      return true;
    case Value::Kind::AutoType:
      // `auto` isn't a concrete type, it's a pattern that matches types.
      return false;
    case Value::Kind::TupleValue: {
      for (Nonnull<const Value*> field : cast<TupleValue>(*value).elements()) {
        if (!IsConcreteType(field)) {
          return false;
        }
      }
      return true;
    }
  }
}

auto TypeChecker::ExpectIsConcreteType(SourceLocation source_loc,
                                       Nonnull<const Value*> value)
    -> ErrorOr<Success> {
  if (!IsConcreteType(value)) {
    return CompilationError(source_loc)
           << "Expected a type, but got " << *value;
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
    llvm::ArrayRef<NamedValue> destination_fields) const -> bool {
  if (source_fields.size() != destination_fields.size()) {
    return false;
  }
  for (const auto& source_field : source_fields) {
    std::optional<NamedValue> destination_field =
        FindField(destination_fields, source_field.name);
    if (!destination_field.has_value() ||
        !IsImplicitlyConvertible(source_field.value,
                                 destination_field.value().value,
                                 // FIXME: We don't have a way to perform
                                 // user-defined conversions of a struct field
                                 // yet, because we can't write a suitable impl
                                 // for ImplicitAs.
                                 std::nullopt)) {
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
            Substitute(class_type.type_args(), &var.binding().static_type());
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
    std::optional<Nonnull<const ImplScope*>> impl_scope) const -> bool {
  // Check for an exact match or for an implicit conversion.
  // FIXME: `impl`s of `ImplicitAs` should be provided to cover these
  // conversions.
  CARBON_CHECK(IsConcreteType(source));
  CARBON_CHECK(IsConcreteType(destination));
  if (TypeEqual(source, destination)) {
    return true;
  }
  switch (source->kind()) {
    case Value::Kind::StructType:
      switch (destination->kind()) {
        case Value::Kind::StructType:
          if (FieldTypesImplicitlyConvertible(
                  cast<StructType>(*source).fields(),
                  cast<StructType>(*destination).fields())) {
            return true;
          }
          break;
        case Value::Kind::NominalClassType:
          if (FieldTypesImplicitlyConvertible(
                  cast<StructType>(*source).fields(),
                  FieldTypes(cast<NominalClassType>(*destination)))) {
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
            if (!IsImplicitlyConvertible(source_tuple.elements()[i],
                                         destination_tuple.elements()[i],
                                         impl_scope)) {
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
            if (!IsImplicitlyConvertible(source_element,
                                         &destination_array.element_type(),
                                         impl_scope)) {
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
            if (!IsImplicitlyConvertible(source_element, destination,
                                         impl_scope)) {
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
      // FIXME: This seems suspicious. Shouldn't this require that the type
      // implements the interface?
      if (destination->kind() == Value::Kind::InterfaceType) {
        return true;
      }
      break;
    case Value::Kind::InterfaceType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfChoiceType:
      // FIXME: These types should presumably also convert to interface types.
      if (destination->kind() == Value::Kind::TypeType) {
        return true;
      }
      break;
    default:
      break;
  }

  // If we weren't given an impl scope, only look for builtin conversions.
  if (!impl_scope.has_value()) {
    return false;
  }

  // We didn't find a builtin implicit conversion. Try a user-defined one.
  // The source location doesn't matter, we're discarding the diagnostics.
  SourceLocation source_loc("", 0);
  ErrorOr<Nonnull<const InterfaceType*>> iface_type = GetBuiltinInterfaceType(
      source_loc, BuiltinInterfaceName{Builtins::ImplicitAs, destination});
  return iface_type.ok() &&
         (*impl_scope)->Resolve(*iface_type, source, source_loc, *this).ok();
}

auto TypeChecker::ImplicitlyConvert(const std::string& context,
                                    const ImplScope& impl_scope,
                                    Nonnull<Expression*> source,
                                    Nonnull<const Value*> destination)
    -> ErrorOr<Nonnull<Expression*>> {
  // FIXME: If a builtin conversion works, for now we don't create any
  // expression to do the conversion and rely on the interpreter to know how to
  // do it.
  // FIXME: This doesn't work for cases of combined built-in and user-defined
  // conversion, such as converting a struct element via an `ImplicitAs` impl.
  if (IsImplicitlyConvertible(&source->static_type(), destination,
                              std::nullopt)) {
    return source;
  }
  ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
      impl_scope, source,
      BuiltinInterfaceName{Builtins::ImplicitAs, destination},
      BuiltinMethodCall{"Convert"});
  if (!converted.ok()) {
    // We couldn't find a matching `impl`.
    return CompilationError(source->source_loc())
           << "type error in " << context << ": "
           << "'" << source->static_type()
           << "' is not implicitly convertible to '" << *destination << "'";
  }
  return *converted;
}

auto TypeChecker::GetBuiltinInterfaceType(SourceLocation source_loc,
                                          BuiltinInterfaceName interface) const
    -> ErrorOr<Nonnull<const InterfaceType*>> {
  auto bad_builtin = [&]() -> Error {
    return CompilationError(source_loc)
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
  BindingMap bindings;
  if (has_arguments) {
    TupleValue args(interface.arguments);
    if (!PatternMatch(&iface_decl->params().value()->value(), &args, source_loc,
                      std::nullopt, bindings, trace_stream_)) {
      return bad_builtin();
    }
  }
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
  Nonnull<Expression*> iface_member =
      arena_->New<SimpleMemberAccessExpression>(source_loc, iface_expr, method.name);
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

auto TypeChecker::ExpectType(
    SourceLocation source_loc, const std::string& context,
    Nonnull<const Value*> expected, Nonnull<const Value*> actual,
    std::optional<Nonnull<const ImplScope*>> impl_scope) const
    -> ErrorOr<Success> {
  if (!IsImplicitlyConvertible(actual, expected, impl_scope)) {
    return CompilationError(source_loc)
           << "type error in " << context << ": "
           << "'" << *actual << "' is not implicitly convertible to '"
           << *expected << "'";
  } else {
    return Success();
  }
}

auto TypeChecker::ArgumentDeduction(
    SourceLocation source_loc, const std::string& context,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> bindings_to_deduce,
    BindingMap& deduced, Nonnull<const Value*> param, Nonnull<const Value*> arg,
    bool allow_implicit_conversion, const ImplScope& impl_scope) const
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
      return CompilationError(source_loc) << "type error in " << context << "\n"
                                          << "expected: " << *param << "\n"
                                          << "actual: " << *arg;
    }
    const Value* subst_param_type = Substitute(deduced, param);
    return allow_implicit_conversion
               ? ExpectType(source_loc, context, subst_param_type, arg,
                            &impl_scope)
               : ExpectExactType(source_loc, context, subst_param_type, arg);
  };

  switch (param->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param);
      if (std::find(bindings_to_deduce.begin(), bindings_to_deduce.end(),
                    &var_type.binding()) != bindings_to_deduce.end()) {
        auto [it, success] = deduced.insert({&var_type.binding(), arg});
        if (!success) {
          // All deductions are required to produce the same value.
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              source_loc, "repeated argument deduction", it->second, arg));
        }
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
        return CompilationError(source_loc)
               << "mismatch in tuple sizes, expected "
               << param_tup.elements().size() << " but got "
               << arg_tup.elements().size();
      }
      for (size_t i = 0; i < param_tup.elements().size(); ++i) {
        CARBON_RETURN_IF_ERROR(
            ArgumentDeduction(source_loc, context, bindings_to_deduce, deduced,
                              param_tup.elements()[i], arg_tup.elements()[i],
                              allow_implicit_conversion, impl_scope));
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
        return CompilationError(source_loc)
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
            return CompilationError(source_loc)
                   << "mismatch in field names, `" << param_field.name
                   << "` != `" << arg_field.name << "`";
          }
        }
        CARBON_RETURN_IF_ERROR(ArgumentDeduction(
            source_loc, context, bindings_to_deduce, deduced, param_field.value,
            arg_field.value, allow_implicit_conversion, impl_scope));
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
      CARBON_RETURN_IF_ERROR(
          ArgumentDeduction(source_loc, context, bindings_to_deduce, deduced,
                            &param_fn.parameters(), &arg_fn.parameters(),
                            /*allow_implicit_conversion=*/false, impl_scope));
      CARBON_RETURN_IF_ERROR(
          ArgumentDeduction(source_loc, context, bindings_to_deduce, deduced,
                            &param_fn.return_type(), &arg_fn.return_type(),
                            /*allow_implicit_conversion=*/false, impl_scope));
      return Success();
    }
    case Value::Kind::PointerType: {
      if (arg->kind() != Value::Kind::PointerType) {
        return handle_non_deduced_type();
      }
      return ArgumentDeduction(source_loc, context, bindings_to_deduce, deduced,
                               &cast<PointerType>(*param).type(),
                               &cast<PointerType>(*arg).type(),
                               /*allow_implicit_conversion=*/false, impl_scope);
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
        CARBON_RETURN_IF_ERROR(
            ArgumentDeduction(source_loc, context, bindings_to_deduce, deduced,
                              param_ty, arg_class_type.type_args().at(ty),
                              /*allow_implicit_conversion=*/false, impl_scope));
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
        CARBON_RETURN_IF_ERROR(
            ArgumentDeduction(source_loc, context, bindings_to_deduce, deduced,
                              param_ty, arg_iface_type.args().at(ty),
                              /*allow_implicit_conversion=*/false, impl_scope));
      }
      return Success();
    }
    // For the following cases, we check the type matches.
    case Value::Kind::StaticArrayType:
      // TODO: We could deduce the array type from an array or tuple argument.
    case Value::Kind::ContinuationType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      return handle_non_deduced_type();
    case Value::Kind::Witness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue: {
      // Argument deduction within the parameters of a parameterized class type
      // or interface type can compare values, rather than types.
      // TODO: Deduce within the values where possible.
      if (!ValueEqual(param, arg)) {
        return CompilationError(source_loc)
               << "mismatch in non-type values, `" << *arg << "` != `" << *param
               << "`";
      }
      return Success();
    }
  }
}

auto TypeChecker::Substitute(
    const std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>& dict,
    Nonnull<const Value*> type) const -> Nonnull<const Value*> {
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      auto it = dict.find(&cast<VariableType>(*type).binding());
      if (it == dict.end()) {
        return type;
      } else {
        return it->second;
      }
    }
    case Value::Kind::TupleValue: {
      std::vector<Nonnull<const Value*>> elts;
      for (const auto& elt : cast<TupleValue>(*type).elements()) {
        elts.push_back(Substitute(dict, elt));
      }
      return arena_->New<TupleValue>(elts);
    }
    case Value::Kind::StructType: {
      std::vector<NamedValue> fields;
      for (const auto& [name, value] : cast<StructType>(*type).fields()) {
        auto new_type = Substitute(dict, value);
        fields.push_back({name, new_type});
      }
      return arena_->New<StructType>(std::move(fields));
    }
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*type);
      auto param = Substitute(dict, &fn_type.parameters());
      auto ret = Substitute(dict, &fn_type.return_type());
      // TODO: Only remove the bindings that are in `dict`; we may still need
      // to do deduction.
      return arena_->New<FunctionType>(param, llvm::None, ret, llvm::None,
                                       llvm::None);
    }
    case Value::Kind::PointerType: {
      return arena_->New<PointerType>(
          Substitute(dict, &cast<PointerType>(*type).type()));
    }
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*type);
      BindingMap type_args;
      for (const auto& [name, value] : class_type.type_args()) {
        type_args[name] = Substitute(dict, value);
      }
      Nonnull<const NominalClassType*> new_class_type =
          arena_->New<NominalClassType>(&class_type.declaration(), type_args);
      if (trace_stream_) {
        **trace_stream_ << "substitution: " << class_type << " => "
                        << *new_class_type << "\n";
      }
      return new_class_type;
    }
    case Value::Kind::InterfaceType: {
      const auto& iface_type = cast<InterfaceType>(*type);
      BindingMap args;
      for (const auto& [name, value] : iface_type.args()) {
        args[name] = Substitute(dict, value);
      }
      Nonnull<const InterfaceType*> new_iface_type =
          arena_->New<InterfaceType>(&iface_type.declaration(), args);
      if (trace_stream_) {
        **trace_stream_ << "substitution: " << iface_type << " => "
                        << *new_iface_type << "\n";
      }
      return new_iface_type;
    }
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      return type;
    case Value::Kind::Witness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
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
    -> std::optional<Nonnull<Expression*>> {
  if (trace_stream_) {
    **trace_stream_ << "MatchImpl: looking for " << *impl_type << " as "
                    << iface << "\n";
    **trace_stream_ << "checking " << *impl.type << " as "
                    << *impl.interface << "\n";
  }

  BindingMap deduced_args;

  if (ErrorOr<Success> e = ArgumentDeduction(
          source_loc, "match", impl.deduced, deduced_args, impl.type, impl_type,
          /*allow_implicit_conversion=*/false, impl_scope);
      !e.ok()) {
    if (trace_stream_) {
      **trace_stream_ << "type does not match: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (ErrorOr<Success> e = ArgumentDeduction(
          source_loc, "match", impl.deduced, deduced_args, impl.interface,
          &iface, /*allow_implicit_conversion=*/false, impl_scope);
      !e.ok()) {
    if (trace_stream_) {
      **trace_stream_ << "interface does not match: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (trace_stream_) {
    **trace_stream_ << "match results: {";
    llvm::ListSeparator sep;
    for (const auto& [binding, val] : deduced_args) {
      **trace_stream_ << sep << *binding << " = " << *val;
    }
    **trace_stream_ << "}\n";
  }

  // Ensure the constraints on the `impl` are satisfied by the deduced
  // arguments.
  ImplExpMap impls;
  if (ErrorOr<Success> e = SatisfyImpls(impl.impl_bindings, impl_scope,
                                        source_loc, deduced_args, impls);
      !e.ok()) {
    if (trace_stream_) {
      **trace_stream_ << "missing required impl: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (trace_stream_) {
    **trace_stream_ << "matched with " << *impl.type << " as "
                    << *impl.interface << "\n\n";
  }
  return deduced_args.empty() ? impl.impl
                              : arena_->New<InstantiateImpl>(
                                    source_loc, impl.impl, deduced_args, impls);
}

auto TypeChecker::SatisfyImpls(
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
    const ImplScope& impl_scope, SourceLocation source_loc,
    const BindingMap& deduced_type_args, ImplExpMap& impls) const
    -> ErrorOr<Success> {
  for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
    Nonnull<const Value*> interface =
        Substitute(deduced_type_args, impl_binding->interface());
    CARBON_ASSIGN_OR_RETURN(
        Nonnull<Expression*> impl,
        impl_scope.Resolve(interface,
                           deduced_type_args.at(impl_binding->type_var()),
                           source_loc, *this));
    impls.emplace(impl_binding, impl);
  }
  return Success();
}

auto TypeChecker::DeduceCallBindings(
    CallExpression& call, Nonnull<const Value*> params_type,
    llvm::ArrayRef<FunctionType::GenericParameter> generic_params,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
  llvm::ArrayRef<Nonnull<const Value*>> params =
      cast<TupleValue>(*params_type).elements();
  llvm::ArrayRef<Nonnull<const Expression*>> args =
      cast<TupleLiteral>(call.argument()).fields();
  if (params.size() != args.size()) {
    return CompilationError(call.source_loc())
           << "wrong number of arguments in function call, expected "
           << params.size() << " but got " << args.size();
  }

  // Bindings for deduced parameters and generic parameters.
  BindingMap generic_bindings;

  // Deduce and/or convert each argument to the corresponding
  // parameter.
  for (size_t i = 0; i < params.size(); ++i) {
    const Value* param = params[i];
    const Expression* arg = args[i];
    CARBON_RETURN_IF_ERROR(
        ArgumentDeduction(arg->source_loc(), "call", deduced_bindings,
                          generic_bindings, param, &arg->static_type(),
                          /*allow_implicit_conversion=*/true, impl_scope));
    // If the parameter is a `:!` binding, evaluate and collect its
    // value for use in later parameters and in the function body.
    if (!generic_params.empty() && generic_params.front().index == i) {
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> arg_value,
                              InterpExp(arg, arena_, trace_stream_));
      if (trace_stream_) {
        **trace_stream_ << "evaluated generic parameter "
                        << *generic_params.front().binding << " as "
                        << *arg_value << "\n";
      }
      bool newly_added =
          generic_bindings.insert({generic_params.front().binding, arg_value})
              .second;
      CARBON_CHECK(newly_added) << "generic parameter should not be deduced";
      generic_params = generic_params.drop_front();
    }
  }
  CARBON_CHECK(generic_params.empty())
      << "did not find all generic parameters in parameter list";

  call.set_deduced_args(generic_bindings);
  for (Nonnull<const GenericBinding*> deduced_param : deduced_bindings) {
    // TODO: change the following to a CHECK once the real checking
    // has been added to the type checking of function signatures.
    if (auto it = generic_bindings.find(deduced_param);
        it == generic_bindings.end()) {
      return CompilationError(call.source_loc())
             << "could not deduce type argument for type parameter "
             << deduced_param->name() << "\n"
             << "in " << call;
    }
  }

  // Find impls for all the required impl bindings.
  ImplExpMap impls;
  CARBON_RETURN_IF_ERROR(SatisfyImpls(
      impl_bindings, impl_scope, call.source_loc(), generic_bindings, impls));
  call.set_impls(impls);

  // Convert the arguments to the parameter type.
  Nonnull<const Value*> param_type = Substitute(generic_bindings, params_type);
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<Expression*> converted_argument,
      ImplicitlyConvert("call", impl_scope, &call.argument(), param_type));
  call.set_argument(converted_argument);

  return Success();
}

auto TypeChecker::TypeCheckExp(Nonnull<Expression*> e,
                               const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking expression " << *e;
    **trace_stream_ << "\nconstants: ";
    PrintConstants(**trace_stream_);
    **trace_stream_ << "\n";
  }
  if (e->is_type_checked()) {
    if (trace_stream_) {
      **trace_stream_ << "expression has already been type-checked\n";
    }
    return Success();
  }
  switch (e->kind()) {
    case ExpressionKind::InstantiateImpl:
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
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              index.offset().source_loc(), "tuple index",
              arena_->New<IntType>(), &index.offset().static_type()));
          CARBON_ASSIGN_OR_RETURN(
              auto offset_value,
              InterpExp(&index.offset(), arena_, trace_stream_));
          int i = cast<IntValue>(*offset_value).value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            return CompilationError(e->source_loc())
                   << "index " << i << " is out of range for type "
                   << tuple_type;
          }
          index.set_static_type(tuple_type.elements()[i]);
          index.set_value_category(index.object().value_category());
          return Success();
        }
        case Value::Kind::StaticArrayType: {
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              index.offset().source_loc(), "array index",
              arena_->New<IntType>(), &index.offset().static_type()));
          index.set_static_type(
              &cast<StaticArrayType>(object_type).element_type());
          index.set_value_category(index.object().value_category());
          return Success();
        }
        default:
          return CompilationError(e->source_loc()) << "expected a tuple";
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
      switch (object_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(object_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              access.set_static_type(field_type);
              access.set_value_category(access.object().value_category());
              return Success();
            }
          }
          return CompilationError(access.source_loc())
                 << "struct " << struct_type << " does not have a field named "
                 << access.field();
        }
        case Value::Kind::TypeType: {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              InterpExp(&access.object(), arena_, trace_stream_));
          if (const auto* struct_type = dyn_cast<StructType>(type)) {
            for (const auto& field : struct_type->fields()) {
              if (access.field() == field.name) {
                access.set_static_type(
                    arena_->New<TypeOfMemberName>(Member(&field)));
                access.set_value_category(ValueCategory::Let);
                return Success();
              }
            }
            return CompilationError(access.source_loc())
                   << "struct " << *struct_type
                   << " does not have a field named " << access.field();
          }
          // TODO: We should handle all types here, not only structs. For
          // example:
          //   fn Main() -> i32 {
          //     class Class { var n: i32; };
          //     let T:! Type = Class;
          //     let x: T = {.n = 0};
          //     return x.(T.n);
          //   }
          // is valid, and the type of `T` here is `Type`, not `typeof(Class)`.
          return CompilationError(access.source_loc())
                 << "unsupported member access into type " << *type;
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(object_type);
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), t_class.declaration().members());
              member.has_value()) {
            Nonnull<const Value*> field_type =
                Substitute(t_class.type_args(), &(*member)->static_type());
            access.set_static_type(field_type);
            switch ((*member)->kind()) {
              case DeclarationKind::VariableDeclaration:
                access.set_value_category(access.object().value_category());
                break;
              case DeclarationKind::FunctionDeclaration:
                access.set_value_category(ValueCategory::Let);
                break;
              default:
                CARBON_FATAL() << "member " << access.field()
                               << " is not a field or method";
                break;
            }
            return Success();
          } else {
            return CompilationError(e->source_loc())
                   << "class " << t_class.declaration().name()
                   << " does not have a field named " << access.field();
          }
        }
        case Value::Kind::TypeOfChoiceType: {
          const ChoiceType& choice =
              cast<TypeOfChoiceType>(object_type).choice_type();
          std::optional<Nonnull<const Value*>> parameter_types =
              choice.FindAlternative(access.field());
          if (!parameter_types.has_value()) {
            return CompilationError(e->source_loc())
                   << "choice " << choice.name()
                   << " does not have a field named " << access.field();
          }
          access.set_static_type(arena_->New<FunctionType>(
              *parameter_types, llvm::None, &choice, llvm::None, llvm::None));
          access.set_value_category(ValueCategory::Let);
          return Success();
        }
        case Value::Kind::TypeOfClassType: {
          const NominalClassType& class_type =
              cast<TypeOfClassType>(object_type).class_type();
          if (std::optional<Nonnull<const Declaration*>> member = FindMember(
                  access.field(), class_type.declaration().members());
              member.has_value()) {
            switch ((*member)->kind()) {
              case DeclarationKind::FunctionDeclaration: {
                const auto& func = cast<FunctionDeclaration>(*member);
                if (func->is_method()) {
                  break;
                }
                Nonnull<const Value*> field_type = Substitute(
                    class_type.type_args(), &(*member)->static_type());
                access.set_static_type(field_type);
                access.set_value_category(ValueCategory::Let);
                return Success();
              }
              default:
                break;
            }
            access.set_static_type(
                arena_->New<TypeOfMemberName>(Member(*member)));
            access.set_value_category(ValueCategory::Let);
            return Success();
          } else {
            return CompilationError(access.source_loc())
                   << class_type << " does not have a member named "
                   << access.field();
          }
        }
        case Value::Kind::TypeOfInterfaceType: {
          const InterfaceType& iface_type =
              cast<TypeOfInterfaceType>(object_type).interface_type();
          if (std::optional<Nonnull<const Declaration*>> member = FindMember(
                  access.field(), iface_type.declaration().members());
              member.has_value()) {
            access.set_static_type(
                arena_->New<TypeOfMemberName>(Member(*member)));
            access.set_value_category(ValueCategory::Let);
            return Success();
          } else {
            return CompilationError(access.source_loc())
                   << iface_type << " does not have a member named "
                   << access.field();
          }
        }
        case Value::Kind::VariableType: {
          // This case handles access to a method on a receiver whose type
          // is a type variable. For example, `x.foo` where the type of
          // `x` is `T` and `foo` and `T` implements an interface that
          // includes `foo`.
          const VariableType& var_type = cast<VariableType>(object_type);
          const Value& typeof_var = var_type.binding().static_type();
          switch (typeof_var.kind()) {
            case Value::Kind::InterfaceType: {
              const auto& iface_type = cast<InterfaceType>(typeof_var);
              const InterfaceDeclaration& iface_decl = iface_type.declaration();
              if (std::optional<Nonnull<const Declaration*>> member =
                      FindMember(access.field(), iface_decl.members());
                  member.has_value()) {
                const Value& member_type = (*member)->static_type();
                BindingMap binding_map = iface_type.args();
                binding_map[iface_decl.self()] = &var_type;
                Nonnull<const Value*> inst_member_type =
                    Substitute(binding_map, &member_type);
                access.set_static_type(inst_member_type);
                CARBON_CHECK(var_type.binding().impl_binding().has_value());
                access.set_impl(*var_type.binding().impl_binding());
                return Success();
              } else {
                return CompilationError(e->source_loc())
                       << "field access, " << access.field() << " not in "
                       << iface_decl.name();
              }
              break;
            }
            default:
              return CompilationError(e->source_loc())
                     << "field access, unexpected " << object_type
                     << " of non-interface type " << typeof_var << " in " << *e;
          }
          break;
        }
        case Value::Kind::InterfaceType: {
          // This case handles access to a class function from a type variable.
          // If `T` is a type variable and `foo` is a class function in an
          // interface implemented by `T`, then `T.foo` accesses the `foo` class
          // function of `T`.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> var_addr,
              InterpExp(&access.object(), arena_, trace_stream_));
          const VariableType& var_type = cast<VariableType>(*var_addr);
          const InterfaceType& iface_type = cast<InterfaceType>(object_type);
          const InterfaceDeclaration& iface_decl = iface_type.declaration();
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), iface_decl.members());
              member.has_value()) {
            CARBON_CHECK(var_type.binding().impl_binding().has_value());
            access.set_impl(*var_type.binding().impl_binding());

            switch ((*member)->kind()) {
              case DeclarationKind::FunctionDeclaration: {
                const auto& func = cast<FunctionDeclaration>(*member);
                if (func->is_method()) {
                  break;
                }
                const Value& member_type = (*member)->static_type();
                BindingMap binding_map = iface_type.args();
                binding_map[iface_decl.self()] = &var_type;
                Nonnull<const Value*> inst_member_type =
                    Substitute(binding_map, &member_type);
                access.set_static_type(inst_member_type);
                return Success();
              }
              default:
                break;
            }
            // TODO: Consider setting the static type of all interface member
            // declarations and instance member declarations to be member name
            // types, rather than special-casing member accesses that name
            // them.
            access.set_static_type(
                arena_->New<TypeOfMemberName>(Member(*member)));
            access.set_value_category(ValueCategory::Let);
            return Success();
          } else {
            return CompilationError(e->source_loc())
                   << "field access, " << access.field() << " not in "
                   << iface_decl.name();
          }
          break;
        }
        default:
          return CompilationError(e->source_loc())
                 << "field access, unexpected " << object_type << " in "
                 << *e;
      }
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      auto& access = cast<CompoundMemberAccessExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.object(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.path(), impl_scope));
      if (!isa<TypeOfMemberName>(access.path().static_type())) {
        return CompilationError(e->source_loc())
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

      // Perform impl selection if necessary.
      if (std::optional<Nonnull<const Value*>> iface =
              member_name.interface()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> impl,
            impl_scope.Resolve(*iface, *base_type, e->source_loc(), *this));
        access.set_impl(impl);
      }

      auto SubstituteIntoMemberType = [&]() {
        Nonnull<const Value*> member_type = &member_name.member().type();
        if (member_name.interface()) {
          Nonnull<const InterfaceType*> iface_type = *member_name.interface();
          BindingMap binding_map = iface_type->args();
          binding_map[iface_type->declaration().self()] = *base_type;
          return Substitute(binding_map, member_type);
        }
        if (auto* class_type = dyn_cast<NominalClassType>(base_type.value())) {
          return Substitute(class_type->type_args(), member_type);
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
          bool is_method = cast<FunctionDeclaration>(*decl.value()).is_method();
          if (has_instance || !is_method) {
            // This should not be possible: the name of a static member
            // function should have function type not member name type.
            CARBON_CHECK(!has_instance || is_method ||
                         !member_name.base_type().has_value())
                << "vacuous compound member access";
            access.set_static_type(SubstituteIntoMemberType());
            access.set_value_category(ValueCategory::Let);
            return Success();
          }
          break;
        }
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
          return CompilationError(ident.source_loc())
                 << "Function calls itself, but has a deduced return type";
        }
      }
      ident.set_static_type(&ident.value_node().static_type());
      ident.set_value_category(ident.value_node().value_category());
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
    case ExpressionKind::PrimitiveOperatorExpression: {
      auto& op = cast<PrimitiveOperatorExpression>(*e);
      std::vector<Nonnull<const Value*>> ts;
      for (Nonnull<Expression*> argument : op.arguments()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(argument, impl_scope));
        ts.push_back(&argument->static_type());
      }
      switch (op.op()) {
        case Operator::Neg:
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "negation", arena_->New<IntType>(), ts[0]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Add:
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "addition(1)", arena_->New<IntType>(), ts[0]));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "addition(2)", arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Sub:
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(e->source_loc(), "subtraction(1)",
                              arena_->New<IntType>(), ts[0]));
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(e->source_loc(), "subtraction(2)",
                              arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Mul:
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(e->source_loc(), "multiplication(1)",
                              arena_->New<IntType>(), ts[0]));
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(e->source_loc(), "multiplication(2)",
                              arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::And:
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "&&(1)", arena_->New<BoolType>(), ts[0]));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "&&(2)", arena_->New<BoolType>(), ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Or:
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "||(1)", arena_->New<BoolType>(), ts[0]));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "||(2)", arena_->New<BoolType>(), ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Not:
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "!", arena_->New<BoolType>(), ts[0]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Eq:
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(e->source_loc(), "==", ts[0], ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Deref:
          CARBON_RETURN_IF_ERROR(
              ExpectPointerType(e->source_loc(), "*", ts[0]));
          op.set_static_type(&cast<PointerType>(*ts[0]).type());
          op.set_value_category(ValueCategory::Var);
          return Success();
        case Operator::Ptr:
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "*", arena_->New<TypeType>(), ts[0]));
          op.set_static_type(arena_->New<TypeType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::AddressOf:
          if (op.arguments()[0]->value_category() != ValueCategory::Var) {
            return CompilationError(op.arguments()[0]->source_loc())
                   << "Argument to " << ToString(op.op())
                   << " should be an lvalue.";
          }
          op.set_static_type(arena_->New<PointerType>(ts[0]));
          op.set_value_category(ValueCategory::Let);
          return Success();
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
          CARBON_RETURN_IF_ERROR(DeduceCallBindings(
              call, &fun_t.parameters(), fun_t.generic_parameters(),
              fun_t.deduced_bindings(), fun_t.impl_bindings(), impl_scope));
          const BindingMap& generic_bindings = call.deduced_args();

          // Substitute into the return type to determine the type of the call
          // expression.
          Nonnull<const Value*> return_type =
              Substitute(generic_bindings, &fun_t.return_type());
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
            // FIXME: Should we disallow all other kinds of top-level params?
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

          const Declaration& decl = param_name.declaration();
          switch (decl.kind()) {
            case DeclarationKind::ClassDeclaration: {
              Nonnull<NominalClassType*> inst_class_type =
                  arena_->New<NominalClassType>(&cast<ClassDeclaration>(decl),
                                                call.deduced_args(),
                                                call.impls());
              call.set_static_type(
                  arena_->New<TypeOfClassType>(inst_class_type));
              call.set_value_category(ValueCategory::Let);
              break;
            }
            case DeclarationKind::InterfaceDeclaration: {
              Nonnull<InterfaceType*> inst_iface_type =
                  arena_->New<InterfaceType>(&cast<InterfaceDeclaration>(decl),
                                             call.deduced_args(), call.impls());
              call.set_static_type(
                  arena_->New<TypeOfInterfaceType>(inst_iface_type));
              call.set_value_category(ValueCategory::Let);
              break;
            }
            default:
              CARBON_FATAL()
                  << "unknown type of ParameterizedEntityName for " << decl;
          }
          return Success();
        }
        default: {
          return CompilationError(e->source_loc())
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
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          if (intrinsic_exp.args().fields().size() != 1) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_print takes 1 argument";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_print argument",
              arena_->New<StringType>(),
              &intrinsic_exp.args().fields()[0]->static_type()));
          e->set_static_type(TupleValue::Empty());
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
      CARBON_RETURN_IF_ERROR(
          ExpectExactType(e->source_loc(), "expression of `if` expression",
                          &if_expr.then_expression().static_type(),
                          &if_expr.else_expression().static_type()));
      e->set_static_type(&if_expr.then_expression().static_type());
      e->set_value_category(ValueCategory::Let);
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
      CARBON_RETURN_IF_ERROR(
          ExpectExactType(array_literal.size_expression().source_loc(),
                          "array size", arena_->New<IntType>(),
                          &array_literal.size_expression().static_type()));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> size_value,
          InterpExp(&array_literal.size_expression(), arena_, trace_stream_));
      if (cast<IntValue>(size_value)->value() < 0) {
        return CompilationError(array_literal.size_expression().source_loc())
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

auto TypeChecker::CreateImplReference(Nonnull<const ImplBinding*> impl_binding)
    -> Nonnull<Expression*> {
  auto impl_id =
      arena_->New<IdentifierExpression>(impl_binding->source_loc(), "impl");
  impl_id->set_value_node(impl_binding);
  return impl_id;
}

void TypeChecker::BringImplIntoScope(Nonnull<const ImplBinding*> impl_binding,
                                     ImplScope& impl_scope) {
  CARBON_CHECK(impl_binding->type_var()->symbolic_identity().has_value());
  impl_scope.Add(impl_binding->interface(),
                 *impl_binding->type_var()->symbolic_identity(),
                 CreateImplReference(impl_binding));
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

auto TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, std::optional<Nonnull<const Value*>> expected,
    ImplScope& impl_scope, ValueCategory enclosing_value_category)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking pattern " << *p;
    if (expected) {
      **trace_stream_ << ", expecting " << **expected;
    }
    **trace_stream_ << "\nconstants: ";
    PrintConstants(**trace_stream_);
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
        return CompilationError(binding.type().source_loc())
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
                                            type, *expected, &impl_scope));
        } else {
          BindingMap generic_args;
          if (!PatternMatch(type, *expected, binding.type().source_loc(),
                            std::nullopt, generic_args, trace_stream_)) {
            return CompilationError(binding.type().source_loc())
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
        return CompilationError(binding.type().source_loc())
               << "Generic binding may not occur in pattern with expected "
                  "type: "
               << binding;
      }
      binding.set_static_type(type);
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                              InterpPattern(&binding, arena_, trace_stream_));
      binding.set_symbolic_identity(val);
      SetValue(&binding, val);

      if (isa<InterfaceType>(type)) {
        Nonnull<ImplBinding*> impl_binding =
            arena_->New<ImplBinding>(binding.source_loc(), &binding, type);
        binding.set_impl_binding(impl_binding);
        BringImplIntoScope(impl_binding, impl_scope);
      }
      return Success();
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      std::vector<Nonnull<const Value*>> field_types;
      if (expected && (*expected)->kind() != Value::Kind::TupleValue) {
        return CompilationError(p->source_loc()) << "didn't expect a tuple";
      }
      if (expected && tuple.fields().size() !=
                          cast<TupleValue>(**expected).elements().size()) {
        return CompilationError(tuple.source_loc())
               << "tuples of different length";
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
      if (alternative.choice_type().static_type().kind() !=
          Value::Kind::TypeOfChoiceType) {
        return CompilationError(alternative.source_loc())
               << "alternative pattern does not name a choice type.";
      }
      const ChoiceType& choice_type =
          cast<TypeOfChoiceType>(alternative.choice_type().static_type())
              .choice_type();
      if (expected) {
        CARBON_RETURN_IF_ERROR(ExpectType(alternative.source_loc(),
                                          "alternative pattern", &choice_type,
                                          *expected, &impl_scope));
      }
      std::optional<Nonnull<const Value*>> parameter_types =
          choice_type.FindAlternative(alternative.alternative_name());
      if (parameter_types == std::nullopt) {
        return CompilationError(alternative.source_loc())
               << "'" << alternative.alternative_name()
               << "' is not an alternative of " << choice_type;
      }
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&alternative.arguments(),
                                              *parameter_types, impl_scope,
                                              enclosing_value_category));
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
    case PatternKind::VarPattern:
      auto& let_var_pattern = cast<VarPattern>(*p);

      CARBON_RETURN_IF_ERROR(
          TypeCheckPattern(&let_var_pattern.pattern(), expected, impl_scope,
                           let_var_pattern.value_category()));
      let_var_pattern.set_static_type(&let_var_pattern.pattern().static_type());
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> pattern_value,
          InterpPattern(&let_var_pattern, arena_, trace_stream_));
      SetValue(&let_var_pattern, pattern_value);
      return Success();
  }
}

auto TypeChecker::TypeCheckStmt(Nonnull<Statement*> s,
                                const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking statement " << *s << "\n";
  }
  switch (s->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&match.expression(), impl_scope));
      std::vector<Match::Clause> new_clauses;
      std::optional<Nonnull<const Value*>> expected_type;
      for (auto& clause : match.clauses()) {
        ImplScope clause_scope;
        clause_scope.AddParent(&impl_scope);
        // FIXME: Should user-defined conversions be permitted in `match`
        // statements? When would we run them? See #1283.
        CARBON_RETURN_IF_ERROR(TypeCheckPattern(
            &clause.pattern(), &match.expression().static_type(), clause_scope,
            ValueCategory::Let));
        if (expected_type.has_value()) {
          // FIXME: For now, we require all patterns to have the same type. If
          // that's not the same type as the scrutinee, we will convert the
          // scrutinee. We might want to instead allow a different conversion
          // to be performed for each pattern.
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              clause.pattern().source_loc(), "`match` pattern type",
              expected_type.value(), &clause.pattern().static_type()));
        } else {
          expected_type = &clause.pattern().static_type();
        }
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
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&var.init(), impl_scope));
      const Value& rhs_ty = var.init().static_type();
      // TODO: If the pattern contains a binding that implies a new impl is
      // available, should that remain in scope for as long as its binding?
      // ```
      // var a: (T:! Widget) = ...;
      // // Is the `impl T as Widget` in scope here?
      // a.(Widget.F)();
      // ```
      ImplScope var_scope;
      var_scope.AddParent(&impl_scope);
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var.pattern(), &rhs_ty,
                                              var_scope, var.value_category()));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_init,
          ImplicitlyConvert("initializer of variable", impl_scope, &var.init(),
                            &var.pattern().static_type()));
      var.set_init(converted_init);
      return Success();
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&assign.rhs(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&assign.lhs(), impl_scope));
      if (assign.lhs().value_category() != ValueCategory::Var) {
        return CompilationError(assign.source_loc())
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
    case StatementKind::Return: {
      auto& ret = cast<Return>(*s);
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
//
// TODO: the current rule is an extremely simplistic placeholder, with
// many false negatives.
static auto IsExhaustive(const Match& match) -> bool {
  for (const Match::Clause& clause : match.clauses()) {
    // A pattern consisting of a single variable binding is guaranteed to match.
    if (clause.pattern().kind() == PatternKind::BindingPattern) {
      return true;
    }
  }
  return false;
}

auto TypeChecker::ExpectReturnOnAllPaths(
    std::optional<Nonnull<Statement*>> opt_stmt, SourceLocation source_loc)
    -> ErrorOr<Success> {
  if (!opt_stmt) {
    return CompilationError(source_loc)
           << "control-flow reaches end of function that provides a `->` "
              "return type without reaching a return statement";
  }
  Nonnull<Statement*> stmt = *opt_stmt;
  switch (stmt->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*stmt);
      if (!IsExhaustive(match)) {
        return CompilationError(source_loc)
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
        return CompilationError(stmt->source_loc())
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
    case StatementKind::Return:
      return Success();
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
    case StatementKind::Assign:
    case StatementKind::ExpressionStatement:
    case StatementKind::While:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::VariableDefinition:
      return CompilationError(stmt->source_loc())
             << "control-flow reaches end of function that provides a `->` "
                "return type without reaching a return statement";
  }
}

// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
auto TypeChecker::DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
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
    // We ignore the return value because return type expressions can't bring
    // new types into scope.
    // Should we be doing SetConstantValue instead? -Jeremy
    // And shouldn't the type of this be Type?
    CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> ret_type,
                            TypeCheckTypeExp(*return_expression, function_scope,
                                             /*concrete=*/false));
    f->return_term().set_static_type(ret_type);
  } else if (f->return_term().is_omitted()) {
    f->return_term().set_static_type(TupleValue::Empty());
  } else {
    // We have to type-check the body in order to determine the return type.
    if (!f->body().has_value()) {
      return CompilationError(f->return_term().source_loc())
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
      &f->param_pattern().static_type(), generic_parameters,
      &f->return_term().static_type(), deduced_bindings, impl_bindings));
  SetConstantValue(f, arena_->New<FunctionValue>(f));

  if (f->name() == "Main") {
    if (!f->return_term().type_expression().has_value()) {
      return CompilationError(f->return_term().source_loc())
             << "`Main` must have an explicit return type";
    }
    CARBON_RETURN_IF_ERROR(ExpectExactType(
        f->return_term().source_loc(), "return type of `Main`",
        arena_->New<IntType>(), &f->return_term().static_type()));
    // TODO: Check that main doesn't have any parameters.
  }

  if (trace_stream_) {
    **trace_stream_ << "** finished declaring function " << f->name()
                    << " of type " << f->static_type() << "\n";
  }
  return Success();
}

auto TypeChecker::TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                               const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** checking function " << f->name() << "\n";
  }
  // if f->return_term().is_auto(), the function body was already
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
  BindingMap generic_args;
  for (auto* binding : bindings) {
    // binding.symbolic_identity() set by call to `TypeCheckPattern(...)`
    // above and/or by any enclosing generic classes.
    generic_args[binding] = *binding->symbolic_identity();
  }
  Nonnull<NominalClassType*> self_type =
      arena_->New<NominalClassType>(class_decl, generic_args);
  SetConstantValue(self, self_type);
  self->set_static_type(arena_->New<TypeOfClassType>(self_type));

  // The declarations of the members may refer to the class, so we must set the
  // constant value of the class and its static type before we start processing
  // the members.
  if (class_decl->type_params().has_value()) {
    // TODO: The `enclosing_bindings` should be tracked in the parameterized
    // entity name so that they can be included in the eventual type.
    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(class_decl,
                                             *class_decl->type_params());
    SetConstantValue(class_decl, param_name);
    class_decl->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
  } else {
    SetConstantValue(class_decl, self_type);
    class_decl->set_static_type(&self->static_type());
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
  for (Nonnull<Declaration*> m : class_decl->members()) {
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, class_scope));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking class " << class_decl->name()
                    << "\n";
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

  if (iface_decl->params().has_value()) {
    CARBON_RETURN_IF_ERROR(TypeCheckPattern(*iface_decl->params(), std::nullopt,
                                            iface_scope, ValueCategory::Let));
    if (trace_stream_) {
      **trace_stream_ << iface_scope;
    }

    Nonnull<ParameterizedEntityName*> param_name =
        arena_->New<ParameterizedEntityName>(iface_decl, *iface_decl->params());
    SetConstantValue(iface_decl, param_name);
    iface_decl->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
  } else {
    Nonnull<InterfaceType*> iface_type = arena_->New<InterfaceType>(iface_decl);
    SetConstantValue(iface_decl, iface_type);
    iface_decl->set_static_type(arena_->New<TypeOfInterfaceType>(iface_type));
  }

  // Process the Self parameter.
  CARBON_RETURN_IF_ERROR(TypeCheckPattern(iface_decl->self(), std::nullopt,
                                          iface_scope, ValueCategory::Let));

  ScopeInfo iface_scope_info = ScopeInfo::ForNonClassScope(&iface_scope);
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, iface_scope_info));
  }
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
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, iface_scope));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking interface " << iface_decl->name()
                    << "\n";
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
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;

  // Bring the deduced parameters into scope.
  for (Nonnull<GenericBinding*> deduced : impl_decl->deduced_parameters()) {
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
  // Static type set in call to `TypeCheckExp(...)` above.
  self->set_static_type(&impl_decl->impl_type()->static_type());

  // Check and interpret the interface.
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<const Value*> written_iface_type,
      TypeCheckTypeExp(&impl_decl->interface(), impl_scope));
  const auto* iface_type = dyn_cast<InterfaceType>(written_iface_type);
  if (!iface_type) {
    return CompilationError(impl_decl->interface().source_loc())
           << "expected constraint after `as`, found value of type "
           << *written_iface_type;
  }

  const auto& iface_decl = iface_type->declaration();
  impl_decl->set_interface_type(iface_type);

  // Bring this impl into the enclosing non-class scope.
  auto impl_id =
      arena_->New<IdentifierExpression>(impl_decl->source_loc(), "impl");
  impl_id->set_value_node(impl_decl);
  {
    // The deduced bindings are the parameters for all enclosing classes
    // followed by any deduced parameters written on the `impl` declaration
    // itself.
    std::vector<Nonnull<const GenericBinding*>> deduced_bindings =
        scope_info.bindings;
    deduced_bindings.insert(deduced_bindings.end(),
                            impl_decl->deduced_parameters().begin(),
                            impl_decl->deduced_parameters().end());
    scope_info.innermost_non_class_scope->Add(
        iface_type, std::move(deduced_bindings), impl_type_value, impl_bindings,
        impl_id);
  }

  // Declare the impl members.
  ScopeInfo impl_scope_info = ScopeInfo::ForNonClassScope(&impl_scope);
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, impl_scope_info));
  }
  // Check that the interface is satisfied by the impl members.
  for (Nonnull<Declaration*> m : iface_decl.members()) {
    if (std::optional<std::string> mem_name = GetName(*m);
        mem_name.has_value()) {
      if (std::optional<Nonnull<const Declaration*>> mem =
              FindMember(*mem_name, impl_decl->members());
          mem.has_value()) {
        BindingMap binding_map = iface_type->args();
        binding_map[iface_decl.self()] = impl_type_value;
        Nonnull<const Value*> iface_mem_type =
            Substitute(binding_map, &m->static_type());
        // FIXME: How should the signature in the implementation be permitted
        // to differ from the signature in the interface?
        CARBON_RETURN_IF_ERROR(
            ExpectExactType((*mem)->source_loc(), "member of implementation",
                            iface_mem_type, &(*mem)->static_type()));
      } else {
        return CompilationError(impl_decl->source_loc())
               << "implementation missing " << *mem_name;
      }
    }
  }
  impl_decl->set_constant_value(arena_->New<Witness>(impl_decl));
  if (trace_stream_) {
    **trace_stream_ << "** finished declaring impl " << *impl_decl->impl_type()
                    << " as " << impl_decl->interface() << "\n";
  }
  return Success();
}

auto TypeChecker::TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                           const ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << *impl_decl << "\n";
  }
  // Bring the impls from the parameters into scope.
  ImplScope impl_scope;
  impl_scope.AddParent(&enclosing_scope);
  BringImplsIntoScope(impl_decl->impl_bindings(), impl_scope);
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, impl_scope));
  }
  if (trace_stream_) {
    **trace_stream_ << "finished checking impl\n";
  }
  return Success();
}

auto TypeChecker::DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                           const ScopeInfo& scope_info)
    -> ErrorOr<Success> {
  std::vector<NamedValue> alternatives;
  for (Nonnull<AlternativeSignature*> alternative : choice->alternatives()) {
    CARBON_ASSIGN_OR_RETURN(auto signature,
                            TypeCheckTypeExp(&alternative->signature(),
                                             *scope_info.innermost_scope));
    alternatives.push_back({.name = alternative->name(), .value = signature});
  }
  auto ct = arena_->New<ChoiceType>(choice->name(), std::move(alternatives));
  SetConstantValue(choice, ct);
  choice->set_static_type(arena_->New<TypeOfChoiceType>(ct));
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
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::TupleValue:
    case Value::Kind::Witness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::MemberName:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
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
      return false;

    case Value::Kind::FunctionType:
    case Value::Kind::InterfaceType:
    case Value::Kind::TypeType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
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
    return CompilationError(alias->source_loc())
           << "invalid target for alias declaration";
  }

  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> target,
                          InterpExp(&alias->target(), arena_, trace_stream_));

  SetConstantValue(alias, target);
  alias->set_static_type(&alias->target().static_type());
  return Success();
}

auto TypeChecker::TypeCheck(AST& ast) -> ErrorOr<Success> {
  ImplScope impl_scope;
  ScopeInfo top_level_scope_info = ScopeInfo::ForNonClassScope(&impl_scope);
  for (Nonnull<Declaration*> declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(
        DeclareDeclaration(declaration, top_level_scope_info));
  }
  for (Nonnull<Declaration*> decl : ast.declarations) {
    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(decl, impl_scope));
    // Check to see if this declaration is a builtin.
    // FIXME: Only do this when type-checking the prelude.
    builtins_.Register(decl);
  }
  CARBON_RETURN_IF_ERROR(TypeCheckExp(*ast.main_call, impl_scope));
  return Success();
}

auto TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d,
                                       const ImplScope& impl_scope)
    -> ErrorOr<Success> {
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
    case DeclarationKind::FunctionDeclaration:
      CARBON_RETURN_IF_ERROR(TypeCheckFunctionDeclaration(
          &cast<FunctionDeclaration>(*d), impl_scope));
      return Success();
    case DeclarationKind::ClassDeclaration:
      CARBON_RETURN_IF_ERROR(
          TypeCheckClassDeclaration(&cast<ClassDeclaration>(*d), impl_scope));
      return Success();
    case DeclarationKind::ChoiceDeclaration:
      CARBON_RETURN_IF_ERROR(
          TypeCheckChoiceDeclaration(&cast<ChoiceDeclaration>(*d), impl_scope));
      return Success();
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      if (var.has_initializer()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&var.initializer(), impl_scope));
      }
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        return CompilationError(var.source_loc())
               << "Type of a top-level variable must be an expression.";
      }
      if (var.has_initializer()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_initializer,
            ImplicitlyConvert("initializer of variable", impl_scope,
                              &var.initializer(), &var.static_type()));
        var.set_initializer(converted_initializer);
      }
      return Success();
    }
    case DeclarationKind::SelfDeclaration: {
      CARBON_FATAL() << "Unreachable TypeChecker `Self` declaration";
    }
    case DeclarationKind::AliasDeclaration: {
      return Success();
    }
  }
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
      auto& func_def = cast<FunctionDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareFunctionDeclaration(&func_def, scope_info));
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      CARBON_RETURN_IF_ERROR(DeclareClassDeclaration(&class_decl, scope_info));
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
        return CompilationError(var.binding().type().source_loc())
               << "Expected expression for variable type";
      }
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&var.binding(), std::nullopt,
                                              *scope_info.innermost_scope,
                                              var.value_category()));
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> declared_type,
                              InterpExp(&type, arena_, trace_stream_));
      var.set_static_type(declared_type);
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
  return Success();
}

template <typename T>
void TypeChecker::SetConstantValue(Nonnull<T*> value_node,
                                   Nonnull<const Value*> value) {
  std::optional<Nonnull<const Value*>> old_value = value_node->constant_value();
  CARBON_CHECK(!old_value.has_value());
  value_node->set_constant_value(value);
  CARBON_CHECK(constants_.insert(value_node).second);
}

void TypeChecker::PrintConstants(llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& value_node : constants_) {
    out << sep << value_node;
  }
}

}  // namespace Carbon
