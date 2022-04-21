// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/type_checker.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error_builders.h"
#include "executable_semantics/interpreter/impl_scope.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

using llvm::cast;
using llvm::dyn_cast;
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
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::InterfaceType:
    case Value::Kind::Witness:
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
    case Value::Kind::NominalClassType: {
      const NominalClassType& class_type = cast<NominalClassType>(*value);
      // A NominalClassType is concrete if
      // 1) it is not a generic class (has no type parameters), or
      // 2) it is a generic class applied to some type arguments.
      return !class_type.declaration().type_params().has_value() ||
             !class_type.type_args().empty();
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
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::InterfaceType:
    case Value::Kind::Witness:
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
    case Value::Kind::NominalClassType: {
      const NominalClassType& class_type = cast<NominalClassType>(*value);
      // A NominalClassType is concrete if
      // 1) it is not a generic class (has no type parameters), or
      // 2) it is a generic class applied to some type arguments.
      return !class_type.declaration().type_params().has_value() ||
             !class_type.type_args().empty();
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

auto TypeChecker::FieldTypesImplicitlyConvertible(
    llvm::ArrayRef<NamedValue> source_fields,
    llvm::ArrayRef<NamedValue> destination_fields) const {
  if (source_fields.size() != destination_fields.size()) {
    return false;
  }
  for (const auto& source_field : source_fields) {
    auto it = std::find_if(destination_fields.begin(), destination_fields.end(),
                           [&](const NamedValue& field) {
                             return field.name == source_field.name;
                           });
    if (it == destination_fields.end() ||
        !IsImplicitlyConvertible(source_field.value, it->value)) {
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
    Nonnull<const Value*> source, Nonnull<const Value*> destination) const
    -> bool {
  CHECK(IsConcreteType(source));
  CHECK(IsConcreteType(destination));
  if (TypeEqual(source, destination)) {
    return true;
  }
  switch (source->kind()) {
    case Value::Kind::StructType:
      switch (destination->kind()) {
        case Value::Kind::StructType:
          return FieldTypesImplicitlyConvertible(
              cast<StructType>(*source).fields(),
              cast<StructType>(*destination).fields());
        case Value::Kind::NominalClassType:
          return FieldTypesImplicitlyConvertible(
              cast<StructType>(*source).fields(),
              FieldTypes(cast<NominalClassType>(*destination)));
        default:
          return false;
      }
    case Value::Kind::TupleValue: {
      const auto& source_tuple = cast<TupleValue>(*source);
      switch (destination->kind()) {
        case Value::Kind::TupleValue: {
          const auto& destination_tuple = cast<TupleValue>(*destination);
          if (source_tuple.elements().size() !=
              destination_tuple.elements().size()) {
            return false;
          }
          for (size_t i = 0; i < source_tuple.elements().size(); ++i) {
            if (!IsImplicitlyConvertible(source_tuple.elements()[i],
                                         destination_tuple.elements()[i])) {
              return false;
            }
          }
          return true;
        }
        case Value::Kind::StaticArrayType: {
          const auto& destination_array = cast<StaticArrayType>(*destination);
          if (destination_array.size() != source_tuple.elements().size()) {
            return false;
          }
          for (Nonnull<const Value*> source_element : source_tuple.elements()) {
            if (!IsImplicitlyConvertible(source_element,
                                         &destination_array.element_type())) {
              return false;
            }
          }
          return true;
        }
        case Value::Kind::TypeType: {
          for (Nonnull<const Value*> source_element : source_tuple.elements()) {
            if (!IsImplicitlyConvertible(source_element, destination)) {
              return false;
            }
          }
          return true;
        }
        default:
          return false;
      }
    }
    case Value::Kind::TypeType:
      return destination->kind() == Value::Kind::InterfaceType;
    case Value::Kind::InterfaceType:
      return destination->kind() == Value::Kind::TypeType;
    case Value::Kind::TypeOfClassType: {
      const auto& class_type = cast<TypeOfClassType>(*source).class_type();
      return ((!class_type.declaration().type_params().has_value()) ||
              (!class_type.type_args().empty())) &&
             destination->kind() == Value::Kind::TypeType;
    }
    default:
      return false;
  }
}

auto TypeChecker::ExpectType(SourceLocation source_loc,
                             const std::string& context,
                             Nonnull<const Value*> expected,
                             Nonnull<const Value*> actual) const
    -> ErrorOr<Success> {
  if (!IsImplicitlyConvertible(actual, expected)) {
    return CompilationError(source_loc)
           << "type error in " << context << ": "
           << "'" << *actual << "' is not implicitly convertible to '"
           << *expected << "'";
  } else {
    return Success();
  }
}

auto TypeChecker::ArgumentDeduction(
    SourceLocation source_loc,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> type_params,
    BindingMap& deduced, Nonnull<const Value*> param_type,
    Nonnull<const Value*> arg_type) const -> ErrorOr<Success> {
  switch (param_type->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param_type);
      if (std::find(type_params.begin(), type_params.end(),
                    &var_type.binding()) != type_params.end()) {
        auto [it, success] = deduced.insert({&var_type.binding(), arg_type});
        if (!success) {
          // Variable already has a match.
          // TODO: can we allow implicit conversions here?
          RETURN_IF_ERROR(ExpectExactType(source_loc, "argument deduction",
                                          it->second, arg_type));
        }
      } else {
        RETURN_IF_ERROR(ExpectExactType(source_loc, "argument deduction",
                                        param_type, arg_type));
      }
      return Success();
    }
    case Value::Kind::TupleValue: {
      if (arg_type->kind() != Value::Kind::TupleValue) {
        return CompilationError(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param_type << "\n"
               << "actual: " << *arg_type;
      }
      const auto& param_tup = cast<TupleValue>(*param_type);
      const auto& arg_tup = cast<TupleValue>(*arg_type);
      if (param_tup.elements().size() != arg_tup.elements().size()) {
        return CompilationError(source_loc)
               << "mismatch in tuple sizes, expected "
               << param_tup.elements().size() << " but got "
               << arg_tup.elements().size();
      }
      for (size_t i = 0; i < param_tup.elements().size(); ++i) {
        RETURN_IF_ERROR(ArgumentDeduction(source_loc, type_params, deduced,
                                          param_tup.elements()[i],
                                          arg_tup.elements()[i]));
      }
      return Success();
    }
    case Value::Kind::StructType: {
      if (arg_type->kind() != Value::Kind::StructType) {
        return CompilationError(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param_type << "\n"
               << "actual: " << *arg_type;
      }
      const auto& param_struct = cast<StructType>(*param_type);
      const auto& arg_struct = cast<StructType>(*arg_type);
      if (param_struct.fields().size() != arg_struct.fields().size()) {
        return CompilationError(source_loc)
               << "mismatch in struct field counts, expected "
               << param_struct.fields().size() << " but got "
               << arg_struct.fields().size();
      }
      for (size_t i = 0; i < param_struct.fields().size(); ++i) {
        if (param_struct.fields()[i].name != arg_struct.fields()[i].name) {
          return CompilationError(source_loc)
                 << "mismatch in field names, " << param_struct.fields()[i].name
                 << " != " << arg_struct.fields()[i].name;
        }
        RETURN_IF_ERROR(ArgumentDeduction(source_loc, type_params, deduced,
                                          param_struct.fields()[i].value,
                                          arg_struct.fields()[i].value));
      }
      return Success();
    }
    case Value::Kind::FunctionType: {
      if (arg_type->kind() != Value::Kind::FunctionType) {
        return CompilationError(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param_type << "\n"
               << "actual: " << *arg_type;
      }
      const auto& param_fn = cast<FunctionType>(*param_type);
      const auto& arg_fn = cast<FunctionType>(*arg_type);
      // TODO: handle situation when arg has deduced parameters.
      RETURN_IF_ERROR(ArgumentDeduction(source_loc, type_params, deduced,
                                        &param_fn.parameters(),
                                        &arg_fn.parameters()));
      RETURN_IF_ERROR(ArgumentDeduction(source_loc, type_params, deduced,
                                        &param_fn.return_type(),
                                        &arg_fn.return_type()));
      return Success();
    }
    case Value::Kind::PointerType: {
      if (arg_type->kind() != Value::Kind::PointerType) {
        return CompilationError(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param_type << "\n"
               << "actual: " << *arg_type;
      }
      return ArgumentDeduction(source_loc, type_params, deduced,
                               &cast<PointerType>(*param_type).type(),
                               &cast<PointerType>(*arg_type).type());
    }
    // Nothing to do in the case for `auto`.
    case Value::Kind::AutoType: {
      return Success();
    }
    case Value::Kind::NominalClassType: {
      const auto& param_class_type = cast<NominalClassType>(*param_type);
      if (arg_type->kind() == Value::Kind::NominalClassType) {
        const auto& arg_class_type = cast<NominalClassType>(*arg_type);
        if (param_class_type.declaration().name() ==
            arg_class_type.declaration().name()) {
          for (const auto& [ty, param_ty] : param_class_type.type_args()) {
            RETURN_IF_ERROR(
                ArgumentDeduction(source_loc, type_params, deduced, param_ty,
                                  arg_class_type.type_args().at(ty)));
          }
          return Success();
        }
      }
      return CompilationError(source_loc)
             << "type error in argument deduction\n"
             << "expected: " << *param_type << "\n"
             << "actual: " << *arg_type;
    }
    // For the following cases, we check for type convertability.
    case Value::Kind::StaticArrayType:
    case Value::Kind::ContinuationType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      return ExpectType(source_loc, "argument deduction", param_type, arg_type);
    // The rest of these cases should never happen.
    case Value::Kind::Witness:
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
      FATAL() << "In ArgumentDeduction: expected type, not value "
              << *param_type;
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
      return arena_->New<FunctionType>(
          std::vector<Nonnull<const GenericBinding*>>(), param, ret,
          std::vector<Nonnull<const ImplBinding*>>());
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
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      return type;
    // The rest of these cases should never happen.
    case Value::Kind::Witness:
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
      FATAL() << "In Substitute: expected type, not value " << *type;
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
    **trace_stream_ << "checking [";
    llvm::ListSeparator sep;
    for (Nonnull<const GenericBinding*> deduced_param : impl.deduced) {
      **trace_stream_ << sep << *deduced_param;
    }
    **trace_stream_ << "] " << *impl.type << " as " << *impl.interface << "\n";
  }
  if (!TypeEqual(&iface, impl.interface)) {
    return std::nullopt;
  }
  if (impl.deduced.empty() && impl.impl_bindings.empty()) {
    // case: impl is a non-generic impl.
    if (!TypeEqual(impl_type, impl.type)) {
      return std::nullopt;
    }
    return impl.impl;
  } else {
    // case: impl is a generic impl.
    BindingMap deduced_type_args;
    ErrorOr<Success> e = ArgumentDeduction(
        source_loc, impl.deduced, deduced_type_args, impl.type, impl_type);
    if (trace_stream_) {
      **trace_stream_ << "match results: {";
      llvm::ListSeparator sep;
      for (const auto& [binding, val] : deduced_type_args) {
        **trace_stream_ << sep << *binding << " = " << *val;
      }
      **trace_stream_ << "}\n";
    }
    if (!e.ok()) {
      return std::nullopt;
    }
    // Check that all the type parameters were deduced.
    // Find impls for all the impls bindings.
    ImplExpMap impls;
    ErrorOr<Success> m = SatisfyImpls(impl.impl_bindings, impl_scope,
                                      source_loc, deduced_type_args, impls);
    if (!m.ok()) {
      return std::nullopt;
    }
    if (trace_stream_) {
      **trace_stream_ << "matched with " << *impl.type << " as "
                      << *impl.interface << "\n\n";
    }
    return arena_->New<InstantiateImpl>(source_loc, impl.impl,
                                        deduced_type_args, impls);
  }
}

auto TypeChecker::SatisfyImpls(
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
    const ImplScope& impl_scope, SourceLocation source_loc,
    BindingMap& deduced_type_args, ImplExpMap& impls) const
    -> ErrorOr<Success> {
  for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
    ASSIGN_OR_RETURN(
        Nonnull<Expression*> impl,
        impl_scope.Resolve(impl_binding->interface(),
                           deduced_type_args[impl_binding->type_var()],
                           source_loc, *this));
    impls.emplace(impl_binding, impl);
  }
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
  switch (e->kind()) {
    case ExpressionKind::InstantiateImpl: {
      FATAL() << "instantiate impl nodes are generated during type checking";
    }
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(&index.aggregate(), impl_scope));
      RETURN_IF_ERROR(TypeCheckExp(&index.offset(), impl_scope));
      const Value& aggregate_type = index.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::TupleValue: {
          const auto& tuple_type = cast<TupleValue>(aggregate_type);
          RETURN_IF_ERROR(ExpectExactType(index.offset().source_loc(),
                                          "tuple index", arena_->New<IntType>(),
                                          &index.offset().static_type()));
          ASSIGN_OR_RETURN(auto offset_value,
                           InterpExp(&index.offset(), arena_, trace_stream_));
          int i = cast<IntValue>(*offset_value).value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            return CompilationError(e->source_loc())
                   << "index " << i << " is out of range for type "
                   << tuple_type;
          }
          index.set_static_type(tuple_type.elements()[i]);
          index.set_value_category(index.aggregate().value_category());
          return Success();
        }
        case Value::Kind::StaticArrayType: {
          RETURN_IF_ERROR(ExpectExactType(index.offset().source_loc(),
                                          "array index", arena_->New<IntType>(),
                                          &index.offset().static_type()));
          index.set_static_type(
              &cast<StaticArrayType>(aggregate_type).element_type());
          index.set_value_category(index.aggregate().value_category());
          return Success();
        }
        default:
          return CompilationError(e->source_loc()) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      for (auto& arg : cast<TupleLiteral>(*e).fields()) {
        RETURN_IF_ERROR(TypeCheckExp(arg, impl_scope));
        arg_types.push_back(&arg->static_type());
      }
      e->set_static_type(arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      e->set_static_type(arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      for (auto& arg : struct_type.fields()) {
        RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        ASSIGN_OR_RETURN(auto value,
                         InterpExp(&arg.expression(), arena_, trace_stream_));
        RETURN_IF_ERROR(
            ExpectIsConcreteType(arg.expression().source_loc(), value));
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
    case ExpressionKind::FieldAccessExpression: {
      auto& access = cast<FieldAccessExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(&access.aggregate(), impl_scope));
      const Value& aggregate_type = access.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(aggregate_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              access.set_static_type(field_type);
              access.set_value_category(access.aggregate().value_category());
              return Success();
            }
          }
          return CompilationError(access.source_loc())
                 << "struct " << struct_type << " does not have a field named "
                 << access.field();
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(aggregate_type);
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), t_class.declaration().members());
              member.has_value()) {
            Nonnull<const Value*> field_type =
                Substitute(t_class.type_args(), &(*member)->static_type());
            access.set_static_type(field_type);
            switch ((*member)->kind()) {
              case DeclarationKind::VariableDeclaration:
                access.set_value_category(access.aggregate().value_category());
                break;
              case DeclarationKind::FunctionDeclaration:
                access.set_value_category(ValueCategory::Let);
                break;
              default:
                FATAL() << "member " << access.field()
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
              cast<TypeOfChoiceType>(aggregate_type).choice_type();
          std::optional<Nonnull<const Value*>> parameter_types =
              choice.FindAlternative(access.field());
          if (!parameter_types.has_value()) {
            return CompilationError(e->source_loc())
                   << "choice " << choice.name()
                   << " does not have a field named " << access.field();
          }
          access.set_static_type(arena_->New<FunctionType>(
              std::vector<Nonnull<const GenericBinding*>>(), *parameter_types,
              &aggregate_type, std::vector<Nonnull<const ImplBinding*>>()));
          access.set_value_category(ValueCategory::Let);
          return Success();
        }
        case Value::Kind::TypeOfClassType: {
          const NominalClassType& class_type =
              cast<TypeOfClassType>(aggregate_type).class_type();
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
            return CompilationError(access.source_loc())
                   << access.field() << " is not a class function";
          } else {
            return CompilationError(access.source_loc())
                   << class_type << " does not have a class function named "
                   << access.field();
          }
        }
        case Value::Kind::VariableType: {
          // This case handles access to a method on a receiver whose type
          // is a type variable. For example, `x.foo` where the type of
          // `x` is `T` and `foo` and `T` implements an interface that
          // includes `foo`.
          const VariableType& var_type = cast<VariableType>(aggregate_type);
          const Value& typeof_var = var_type.binding().static_type();
          switch (typeof_var.kind()) {
            case Value::Kind::InterfaceType: {
              const auto& iface_type = cast<InterfaceType>(typeof_var);
              const InterfaceDeclaration& iface_decl = iface_type.declaration();
              if (std::optional<Nonnull<const Declaration*>> member =
                      FindMember(access.field(), iface_decl.members());
                  member.has_value()) {
                const Value& member_type = (*member)->static_type();
                std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
                    self_map;
                self_map[iface_decl.self()] = &var_type;
                Nonnull<const Value*> inst_member_type =
                    Substitute(self_map, &member_type);
                access.set_static_type(inst_member_type);
                CHECK(var_type.binding().impl_binding().has_value());
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
                     << "field access, unexpected " << aggregate_type
                     << " of non-interface type " << typeof_var << " in " << *e;
          }
          break;
        }
        case Value::Kind::InterfaceType: {
          // This case handles access to a class function from a type variable.
          // If `T` is a type variable and `foo` is a class function in an
          // interface implemented by `T`, then `T.foo` accesses the `foo` class
          // function of `T`.
          ASSIGN_OR_RETURN(
              Nonnull<const Value*> var_addr,
              InterpExp(&access.aggregate(), arena_, trace_stream_));
          const VariableType& var_type = cast<VariableType>(*var_addr);
          const InterfaceType& iface_type = cast<InterfaceType>(aggregate_type);
          const InterfaceDeclaration& iface_decl = iface_type.declaration();
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), iface_decl.members());
              member.has_value()) {
            const Value& member_type = (*member)->static_type();
            Nonnull<const Value*> inst_member_type =
                Substitute({{iface_decl.self(), &var_type}}, &member_type);
            access.set_static_type(inst_member_type);
            CHECK(var_type.binding().impl_binding().has_value());
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
                 << "field access, unexpected " << aggregate_type << " in "
                 << *e;
      }
    }
    case ExpressionKind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      if (ident.value_node().base().kind() ==
          AstNodeKind::FunctionDeclaration) {
        const auto& function =
            cast<FunctionDeclaration>(ident.value_node().base());
        if (!function.has_static_type()) {
          CHECK(function.return_term().is_auto());
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
        RETURN_IF_ERROR(TypeCheckExp(argument, impl_scope));
        ts.push_back(&argument->static_type());
      }
      switch (op.op()) {
        case Operator::Neg:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "negation",
                                          arena_->New<IntType>(), ts[0]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Add:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "addition(1)",
                                          arena_->New<IntType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "addition(2)",
                                          arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Sub:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "subtraction(1)",
                                          arena_->New<IntType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "subtraction(2)",
                                          arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Mul:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "multiplication(1)",
                                          arena_->New<IntType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "multiplication(2)",
                                          arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::And:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(1)",
                                          arena_->New<BoolType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(2)",
                                          arena_->New<BoolType>(), ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Or:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(1)",
                                          arena_->New<BoolType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(2)",
                                          arena_->New<BoolType>(), ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Not:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "!",
                                          arena_->New<BoolType>(), ts[0]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Eq:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "==", ts[0], ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Deref:
          RETURN_IF_ERROR(ExpectPointerType(e->source_loc(), "*", ts[0]));
          op.set_static_type(&cast<PointerType>(*ts[0]).type());
          op.set_value_category(ValueCategory::Var);
          return Success();
        case Operator::Ptr:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "*",
                                          arena_->New<TypeType>(), ts[0]));
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
      RETURN_IF_ERROR(TypeCheckExp(&call.function(), impl_scope));
      RETURN_IF_ERROR(TypeCheckExp(&call.argument(), impl_scope));
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          Nonnull<const Value*> parameters = &fun_t.parameters();
          Nonnull<const Value*> return_type = &fun_t.return_type();
          if (!fun_t.deduced().empty()) {
            BindingMap deduced_type_args;
            RETURN_IF_ERROR(ArgumentDeduction(e->source_loc(), fun_t.deduced(),
                                              deduced_type_args, parameters,
                                              &call.argument().static_type()));
            call.set_deduced_args(deduced_type_args);
            for (Nonnull<const GenericBinding*> deduced_param :
                 fun_t.deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (auto it = deduced_type_args.find(deduced_param);
                  it == deduced_type_args.end()) {
                return CompilationError(e->source_loc())
                       << "could not deduce type argument for type parameter "
                       << deduced_param->name() << "\n"
                       << "in " << call;
              }
            }
            parameters = Substitute(deduced_type_args, parameters);
            return_type = Substitute(deduced_type_args, return_type);
            // Find impls for all the impl bindings of the function.
            ImplExpMap impls;
            RETURN_IF_ERROR(SatisfyImpls(fun_t.impl_bindings(), impl_scope,
                                         e->source_loc(), deduced_type_args,
                                         impls));
            call.set_impls(impls);
          } else {
            // No deduced parameters. Check that the argument types
            // are convertible to the parameter types.
            RETURN_IF_ERROR(ExpectType(e->source_loc(), "call", parameters,
                                       &call.argument().static_type()));
          }
          call.set_static_type(return_type);
          call.set_value_category(ValueCategory::Let);
          return Success();
        }
        case Value::Kind::TypeOfClassType: {
          // This case handles the application of a generic class to
          // a type argument, such as Point(i32).
          const ClassDeclaration& class_decl =
              cast<TypeOfClassType>(call.function().static_type())
                  .class_type()
                  .declaration();
          BindingMap generic_args;
          if (class_decl.type_params().has_value()) {
            if (trace_stream_) {
              **trace_stream_ << "pattern matching type params and args\n";
            }
            RETURN_IF_ERROR(
                ExpectType(call.source_loc(), "call",
                           &(*class_decl.type_params())->static_type(),
                           &call.argument().static_type()));
            ASSIGN_OR_RETURN(
                Nonnull<const Value*> arg,
                InterpExp(&call.argument(), arena_, trace_stream_));
            CHECK(PatternMatch(&(*class_decl.type_params())->value(), arg,
                               call.source_loc(), std::nullopt, generic_args,
                               trace_stream_));
          } else {
            return CompilationError(call.source_loc())
                   << "attempt to instantiate a non-generic class: " << *e;
          }
          // Find impls for all the impl bindings of the class.
          ImplExpMap impls;
          for (const auto& [binding, val] : generic_args) {
            if (binding->impl_binding().has_value()) {
              Nonnull<const ImplBinding*> impl_binding =
                  *binding->impl_binding();
              switch (impl_binding->interface()->kind()) {
                case Value::Kind::InterfaceType: {
                  ASSIGN_OR_RETURN(
                      Nonnull<Expression*> impl,
                      impl_scope.Resolve(impl_binding->interface(),
                                         generic_args[binding],
                                         call.source_loc(), *this));
                  impls.emplace(impl_binding, impl);
                  break;
                }
                case Value::Kind::TypeType:
                  break;
                default:
                  return CompilationError(e->source_loc())
                         << "unexpected type of deduced parameter "
                         << *impl_binding->interface();
              }
            }
          }
          Nonnull<NominalClassType*> class_type =
              arena_->New<NominalClassType>(&class_decl, generic_args, impls);
          call.set_impls(impls);
          call.set_static_type(arena_->New<TypeOfClassType>(class_type));
          call.set_value_category(ValueCategory::Let);
          return Success();
        }
        default: {
          return CompilationError(e->source_loc())
                 << "in call, expected a function\n"
                 << *e << "\nnot an operator of type "
                 << call.function().static_type() << "\n";
        }
      }
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fn = cast<FunctionTypeLiteral>(*e);
      ASSIGN_OR_RETURN(Nonnull<const Value*> param_type,
                       InterpExp(&fn.parameter(), arena_, trace_stream_));
      RETURN_IF_ERROR(
          ExpectIsConcreteType(fn.parameter().source_loc(), param_type));
      ASSIGN_OR_RETURN(Nonnull<const Value*> ret_type,
                       InterpExp(&fn.return_type(), arena_, trace_stream_));
      RETURN_IF_ERROR(
          ExpectIsConcreteType(fn.return_type().source_loc(), ret_type));
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
      RETURN_IF_ERROR(TypeCheckExp(&intrinsic_exp.args(), impl_scope));
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          if (intrinsic_exp.args().fields().size() != 1) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_print takes 1 argument";
          }
          RETURN_IF_ERROR(
              ExpectType(e->source_loc(), "__intrinsic_print argument",
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
      RETURN_IF_ERROR(TypeCheckExp(&if_expr.condition(), impl_scope));
      RETURN_IF_ERROR(ExpectType(if_expr.source_loc(), "condition of `if`",
                                 arena_->New<BoolType>(),
                                 &if_expr.condition().static_type()));

      // TODO: Compute the common type and convert both operands to it.
      RETURN_IF_ERROR(TypeCheckExp(&if_expr.then_expression(), impl_scope));
      RETURN_IF_ERROR(TypeCheckExp(&if_expr.else_expression(), impl_scope));
      RETURN_IF_ERROR(
          ExpectExactType(e->source_loc(), "expression of `if` expression",
                          &if_expr.then_expression().static_type(),
                          &if_expr.else_expression().static_type()));
      e->set_static_type(&if_expr.then_expression().static_type());
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << *e;
    case ExpressionKind::ArrayTypeLiteral: {
      auto& array_literal = cast<ArrayTypeLiteral>(*e);
      RETURN_IF_ERROR(
          TypeCheckExp(&array_literal.element_type_expression(), impl_scope));
      ASSIGN_OR_RETURN(Nonnull<const Value*> element_type,
                       InterpExp(&array_literal.element_type_expression(),
                                 arena_, trace_stream_));
      RETURN_IF_ERROR(ExpectIsConcreteType(
          array_literal.element_type_expression().source_loc(), element_type));

      RETURN_IF_ERROR(
          TypeCheckExp(&array_literal.size_expression(), impl_scope));
      RETURN_IF_ERROR(
          ExpectExactType(array_literal.size_expression().source_loc(),
                          "array size", arena_->New<IntType>(),
                          &array_literal.size_expression().static_type()));
      ASSIGN_OR_RETURN(
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

void TypeChecker::AddPatternImpls(Nonnull<Pattern*> p, ImplScope& impl_scope) {
  switch (p->kind()) {
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(*p);
      CHECK(binding.impl_binding().has_value());
      Nonnull<const ImplBinding*> impl_binding = *binding.impl_binding();
      auto impl_id = arena_->New<IdentifierExpression>(p->source_loc(), "impl");
      impl_id->set_value_node(impl_binding);
      impl_scope.Add(impl_binding->interface(),
                     *impl_binding->type_var()->symbolic_identity(), impl_id);
      return;
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      for (Nonnull<Pattern*> field : tuple.fields()) {
        AddPatternImpls(field, impl_scope);
      }
      return;
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      AddPatternImpls(&alternative.arguments(), impl_scope);
      return;
    }
    case PatternKind::VarPattern: {
      auto& var_pattern = cast<VarPattern>(*p);
      AddPatternImpls(&var_pattern.pattern(), impl_scope);
      return;
    }
    case PatternKind::ExpressionPattern:
    case PatternKind::AutoPattern:
    case PatternKind::BindingPattern:
      return;
  }
}

auto TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, std::optional<Nonnull<const Value*>> expected,
    const ImplScope& impl_scope, ValueCategory enclosing_value_category)
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
      if (!GetBindings(binding.type()).empty()) {
        return CompilationError(binding.type().source_loc())
               << "The type of a binding pattern cannot contain bindings.";
      }
      RETURN_IF_ERROR(TypeCheckPattern(&binding.type(), std::nullopt,
                                       impl_scope, enclosing_value_category));
      ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                       InterpPattern(&binding.type(), arena_, trace_stream_));
      RETURN_IF_ERROR(ExpectIsType(binding.source_loc(), type));
      if (expected) {
        if (IsConcreteType(type)) {
          RETURN_IF_ERROR(
              ExpectType(p->source_loc(), "name binding", type, *expected));
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
      RETURN_IF_ERROR(ExpectIsConcreteType(binding.source_loc(), type));
      binding.set_static_type(type);
      ASSIGN_OR_RETURN(Nonnull<const Value*> binding_value,
                       InterpPattern(&binding, arena_, trace_stream_));
      SetValue(&binding, binding_value);

      if (!binding.has_value_category()) {
        binding.set_value_category(enclosing_value_category);
      }
      return Success();
    }
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(*p);
      RETURN_IF_ERROR(TypeCheckExp(&binding.type(), impl_scope));
      ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                       InterpExp(&binding.type(), arena_, trace_stream_));
      if (expected) {
        return CompilationError(binding.type().source_loc())
               << "Generic binding may not occur in pattern with expected "
                  "type: "
               << binding;
      }
      binding.set_static_type(type);
      ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                       InterpPattern(&binding, arena_, trace_stream_));
      binding.set_symbolic_identity(val);
      Nonnull<ImplBinding*> impl_binding = arena_->New<ImplBinding>(
          binding.source_loc(), &binding, &binding.static_type());
      binding.set_impl_binding(impl_binding);
      SetValue(&binding, val);
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
        RETURN_IF_ERROR(TypeCheckPattern(field, expected_field_type, impl_scope,
                                         enclosing_value_category));
        if (trace_stream_)
          **trace_stream_ << "finished checking tuple pattern field " << *field
                          << "\n";
        field_types.push_back(&field->static_type());
      }
      tuple.set_static_type(arena_->New<TupleValue>(std::move(field_types)));
      ASSIGN_OR_RETURN(Nonnull<const Value*> tuple_value,
                       InterpPattern(&tuple, arena_, trace_stream_));
      SetValue(&tuple, tuple_value);
      return Success();
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      RETURN_IF_ERROR(TypeCheckExp(&alternative.choice_type(), impl_scope));
      if (alternative.choice_type().static_type().kind() !=
          Value::Kind::TypeOfChoiceType) {
        return CompilationError(alternative.source_loc())
               << "alternative pattern does not name a choice type.";
      }
      if (expected) {
        RETURN_IF_ERROR(ExpectExactType(
            alternative.source_loc(), "alternative pattern", *expected,
            &alternative.choice_type().static_type()));
      }
      const ChoiceType& choice_type =
          cast<TypeOfChoiceType>(alternative.choice_type().static_type())
              .choice_type();
      std::optional<Nonnull<const Value*>> parameter_types =
          cast<ChoiceType>(choice_type)
              .FindAlternative(alternative.alternative_name());
      if (parameter_types == std::nullopt) {
        return CompilationError(alternative.source_loc())
               << "'" << alternative.alternative_name()
               << "' is not an alternative of " << choice_type;
      }
      RETURN_IF_ERROR(TypeCheckPattern(&alternative.arguments(),
                                       *parameter_types, impl_scope,
                                       enclosing_value_category));
      alternative.set_static_type(&choice_type);
      ASSIGN_OR_RETURN(Nonnull<const Value*> alternative_value,
                       InterpPattern(&alternative, arena_, trace_stream_));
      SetValue(&alternative, alternative_value);
      return Success();
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      RETURN_IF_ERROR(TypeCheckExp(&expression, impl_scope));
      p->set_static_type(&expression.static_type());
      ASSIGN_OR_RETURN(Nonnull<const Value*> expr_value,
                       InterpPattern(p, arena_, trace_stream_));
      SetValue(p, expr_value);
      return Success();
    }
    case PatternKind::VarPattern:
      auto& let_var_pattern = cast<VarPattern>(*p);

      RETURN_IF_ERROR(TypeCheckPattern(&let_var_pattern.pattern(), expected,
                                       impl_scope,
                                       let_var_pattern.value_category()));
      let_var_pattern.set_static_type(&let_var_pattern.pattern().static_type());
      ASSIGN_OR_RETURN(Nonnull<const Value*> pattern_value,
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
      RETURN_IF_ERROR(TypeCheckExp(&match.expression(), impl_scope));
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        RETURN_IF_ERROR(TypeCheckPattern(&clause.pattern(),
                                         &match.expression().static_type(),
                                         impl_scope, ValueCategory::Let));
        RETURN_IF_ERROR(TypeCheckStmt(&clause.statement(), impl_scope));
      }
      return Success();
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&while_stmt.condition(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "condition of `while`",
                                 arena_->New<BoolType>(),
                                 &while_stmt.condition().static_type()));
      RETURN_IF_ERROR(TypeCheckStmt(&while_stmt.body(), impl_scope));
      return Success();
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return Success();
    case StatementKind::Block: {
      auto& block = cast<Block>(*s);
      for (auto* block_statement : block.statements()) {
        RETURN_IF_ERROR(TypeCheckStmt(block_statement, impl_scope));
      }
      return Success();
    }
    case StatementKind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&var.init(), impl_scope));
      const Value& rhs_ty = var.init().static_type();
      RETURN_IF_ERROR(TypeCheckPattern(&var.pattern(), &rhs_ty, impl_scope,
                                       var.value_category()));
      return Success();
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&assign.rhs(), impl_scope));
      RETURN_IF_ERROR(TypeCheckExp(&assign.lhs(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "assign",
                                 &assign.lhs().static_type(),
                                 &assign.rhs().static_type()));
      if (assign.lhs().value_category() != ValueCategory::Var) {
        return CompilationError(assign.source_loc())
               << "Cannot assign to rvalue '" << assign.lhs() << "'";
      }
      return Success();
    }
    case StatementKind::ExpressionStatement: {
      RETURN_IF_ERROR(TypeCheckExp(&cast<ExpressionStatement>(*s).expression(),
                                   impl_scope));
      return Success();
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&if_stmt.condition(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "condition of `if`",
                                 arena_->New<BoolType>(),
                                 &if_stmt.condition().static_type()));
      RETURN_IF_ERROR(TypeCheckStmt(&if_stmt.then_block(), impl_scope));
      if (if_stmt.else_block()) {
        RETURN_IF_ERROR(TypeCheckStmt(*if_stmt.else_block(), impl_scope));
      }
      return Success();
    }
    case StatementKind::Return: {
      auto& ret = cast<Return>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&ret.expression(), impl_scope));
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        return_term.set_static_type(&ret.expression().static_type());
      } else {
        RETURN_IF_ERROR(ExpectType(s->source_loc(), "return",
                                   &return_term.static_type(),
                                   &ret.expression().static_type()));
      }
      return Success();
    }
    case StatementKind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      RETURN_IF_ERROR(TypeCheckStmt(&cont.body(), impl_scope));
      cont.set_static_type(arena_->New<ContinuationType>());
      return Success();
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&run.argument(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "argument of `run`",
                                 arena_->New<ContinuationType>(),
                                 &run.argument().static_type()));
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
              "return "
              "type without reaching a return statement";
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
        RETURN_IF_ERROR(
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
      RETURN_IF_ERROR(ExpectReturnOnAllPaths(
          block.statements()[block.statements().size() - 1],
          block.source_loc()));
      return Success();
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*stmt);
      RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(&if_stmt.then_block(), stmt->source_loc()));
      RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(if_stmt.else_block(), stmt->source_loc()));
      return Success();
    }
    case StatementKind::Return:
      return Success();
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
      return Success();
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

auto TypeChecker::CreateImplBindings(
    llvm::ArrayRef<Nonnull<GenericBinding*>> deduced_parameters,
    SourceLocation source_loc,
    std::vector<Nonnull<const ImplBinding*>>& impl_bindings)
    -> ErrorOr<Success> {
  for (Nonnull<GenericBinding*> deduced : deduced_parameters) {
    switch (deduced->static_type().kind()) {
      case Value::Kind::InterfaceType: {
        Nonnull<ImplBinding*> impl_binding = arena_->New<ImplBinding>(
            deduced->source_loc(), deduced, &deduced->static_type());
        deduced->set_impl_binding(impl_binding);
        impl_binding->set_static_type(&deduced->static_type());
        impl_bindings.push_back(impl_binding);
        break;
      }
      case Value::Kind::TypeType:
        // No `impl` binding needed for type parameter with bound `Type`.
        break;
      default:
        return CompilationError(source_loc)
               << "unexpected type of deduced parameter "
               << deduced->static_type();
    }
  }
  return Success();
}

void TypeChecker::BringImplsIntoScope(
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings, ImplScope& scope,
    SourceLocation source_loc) {
  for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
    CHECK(impl_binding->type_var()->symbolic_identity().has_value());
    auto impl_id = arena_->New<IdentifierExpression>(source_loc, "impl");
    impl_id->set_value_node(impl_binding);
    scope.Add(impl_binding->interface(),
              *impl_binding->type_var()->symbolic_identity(), impl_id);
  }
}

// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
auto TypeChecker::DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                             const ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** declaring function " << f->name() << "\n";
  }
  // Bring the deduced parameters into scope.
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    RETURN_IF_ERROR(TypeCheckExp(&deduced->type(), enclosing_scope));
    deduced->set_symbolic_identity(arena_->New<VariableType>(deduced));
    ASSIGN_OR_RETURN(Nonnull<const Value*> type_of_type,
                     InterpExp(&deduced->type(), arena_, trace_stream_));
    deduced->set_static_type(type_of_type);
  }
  // Create the impl_bindings.
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;
  RETURN_IF_ERROR(CreateImplBindings(f->deduced_parameters(), f->source_loc(),
                                     impl_bindings));
  // Bring the impl bindings into scope.
  ImplScope function_scope;
  function_scope.AddParent(&enclosing_scope);
  BringImplsIntoScope(impl_bindings, function_scope, f->source_loc());
  // Type check the receiver pattern.
  if (f->is_method()) {
    RETURN_IF_ERROR(TypeCheckPattern(&f->me_pattern(), std::nullopt,
                                     function_scope, ValueCategory::Let));
  }
  // Type check the parameter pattern.
  RETURN_IF_ERROR(TypeCheckPattern(&f->param_pattern(), std::nullopt,
                                   function_scope, ValueCategory::Let));

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    // We ignore the return value because return type expressions can't bring
    // new types into scope.
    RETURN_IF_ERROR(TypeCheckExp(*return_expression, function_scope));
    // Should we be doing SetConstantValue instead? -Jeremy
    // And shouldn't the type of this be Type?
    ASSIGN_OR_RETURN(Nonnull<const Value*> ret_type,
                     InterpExp(*return_expression, arena_, trace_stream_));
    RETURN_IF_ERROR(ExpectIsType(f->source_loc(), ret_type));
    f->return_term().set_static_type(ret_type);
  } else if (f->return_term().is_omitted()) {
    f->return_term().set_static_type(TupleValue::Empty());
  } else {
    // We have to type-check the body in order to determine the return type.
    if (!f->body().has_value()) {
      return CompilationError(f->return_term().source_loc())
             << "Function declaration has deduced return type but no body";
    }
    RETURN_IF_ERROR(TypeCheckStmt(*f->body(), function_scope));
    if (!f->return_term().is_omitted()) {
      RETURN_IF_ERROR(ExpectReturnOnAllPaths(f->body(), f->source_loc()));
    }
  }

  RETURN_IF_ERROR(
      ExpectIsConcreteType(f->source_loc(), &f->return_term().static_type()));
  f->set_static_type(arena_->New<FunctionType>(
      f->deduced_parameters(), &f->param_pattern().static_type(),
      &f->return_term().static_type(), impl_bindings));
  SetConstantValue(f, arena_->New<FunctionValue>(f));

  if (f->name() == "Main") {
    if (!f->return_term().type_expression().has_value()) {
      return CompilationError(f->return_term().source_loc())
             << "`Main` must have an explicit return type";
    }
    RETURN_IF_ERROR(ExpectExactType(
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
    // Bring the impl's into scope.
    ImplScope function_scope;
    function_scope.AddParent(&impl_scope);
    BringImplsIntoScope(cast<FunctionType>(f->static_type()).impl_bindings(),
                        function_scope, f->source_loc());
    if (trace_stream_)
      **trace_stream_ << function_scope;
    RETURN_IF_ERROR(TypeCheckStmt(*f->body(), function_scope));
    if (!f->return_term().is_omitted()) {
      RETURN_IF_ERROR(ExpectReturnOnAllPaths(f->body(), f->source_loc()));
    }
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking function " << f->name() << "\n";
  }
  return Success();
}

auto TypeChecker::DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                          ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "** declaring class " << class_decl->name() << "\n";
  }
  if (class_decl->type_params().has_value()) {
    ImplScope class_scope;
    class_scope.AddParent(&enclosing_scope);
    RETURN_IF_ERROR(TypeCheckPattern(*class_decl->type_params(), std::nullopt,
                                     class_scope, ValueCategory::Let));
    AddPatternImpls(*class_decl->type_params(), class_scope);
    if (trace_stream_) {
      **trace_stream_ << class_scope;
    }

    Nonnull<NominalClassType*> class_type =
        arena_->New<NominalClassType>(class_decl);
    SetConstantValue(class_decl, class_type);
    class_decl->set_static_type(arena_->New<TypeOfClassType>(class_type));

    for (Nonnull<Declaration*> m : class_decl->members()) {
      RETURN_IF_ERROR(DeclareDeclaration(m, class_scope));
    }

    // TODO: when/how to bring impls in generic class into scope?
  } else {
    // The declarations of the members may refer to the class, so we
    // must set the constant value of the class and its static type
    // before we start processing the members.
    Nonnull<NominalClassType*> class_type =
        arena_->New<NominalClassType>(class_decl);
    SetConstantValue(class_decl, class_type);
    class_decl->set_static_type(arena_->New<TypeOfClassType>(class_type));

    for (Nonnull<Declaration*> m : class_decl->members()) {
      RETURN_IF_ERROR(DeclareDeclaration(m, enclosing_scope));
    }
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
    AddPatternImpls(*class_decl->type_params(), class_scope);
  }
  if (trace_stream_) {
    **trace_stream_ << class_scope;
  }
  for (Nonnull<Declaration*> m : class_decl->members()) {
    RETURN_IF_ERROR(TypeCheckDeclaration(m, class_scope));
  }
  if (trace_stream_) {
    **trace_stream_ << "** finished checking class " << class_decl->name()
                    << "\n";
  }
  return Success();
}

auto TypeChecker::DeclareInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  Nonnull<InterfaceType*> iface_type = arena_->New<InterfaceType>(iface_decl);
  SetConstantValue(iface_decl, iface_type);
  iface_decl->set_static_type(arena_->New<TypeOfInterfaceType>(iface_type));

  // Process the Self parameter.
  RETURN_IF_ERROR(TypeCheckPattern(iface_decl->self(), std::nullopt,
                                   enclosing_scope, ValueCategory::Let));
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    RETURN_IF_ERROR(DeclareDeclaration(m, enclosing_scope));
  }
  return Success();
}

auto TypeChecker::TypeCheckInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    RETURN_IF_ERROR(TypeCheckDeclaration(m, impl_scope));
  }
  return Success();
}

auto TypeChecker::DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                         ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "declaring " << *impl_decl << "\n";
  }
  RETURN_IF_ERROR(TypeCheckExp(&impl_decl->interface(), enclosing_scope));
  ASSIGN_OR_RETURN(Nonnull<const Value*> iface_type,
                   InterpExp(&impl_decl->interface(), arena_, trace_stream_));
  const auto& iface_decl = cast<InterfaceType>(*iface_type).declaration();
  impl_decl->set_interface_type(iface_type);

  // Bring the deduced parameters into scope.
  for (Nonnull<GenericBinding*> deduced : impl_decl->deduced_parameters()) {
    RETURN_IF_ERROR(TypeCheckExp(&deduced->type(), enclosing_scope));
    deduced->set_symbolic_identity(arena_->New<VariableType>(deduced));
    ASSIGN_OR_RETURN(Nonnull<const Value*> type_of_type,
                     InterpExp(&deduced->type(), arena_, trace_stream_));
    deduced->set_static_type(type_of_type);
  }
  // Create the impl_bindings.
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;
  RETURN_IF_ERROR(CreateImplBindings(impl_decl->deduced_parameters(),
                                     impl_decl->source_loc(), impl_bindings));
  impl_decl->set_impl_bindings(impl_bindings);

  // Bring the impl bindings into scope for the impl body.
  ImplScope impl_scope;
  impl_scope.AddParent(&enclosing_scope);
  BringImplsIntoScope(impl_bindings, impl_scope, impl_decl->source_loc());
  // Check and interpret the impl_type
  RETURN_IF_ERROR(TypeCheckExp(impl_decl->impl_type(), impl_scope));
  ASSIGN_OR_RETURN(Nonnull<const Value*> impl_type_value,
                   InterpExp(impl_decl->impl_type(), arena_, trace_stream_));
  // Bring this impl into the enclosing scope.
  auto impl_id =
      arena_->New<IdentifierExpression>(impl_decl->source_loc(), "impl");
  impl_id->set_value_node(impl_decl);
  enclosing_scope.Add(iface_type, impl_decl->deduced_parameters(),
                      impl_type_value, impl_bindings, impl_id);

  // Declare the impl members.
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    RETURN_IF_ERROR(DeclareDeclaration(m, impl_scope));
  }
  // Check that the interface is satisfied by the impl members.
  for (Nonnull<Declaration*> m : iface_decl.members()) {
    if (std::optional<std::string> mem_name = GetName(*m);
        mem_name.has_value()) {
      if (std::optional<Nonnull<const Declaration*>> mem =
              FindMember(*mem_name, impl_decl->members());
          mem.has_value()) {
        std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
            self_map;
        self_map[iface_decl.self()] = impl_type_value;
        Nonnull<const Value*> iface_mem_type =
            Substitute(self_map, &m->static_type());
        RETURN_IF_ERROR(ExpectType((*mem)->source_loc(),
                                   "member of implementation", iface_mem_type,
                                   &(*mem)->static_type()));
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
  // Bring the impl's from the parameters into scope.
  ImplScope impl_scope;
  impl_scope.AddParent(&enclosing_scope);
  BringImplsIntoScope(impl_decl->impl_bindings(), impl_scope,
                      impl_decl->source_loc());
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    RETURN_IF_ERROR(TypeCheckDeclaration(m, impl_scope));
  }
  if (trace_stream_) {
    **trace_stream_ << "finished checking impl\n";
  }
  return Success();
}

auto TypeChecker::DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                           const ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  std::vector<NamedValue> alternatives;
  for (Nonnull<AlternativeSignature*> alternative : choice->alternatives()) {
    RETURN_IF_ERROR(TypeCheckExp(&alternative->signature(), enclosing_scope));
    ASSIGN_OR_RETURN(auto signature, InterpExp(&alternative->signature(),
                                               arena_, trace_stream_));
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

auto TypeChecker::TypeCheck(AST& ast) -> ErrorOr<Success> {
  ImplScope impl_scope;
  for (Nonnull<Declaration*> declaration : ast.declarations) {
    RETURN_IF_ERROR(DeclareDeclaration(declaration, impl_scope));
  }
  for (Nonnull<Declaration*> decl : ast.declarations) {
    RETURN_IF_ERROR(TypeCheckDeclaration(decl, impl_scope));
  }
  RETURN_IF_ERROR(TypeCheckExp(*ast.main_call, impl_scope));
  return Success();
}

auto TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d,
                                       const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      RETURN_IF_ERROR(TypeCheckInterfaceDeclaration(
          &cast<InterfaceDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      RETURN_IF_ERROR(
          TypeCheckImplDeclaration(&cast<ImplDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::FunctionDeclaration:
      RETURN_IF_ERROR(TypeCheckFunctionDeclaration(
          &cast<FunctionDeclaration>(*d), impl_scope));
      return Success();
    case DeclarationKind::ClassDeclaration:
      RETURN_IF_ERROR(
          TypeCheckClassDeclaration(&cast<ClassDeclaration>(*d), impl_scope));
      return Success();
    case DeclarationKind::ChoiceDeclaration:
      RETURN_IF_ERROR(
          TypeCheckChoiceDeclaration(&cast<ChoiceDeclaration>(*d), impl_scope));
      return Success();
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      if (var.has_initializer()) {
        RETURN_IF_ERROR(TypeCheckExp(&var.initializer(), impl_scope));
      }
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        return CompilationError(var.source_loc())
               << "Type of a top-level variable must be an expression.";
      }
      if (var.has_initializer()) {
        RETURN_IF_ERROR(ExpectType(var.source_loc(), "initializer of variable",
                                   &var.static_type(),
                                   &var.initializer().static_type()));
      }
      return Success();
    }
  }
  return Success();
}

auto TypeChecker::DeclareDeclaration(Nonnull<Declaration*> d,
                                     ImplScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface_decl = cast<InterfaceDeclaration>(*d);
      RETURN_IF_ERROR(
          DeclareInterfaceDeclaration(&iface_decl, enclosing_scope));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl_decl = cast<ImplDeclaration>(*d);
      RETURN_IF_ERROR(DeclareImplDeclaration(&impl_decl, enclosing_scope));
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      auto& func_def = cast<FunctionDeclaration>(*d);
      RETURN_IF_ERROR(DeclareFunctionDeclaration(&func_def, enclosing_scope));
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      RETURN_IF_ERROR(DeclareClassDeclaration(&class_decl, enclosing_scope));
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(*d);
      RETURN_IF_ERROR(DeclareChoiceDeclaration(&choice, enclosing_scope));
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
      RETURN_IF_ERROR(TypeCheckPattern(&var.binding(), std::nullopt,
                                       enclosing_scope, var.value_category()));
      ASSIGN_OR_RETURN(Nonnull<const Value*> declared_type,
                       InterpExp(&type, arena_, trace_stream_));
      var.set_static_type(declared_type);
      break;
    }
  }
  return Success();
}

template <typename T>
void TypeChecker::SetConstantValue(Nonnull<T*> value_node,
                                   Nonnull<const Value*> value) {
  std::optional<Nonnull<const Value*>> old_value = value_node->constant_value();
  CHECK(!old_value.has_value());
  value_node->set_constant_value(value);
  CHECK(constants_.insert(value_node).second);
}

void TypeChecker::PrintConstants(llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& value_node : constants_) {
    out << sep << value_node;
  }
}

}  // namespace Carbon
