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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace Carbon {

using ::llvm::cast;
using ::llvm::dyn_cast;
using ::llvm::isa;
static void SetValue(Nonnull<Pattern*> pattern, Nonnull<const Value*> value) {
  // TODO: find some way to CHECK that `value` is identical to pattern->value(),
  // if it's already set. Unclear if `ValueEqual` is suitable, because it
  // currently focuses more on "real" values, and disallows the pseudo-values
  // like `BindingPlaceholderValue` that we get in pattern evaluation.
  if (!pattern->has_value()) {
    pattern->set_value(value);
  }
}

static auto ExpectPointerType(SourceLocation source_loc,
                              const std::string& context,
                              Nonnull<const Value*> actual)
    -> ErrorOr<Success> {
  // TODO: Try to resolve in equality context.
  if (actual->kind() != Value::Kind::PointerType) {
    return CompilationError(source_loc) << "type error in " << context << "\n"
                                        << "expected a pointer type\n"
                                        << "actual: " << *actual;
  }
  return Success();
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

auto TypeChecker::ImplicitlyConvert(const std::string& context,
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
    return CompilationError(source->source_loc())
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

auto TypeChecker::SatisfyImpls(
    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
    const ImplScope& impl_scope, SourceLocation source_loc,
    const BindingMap& deduced_type_args, ImplExpMap& impls) const
    -> ErrorOr<Success> {
  for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
    Nonnull<const Value*> interface =
        Substitute(deduced_type_args, impl_binding->interface());
    CARBON_CHECK(deduced_type_args.find(impl_binding->type_var()) !=
                 deduced_type_args.end());
    CARBON_ASSIGN_OR_RETURN(
        Nonnull<Expression*> impl,
        impl_scope.Resolve(interface,
                           deduced_type_args.at(impl_binding->type_var()),
                           source_loc, *this));
    impls.emplace(impl_binding, impl);
  }
  return Success();
}

auto TypeChecker::MakeConstraintForInterface(
    SourceLocation source_loc, Nonnull<const InterfaceType*> iface_type)
    -> Nonnull<const ConstraintType*> {
  ConstraintTypeBuilder builder(arena_, source_loc);
  builder.AddImplConstraint(
      {.type = builder.GetSelfType(arena_), .interface = iface_type});
  builder.AddLookupContext({.context = iface_type});
  return std::move(builder).Build(arena_);
}

auto TypeChecker::CombineConstraints(
    SourceLocation source_loc,
    llvm::ArrayRef<Nonnull<const ConstraintType*>> constraints)
    -> Nonnull<const ConstraintType*> {
  ConstraintTypeBuilder builder(arena_, source_loc);
  auto* self = builder.GetSelfType(arena_);
  for (Nonnull<const ConstraintType*> constraint : constraints) {
    BindingMap map;
    map[constraint->self_binding()] = self;
    builder.Add(cast<ConstraintType>(Substitute(map, constraint)));
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

  // TODO: Ensure any equality constraints are satisfied.

  // Convert the arguments to the parameter type.
  Nonnull<const Value*> param_type = Substitute(generic_bindings, params_type);

  // Convert the arguments to the deduced and substituted parameter type.
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<Expression*> converted_argument,
      ImplicitlyConvert("call", impl_scope, &call.argument(), param_type));

  call.set_argument(converted_argument);

  return Success();
}

struct ConstraintLookupResult {
  Nonnull<const InterfaceType*> interface;
  Nonnull<const Declaration*> member;
};

/// Look up a member name in a constraint, which might be a single interface or
/// a compound constraint.
static auto LookupInConstraint(SourceLocation source_loc,
                               std::string_view lookup_kind,
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
                 CreateImplReference(impl_binding), *this);
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
        return CompilationError(is_clause.constraint().source_loc())
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
        return CompilationError(clause->source_loc())
               << "type mismatch between values in `where LHS == RHS`\n"
               << "  LHS type: " << *lhs_type << "\n"
               << "  RHS type: " << *rhs_type;
      }
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
                                            type, *expected, impl_scope));
        } else {
          BindingMap generic_args;
          if (!PatternMatch(type, *expected, binding.type().source_loc(),
                            std::nullopt, generic_args, trace_stream_,
                            this->arena_)) {
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
      if (binding.named_as_type_via_dot_self() && !IsTypeOfType(type)) {
        return CompilationError(binding.type().source_loc())
               << "`.Self` used in type of non-type binding `" << binding.name()
               << "`";
      }
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                              InterpPattern(&binding, arena_, trace_stream_));
      binding.set_symbolic_identity(val);
      SetValue(&binding, val);

      if (isa<InterfaceType, ConstraintType>(type)) {
        Nonnull<ImplBinding*> impl_binding =
            arena_->New<ImplBinding>(binding.source_loc(), &binding, type);
        impl_binding->set_symbolic_identity(
            arena_->New<SymbolicWitness>(CreateImplReference(impl_binding)));
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
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> type,
          InterpExp(&alternative.choice_type(), arena_, trace_stream_));
      if (!isa<ChoiceType>(type)) {
        return CompilationError(alternative.source_loc())
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
        return CompilationError(alternative.source_loc())
               << "'" << alternative.alternative_name()
               << "' is not an alternative of " << choice_type;
      }

      Nonnull<const Value*> substituted_parameter_type = *parameter_types;
      if (choice_type.IsParameterized()) {
        substituted_parameter_type =
            Substitute(choice_type.type_args(), *parameter_types);
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
        return CompilationError(addr_pattern.source_loc())
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
        return CompilationError(for_stmt.source_loc())
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
      ImplScope var_scope;
      var_scope.AddParent(&impl_scope);
      if (var.has_init()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&var.init(), impl_scope));
        const Value& rhs_ty = var.init().static_type();
        // TODO: If the pattern contains a binding that implies a new impl is
        // available, should that remain in scope for as long as its binding?
        // ```
        // var a: (T:! Widget) = ...;
        // // Is the `impl T as Widget` in scope here?
        // a.(Widget.F)();
        // ```
        CARBON_RETURN_IF_ERROR(TypeCheckPattern(
            &var.pattern(), &rhs_ty, var_scope, var.value_category()));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_init,
            ImplicitlyConvert("initializer of variable", impl_scope,
                              &var.init(), &var.pattern().static_type()));
        var.set_init(converted_init);
      } else {
        CARBON_RETURN_IF_ERROR(TypeCheckPattern(
            &var.pattern(), std::nullopt, var_scope, var.value_category()));
      }
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
          return CompilationError(ret.value_node().base().source_loc())
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

  if (class_decl->extensibility() != ClassExtensibility::None) {
    return CompilationError(class_decl->source_loc())
           << "Class prefixes `base` and `abstract` are not supported yet";
  }
  if (class_decl->extends()) {
    return CompilationError(class_decl->source_loc())
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
  BindingMap generic_args;
  for (auto* binding : bindings) {
    // binding.symbolic_identity() set by call to `TypeCheckPattern(...)`
    // above and/or by any enclosing generic classes.
    generic_args[binding] = *binding->symbolic_identity();
  }
  Nonnull<NominalClassType*> self_type = arena_->New<NominalClassType>(
      class_decl,
      arena_->New<Bindings>(std::move(generic_args), Bindings::NoWitnesses));
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

  Nonnull<InterfaceType*> iface_type;
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

    // Form the full symbolic type of the interface. This is used as part of
    // the value of associated constants, if they're referenced within the
    // interface itself.
    std::vector<Nonnull<const GenericBinding*>> bindings = scope_info.bindings;
    CollectGenericBindingsInPattern(*iface_decl->params(), bindings);
    BindingMap generic_args;
    for (auto* binding : bindings) {
      generic_args[binding] = *binding->symbolic_identity();
    }
    iface_type = arena_->New<InterfaceType>(
        iface_decl,
        arena_->New<Bindings>(std::move(generic_args), Bindings::NoWitnesses));
  } else {
    iface_type = arena_->New<InterfaceType>(iface_decl);
    SetConstantValue(iface_decl, iface_type);
    iface_decl->set_static_type(arena_->New<TypeOfInterfaceType>(iface_type));
  }

  // Process the Self parameter.
  CARBON_RETURN_IF_ERROR(TypeCheckPattern(iface_decl->self(), std::nullopt,
                                          iface_scope, ValueCategory::Let));

  ScopeInfo iface_scope_info = ScopeInfo::ForNonClassScope(&iface_scope);
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, iface_scope_info));

    if (auto* assoc = dyn_cast<AssociatedConstantDeclaration>(m)) {
      // TODO: The witness should be optional in AssociatedConstant.
      Nonnull<const Expression*> witness_expr =
          arena_->New<DotSelfExpression>(iface_decl->source_loc());
      assoc->binding().set_symbolic_identity(arena_->New<AssociatedConstant>(
          &iface_decl->self()->value(), iface_type, assoc,
          arena_->New<SymbolicWitness>(witness_expr)));
    }
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

auto TypeChecker::CheckImplIsDeducible(
    SourceLocation source_loc, Nonnull<const Value*> impl_type,
    Nonnull<const InterfaceType*> impl_iface,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
    const ImplScope& impl_scope) -> ErrorOr<Success> {
  BindingMap deduced_args;
  CARBON_RETURN_IF_ERROR(ArgumentDeduction(
      source_loc, "impl", deduced_bindings, deduced_args, impl_type, impl_type,
      /*allow_implicit_conversion=*/false, impl_scope));
  CARBON_RETURN_IF_ERROR(ArgumentDeduction(source_loc, "impl", deduced_bindings,
                                           deduced_args, impl_iface, impl_iface,
                                           /*allow_implicit_conversion=*/false,
                                           impl_scope));
  for (auto* expected_deduced : deduced_bindings) {
    if (!deduced_args.count(expected_deduced)) {
      return CompilationError(source_loc)
             << "parameter `" << *expected_deduced
             << "` is not deducible from `impl " << *impl_type << " as "
             << *impl_iface << "`";
    }
  }
  return Success();
}

auto TypeChecker::CheckImplIsComplete(Nonnull<const InterfaceType*> iface_type,
                                      Nonnull<const ImplDeclaration*> impl_decl,
                                      Nonnull<const Value*> self_type,
                                      const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  const auto& iface_decl = iface_type->declaration();
  for (Nonnull<Declaration*> m : iface_decl.members()) {
    if (auto* assoc = dyn_cast<AssociatedConstantDeclaration>(m)) {
      // An associated constant must be given exactly one value.
      Nonnull<const GenericBinding*> symbolic_self =
          impl_decl->constraint_type()->self_binding();
      Nonnull<const Value*> expected = arena_->New<AssociatedConstant>(
          &symbolic_self->value(), iface_type, assoc,
          arena_->New<ImplWitness>(impl_decl));

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
        return CompilationError(impl_decl->source_loc())
               << "implementation missing " << *expected;
      } else if (!found_value) {
        // TODO: It's not clear what the right rule is here. Clearly
        //   impl T as HasX & HasY where .X == .Y {}
        // ... is insufficient to establish a value for either X or Y.
        // But perhaps we can allow
        //   impl forall [T:! HasX] T as HasY where .Y == .X {}
        return CompilationError(impl_decl->source_loc())
               << "implementation doesn't provide a concrete value for "
               << *expected;
      } else if (second_value) {
        return CompilationError(impl_decl->source_loc())
               << "implementation provides multiple values for " << *expected
               << ": " << **found_value << " and " << **second_value;
      }
    } else {
      // Every member function must be declared.
      std::optional<std::string_view> mem_name = GetName(*m);
      CARBON_CHECK(mem_name.has_value()) << "unnamed interface member " << *m;

      std::optional<Nonnull<const Declaration*>> mem =
          FindMember(*mem_name, impl_decl->members());
      if (!mem.has_value()) {
        return CompilationError(impl_decl->source_loc())
               << "implementation missing " << *mem_name;
      }

      BindingMap binding_map = iface_type->args();
      binding_map[iface_decl.self()] = self_type;
      Nonnull<const Value*> iface_mem_type =
          Substitute(binding_map, &m->static_type());
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
    const ScopeInfo& scope_info) -> ErrorOr<Success> {
  // The deduced bindings are the parameters for all enclosing classes followed
  // by any deduced parameters written on the `impl` declaration itself.
  std::vector<Nonnull<const GenericBinding*>> deduced_bindings =
      scope_info.bindings;
  deduced_bindings.insert(deduced_bindings.end(),
                          impl_decl->deduced_parameters().begin(),
                          impl_decl->deduced_parameters().end());

  // An expression that evaluates to this impl's witness.
  // TODO: Store witnesses as `Witness*` rather than `Expression*` everywhere
  // so we don't need to create this.
  auto* impl_expr = arena_->New<ValueLiteral>(
      impl_decl->source_loc(), arena_->New<ImplWitness>(impl_decl),
      arena_->New<TypeType>(), ValueCategory::Let);

  // Form the resolved constraint type by substituting `Self` for `.Self`.
  Nonnull<const Value*> self = *impl_decl->self()->constant_value();
  BindingMap constraint_self_map;
  constraint_self_map[impl_decl->constraint_type()->self_binding()] = self;
  Nonnull<const ConstraintType*> constraint = cast<ConstraintType>(
      Substitute(constraint_self_map, impl_decl->constraint_type()));

  // Each interface that is a lookup context is required to be implemented by
  // the impl members. Other constraints are required to be satisfied by
  // either those impls or impls available elsewhere.
  for (auto lookup : constraint->lookup_contexts()) {
    if (auto* iface_type = dyn_cast<InterfaceType>(lookup.context)) {
      CARBON_RETURN_IF_ERROR(
          CheckImplIsDeducible(impl_decl->source_loc(), impl_type, iface_type,
                               deduced_bindings, *scope_info.innermost_scope));

      // Bring the associated constant values for this interface into scope. We
      // know that if the methods of this interface are used, they will use
      // these values.
      ImplScope iface_scope;
      iface_scope.AddParent(scope_info.innermost_scope);
      BringAssociatedConstantsIntoScope(cast<ConstraintType>(constraint), self,
                                        iface_type, iface_scope);

      CARBON_RETURN_IF_ERROR(
          CheckImplIsComplete(iface_type, impl_decl, impl_type, iface_scope));

      scope_info.innermost_non_class_scope->Add(
          iface_type, deduced_bindings, impl_type, impl_decl->impl_bindings(),
          impl_expr, *this);
    } else {
      // TODO: Add support for implementing `adapter`s.
      return CompilationError(impl_decl->source_loc())
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
      Nonnull<const Value*> constraint_type,
      TypeCheckTypeExp(&impl_decl->interface(), impl_scope));
  if (auto* iface_type = dyn_cast<InterfaceType>(constraint_type)) {
    constraint_type = MakeConstraintForInterface(
        impl_decl->interface().source_loc(), iface_type);
  }
  if (!isa<ConstraintType>(constraint_type)) {
    return CompilationError(impl_decl->interface().source_loc())
           << "expected constraint after `as`, found value of type "
           << *constraint_type;
  }
  impl_decl->set_constraint_type(cast<ConstraintType>(constraint_type));

  // Declare the impl members.
  ScopeInfo impl_scope_info = ScopeInfo::ForNonClassScope(&impl_scope);
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    CARBON_RETURN_IF_ERROR(DeclareDeclaration(m, impl_scope_info));
  }

  // Create the implied impl bindings.
  CARBON_RETURN_IF_ERROR(
      CheckAndAddImplBindings(impl_decl, impl_type_value, scope_info));

  // Check the constraint is satisfied by the `impl`s we just created. This
  // serves a couple of purposes:
  //  - It ensures that any constraints in a `ConstraintType` are met.
  //  - It rejects `impl`s that immediately introduce ambiguity.
  CARBON_RETURN_IF_ERROR(impl_scope.Resolve(constraint_type, impl_type_value,
                                            impl_decl->source_loc(), *this));

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

  // Form the resolved constraint type by substituting `Self` for `.Self`.
  Nonnull<const Value*> self = *impl_decl->self()->constant_value();
  BindingMap constraint_self_map;
  constraint_self_map[impl_decl->constraint_type()->self_binding()] = self;
  Nonnull<const ConstraintType*> constraint = cast<ConstraintType>(
      Substitute(constraint_self_map, impl_decl->constraint_type()));

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

    CARBON_RETURN_IF_ERROR(TypeCheckDeclaration(m, member_scope));
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
  BindingMap generic_args;
  for (auto* binding : bindings) {
    generic_args[binding] = *binding->symbolic_identity();
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
    SetConstantValue(choice, param_name);
    choice->set_static_type(
        arena_->New<TypeOfParameterizedEntityName>(param_name));
    return Success();
  }

  auto ct = arena_->New<ChoiceType>(
      choice,
      arena_->New<Bindings>(std::move(generic_args), Bindings::NoWitnesses));

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
    // TODO: Only do this when type-checking the prelude.
    builtins_.Register(decl);
  }
  CARBON_RETURN_IF_ERROR(TypeCheckExp(*ast.main_call, impl_scope));
  return Success();
}

auto TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d,
                                       const ImplScope& impl_scope)
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
    case DeclarationKind::AssociatedConstantDeclaration:
      return Success();
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
