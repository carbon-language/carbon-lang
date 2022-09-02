// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/generic_checker.h"

#include "explorer/interpreter/constraint_type_builder.h"
#include "explorer/interpreter/type_utils.h"

namespace Carbon {

using ::llvm::cast;

auto GenericChecker::ArgumentDeduction(
    SourceLocation source_loc, const std::string& context,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> bindings_to_deduce,
    BindingMap& deduced, Nonnull<const Value*> param, Nonnull<const Value*> arg,
    bool allow_implicit_conversion, const ImplScope& impl_scope, Globals g)
    -> ErrorOr<Success> {
  if (g.trace_stream()) {
    **g.trace_stream() << "deducing " << *param << " from " << *arg << "\n";
    **g.trace_stream() << "bindings: ";
    llvm::ListSeparator sep;
    for (auto binding : bindings_to_deduce) {
      **g.trace_stream() << sep << *binding;
    }
    **g.trace_stream() << "\n";
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
    const Value* subst_param_type = Substitute(deduced, param, g);
    return allow_implicit_conversion
               ? ExpectType(source_loc, context, subst_param_type, arg,
                            impl_scope)
               : ExpectExactType(source_loc, context, subst_param_type, arg,
                                 impl_scope);
  };

  switch (param->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param);
      const auto& binding = cast<VariableType>(*param).binding();
      if (binding.has_static_type()) {
        const Value* binding_type =
            Substitute(deduced, &binding.static_type(), g);
        if (!IsTypeOfType(binding_type)) {
          if (!IsImplicitlyConvertible(arg, binding_type, impl_scope, false)) {
            return CompilationError(source_loc)
                   << "cannot convert deduced value " << *arg << " for "
                   << binding.name() << " to parameter type " << *binding_type;
          }
        }
      }

      if (std::find(bindings_to_deduce.begin(), bindings_to_deduce.end(),
                    &var_type.binding()) != bindings_to_deduce.end()) {
        auto [it, success] = deduced.insert({&var_type.binding(), arg});
        if (!success) {
          // All deductions are required to produce the same value. Note that
          // we intentionally don't consider type equality here; we need the
          // same symbolic type, otherwise it would be ambiguous which spelling
          // should be used, and we'd need to check all pairs of types for
          // equality because our notion of equality is non-transitive.
          if (!TypeEqual(it->second, arg, std::nullopt)) {
            return CompilationError(source_loc)
                   << "deduced multiple different values for "
                   << var_type.binding() << ":\n  " << *it->second << "\n  "
                   << *arg;
          }
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
                              allow_implicit_conversion, impl_scope, g));
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
            arg_field.value, allow_implicit_conversion, impl_scope, g));
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
      CARBON_RETURN_IF_ERROR(ArgumentDeduction(
          source_loc, context, bindings_to_deduce, deduced,
          &param_fn.parameters(), &arg_fn.parameters(),
          /*allow_implicit_conversion=*/false, impl_scope, g));
      CARBON_RETURN_IF_ERROR(ArgumentDeduction(
          source_loc, context, bindings_to_deduce, deduced,
          &param_fn.return_type(), &arg_fn.return_type(),
          /*allow_implicit_conversion=*/false, impl_scope, g));
      return Success();
    }
    case Value::Kind::PointerType: {
      if (arg->kind() != Value::Kind::PointerType) {
        return handle_non_deduced_type();
      }
      return ArgumentDeduction(
          source_loc, context, bindings_to_deduce, deduced,
          &cast<PointerType>(*param).type(), &cast<PointerType>(*arg).type(),
          /*allow_implicit_conversion=*/false, impl_scope, g);
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
        CARBON_RETURN_IF_ERROR(ArgumentDeduction(
            source_loc, context, bindings_to_deduce, deduced, param_ty,
            arg_class_type.type_args().at(ty),
            /*allow_implicit_conversion=*/false, impl_scope, g));
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
        CARBON_RETURN_IF_ERROR(ArgumentDeduction(
            source_loc, context, bindings_to_deduce, deduced, param_ty,
            arg_iface_type.args().at(ty),
            /*allow_implicit_conversion=*/false, impl_scope, g));
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
    case Value::Kind::SymbolicWitness:
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
        return CompilationError(source_loc)
               << "mismatch in non-type values, `" << *arg << "` != `" << *param
               << "`";
      }
      return Success();
    }
  }
}

auto GenericChecker::Substitute(
    const std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>& dict,
    Nonnull<const Value*> type, Globals g) -> Nonnull<const Value*> {
  auto SubstituteIntoBindings =
      [&](const Bindings& bindings) -> Nonnull<const Bindings*> {
    BindingMap result;
    for (const auto& [name, value] : bindings.args()) {
      result[name] = Substitute(dict, value, g);
    }
    return g.arena()->New<Bindings>(std::move(result), Bindings::NoWitnesses);
  };

  switch (type->kind()) {
    case Value::Kind::VariableType: {
      auto it = dict.find(&cast<VariableType>(*type).binding());
      if (it == dict.end()) {
        return type;
      } else {
        return it->second;
      }
    }
    case Value::Kind::AssociatedConstant: {
      const auto& assoc = cast<AssociatedConstant>(*type);
      Nonnull<const Value*> base = Substitute(dict, &assoc.base(), g);
      Nonnull<const Value*> interface = Substitute(dict, &assoc.interface(), g);
      Nonnull<const Value*> witness = Substitute(dict, &assoc.witness(), g);
      return g.arena()->New<AssociatedConstant>(
          base, cast<InterfaceType>(interface), &assoc.constant(),
          cast<Witness>(witness));
    }
    case Value::Kind::TupleValue: {
      std::vector<Nonnull<const Value*>> elts;
      for (const auto& elt : cast<TupleValue>(*type).elements()) {
        elts.push_back(Substitute(dict, elt, g));
      }
      return g.arena()->New<TupleValue>(elts);
    }
    case Value::Kind::StructType: {
      std::vector<NamedValue> fields;
      for (const auto& [name, value] : cast<StructType>(*type).fields()) {
        auto new_type = Substitute(dict, value, g);
        fields.push_back({name, new_type});
      }
      return g.arena()->New<StructType>(std::move(fields));
    }
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*type);
      std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>> new_dict(
          dict);
      // Create new generic parameters and generic bindings
      // and add them to new_dict.
      std::vector<FunctionType::GenericParameter> generic_parameters;
      std::vector<Nonnull<const GenericBinding*>> deduced_bindings;
      std::map<Nonnull<const GenericBinding*>, Nonnull<const GenericBinding*>>
          bind_map;  // Map old generic bindings to new ones.
      for (const FunctionType::GenericParameter& gp :
           fn_type.generic_parameters()) {
        Nonnull<const Value*> new_type =
            Substitute(dict, &gp.binding->static_type(), g);
        Nonnull<GenericBinding*> new_gb = g.arena()->New<GenericBinding>(
            gp.binding->source_loc(), gp.binding->name(),
            const_cast<Expression*>(
                &gp.binding->type()));  // How to avoid the cast? -jsiek
        new_gb->set_original(gp.binding->original());
        new_gb->set_static_type(new_type);
        FunctionType::GenericParameter new_gp = {.index = gp.index,
                                                 .binding = new_gb};
        generic_parameters.push_back(new_gp);
        new_dict[gp.binding] = g.arena()->New<VariableType>(new_gp.binding);
        bind_map[gp.binding] = new_gb;
      }
      for (Nonnull<const GenericBinding*> gb : fn_type.deduced_bindings()) {
        Nonnull<const Value*> new_type =
            Substitute(dict, &gb->static_type(), g);
        Nonnull<GenericBinding*> new_gb = g.arena()->New<GenericBinding>(
            gb->source_loc(), gb->name(),
            const_cast<Expression*>(
                &gb->type()));  // How to avoid the cast? -jsiek
        new_gb->set_original(gb->original());
        new_gb->set_static_type(new_type);
        deduced_bindings.push_back(new_gb);
        new_dict[gb] = g.arena()->New<VariableType>(new_gb);
        bind_map[gb] = new_gb;
      }
      // Apply substitution to impl bindings and update their
      // `type_var` pointers to the new generic bindings.
      std::vector<Nonnull<const ImplBinding*>> impl_bindings;
      for (auto ib : fn_type.impl_bindings()) {
        Nonnull<ImplBinding*> new_ib = g.arena()->New<ImplBinding>(
            ib->source_loc(), bind_map[ib->type_var()],
            Substitute(new_dict, ib->interface(), g));
        new_ib->set_original(ib->original());
        impl_bindings.push_back(new_ib);
      }
      // Apply substitution to parameter types
      auto param = Substitute(new_dict, &fn_type.parameters(), g);
      // Apply substitution to return type
      auto ret = Substitute(new_dict, &fn_type.return_type(), g);
      // Create the new FunctionType
      Nonnull<const Value*> new_fn_type = g.arena()->New<FunctionType>(
          param, generic_parameters, ret, deduced_bindings, impl_bindings);
      return new_fn_type;
    }
    case Value::Kind::PointerType: {
      return g.arena()->New<PointerType>(
          Substitute(dict, &cast<PointerType>(*type).type(), g));
    }
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*type);
      Nonnull<const NominalClassType*> new_class_type =
          g.arena()->New<NominalClassType>(
              &class_type.declaration(),
              SubstituteIntoBindings(class_type.bindings()));
      return new_class_type;
    }
    case Value::Kind::InterfaceType: {
      const auto& iface_type = cast<InterfaceType>(*type);
      Nonnull<const InterfaceType*> new_iface_type =
          g.arena()->New<InterfaceType>(
              &iface_type.declaration(),
              SubstituteIntoBindings(iface_type.bindings()));
      return new_iface_type;
    }
    case Value::Kind::ConstraintType: {
      const auto& constraint = cast<ConstraintType>(*type);
      ConstraintTypeBuilder builder(constraint.self_binding());
      for (const auto& impl_constraint : constraint.impl_constraints()) {
        builder.AddImplConstraint(
            {.type = Substitute(dict, impl_constraint.type, g),
             .interface = cast<InterfaceType>(
                 Substitute(dict, impl_constraint.interface, g))});
      }

      for (const auto& equality_constraint :
           constraint.equality_constraints()) {
        std::vector<Nonnull<const Value*>> values;
        for (const Value* value : equality_constraint.values) {
          // Ensure we don't create any duplicates through substitution.
          if (std::find_if(values.begin(), values.end(), [&](const Value* v) {
                return ValueEqual(v, value, std::nullopt);
              }) == values.end()) {
            values.push_back(Substitute(dict, value, g));
          }
        }
        builder.AddEqualityConstraint({.values = std::move(values)});
      }

      for (const auto& lookup_context : constraint.lookup_contexts()) {
        builder.AddLookupContext(
            {.context = Substitute(dict, lookup_context.context, g)});
      }
      Nonnull<const ConstraintType*> new_constraint =
          std::move(builder).Build(g.arena());
      if (g.trace_stream()) {
        **g.trace_stream() << "substitution: " << constraint << " => "
                           << *new_constraint << "\n";
      }
      return new_constraint;
    }
    case Value::Kind::StaticArrayType:
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
      return type;
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfConstraintType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
      // TODO: We should substitute into the value and produce a new type of
      // type for it.
      return type;
    case Value::Kind::ImplWitness:
    case Value::Kind::SymbolicWitness:
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

auto GenericChecker::MatchImpl(const InterfaceType& iface,
                               Nonnull<const Value*> impl_type,
                               const ImplScope::Impl& impl,
                               const ImplScope& impl_scope,
                               SourceLocation source_loc, Globals g)
    -> std::optional<Nonnull<Expression*>> {
  if (g.trace_stream()) {
    **g.trace_stream() << "MatchImpl: looking for " << *impl_type << " as "
                       << iface << "\n";
    **g.trace_stream() << "checking " << *impl.type << " as "
                       << *impl.interface << "\n";
  }

  BindingMap deduced_args;

  if (ErrorOr<Success> e = ArgumentDeduction(
          source_loc, "match", impl.deduced, deduced_args, impl.type, impl_type,
          /*allow_implicit_conversion=*/false, impl_scope, g);
      !e.ok()) {
    if (g.trace_stream()) {
      **g.trace_stream() << "type does not match: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (ErrorOr<Success> e = ArgumentDeduction(
          source_loc, "match", impl.deduced, deduced_args, impl.interface,
          &iface, /*allow_implicit_conversion=*/false, impl_scope, g);
      !e.ok()) {
    if (g.trace_stream()) {
      **g.trace_stream() << "interface does not match: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (g.trace_stream()) {
    **g.trace_stream() << "match results: {";
    llvm::ListSeparator sep;
    for (const auto& [binding, val] : deduced_args) {
      **g.trace_stream() << sep << *binding << " = " << *val;
    }
    **g.trace_stream() << "}\n";
  }

  CARBON_CHECK(impl.deduced.size() == deduced_args.size())
      << "failed to deduce all expected deduced arguments";

  // Ensure the constraints on the `impl` are satisfied by the deduced
  // arguments.
  ImplExpMap impls;
  if (ErrorOr<Success> e = SatisfyImpls(impl.impl_bindings, impl_scope,
                                        source_loc, deduced_args, impls, g);
      !e.ok()) {
    if (g.trace_stream()) {
      **g.trace_stream() << "missing required impl: " << e.error() << "\n";
    }
    return std::nullopt;
  }

  if (g.trace_stream()) {
    **g.trace_stream() << "matched with " << *impl.type << " as "
                       << *impl.interface << "\n\n";
  }
  return deduced_args.empty() ? impl.impl
                              : g.arena()->New<InstantiateImpl>(
                                    source_loc, impl.impl, deduced_args, impls);
}

}  // namespace Carbon
