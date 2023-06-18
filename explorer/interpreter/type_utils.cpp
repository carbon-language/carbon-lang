// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_utils.h"

#include "explorer/ast/value.h"
#include "explorer/common/arena.h"
#include "explorer/common/trace_stream.h"
#include "explorer/interpreter/action.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

auto PatternMatch(Nonnull<const Value*> p, ExpressionResult v,
                  SourceLocation source_loc,
                  std::optional<Nonnull<RuntimeScope*>> bindings,
                  BindingMap& generic_args, Nonnull<TraceStream*> trace_stream,
                  Nonnull<Arena*> arena) -> bool {
  if (trace_stream->is_enabled()) {
    *trace_stream << "match pattern " << *p << "\nfrom "
                  << ExpressionCategoryToString(v.expression_category())
                  << " expression with value " << *v.value() << "\n";
  }
  const auto make_expr_result =
      [](Nonnull<const Value*> v) -> ExpressionResult {
    if (const auto* expr_v = dyn_cast<ReferenceExpressionValue>(v)) {
      return ExpressionResult::Reference(expr_v->value(), expr_v->address());
    }
    return ExpressionResult::Value(v);
  };
  if (v.value()->kind() == Value::Kind::ReferenceExpressionValue) {
    return PatternMatch(p, make_expr_result(v.value()), source_loc, bindings,
                        generic_args, trace_stream, arena);
  }
  switch (p->kind()) {
    case Value::Kind::BindingPlaceholderValue: {
      CARBON_CHECK(bindings.has_value());
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      if (placeholder.value_node().has_value()) {
        InitializePlaceholderValue(*placeholder.value_node(), v, *bindings);
      }
      return true;
    }
    case Value::Kind::AddrValue: {
      const auto& addr = cast<AddrValue>(*p);
      CARBON_CHECK(v.value()->kind() == Value::Kind::LocationValue);
      const auto& location = cast<LocationValue>(*v.value());
      return PatternMatch(
          &addr.pattern(),
          ExpressionResult::Value(arena->New<PointerValue>(location.address())),
          source_loc, bindings, generic_args, trace_stream, arena);
    }
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*p);
      generic_args[&var_type.binding()] = v.value();
      return true;
    }
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue:
      switch (v.value()->kind()) {
        case Value::Kind::TupleType:
        case Value::Kind::TupleValue: {
          const auto& p_tup = cast<TupleValueBase>(*p);
          const auto& v_tup = cast<TupleValueBase>(*v.value());
          CARBON_CHECK(p_tup.elements().size() == v_tup.elements().size());
          for (size_t i = 0; i < p_tup.elements().size(); ++i) {
            if (!PatternMatch(p_tup.elements()[i],
                              make_expr_result(v_tup.elements()[i]), source_loc,
                              bindings, generic_args, trace_stream, arena)) {
              return false;
            }
          }  // for
          return true;
        }
        case Value::Kind::UninitializedValue: {
          const auto& p_tup = cast<TupleValueBase>(*p);
          for (const auto& ele : p_tup.elements()) {
            if (!PatternMatch(ele,
                              ExpressionResult::Value(
                                  arena->New<UninitializedValue>(ele)),
                              source_loc, bindings, generic_args, trace_stream,
                              arena)) {
              return false;
            }
          }
          return true;
        }
        default:
          CARBON_FATAL() << "expected a tuple value in pattern, not "
                         << *v.value();
      }
    case Value::Kind::StructValue: {
      const auto& p_struct = cast<StructValue>(*p);
      const auto& v_struct = cast<StructValue>(*v.value());
      CARBON_CHECK(p_struct.elements().size() == v_struct.elements().size());
      for (size_t i = 0; i < p_struct.elements().size(); ++i) {
        CARBON_CHECK(p_struct.elements()[i].name ==
                     v_struct.elements()[i].name);
        if (!PatternMatch(p_struct.elements()[i].value,
                          ExpressionResult::Value(v_struct.elements()[i].value),
                          source_loc, bindings, generic_args, trace_stream,
                          arena)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::AlternativeValue:
      switch (v.value()->kind()) {
        case Value::Kind::AlternativeValue: {
          const auto& p_alt = cast<AlternativeValue>(*p);
          const auto& v_alt = cast<AlternativeValue>(*v.value());
          if (&p_alt.alternative() != &v_alt.alternative()) {
            return false;
          }
          CARBON_CHECK(p_alt.argument().has_value() ==
                       v_alt.argument().has_value());
          if (!p_alt.argument().has_value()) {
            return true;
          }
          return PatternMatch(
              *p_alt.argument(), ExpressionResult::Value(*v_alt.argument()),
              source_loc, bindings, generic_args, trace_stream, arena);
        }
        default:
          CARBON_FATAL() << "expected a choice alternative in pattern, not "
                         << *v.value();
      }
    case Value::Kind::UninitializedValue:
      CARBON_FATAL() << "uninitialized value is not allowed in pattern "
                     << *v.value();
    case Value::Kind::FunctionType:
      switch (v.value()->kind()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v.value());
          if (!PatternMatch(&p_fn.parameters(),
                            ExpressionResult::Value(&v_fn.parameters()),
                            source_loc, bindings, generic_args, trace_stream,
                            arena)) {
            return false;
          }
          if (!PatternMatch(&p_fn.return_type(),
                            ExpressionResult::Value(&v_fn.return_type()),
                            source_loc, bindings, generic_args, trace_stream,
                            arena)) {
            return false;
          }
          return true;
        }
        default:
          return false;
      }
    case Value::Kind::AutoType:
      // `auto` matches any type, without binding any new names. We rely
      // on the typechecker to ensure that `v.value()` is a type.
      return true;
    case Value::Kind::StaticArrayType: {
      switch (v.value()->kind()) {
        case Value::Kind::TupleType:
        case Value::Kind::TupleValue: {
          return true;
        }
        case Value::Kind::StaticArrayType: {
          const auto& v_arr = cast<StaticArrayType>(*v.value());
          return v_arr.has_size();
        }
        default:
          return false;
      }
    }
    default:
      return ValueEqual(p, v.value(), std::nullopt);
  }
}

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
}  // namespace Carbon
