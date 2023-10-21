// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/pattern_match.h"

#include <algorithm>

#include "explorer/ast/value.h"
#include "explorer/base/arena.h"
#include "explorer/base/trace_stream.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/type_utils.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;

namespace Carbon {

static auto InitializePlaceholderValue(const ValueNodeView& value_node,
                                       ExpressionResult v,
                                       Nonnull<RuntimeScope*> bindings) {
  switch (value_node.expression_category()) {
    case ExpressionCategory::Reference:
      if (v.expression_category() == ExpressionCategory::Value ||
          v.expression_category() == ExpressionCategory::Reference) {
        // Build by copying from value or reference expression.
        bindings->Initialize(value_node, v.value());
      } else {
        // Location initialized by initializing expression, bind node to
        // address.
        CARBON_CHECK(v.address())
            << "Missing location from initializing expression";
        bindings->Bind(value_node, *v.address());
      }
      break;
    case ExpressionCategory::Value:
      if (v.expression_category() == ExpressionCategory::Value) {
        // We assume values are strictly nested for now.
        bindings->BindValue(value_node, v.value());
      } else if (v.expression_category() == ExpressionCategory::Reference) {
        // Bind the reference expression value directly.
        CARBON_CHECK(v.address())
            << "Missing location from reference expression";
        bindings->BindAndPin(value_node, *v.address());
      } else {
        // Location initialized by initializing expression, bind node to
        // address.
        CARBON_CHECK(v.address())
            << "Missing location from initializing expression";
        bindings->Bind(value_node, *v.address());
      }
      break;
    case ExpressionCategory::Initializing:
      CARBON_FATAL() << "Cannot pattern match an initializing expression";
      break;
  }
}

auto PatternMatch(Nonnull<const Value*> p, ExpressionResult v,
                  SourceLocation source_loc,
                  std::optional<Nonnull<RuntimeScope*>> bindings,
                  BindingMap& generic_args, Nonnull<TraceStream*> trace_stream,
                  Nonnull<Arena*> arena) -> bool {
  if (trace_stream->is_enabled()) {
    trace_stream->Match() << "match pattern `" << *p << "`\n";
    trace_stream->Indent() << "from "
                           << ExpressionCategoryToString(
                                  v.expression_category())
                           << " expression with value `" << *v.value() << "`\n";
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
      const auto& p_arr = cast<StaticArrayType>(*p);
      switch (v.value()->kind()) {
        case Value::Kind::TupleType:
        case Value::Kind::TupleValue: {
          const auto& v_tup = cast<TupleValueBase>(*v.value());
          if (v_tup.elements().empty()) {
            return !TypeIsDeduceable(&p_arr.element_type());
          }

          std::vector<Nonnull<const Value*>> deduced_types;
          deduced_types.reserve(v_tup.elements().size());
          for (const auto& tup_elem : v_tup.elements()) {
            if (!PatternMatch(&p_arr.element_type(), make_expr_result(tup_elem),
                              source_loc, bindings, generic_args, trace_stream,
                              arena)) {
              return false;
            }

            deduced_types.emplace_back(
                DeducePatternType(&p_arr.element_type(), tup_elem, arena));
          }  // for

          return std::adjacent_find(deduced_types.begin(), deduced_types.end(),
                                    [](const auto& lhs, const auto& rhs) {
                                      return !TypeEqual(lhs, rhs, std::nullopt);
                                    }) == deduced_types.end();
        }
        case Value::Kind::StaticArrayType: {
          const auto& v_arr = cast<StaticArrayType>(*v.value());
          if (!v_arr.has_size()) {
            return false;
          }
          return PatternMatch(
              &p_arr.element_type(), make_expr_result(&v_arr.element_type()),
              source_loc, bindings, generic_args, trace_stream, arena);
        }
        default:
          return false;
      }
    }
    default:
      return ValueEqual(p, v.value(), std::nullopt);
  }
}
}  // namespace Carbon
