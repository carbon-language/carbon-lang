// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/typecheck.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/common/tracing_flag.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;

namespace Carbon {

void ExpectType(int line_num, const std::string& context, const Value* expected,
                const Value* actual) {
  if (!TypeEqual(expected, actual)) {
    FATAL_COMPILATION_ERROR(line_num) << "type error in " << context << "\n"
                                      << "expected: " << *expected << "\n"
                                      << "actual: " << *actual;
  }
}

void ExpectPointerType(int line_num, const std::string& context,
                       const Value* actual) {
  if (actual->tag() != ValKind::PointerType) {
    FATAL_COMPILATION_ERROR(line_num) << "type error in " << context << "\n"
                                      << "expected a pointer type\n"
                                      << "actual: " << *actual;
  }
}

// Reify type to type expression.
auto ReifyType(const Value* t, int line_num) -> const Expression* {
  switch (t->tag()) {
    case ValKind::IntType:
      return Expression::MakeIntTypeLiteral(0);
    case ValKind::BoolType:
      return Expression::MakeBoolTypeLiteral(0);
    case ValKind::TypeType:
      return Expression::MakeTypeTypeLiteral(0);
    case ValKind::ContinuationType:
      return Expression::MakeContinuationTypeLiteral(0);
    case ValKind::FunctionType:
      return Expression::MakeFunctionTypeLiteral(
          0, ReifyType(t->GetFunctionType().param, line_num),
          ReifyType(t->GetFunctionType().ret, line_num),
          /*is_omitted_return_type=*/false);
    case ValKind::TupleValue: {
      std::vector<FieldInitializer> args;
      for (const TupleElement& field : t->GetTupleValue().elements) {
        args.push_back(
            FieldInitializer(field.name, ReifyType(field.value, line_num)));
      }
      return Expression::MakeTupleLiteral(0, args);
    }
    case ValKind::StructType:
      return Expression::MakeIdentifierExpression(0, t->GetStructType().name);
    case ValKind::ChoiceType:
      return Expression::MakeIdentifierExpression(0, t->GetChoiceType().name);
    case ValKind::PointerType:
      return Expression::MakePrimitiveOperatorExpression(
          0, Operator::Ptr, {ReifyType(t->GetPointerType().type, line_num)});
    case ValKind::VariableType:
      return Expression::MakeIdentifierExpression(0, t->GetVariableType().name);
    default:
      FATAL_INTERNAL_ERROR(line_num) << "expected a type, not " << *t;
  }
}

// Perform type argument deduction, matching the parameter type `param`
// against the argument type `arg`. Whenever there is an VariableType
// in the parameter type, it is deduced to be the corresponding type
// inside the argument type.
// The `deduced` parameter is an accumulator, that is, it holds the
// results so-far.
auto ArgumentDeduction(int line_num, TypeEnv deduced, const Value* param,
                       const Value* arg) -> TypeEnv {
  switch (param->tag()) {
    case ValKind::VariableType: {
      std::optional<const Value*> d =
          deduced.Get(param->GetVariableType().name);
      if (!d) {
        deduced.Set(param->GetVariableType().name, arg);
      } else {
        ExpectType(line_num, "argument deduction", *d, arg);
      }
      return deduced;
    }
    case ValKind::TupleValue: {
      if (arg->tag() != ValKind::TupleValue) {
        ExpectType(line_num, "argument deduction", param, arg);
      }
      if (param->GetTupleValue().elements.size() !=
          arg->GetTupleValue().elements.size()) {
        ExpectType(line_num, "argument deduction", param, arg);
      }
      for (size_t i = 0; i < param->GetTupleValue().elements.size(); ++i) {
        if (param->GetTupleValue().elements[i].name !=
            arg->GetTupleValue().elements[i].name) {
          FATAL_COMPILATION_ERROR(line_num)
              << "mismatch in tuple names, "
              << param->GetTupleValue().elements[i].name
              << " != " << arg->GetTupleValue().elements[i].name;
        }
        deduced = ArgumentDeduction(line_num, deduced,
                                    param->GetTupleValue().elements[i].value,
                                    arg->GetTupleValue().elements[i].value);
      }
      return deduced;
    }
    case ValKind::FunctionType: {
      if (arg->tag() != ValKind::FunctionType) {
        ExpectType(line_num, "argument deduction", param, arg);
      }
      // TODO: handle situation when arg has deduced parameters.
      deduced =
          ArgumentDeduction(line_num, deduced, param->GetFunctionType().param,
                            arg->GetFunctionType().param);
      deduced =
          ArgumentDeduction(line_num, deduced, param->GetFunctionType().ret,
                            arg->GetFunctionType().ret);
      return deduced;
    }
    case ValKind::PointerType: {
      if (arg->tag() != ValKind::PointerType) {
        ExpectType(line_num, "argument deduction", param, arg);
      }
      return ArgumentDeduction(line_num, deduced, param->GetPointerType().type,
                               arg->GetPointerType().type);
    }
    // Nothing to do in the case for `auto`.
    case ValKind::AutoType: {
      return deduced;
    }
    // For the following cases, we check for type equality.
    case ValKind::ContinuationType:
    case ValKind::StructType:
    case ValKind::ChoiceType:
    case ValKind::IntType:
    case ValKind::BoolType:
    case ValKind::TypeType: {
      ExpectType(line_num, "argument deduction", param, arg);
      return deduced;
    }
    // The rest of these cases should never happen.
    case ValKind::IntValue:
    case ValKind::BoolValue:
    case ValKind::FunctionValue:
    case ValKind::PointerValue:
    case ValKind::StructValue:
    case ValKind::AlternativeValue:
    case ValKind::BindingPlaceholderValue:
    case ValKind::AlternativeConstructorValue:
    case ValKind::ContinuationValue:
      FATAL_INTERNAL_ERROR(line_num)
          << "In ArgumentDeduction: expected type, not value " << *param;
  }
}

auto Substitute(TypeEnv dict, const Value* type) -> const Value* {
  switch (type->tag()) {
    case ValKind::VariableType: {
      std::optional<const Value*> t = dict.Get(type->GetVariableType().name);
      if (!t) {
        return type;
      } else {
        return *t;
      }
    }
    case ValKind::TupleValue: {
      std::vector<TupleElement> elts;
      for (const auto& elt : type->GetTupleValue().elements) {
        auto t = Substitute(dict, elt.value);
        elts.push_back({.name = elt.name, .value = t});
      }
      return Value::MakeTupleValue(elts);
    }
    case ValKind::FunctionType: {
      auto param = Substitute(dict, type->GetFunctionType().param);
      auto ret = Substitute(dict, type->GetFunctionType().ret);
      return Value::MakeFunctionType({}, param, ret);
    }
    case ValKind::PointerType: {
      return Value::MakePointerType(
          Substitute(dict, type->GetPointerType().type));
    }
    case ValKind::AutoType:
    case ValKind::IntType:
    case ValKind::BoolType:
    case ValKind::TypeType:
    case ValKind::StructType:
    case ValKind::ChoiceType:
    case ValKind::ContinuationType:
      return type;
    // The rest of these cases should never happen.
    case ValKind::IntValue:
    case ValKind::BoolValue:
    case ValKind::FunctionValue:
    case ValKind::PointerValue:
    case ValKind::StructValue:
    case ValKind::AlternativeValue:
    case ValKind::BindingPlaceholderValue:
    case ValKind::AlternativeConstructorValue:
    case ValKind::ContinuationValue:
      FATAL_INTERNAL_ERROR_NO_LINE()
          << "In Substitute: expected type, not value " << *type;
  }
}

// The TypeCheckExp function performs semantic analysis on an expression.
// It returns a new version of the expression, its type, and an
// updated environment which are bundled into a TCResult object.
// The purpose of the updated environment is
// to bring pattern variables into scope, for example, in a match case.
// The new version of the expression may include more information,
// for example, the type arguments deduced for the type parameters of a
// generic.
//
// e is the expression to be analyzed.
// types maps variable names to the type of their run-time value.
// values maps variable names to their compile-time values. It is not
//    directly used in this function but is passed to InterExp.
auto TypeCheckExp(const Expression* e, TypeEnv types, Env values)
    -> TCExpression {
  if (tracing_output) {
    llvm::outs() << "checking expression " << *e << "\n";
  }
  switch (e->tag()) {
    case ExpressionKind::IndexExpression: {
      auto res = TypeCheckExp(e->GetIndexExpression().aggregate, types, values);
      auto t = res.type;
      switch (t->tag()) {
        case ValKind::TupleValue: {
          auto i =
              InterpExp(values, e->GetIndexExpression().offset)->GetIntValue();
          std::string f = std::to_string(i);
          const Value* field_t = t->GetTupleValue().FindField(f);
          if (field_t == nullptr) {
            FATAL_COMPILATION_ERROR(e->line_num)
                << "field " << f << " is not in the tuple " << *t;
          }
          auto new_e = Expression::MakeIndexExpression(
              e->line_num, res.exp, Expression::MakeIntLiteral(e->line_num, i));
          return TCExpression(new_e, field_t, res.types);
        }
        default:
          FATAL_COMPILATION_ERROR(e->line_num) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<FieldInitializer> new_args;
      std::vector<TupleElement> arg_types;
      auto new_types = types;
      int i = 0;
      for (auto arg = e->GetTupleLiteral().fields.begin();
           arg != e->GetTupleLiteral().fields.end(); ++arg, ++i) {
        auto arg_res = TypeCheckExp(arg->expression, new_types, values);
        new_types = arg_res.types;
        new_args.push_back(FieldInitializer(arg->name, arg_res.exp));
        arg_types.push_back({.name = arg->name, .value = arg_res.type});
      }
      auto tuple_e = Expression::MakeTupleLiteral(e->line_num, new_args);
      auto tuple_t = Value::MakeTupleValue(std::move(arg_types));
      return TCExpression(tuple_e, tuple_t, new_types);
    }
    case ExpressionKind::FieldAccessExpression: {
      auto res =
          TypeCheckExp(e->GetFieldAccessExpression().aggregate, types, values);
      auto t = res.type;
      switch (t->tag()) {
        case ValKind::StructType:
          // Search for a field
          for (auto& field : t->GetStructType().fields) {
            if (e->GetFieldAccessExpression().field == field.first) {
              const Expression* new_e = Expression::MakeFieldAccessExpression(
                  e->line_num, res.exp, e->GetFieldAccessExpression().field);
              return TCExpression(new_e, field.second, res.types);
            }
          }
          // Search for a method
          for (auto& method : t->GetStructType().methods) {
            if (e->GetFieldAccessExpression().field == method.first) {
              const Expression* new_e = Expression::MakeFieldAccessExpression(
                  e->line_num, res.exp, e->GetFieldAccessExpression().field);
              return TCExpression(new_e, method.second, res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->line_num)
              << "struct " << t->GetStructType().name
              << " does not have a field named "
              << e->GetFieldAccessExpression().field;
        case ValKind::TupleValue:
          for (const TupleElement& field : t->GetTupleValue().elements) {
            if (e->GetFieldAccessExpression().field == field.name) {
              auto new_e = Expression::MakeFieldAccessExpression(
                  e->line_num, res.exp, e->GetFieldAccessExpression().field);
              return TCExpression(new_e, field.value, res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->line_num)
              << "struct " << t->GetStructType().name
              << " does not have a field named "
              << e->GetFieldAccessExpression().field;
        case ValKind::ChoiceType:
          for (auto vt = t->GetChoiceType().alternatives.begin();
               vt != t->GetChoiceType().alternatives.end(); ++vt) {
            if (e->GetFieldAccessExpression().field == vt->first) {
              const Expression* new_e = Expression::MakeFieldAccessExpression(
                  e->line_num, res.exp, e->GetFieldAccessExpression().field);
              auto fun_ty = Value::MakeFunctionType({}, vt->second, t);
              return TCExpression(new_e, fun_ty, res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->line_num)
              << "struct " << t->GetStructType().name
              << " does not have a field named "
              << e->GetFieldAccessExpression().field;

        default:
          FATAL_COMPILATION_ERROR(e->line_num)
              << "field access, expected a struct\n"
              << *e;
      }
    }
    case ExpressionKind::IdentifierExpression: {
      std::optional<const Value*> type =
          types.Get(e->GetIdentifierExpression().name);
      if (type) {
        return TCExpression(e, *type, types);
      } else {
        FATAL_COMPILATION_ERROR(e->line_num)
            << "could not find `" << e->GetIdentifierExpression().name << "`";
      }
    }
    case ExpressionKind::IntLiteral:
      return TCExpression(e, Value::MakeIntType(), types);
    case ExpressionKind::BoolLiteral:
      return TCExpression(e, Value::MakeBoolType(), types);
    case ExpressionKind::PrimitiveOperatorExpression: {
      std::vector<const Expression*> es;
      std::vector<const Value*> ts;
      auto new_types = types;
      for (const Expression* argument :
           e->GetPrimitiveOperatorExpression().arguments) {
        auto res = TypeCheckExp(argument, types, values);
        new_types = res.types;
        es.push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e = Expression::MakePrimitiveOperatorExpression(
          e->line_num, e->GetPrimitiveOperatorExpression().op, es);
      switch (e->GetPrimitiveOperatorExpression().op) {
        case Operator::Neg:
          ExpectType(e->line_num, "negation", Value::MakeIntType(), ts[0]);
          return TCExpression(new_e, Value::MakeIntType(), new_types);
        case Operator::Add:
          ExpectType(e->line_num, "addition(1)", Value::MakeIntType(), ts[0]);
          ExpectType(e->line_num, "addition(2)", Value::MakeIntType(), ts[1]);
          return TCExpression(new_e, Value::MakeIntType(), new_types);
        case Operator::Sub:
          ExpectType(e->line_num, "subtraction(1)", Value::MakeIntType(),
                     ts[0]);
          ExpectType(e->line_num, "subtraction(2)", Value::MakeIntType(),
                     ts[1]);
          return TCExpression(new_e, Value::MakeIntType(), new_types);
        case Operator::Mul:
          ExpectType(e->line_num, "multiplication(1)", Value::MakeIntType(),
                     ts[0]);
          ExpectType(e->line_num, "multiplication(2)", Value::MakeIntType(),
                     ts[1]);
          return TCExpression(new_e, Value::MakeIntType(), new_types);
        case Operator::And:
          ExpectType(e->line_num, "&&(1)", Value::MakeBoolType(), ts[0]);
          ExpectType(e->line_num, "&&(2)", Value::MakeBoolType(), ts[1]);
          return TCExpression(new_e, Value::MakeBoolType(), new_types);
        case Operator::Or:
          ExpectType(e->line_num, "||(1)", Value::MakeBoolType(), ts[0]);
          ExpectType(e->line_num, "||(2)", Value::MakeBoolType(), ts[1]);
          return TCExpression(new_e, Value::MakeBoolType(), new_types);
        case Operator::Not:
          ExpectType(e->line_num, "!", Value::MakeBoolType(), ts[0]);
          return TCExpression(new_e, Value::MakeBoolType(), new_types);
        case Operator::Eq:
          ExpectType(e->line_num, "==", ts[0], ts[1]);
          return TCExpression(new_e, Value::MakeBoolType(), new_types);
        case Operator::Deref:
          ExpectPointerType(e->line_num, "*", ts[0]);
          return TCExpression(new_e, ts[0]->GetPointerType().type, new_types);
        case Operator::Ptr:
          ExpectType(e->line_num, "*", Value::MakeTypeType(), ts[0]);
          return TCExpression(new_e, Value::MakeTypeType(), new_types);
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto fun_res =
          TypeCheckExp(e->GetCallExpression().function, types, values);
      switch (fun_res.type->tag()) {
        case ValKind::FunctionType: {
          auto fun_t = fun_res.type;
          auto arg_res = TypeCheckExp(e->GetCallExpression().argument,
                                      fun_res.types, values);
          auto parameter_type = fun_t->GetFunctionType().param;
          auto return_type = fun_t->GetFunctionType().ret;
          if (fun_t->GetFunctionType().deduced.size() > 0) {
            auto deduced_args = ArgumentDeduction(e->line_num, TypeEnv(),
                                                  parameter_type, arg_res.type);
            for (auto& deduced_param : fun_t->GetFunctionType().deduced) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (!deduced_args.Get(deduced_param.name)) {
                FATAL_COMPILATION_ERROR(e->line_num)
                    << "could not deduce type argument for type parameter "
                    << deduced_param.name;
              }
            }
            parameter_type = Substitute(deduced_args, parameter_type);
            return_type = Substitute(deduced_args, return_type);
          } else {
            ExpectType(e->line_num, "call", parameter_type, arg_res.type);
          }
          auto new_e = Expression::MakeCallExpression(e->line_num, fun_res.exp,
                                                      arg_res.exp);
          return TCExpression(new_e, return_type, arg_res.types);
        }
        default: {
          FATAL_COMPILATION_ERROR(e->line_num)
              << "in call, expected a function\n"
              << *e;
        }
      }
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto pt = InterpExp(values, e->GetFunctionTypeLiteral().parameter);
      auto rt = InterpExp(values, e->GetFunctionTypeLiteral().return_type);
      auto new_e = Expression::MakeFunctionTypeLiteral(
          e->line_num, ReifyType(pt, e->line_num), ReifyType(rt, e->line_num),
          /*is_omitted_return_type=*/false);
      return TCExpression(new_e, Value::MakeTypeType(), types);
    }
    case ExpressionKind::IntTypeLiteral:
      return TCExpression(e, Value::MakeTypeType(), types);
    case ExpressionKind::BoolTypeLiteral:
      return TCExpression(e, Value::MakeTypeType(), types);
    case ExpressionKind::TypeTypeLiteral:
      return TCExpression(e, Value::MakeTypeType(), types);
    case ExpressionKind::ContinuationTypeLiteral:
      return TCExpression(e, Value::MakeTypeType(), types);
  }
}

// Equivalent to TypeCheckExp, but operates on Patterns instead of Expressions.
// `expected` is the type that this pattern is expected to have, if the
// surrounding context gives us that information. Otherwise, it is null.
auto TypeCheckPattern(const Pattern* p, TypeEnv types, Env values,
                      const Value* expected) -> TCPattern {
  if (tracing_output) {
    llvm::outs() << "checking pattern, ";
    if (expected) {
      llvm::outs() << "expecting " << *expected;
    }
    llvm::outs() << ", " << *p << "\n";
  }
  switch (p->Tag()) {
    case Pattern::Kind::AutoPattern: {
      return {.pattern = p, .type = Value::MakeTypeType(), .types = types};
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*p);
      const Value* type;
      switch (binding.Type()->Tag()) {
        case Pattern::Kind::AutoPattern: {
          if (expected == nullptr) {
            FATAL_COMPILATION_ERROR(binding.LineNumber())
                << "auto not allowed here";
          } else {
            type = expected;
          }
          break;
        }
        case Pattern::Kind::ExpressionPattern: {
          type = InterpExp(
              values, cast<ExpressionPattern>(binding.Type())->Expression());
          CHECK(type->tag() != ValKind::AutoType);
          if (expected != nullptr) {
            ExpectType(binding.LineNumber(), "pattern variable", type,
                       expected);
          }
          break;
        }
        case Pattern::Kind::TuplePattern:
        case Pattern::Kind::BindingPattern:
        case Pattern::Kind::AlternativePattern:
          FATAL_COMPILATION_ERROR(binding.LineNumber())
              << "Unsupported type pattern";
      }
      auto new_p = global_arena->New<BindingPattern>(
          binding.LineNumber(), binding.Name(),
          global_arena->New<ExpressionPattern>(
              ReifyType(type, binding.LineNumber())));
      if (binding.Name().has_value()) {
        types.Set(*binding.Name(), type);
      }
      return {.pattern = new_p, .type = type, .types = types};
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*p);
      std::vector<TuplePattern::Field> new_fields;
      std::vector<TupleElement> field_types;
      auto new_types = types;
      if (expected && expected->tag() != ValKind::TupleValue) {
        FATAL_COMPILATION_ERROR(p->LineNumber()) << "didn't expect a tuple";
      }
      if (expected &&
          tuple.Fields().size() != expected->GetTupleValue().elements.size()) {
        FATAL_COMPILATION_ERROR(tuple.LineNumber())
            << "tuples of different length";
      }
      for (size_t i = 0; i < tuple.Fields().size(); ++i) {
        const TuplePattern::Field& field = tuple.Fields()[i];
        const Value* expected_field_type = nullptr;
        if (expected != nullptr) {
          const TupleElement& expected_element =
              expected->GetTupleValue().elements[i];
          if (expected_element.name != field.name) {
            FATAL_COMPILATION_ERROR(tuple.LineNumber())
                << "field names do not match, expected "
                << expected_element.name << " but got " << field.name;
          }
          expected_field_type = expected_element.value;
        }
        auto field_result = TypeCheckPattern(field.pattern, new_types, values,
                                             expected_field_type);
        new_types = field_result.types;
        new_fields.push_back(
            TuplePattern::Field(field.name, field_result.pattern));
        field_types.push_back({.name = field.name, .value = field_result.type});
      }
      auto new_tuple =
          global_arena->New<TuplePattern>(tuple.LineNumber(), new_fields);
      auto tuple_t = Value::MakeTupleValue(std::move(field_types));
      return {.pattern = new_tuple, .type = tuple_t, .types = new_types};
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*p);
      const Value* choice_type = InterpExp(values, alternative.ChoiceType());
      if (choice_type->tag() != ValKind::ChoiceType) {
        FATAL_COMPILATION_ERROR(alternative.LineNumber())
            << "alternative pattern does not name a choice type.";
      }
      if (expected != nullptr) {
        ExpectType(alternative.LineNumber(), "alternative pattern", expected,
                   choice_type);
      }
      const Value* parameter_types =
          FindInVarValues(alternative.AlternativeName(),
                          choice_type->GetChoiceType().alternatives);
      if (parameter_types == nullptr) {
        FATAL_COMPILATION_ERROR(alternative.LineNumber())
            << "'" << alternative.AlternativeName()
            << "' is not an alternative of " << choice_type;
      }
      TCPattern arg_results = TypeCheckPattern(alternative.Arguments(), types,
                                               values, parameter_types);
      return {.pattern = global_arena->New<AlternativePattern>(
                  alternative.LineNumber(),
                  ReifyType(choice_type, alternative.LineNumber()),
                  alternative.AlternativeName(),
                  cast<TuplePattern>(arg_results.pattern)),
              .type = choice_type,
              .types = arg_results.types};
    }
    case Pattern::Kind::ExpressionPattern: {
      TCExpression result =
          TypeCheckExp(cast<ExpressionPattern>(p)->Expression(), types, values);
      return {.pattern = global_arena->New<ExpressionPattern>(result.exp),
              .type = result.type,
              .types = result.types};
    }
  }
}

auto TypecheckCase(const Value* expected, const Pattern* pat,
                   const Statement* body, TypeEnv types, Env values,
                   const Value*& ret_type)
    -> std::pair<const Pattern*, const Statement*> {
  auto pat_res = TypeCheckPattern(pat, types, values, expected);
  auto res = TypeCheckStmt(body, pat_res.types, values, ret_type);
  return std::make_pair(pat, res.stmt);
}

// The TypeCheckStmt function performs semantic analysis on a statement.
// It returns a new version of the statement and a new type environment.
//
// The ret_type parameter is used for analyzing return statements.
// It is the declared return type of the enclosing function definition.
// If the return type is "auto", then the return type is inferred from
// the first return statement.
auto TypeCheckStmt(const Statement* s, TypeEnv types, Env values,
                   const Value*& ret_type) -> TCStatement {
  if (!s) {
    return TCStatement(s, types);
  }
  switch (s->tag()) {
    case StatementKind::Match: {
      auto res = TypeCheckExp(s->GetMatch().exp, types, values);
      auto res_type = res.type;
      auto new_clauses =
          global_arena
              ->New<std::list<std::pair<const Pattern*, const Statement*>>>();
      for (auto& clause : *s->GetMatch().clauses) {
        new_clauses->push_back(TypecheckCase(
            res_type, clause.first, clause.second, types, values, ret_type));
      }
      const Statement* new_s =
          Statement::MakeMatch(s->line_num, res.exp, new_clauses);
      return TCStatement(new_s, types);
    }
    case StatementKind::While: {
      auto cnd_res = TypeCheckExp(s->GetWhile().cond, types, values);
      ExpectType(s->line_num, "condition of `while`", Value::MakeBoolType(),
                 cnd_res.type);
      auto body_res =
          TypeCheckStmt(s->GetWhile().body, types, values, ret_type);
      auto new_s =
          Statement::MakeWhile(s->line_num, cnd_res.exp, body_res.stmt);
      return TCStatement(new_s, types);
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return TCStatement(s, types);
    case StatementKind::Block: {
      auto stmt_res =
          TypeCheckStmt(s->GetBlock().stmt, types, values, ret_type);
      return TCStatement(Statement::MakeBlock(s->line_num, stmt_res.stmt),
                         types);
    }
    case StatementKind::VariableDefinition: {
      auto res = TypeCheckExp(s->GetVariableDefinition().init, types, values);
      const Value* rhs_ty = res.type;
      auto lhs_res = TypeCheckPattern(s->GetVariableDefinition().pat, types,
                                      values, rhs_ty);
      const Statement* new_s = Statement::MakeVariableDefinition(
          s->line_num, s->GetVariableDefinition().pat, res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case StatementKind::Sequence: {
      auto stmt_res =
          TypeCheckStmt(s->GetSequence().stmt, types, values, ret_type);
      auto types2 = stmt_res.types;
      auto next_res =
          TypeCheckStmt(s->GetSequence().next, types2, values, ret_type);
      auto types3 = next_res.types;
      return TCStatement(
          Statement::MakeSequence(s->line_num, stmt_res.stmt, next_res.stmt),
          types3);
    }
    case StatementKind::Assign: {
      auto rhs_res = TypeCheckExp(s->GetAssign().rhs, types, values);
      auto rhs_t = rhs_res.type;
      auto lhs_res = TypeCheckExp(s->GetAssign().lhs, types, values);
      auto lhs_t = lhs_res.type;
      ExpectType(s->line_num, "assign", lhs_t, rhs_t);
      auto new_s = Statement::MakeAssign(s->line_num, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case StatementKind::ExpressionStatement: {
      auto res = TypeCheckExp(s->GetExpressionStatement().exp, types, values);
      auto new_s = Statement::MakeExpressionStatement(s->line_num, res.exp);
      return TCStatement(new_s, types);
    }
    case StatementKind::If: {
      auto cnd_res = TypeCheckExp(s->GetIf().cond, types, values);
      ExpectType(s->line_num, "condition of `if`", Value::MakeBoolType(),
                 cnd_res.type);
      auto thn_res =
          TypeCheckStmt(s->GetIf().then_stmt, types, values, ret_type);
      auto els_res =
          TypeCheckStmt(s->GetIf().else_stmt, types, values, ret_type);
      auto new_s = Statement::MakeIf(s->line_num, cnd_res.exp, thn_res.stmt,
                                     els_res.stmt);
      return TCStatement(new_s, types);
    }
    case StatementKind::Return: {
      auto res = TypeCheckExp(s->GetReturn().exp, types, values);
      if (ret_type->tag() == ValKind::AutoType) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        ret_type = res.type;
      } else {
        ExpectType(s->line_num, "return", ret_type, res.type);
      }
      return TCStatement(Statement::MakeReturn(s->line_num, res.exp,
                                               s->GetReturn().is_omitted_exp),
                         types);
    }
    case StatementKind::Continuation: {
      TCStatement body_result =
          TypeCheckStmt(s->GetContinuation().body, types, values, ret_type);
      const Statement* new_continuation = Statement::MakeContinuation(
          s->line_num, s->GetContinuation().continuation_variable,
          body_result.stmt);
      types.Set(s->GetContinuation().continuation_variable,
                Value::MakeContinuationType());
      return TCStatement(new_continuation, types);
    }
    case StatementKind::Run: {
      TCExpression argument_result =
          TypeCheckExp(s->GetRun().argument, types, values);
      ExpectType(s->line_num, "argument of `run`",
                 Value::MakeContinuationType(), argument_result.type);
      const Statement* new_run =
          Statement::MakeRun(s->line_num, argument_result.exp);
      return TCStatement(new_run, types);
    }
    case StatementKind::Await: {
      // nothing to do here
      return TCStatement(s, types);
    }
  }  // switch
}

auto CheckOrEnsureReturn(const Statement* stmt, bool void_return, int line_num)
    -> const Statement* {
  if (!stmt) {
    if (void_return) {
      return Statement::MakeReturn(line_num, nullptr,
                                   /*is_omitted_exp=*/true);
    } else {
      FATAL_COMPILATION_ERROR(line_num)
          << "control-flow reaches end of non-void function without a return";
    }
  }
  switch (stmt->tag()) {
    case StatementKind::Match: {
      auto new_clauses =
          global_arena
              ->New<std::list<std::pair<const Pattern*, const Statement*>>>();
      for (auto i = stmt->GetMatch().clauses->begin();
           i != stmt->GetMatch().clauses->end(); ++i) {
        auto s = CheckOrEnsureReturn(i->second, void_return, stmt->line_num);
        new_clauses->push_back(std::make_pair(i->first, s));
      }
      return Statement::MakeMatch(stmt->line_num, stmt->GetMatch().exp,
                                  new_clauses);
    }
    case StatementKind::Block:
      return Statement::MakeBlock(
          stmt->line_num, CheckOrEnsureReturn(stmt->GetBlock().stmt,
                                              void_return, stmt->line_num));
    case StatementKind::If:
      return Statement::MakeIf(
          stmt->line_num, stmt->GetIf().cond,
          CheckOrEnsureReturn(stmt->GetIf().then_stmt, void_return,
                              stmt->line_num),
          CheckOrEnsureReturn(stmt->GetIf().else_stmt, void_return,
                              stmt->line_num));
    case StatementKind::Return:
      return stmt;
    case StatementKind::Sequence:
      if (stmt->GetSequence().next) {
        return Statement::MakeSequence(
            stmt->line_num, stmt->GetSequence().stmt,
            CheckOrEnsureReturn(stmt->GetSequence().next, void_return,
                                stmt->line_num));
      } else {
        return CheckOrEnsureReturn(stmt->GetSequence().stmt, void_return,
                                   stmt->line_num);
      }
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
      return stmt;
    case StatementKind::Assign:
    case StatementKind::ExpressionStatement:
    case StatementKind::While:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::VariableDefinition:
      if (void_return) {
        return Statement::MakeSequence(
            stmt->line_num, stmt,
            Statement::MakeReturn(line_num, nullptr,
                                  /*is_omitted_exp=*/true));
      } else {
        FATAL_COMPILATION_ERROR(stmt->line_num)
            << "control-flow reaches end of non-void function without a return";
      }
  }
}

// TODO: factor common parts of TypeCheckFunDef and TypeOfFunDef into
// a function.
// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv types, Env values)
    -> struct FunctionDefinition* {
  // Bring the deduced parameters into scope
  for (const auto& deduced : f->deduced_parameters) {
    // auto t = InterpExp(values, deduced.type);
    Address a =
        state->heap.AllocateValue(Value::MakeVariableType(deduced.name));
    values.Set(deduced.name, a);
  }
  // Type check the parameter pattern
  auto param_res = TypeCheckPattern(f->param_pattern, types, values, nullptr);
  // Evaluate the return type expression
  auto return_type = InterpPattern(values, f->return_type);
  if (f->name == "main") {
    ExpectType(f->line_num, "return type of `main`", Value::MakeIntType(),
               return_type);
    // TODO: Check that main doesn't have any parameters.
  }
  auto res = TypeCheckStmt(f->body, param_res.types, values, return_type);
  bool void_return = TypeEqual(return_type, Value::MakeUnitTypeVal());
  auto body = CheckOrEnsureReturn(res.stmt, void_return, f->line_num);
  return global_arena->New<FunctionDefinition>(
      f->line_num, f->name, f->deduced_parameters, f->param_pattern,
      global_arena->New<ExpressionPattern>(ReifyType(return_type, f->line_num)),
      /*is_omitted_return_type=*/false, body);
}

auto TypeOfFunDef(TypeEnv types, Env values, const FunctionDefinition* fun_def)
    -> const Value* {
  // Bring the deduced parameters into scope
  for (const auto& deduced : fun_def->deduced_parameters) {
    // auto t = InterpExp(values, deduced.type);
    Address a =
        state->heap.AllocateValue(Value::MakeVariableType(deduced.name));
    values.Set(deduced.name, a);
  }
  // Type check the parameter pattern
  auto param_res =
      TypeCheckPattern(fun_def->param_pattern, types, values, nullptr);
  // Evaluate the return type expression
  auto ret = InterpPattern(values, fun_def->return_type);
  if (ret->tag() == ValKind::AutoType) {
    auto f = TypeCheckFunDef(fun_def, types, values);
    ret = InterpPattern(values, f->return_type);
  }
  return Value::MakeFunctionType(fun_def->deduced_parameters, param_res.type,
                                 ret);
}

auto TypeOfStructDef(const StructDefinition* sd, TypeEnv /*types*/, Env ct_top)
    -> const Value* {
  VarValues fields;
  VarValues methods;
  for (const Member* m : sd->members) {
    switch (m->tag()) {
      case MemberKind::FieldMember: {
        const BindingPattern* binding = m->GetFieldMember().binding;
        if (!binding->Name().has_value()) {
          FATAL_COMPILATION_ERROR(binding->LineNumber())
              << "Struct members must have names";
        }
        const Expression* type_expression =
            dyn_cast<ExpressionPattern>(binding->Type())->Expression();
        if (type_expression == nullptr) {
          FATAL_COMPILATION_ERROR(binding->LineNumber())
              << "Struct members must have explicit types";
        }
        auto type = InterpExp(ct_top, type_expression);
        fields.push_back(std::make_pair(*binding->Name(), type));
        break;
      }
    }
  }
  return Value::MakeStructType(sd->name, std::move(fields), std::move(methods));
}

static auto GetName(const Declaration& d) -> const std::string& {
  switch (d.tag()) {
    case DeclarationKind::FunctionDeclaration:
      return d.GetFunctionDeclaration().definition.name;
    case DeclarationKind::StructDeclaration:
      return d.GetStructDeclaration().definition.name;
    case DeclarationKind::ChoiceDeclaration:
      return d.GetChoiceDeclaration().name;
    case DeclarationKind::VariableDeclaration: {
      const BindingPattern* binding = d.GetVariableDeclaration().binding;
      if (!binding->Name().has_value()) {
        FATAL_COMPILATION_ERROR(binding->LineNumber())
            << "Top-level variable declarations must have names";
      }
      return *binding->Name();
    }
  }
}

auto MakeTypeChecked(const Declaration& d, const TypeEnv& types,
                     const Env& values) -> Declaration {
  switch (d.tag()) {
    case DeclarationKind::FunctionDeclaration:
      return Declaration::MakeFunctionDeclaration(*TypeCheckFunDef(
          &d.GetFunctionDeclaration().definition, types, values));

    case DeclarationKind::StructDeclaration: {
      const StructDefinition& struct_def = d.GetStructDeclaration().definition;
      std::list<Member*> fields;
      for (Member* m : struct_def.members) {
        switch (m->tag()) {
          case MemberKind::FieldMember:
            // TODO: Interpret the type expression and store the result.
            fields.push_back(m);
            break;
        }
      }
      return Declaration::MakeStructDeclaration(
          struct_def.line_num, struct_def.name, std::move(fields));
    }

    case DeclarationKind::ChoiceDeclaration:
      // TODO
      return d;

    case DeclarationKind::VariableDeclaration: {
      const auto& var = d.GetVariableDeclaration();
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      TCExpression type_checked_initializer =
          TypeCheckExp(var.initializer, types, values);
      const Expression* type =
          dyn_cast<ExpressionPattern>(var.binding->Type())->Expression();
      if (type == nullptr) {
        // TODO: consider adding support for `auto`
        FATAL_COMPILATION_ERROR(var.source_location)
            << "Type of a top-level variable must be an expression.";
      }
      const Value* declared_type = InterpExp(values, type);
      ExpectType(var.source_location, "initializer of variable", declared_type,
                 type_checked_initializer.type);
      return d;
    }
  }
}

static void TopLevel(const Declaration& d, TypeCheckContext* tops) {
  switch (d.tag()) {
    case DeclarationKind::FunctionDeclaration: {
      const FunctionDefinition& func_def =
          d.GetFunctionDeclaration().definition;
      auto t = TypeOfFunDef(tops->types, tops->values, &func_def);
      tops->types.Set(func_def.name, t);
      InitEnv(d, &tops->values);
      break;
    }

    case DeclarationKind::StructDeclaration: {
      const StructDefinition& struct_def = d.GetStructDeclaration().definition;
      auto st = TypeOfStructDef(&struct_def, tops->types, tops->values);
      Address a = state->heap.AllocateValue(st);
      tops->values.Set(struct_def.name, a);  // Is this obsolete?
      std::vector<TupleElement> field_types;
      for (const auto& [field_name, field_value] : st->GetStructType().fields) {
        field_types.push_back({.name = field_name, .value = field_value});
      }
      auto fun_ty = Value::MakeFunctionType(
          {}, Value::MakeTupleValue(std::move(field_types)), st);
      tops->types.Set(struct_def.name, fun_ty);
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = d.GetChoiceDeclaration();
      VarValues alts;
      for (const auto& [name, signature] : choice.alternatives) {
        auto t = InterpExp(tops->values, signature);
        alts.push_back(std::make_pair(name, t));
      }
      auto ct = Value::MakeChoiceType(choice.name, std::move(alts));
      Address a = state->heap.AllocateValue(ct);
      tops->values.Set(choice.name, a);  // Is this obsolete?
      tops->types.Set(choice.name, ct);
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = d.GetVariableDeclaration();
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      const Expression* type =
          cast<ExpressionPattern>(var.binding->Type())->Expression();
      const Value* declared_type = InterpExp(tops->values, type);
      tops->types.Set(*var.binding->Name(), declared_type);
      break;
    }
  }
}

auto TopLevel(std::list<Declaration>* fs) -> TypeCheckContext {
  TypeCheckContext tops;
  bool found_main = false;

  for (auto const& d : *fs) {
    if (GetName(d) == "main") {
      found_main = true;
    }
    TopLevel(d, &tops);
  }

  if (found_main == false) {
    FATAL_COMPILATION_ERROR_NO_LINE()
        << "program must contain a function named `main`";
  }
  return tops;
}

}  // namespace Carbon
