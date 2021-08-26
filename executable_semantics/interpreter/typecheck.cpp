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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;

namespace Carbon {

void PrintTypeEnv(TypeEnv types, llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& [name, type] : types) {
    out << sep << name << ": " << *type;
  }
}

static void ExpectType(SourceLocation loc, const std::string& context,
                       const Value* expected, const Value* actual) {
  if (!TypeEqual(expected, actual)) {
    FATAL_COMPILATION_ERROR(loc) << "type error in " << context << "\n"
                                 << "expected: " << *expected << "\n"
                                 << "actual: " << *actual;
  }
}

static void ExpectPointerType(SourceLocation loc, const std::string& context,
                              const Value* actual) {
  if (actual->Tag() != Value::Kind::PointerType) {
    FATAL_COMPILATION_ERROR(loc) << "type error in " << context << "\n"
                                 << "expected a pointer type\n"
                                 << "actual: " << *actual;
  }
}

static SourceLocation ReifyFakeSourceLoc() {
  return SourceLocation("<reify>", 0);
}

// Reify type to type expression.
static auto ReifyType(const Value* t, SourceLocation loc) -> const Expression* {
  switch (t->Tag()) {
    case Value::Kind::IntType:
      return global_arena->RawNew<IntTypeLiteral>(ReifyFakeSourceLoc());
    case Value::Kind::BoolType:
      return global_arena->RawNew<BoolTypeLiteral>(ReifyFakeSourceLoc());
    case Value::Kind::TypeType:
      return global_arena->RawNew<TypeTypeLiteral>(ReifyFakeSourceLoc());
    case Value::Kind::ContinuationType:
      return global_arena->RawNew<ContinuationTypeLiteral>(
          ReifyFakeSourceLoc());
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*t);
      return global_arena->RawNew<FunctionTypeLiteral>(
          ReifyFakeSourceLoc(), ReifyType(fn_type.Param(), loc),
          ReifyType(fn_type.Ret(), loc),
          /*is_omitted_return_type=*/false);
    }
    case Value::Kind::TupleValue: {
      std::vector<FieldInitializer> args;
      for (const TupleElement& field : cast<TupleValue>(*t).Elements()) {
        args.push_back(
            FieldInitializer(field.name, ReifyType(field.value, loc)));
      }
      return global_arena->RawNew<TupleLiteral>(ReifyFakeSourceLoc(), args);
    }
    case Value::Kind::ClassType:
      return global_arena->RawNew<IdentifierExpression>(
          ReifyFakeSourceLoc(), cast<ClassType>(*t).Name());
    case Value::Kind::ChoiceType:
      return global_arena->RawNew<IdentifierExpression>(
          ReifyFakeSourceLoc(), cast<ChoiceType>(*t).Name());
    case Value::Kind::PointerType:
      return global_arena->RawNew<PrimitiveOperatorExpression>(
          ReifyFakeSourceLoc(), Operator::Ptr,
          std::vector<const Expression*>(
              {ReifyType(cast<PointerType>(*t).Type(), loc)}));
    case Value::Kind::VariableType:
      return global_arena->RawNew<IdentifierExpression>(
          ReifyFakeSourceLoc(), cast<VariableType>(*t).Name());
    case Value::Kind::StringType:
      return global_arena->RawNew<StringTypeLiteral>(ReifyFakeSourceLoc());
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::AutoType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::BoolValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::IntValue:
    case Value::Kind::PointerValue:
    case Value::Kind::StringValue:
    case Value::Kind::StructValue:
      FATAL() << "expected a type, not " << *t;
  }
}

// Perform type argument deduction, matching the parameter type `param`
// against the argument type `arg`. Whenever there is an VariableType
// in the parameter type, it is deduced to be the corresponding type
// inside the argument type.
// The `deduced` parameter is an accumulator, that is, it holds the
// results so-far.
static auto ArgumentDeduction(SourceLocation loc, TypeEnv deduced,
                              const Value* param, const Value* arg) -> TypeEnv {
  switch (param->Tag()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param);
      std::optional<const Value*> d = deduced.Get(var_type.Name());
      if (!d) {
        deduced.Set(var_type.Name(), arg);
      } else {
        ExpectType(loc, "argument deduction", *d, arg);
      }
      return deduced;
    }
    case Value::Kind::TupleValue: {
      if (arg->Tag() != Value::Kind::TupleValue) {
        ExpectType(loc, "argument deduction", param, arg);
      }
      const auto& param_tup = cast<TupleValue>(*param);
      const auto& arg_tup = cast<TupleValue>(*arg);
      if (param_tup.Elements().size() != arg_tup.Elements().size()) {
        ExpectType(loc, "argument deduction", param, arg);
      }
      for (size_t i = 0; i < param_tup.Elements().size(); ++i) {
        if (param_tup.Elements()[i].name != arg_tup.Elements()[i].name) {
          FATAL_COMPILATION_ERROR(loc)
              << "mismatch in tuple names, " << param_tup.Elements()[i].name
              << " != " << arg_tup.Elements()[i].name;
        }
        deduced = ArgumentDeduction(loc, deduced, param_tup.Elements()[i].value,
                                    arg_tup.Elements()[i].value);
      }
      return deduced;
    }
    case Value::Kind::FunctionType: {
      if (arg->Tag() != Value::Kind::FunctionType) {
        ExpectType(loc, "argument deduction", param, arg);
      }
      const auto& param_fn = cast<FunctionType>(*param);
      const auto& arg_fn = cast<FunctionType>(*arg);
      // TODO: handle situation when arg has deduced parameters.
      deduced =
          ArgumentDeduction(loc, deduced, param_fn.Param(), arg_fn.Param());
      deduced = ArgumentDeduction(loc, deduced, param_fn.Ret(), arg_fn.Ret());
      return deduced;
    }
    case Value::Kind::PointerType: {
      if (arg->Tag() != Value::Kind::PointerType) {
        ExpectType(loc, "argument deduction", param, arg);
      }
      return ArgumentDeduction(loc, deduced, cast<PointerType>(*param).Type(),
                               cast<PointerType>(*arg).Type());
    }
    // Nothing to do in the case for `auto`.
    case Value::Kind::AutoType: {
      return deduced;
    }
    // For the following cases, we check for type equality.
    case Value::Kind::ContinuationType:
    case Value::Kind::ClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
      ExpectType(loc, "argument deduction", param, arg);
      return deduced;
    // The rest of these cases should never happen.
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::PointerValue:
    case Value::Kind::StructValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
      FATAL() << "In ArgumentDeduction: expected type, not value " << *param;
  }
}

static auto Substitute(TypeEnv dict, const Value* type) -> const Value* {
  switch (type->Tag()) {
    case Value::Kind::VariableType: {
      std::optional<const Value*> t =
          dict.Get(cast<VariableType>(*type).Name());
      if (!t) {
        return type;
      } else {
        return *t;
      }
    }
    case Value::Kind::TupleValue: {
      std::vector<TupleElement> elts;
      for (const auto& elt : cast<TupleValue>(*type).Elements()) {
        auto t = Substitute(dict, elt.value);
        elts.push_back({.name = elt.name, .value = t});
      }
      return global_arena->RawNew<TupleValue>(elts);
    }
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*type);
      auto param = Substitute(dict, fn_type.Param());
      auto ret = Substitute(dict, fn_type.Ret());
      return global_arena->RawNew<FunctionType>(std::vector<GenericBinding>(),
                                                param, ret);
    }
    case Value::Kind::PointerType: {
      return global_arena->RawNew<PointerType>(
          Substitute(dict, cast<PointerType>(*type).Type()));
    }
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::ClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
      return type;
    // The rest of these cases should never happen.
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::PointerValue:
    case Value::Kind::StructValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
      FATAL() << "In Substitute: expected type, not value " << *type;
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
    llvm::outs() << "checking expression " << *e << "\ntypes: ";
    PrintTypeEnv(types, llvm::outs());
    llvm::outs() << "\nvalues: ";
    PrintEnv(values, llvm::outs());
    llvm::outs() << "\n";
  }
  switch (e->Tag()) {
    case Expression::Kind::IndexExpression: {
      const auto& index = cast<IndexExpression>(*e);
      auto res = TypeCheckExp(index.Aggregate(), types, values);
      auto t = res.type;
      switch (t->Tag()) {
        case Value::Kind::TupleValue: {
          auto i = cast<IntValue>(*InterpExp(values, index.Offset())).Val();
          std::string f = std::to_string(i);
          const Value* field_t = cast<TupleValue>(*t).FindField(f);
          if (field_t == nullptr) {
            FATAL_COMPILATION_ERROR(e->SourceLoc())
                << "field " << f << " is not in the tuple " << *t;
          }
          auto new_e = global_arena->RawNew<IndexExpression>(
              e->SourceLoc(), res.exp,
              global_arena->RawNew<IntLiteral>(e->SourceLoc(), i));
          return TCExpression(new_e, field_t, res.types);
        }
        default:
          FATAL_COMPILATION_ERROR(e->SourceLoc()) << "expected a tuple";
      }
    }
    case Expression::Kind::TupleLiteral: {
      std::vector<FieldInitializer> new_args;
      std::vector<TupleElement> arg_types;
      auto new_types = types;
      for (const auto& arg : cast<TupleLiteral>(*e).Fields()) {
        auto arg_res = TypeCheckExp(arg.expression, new_types, values);
        new_types = arg_res.types;
        new_args.push_back(FieldInitializer(arg.name, arg_res.exp));
        arg_types.push_back({.name = arg.name, .value = arg_res.type});
      }
      auto tuple_e =
          global_arena->RawNew<TupleLiteral>(e->SourceLoc(), new_args);
      auto tuple_t = global_arena->RawNew<TupleValue>(std::move(arg_types));
      return TCExpression(tuple_e, tuple_t, new_types);
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*e);
      auto res = TypeCheckExp(access.Aggregate(), types, values);
      auto t = res.type;
      switch (t->Tag()) {
        case Value::Kind::ClassType: {
          const auto& t_class = cast<ClassType>(*t);
          // Search for a field
          for (auto& field : t_class.Fields()) {
            if (access.Field() == field.first) {
              const Expression* new_e =
                  global_arena->RawNew<FieldAccessExpression>(
                      e->SourceLoc(), res.exp, access.Field());
              return TCExpression(new_e, field.second, res.types);
            }
          }
          // Search for a method
          for (auto& method : t_class.Methods()) {
            if (access.Field() == method.first) {
              const Expression* new_e =
                  global_arena->RawNew<FieldAccessExpression>(
                      e->SourceLoc(), res.exp, access.Field());
              return TCExpression(new_e, method.second, res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->SourceLoc())
              << "class " << t_class.Name() << " does not have a field named "
              << access.Field();
        }
        case Value::Kind::TupleValue: {
          const auto& tup = cast<TupleValue>(*t);
          for (const TupleElement& field : tup.Elements()) {
            if (access.Field() == field.name) {
              auto new_e = global_arena->RawNew<FieldAccessExpression>(
                  e->SourceLoc(), res.exp, access.Field());
              return TCExpression(new_e, field.value, res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->SourceLoc())
              << "tuple " << tup << " does not have a field named "
              << access.Field();
        }
        case Value::Kind::ChoiceType: {
          const auto& choice = cast<ChoiceType>(*t);
          for (const auto& vt : choice.Alternatives()) {
            if (access.Field() == vt.first) {
              const Expression* new_e =
                  global_arena->RawNew<FieldAccessExpression>(
                      e->SourceLoc(), res.exp, access.Field());
              auto fun_ty = global_arena->RawNew<FunctionType>(
                  std::vector<GenericBinding>(), vt.second, t);
              return TCExpression(new_e, fun_ty, res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->SourceLoc())
              << "choice " << choice.Name() << " does not have a field named "
              << access.Field();
        }
        default:
          FATAL_COMPILATION_ERROR(e->SourceLoc())
              << "field access, expected a struct\n"
              << *e;
      }
    }
    case Expression::Kind::IdentifierExpression: {
      const auto& ident = cast<IdentifierExpression>(*e);
      std::optional<const Value*> type = types.Get(ident.Name());
      if (type) {
        return TCExpression(e, *type, types);
      } else {
        FATAL_COMPILATION_ERROR(e->SourceLoc())
            << "could not find `" << ident.Name() << "`";
      }
    }
    case Expression::Kind::IntLiteral:
      return TCExpression(e, global_arena->RawNew<IntType>(), types);
    case Expression::Kind::BoolLiteral:
      return TCExpression(e, global_arena->RawNew<BoolType>(), types);
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(*e);
      std::vector<const Expression*> es;
      std::vector<const Value*> ts;
      auto new_types = types;
      for (const Expression* argument : op.Arguments()) {
        auto res = TypeCheckExp(argument, types, values);
        new_types = res.types;
        es.push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e = global_arena->RawNew<PrimitiveOperatorExpression>(
          e->SourceLoc(), op.Op(), es);
      switch (op.Op()) {
        case Operator::Neg:
          ExpectType(e->SourceLoc(), "negation",
                     global_arena->RawNew<IntType>(), ts[0]);
          return TCExpression(new_e, global_arena->RawNew<IntType>(),
                              new_types);
        case Operator::Add:
          ExpectType(e->SourceLoc(), "addition(1)",
                     global_arena->RawNew<IntType>(), ts[0]);
          ExpectType(e->SourceLoc(), "addition(2)",
                     global_arena->RawNew<IntType>(), ts[1]);
          return TCExpression(new_e, global_arena->RawNew<IntType>(),
                              new_types);
        case Operator::Sub:
          ExpectType(e->SourceLoc(), "subtraction(1)",
                     global_arena->RawNew<IntType>(), ts[0]);
          ExpectType(e->SourceLoc(), "subtraction(2)",
                     global_arena->RawNew<IntType>(), ts[1]);
          return TCExpression(new_e, global_arena->RawNew<IntType>(),
                              new_types);
        case Operator::Mul:
          ExpectType(e->SourceLoc(), "multiplication(1)",
                     global_arena->RawNew<IntType>(), ts[0]);
          ExpectType(e->SourceLoc(), "multiplication(2)",
                     global_arena->RawNew<IntType>(), ts[1]);
          return TCExpression(new_e, global_arena->RawNew<IntType>(),
                              new_types);
        case Operator::And:
          ExpectType(e->SourceLoc(), "&&(1)", global_arena->RawNew<BoolType>(),
                     ts[0]);
          ExpectType(e->SourceLoc(), "&&(2)", global_arena->RawNew<BoolType>(),
                     ts[1]);
          return TCExpression(new_e, global_arena->RawNew<BoolType>(),
                              new_types);
        case Operator::Or:
          ExpectType(e->SourceLoc(), "||(1)", global_arena->RawNew<BoolType>(),
                     ts[0]);
          ExpectType(e->SourceLoc(), "||(2)", global_arena->RawNew<BoolType>(),
                     ts[1]);
          return TCExpression(new_e, global_arena->RawNew<BoolType>(),
                              new_types);
        case Operator::Not:
          ExpectType(e->SourceLoc(), "!", global_arena->RawNew<BoolType>(),
                     ts[0]);
          return TCExpression(new_e, global_arena->RawNew<BoolType>(),
                              new_types);
        case Operator::Eq:
          ExpectType(e->SourceLoc(), "==", ts[0], ts[1]);
          return TCExpression(new_e, global_arena->RawNew<BoolType>(),
                              new_types);
        case Operator::Deref:
          ExpectPointerType(e->SourceLoc(), "*", ts[0]);
          return TCExpression(new_e, cast<PointerType>(*ts[0]).Type(),
                              new_types);
        case Operator::Ptr:
          ExpectType(e->SourceLoc(), "*", global_arena->RawNew<TypeType>(),
                     ts[0]);
          return TCExpression(new_e, global_arena->RawNew<TypeType>(),
                              new_types);
      }
      break;
    }
    case Expression::Kind::CallExpression: {
      const auto& call = cast<CallExpression>(*e);
      auto fun_res = TypeCheckExp(call.Function(), types, values);
      switch (fun_res.type->Tag()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(*fun_res.type);
          auto arg_res = TypeCheckExp(call.Argument(), fun_res.types, values);
          auto parameter_type = fun_t.Param();
          auto return_type = fun_t.Ret();
          if (!fun_t.Deduced().empty()) {
            auto deduced_args = ArgumentDeduction(e->SourceLoc(), TypeEnv(),
                                                  parameter_type, arg_res.type);
            for (auto& deduced_param : fun_t.Deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (!deduced_args.Get(deduced_param.name)) {
                FATAL_COMPILATION_ERROR(e->SourceLoc())
                    << "could not deduce type argument for type parameter "
                    << deduced_param.name;
              }
            }
            parameter_type = Substitute(deduced_args, parameter_type);
            return_type = Substitute(deduced_args, return_type);
          } else {
            ExpectType(e->SourceLoc(), "call", parameter_type, arg_res.type);
          }
          auto new_e = global_arena->RawNew<CallExpression>(
              e->SourceLoc(), fun_res.exp, arg_res.exp);
          return TCExpression(new_e, return_type, arg_res.types);
        }
        default: {
          FATAL_COMPILATION_ERROR(e->SourceLoc())
              << "in call, expected a function\n"
              << *e;
        }
      }
      break;
    }
    case Expression::Kind::FunctionTypeLiteral: {
      const auto& fn = cast<FunctionTypeLiteral>(*e);
      auto pt = InterpExp(values, fn.Parameter());
      auto rt = InterpExp(values, fn.ReturnType());
      auto new_e = global_arena->RawNew<FunctionTypeLiteral>(
          e->SourceLoc(), ReifyType(pt, e->SourceLoc()),
          ReifyType(rt, e->SourceLoc()),
          /*is_omitted_return_type=*/false);
      return TCExpression(new_e, global_arena->RawNew<TypeType>(), types);
    }
    case Expression::Kind::StringLiteral:
      return TCExpression(e, global_arena->RawNew<StringType>(), types);
    case Expression::Kind::IntrinsicExpression:
      switch (cast<IntrinsicExpression>(*e).Intrinsic()) {
        case IntrinsicExpression::IntrinsicKind::Print:
          return TCExpression(e, &TupleValue::Empty(), types);
      }
    case Expression::Kind::IntTypeLiteral:
    case Expression::Kind::BoolTypeLiteral:
    case Expression::Kind::StringTypeLiteral:
    case Expression::Kind::TypeTypeLiteral:
    case Expression::Kind::ContinuationTypeLiteral:
      return TCExpression(e, global_arena->RawNew<TypeType>(), types);
  }
}

// Equivalent to TypeCheckExp, but operates on Patterns instead of Expressions.
// `expected` is the type that this pattern is expected to have, if the
// surrounding context gives us that information. Otherwise, it is null.
auto TypeCheckPattern(const Pattern* p, TypeEnv types, Env values,
                      const Value* expected) -> TCPattern {
  if (tracing_output) {
    llvm::outs() << "checking pattern " << *p;
    if (expected) {
      llvm::outs() << ", expecting " << *expected;
    }
    llvm::outs() << "\ntypes: ";
    PrintTypeEnv(types, llvm::outs());
    llvm::outs() << "\nvalues: ";
    PrintEnv(values, llvm::outs());
    llvm::outs() << "\n";
  }
  switch (p->Tag()) {
    case Pattern::Kind::AutoPattern: {
      return {.pattern = p,
              .type = global_arena->RawNew<TypeType>(),
              .types = types};
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*p);
      TCPattern binding_type_result =
          TypeCheckPattern(binding.Type(), types, values, nullptr);
      const Value* type = InterpPattern(values, binding_type_result.pattern);
      if (expected != nullptr) {
        std::optional<Env> values =
            PatternMatch(type, expected, binding.Type()->SourceLoc());
        if (values == std::nullopt) {
          FATAL_COMPILATION_ERROR(binding.Type()->SourceLoc())
              << "Type pattern '" << *type << "' does not match actual type '"
              << *expected << "'";
        }
        CHECK(values->begin() == values->end())
            << "Name bindings within type patterns are unsupported";
        type = expected;
      }
      auto new_p = global_arena->RawNew<BindingPattern>(
          binding.SourceLoc(), binding.Name(),
          global_arena->RawNew<ExpressionPattern>(
              ReifyType(type, binding.SourceLoc())));
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
      if (expected && expected->Tag() != Value::Kind::TupleValue) {
        FATAL_COMPILATION_ERROR(p->SourceLoc()) << "didn't expect a tuple";
      }
      if (expected && tuple.Fields().size() !=
                          cast<TupleValue>(*expected).Elements().size()) {
        FATAL_COMPILATION_ERROR(tuple.SourceLoc())
            << "tuples of different length";
      }
      for (size_t i = 0; i < tuple.Fields().size(); ++i) {
        const TuplePattern::Field& field = tuple.Fields()[i];
        const Value* expected_field_type = nullptr;
        if (expected != nullptr) {
          const TupleElement& expected_element =
              cast<TupleValue>(*expected).Elements()[i];
          if (expected_element.name != field.name) {
            FATAL_COMPILATION_ERROR(tuple.SourceLoc())
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
          global_arena->RawNew<TuplePattern>(tuple.SourceLoc(), new_fields);
      auto tuple_t = global_arena->RawNew<TupleValue>(std::move(field_types));
      return {.pattern = new_tuple, .type = tuple_t, .types = new_types};
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*p);
      const Value* choice_type = InterpExp(values, alternative.ChoiceType());
      if (choice_type->Tag() != Value::Kind::ChoiceType) {
        FATAL_COMPILATION_ERROR(alternative.SourceLoc())
            << "alternative pattern does not name a choice type.";
      }
      if (expected != nullptr) {
        ExpectType(alternative.SourceLoc(), "alternative pattern", expected,
                   choice_type);
      }
      const Value* parameter_types =
          FindInVarValues(alternative.AlternativeName(),
                          cast<ChoiceType>(*choice_type).Alternatives());
      if (parameter_types == nullptr) {
        FATAL_COMPILATION_ERROR(alternative.SourceLoc())
            << "'" << alternative.AlternativeName()
            << "' is not an alternative of " << choice_type;
      }
      TCPattern arg_results = TypeCheckPattern(alternative.Arguments(), types,
                                               values, parameter_types);
      return {.pattern = global_arena->RawNew<AlternativePattern>(
                  alternative.SourceLoc(),
                  ReifyType(choice_type, alternative.SourceLoc()),
                  alternative.AlternativeName(),
                  cast<TuplePattern>(arg_results.pattern)),
              .type = choice_type,
              .types = arg_results.types};
    }
    case Pattern::Kind::ExpressionPattern: {
      TCExpression result =
          TypeCheckExp(cast<ExpressionPattern>(p)->Expression(), types, values);
      return {.pattern = global_arena->RawNew<ExpressionPattern>(result.exp),
              .type = result.type,
              .types = result.types};
    }
  }
}

static auto TypecheckCase(const Value* expected, const Pattern* pat,
                          const Statement* body, TypeEnv types, Env values,
                          const Value*& ret_type, bool is_omitted_ret_type)
    -> std::pair<const Pattern*, const Statement*> {
  auto pat_res = TypeCheckPattern(pat, types, values, expected);
  auto res =
      TypeCheckStmt(body, pat_res.types, values, ret_type, is_omitted_ret_type);
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
                   const Value*& ret_type, bool is_omitted_ret_type)
    -> TCStatement {
  if (!s) {
    return TCStatement(s, types);
  }
  switch (s->Tag()) {
    case Statement::Kind::Match: {
      const auto& match = cast<Match>(*s);
      auto res = TypeCheckExp(match.Exp(), types, values);
      auto res_type = res.type;
      auto new_clauses = global_arena->RawNew<
          std::list<std::pair<const Pattern*, const Statement*>>>();
      for (auto& clause : *match.Clauses()) {
        new_clauses->push_back(TypecheckCase(res_type, clause.first,
                                             clause.second, types, values,
                                             ret_type, is_omitted_ret_type));
      }
      const Statement* new_s =
          global_arena->RawNew<Match>(s->SourceLoc(), res.exp, new_clauses);
      return TCStatement(new_s, types);
    }
    case Statement::Kind::While: {
      const auto& while_stmt = cast<While>(*s);
      auto cnd_res = TypeCheckExp(while_stmt.Cond(), types, values);
      ExpectType(s->SourceLoc(), "condition of `while`",
                 global_arena->RawNew<BoolType>(), cnd_res.type);
      auto body_res = TypeCheckStmt(while_stmt.Body(), types, values, ret_type,
                                    is_omitted_ret_type);
      auto new_s = global_arena->RawNew<While>(s->SourceLoc(), cnd_res.exp,
                                               body_res.stmt);
      return TCStatement(new_s, types);
    }
    case Statement::Kind::Break:
    case Statement::Kind::Continue:
      return TCStatement(s, types);
    case Statement::Kind::Block: {
      auto stmt_res = TypeCheckStmt(cast<Block>(*s).Stmt(), types, values,
                                    ret_type, is_omitted_ret_type);
      return TCStatement(
          global_arena->RawNew<Block>(s->SourceLoc(), stmt_res.stmt), types);
    }
    case Statement::Kind::VariableDefinition: {
      const auto& var = cast<VariableDefinition>(*s);
      auto res = TypeCheckExp(var.Init(), types, values);
      const Value* rhs_ty = res.type;
      auto lhs_res = TypeCheckPattern(var.Pat(), types, values, rhs_ty);
      const Statement* new_s = global_arena->RawNew<VariableDefinition>(
          s->SourceLoc(), var.Pat(), res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case Statement::Kind::Sequence: {
      const auto& seq = cast<Sequence>(*s);
      auto stmt_res = TypeCheckStmt(seq.Stmt(), types, values, ret_type,
                                    is_omitted_ret_type);
      auto types2 = stmt_res.types;
      auto next_res = TypeCheckStmt(seq.Next(), types2, values, ret_type,
                                    is_omitted_ret_type);
      auto types3 = next_res.types;
      return TCStatement(global_arena->RawNew<Sequence>(
                             s->SourceLoc(), stmt_res.stmt, next_res.stmt),
                         types3);
    }
    case Statement::Kind::Assign: {
      const auto& assign = cast<Assign>(*s);
      auto rhs_res = TypeCheckExp(assign.Rhs(), types, values);
      auto rhs_t = rhs_res.type;
      auto lhs_res = TypeCheckExp(assign.Lhs(), types, values);
      auto lhs_t = lhs_res.type;
      ExpectType(s->SourceLoc(), "assign", lhs_t, rhs_t);
      auto new_s = global_arena->RawNew<Assign>(s->SourceLoc(), lhs_res.exp,
                                                rhs_res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case Statement::Kind::ExpressionStatement: {
      auto res =
          TypeCheckExp(cast<ExpressionStatement>(*s).Exp(), types, values);
      auto new_s =
          global_arena->RawNew<ExpressionStatement>(s->SourceLoc(), res.exp);
      return TCStatement(new_s, types);
    }
    case Statement::Kind::If: {
      const auto& if_stmt = cast<If>(*s);
      auto cnd_res = TypeCheckExp(if_stmt.Cond(), types, values);
      ExpectType(s->SourceLoc(), "condition of `if`",
                 global_arena->RawNew<BoolType>(), cnd_res.type);
      auto then_res = TypeCheckStmt(if_stmt.ThenStmt(), types, values, ret_type,
                                    is_omitted_ret_type);
      auto else_res = TypeCheckStmt(if_stmt.ElseStmt(), types, values, ret_type,
                                    is_omitted_ret_type);
      auto new_s = global_arena->RawNew<If>(s->SourceLoc(), cnd_res.exp,
                                            then_res.stmt, else_res.stmt);
      return TCStatement(new_s, types);
    }
    case Statement::Kind::Return: {
      const auto& ret = cast<Return>(*s);
      auto res = TypeCheckExp(ret.Exp(), types, values);
      if (ret_type->Tag() == Value::Kind::AutoType) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        ret_type = res.type;
      } else {
        ExpectType(s->SourceLoc(), "return", ret_type, res.type);
      }
      if (ret.IsOmittedExp() != is_omitted_ret_type) {
        FATAL_COMPILATION_ERROR(s->SourceLoc())
            << *s << " should" << (is_omitted_ret_type ? " not" : "")
            << " provide a return value, to match the function's signature.";
      }
      return TCStatement(global_arena->RawNew<Return>(s->SourceLoc(), res.exp,
                                                      ret.IsOmittedExp()),
                         types);
    }
    case Statement::Kind::Continuation: {
      const auto& cont = cast<Continuation>(*s);
      TCStatement body_result = TypeCheckStmt(cont.Body(), types, values,
                                              ret_type, is_omitted_ret_type);
      const Statement* new_continuation = global_arena->RawNew<Continuation>(
          s->SourceLoc(), cont.ContinuationVariable(), body_result.stmt);
      types.Set(cont.ContinuationVariable(),
                global_arena->RawNew<ContinuationType>());
      return TCStatement(new_continuation, types);
    }
    case Statement::Kind::Run: {
      TCExpression argument_result =
          TypeCheckExp(cast<Run>(*s).Argument(), types, values);
      ExpectType(s->SourceLoc(), "argument of `run`",
                 global_arena->RawNew<ContinuationType>(),
                 argument_result.type);
      const Statement* new_run =
          global_arena->RawNew<Run>(s->SourceLoc(), argument_result.exp);
      return TCStatement(new_run, types);
    }
    case Statement::Kind::Await: {
      // nothing to do here
      return TCStatement(s, types);
    }
  }  // switch
}

static auto CheckOrEnsureReturn(const Statement* stmt, bool omitted_ret_type,
                                SourceLocation loc) -> const Statement* {
  if (!stmt) {
    if (omitted_ret_type) {
      return global_arena->RawNew<Return>(loc, nullptr,
                                          /*is_omitted_exp=*/true);
    } else {
      FATAL_COMPILATION_ERROR(loc)
          << "control-flow reaches end of function that provides a `->` return "
             "type without reaching a return statement";
    }
  }
  switch (stmt->Tag()) {
    case Statement::Kind::Match: {
      const auto& match = cast<Match>(*stmt);
      auto new_clauses = global_arena->RawNew<
          std::list<std::pair<const Pattern*, const Statement*>>>();
      for (const auto& clause : *match.Clauses()) {
        auto s = CheckOrEnsureReturn(clause.second, omitted_ret_type,
                                     stmt->SourceLoc());
        new_clauses->push_back(std::make_pair(clause.first, s));
      }
      return global_arena->RawNew<Match>(stmt->SourceLoc(), match.Exp(),
                                         new_clauses);
    }
    case Statement::Kind::Block:
      return global_arena->RawNew<Block>(
          stmt->SourceLoc(),
          CheckOrEnsureReturn(cast<Block>(*stmt).Stmt(), omitted_ret_type,
                              stmt->SourceLoc()));
    case Statement::Kind::If: {
      const auto& if_stmt = cast<If>(*stmt);
      return global_arena->RawNew<If>(
          stmt->SourceLoc(), if_stmt.Cond(),
          CheckOrEnsureReturn(if_stmt.ThenStmt(), omitted_ret_type,
                              stmt->SourceLoc()),
          CheckOrEnsureReturn(if_stmt.ElseStmt(), omitted_ret_type,
                              stmt->SourceLoc()));
    }
    case Statement::Kind::Return:
      return stmt;
    case Statement::Kind::Sequence: {
      const auto& seq = cast<Sequence>(*stmt);
      if (seq.Next()) {
        return global_arena->RawNew<Sequence>(
            stmt->SourceLoc(), seq.Stmt(),
            CheckOrEnsureReturn(seq.Next(), omitted_ret_type,
                                stmt->SourceLoc()));
      } else {
        return CheckOrEnsureReturn(seq.Stmt(), omitted_ret_type,
                                   stmt->SourceLoc());
      }
    }
    case Statement::Kind::Continuation:
    case Statement::Kind::Run:
    case Statement::Kind::Await:
      return stmt;
    case Statement::Kind::Assign:
    case Statement::Kind::ExpressionStatement:
    case Statement::Kind::While:
    case Statement::Kind::Break:
    case Statement::Kind::Continue:
    case Statement::Kind::VariableDefinition:
      if (omitted_ret_type) {
        return global_arena->RawNew<Sequence>(
            stmt->SourceLoc(), stmt,
            global_arena->RawNew<Return>(loc, nullptr,
                                         /*is_omitted_exp=*/true));
      } else {
        FATAL_COMPILATION_ERROR(stmt->SourceLoc())
            << "control-flow reaches end of function that provides a `->` "
               "return type without reaching a return statement";
      }
  }
}

// TODO: factor common parts of TypeCheckFunDef and TypeOfFunDef into
// a function.
// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
static auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv types,
                            Env values) -> Ptr<const FunctionDefinition> {
  // Bring the deduced parameters into scope
  for (const auto& deduced : f->deduced_parameters) {
    // auto t = InterpExp(values, deduced.type);
    types.Set(deduced.name, global_arena->RawNew<VariableType>(deduced.name));
    Address a = state->heap.AllocateValue(*types.Get(deduced.name));
    values.Set(deduced.name, a);
  }
  // Type check the parameter pattern
  auto param_res = TypeCheckPattern(f->param_pattern, types, values, nullptr);
  // Evaluate the return type expression
  auto return_type = InterpPattern(values, f->return_type);
  if (f->name == "main") {
    ExpectType(f->source_location, "return type of `main`",
               global_arena->RawNew<IntType>(), return_type);
    // TODO: Check that main doesn't have any parameters.
  }
  auto res = TypeCheckStmt(f->body, param_res.types, values, return_type,
                           f->is_omitted_return_type);
  auto body = CheckOrEnsureReturn(res.stmt, f->is_omitted_return_type,
                                  f->source_location);
  return global_arena->New<FunctionDefinition>(
      f->source_location, f->name, f->deduced_parameters, f->param_pattern,
      global_arena->RawNew<ExpressionPattern>(
          ReifyType(return_type, f->source_location)),
      /*is_omitted_return_type=*/false, body);
}

static auto TypeOfFunDef(TypeEnv types, Env values,
                         const FunctionDefinition* fun_def) -> const Value* {
  // Bring the deduced parameters into scope
  for (const auto& deduced : fun_def->deduced_parameters) {
    // auto t = InterpExp(values, deduced.type);
    types.Set(deduced.name, global_arena->RawNew<VariableType>(deduced.name));
    Address a = state->heap.AllocateValue(*types.Get(deduced.name));
    values.Set(deduced.name, a);
  }
  // Type check the parameter pattern
  auto param_res =
      TypeCheckPattern(fun_def->param_pattern, types, values, nullptr);
  // Evaluate the return type expression
  auto ret = InterpPattern(values, fun_def->return_type);
  if (ret->Tag() == Value::Kind::AutoType) {
    auto f = TypeCheckFunDef(fun_def, types, values);
    ret = InterpPattern(values, f->return_type);
  }
  return global_arena->RawNew<FunctionType>(fun_def->deduced_parameters,
                                            param_res.type, ret);
}

static auto TypeOfClassDef(const ClassDefinition* sd, TypeEnv /*types*/,
                           Env ct_top) -> const Value* {
  VarValues fields;
  VarValues methods;
  for (Ptr<const Member> m : sd->members) {
    switch (m->Tag()) {
      case Member::Kind::FieldMember: {
        const BindingPattern* binding = cast<FieldMember>(*m).Binding();
        if (!binding->Name().has_value()) {
          FATAL_COMPILATION_ERROR(binding->SourceLoc())
              << "Struct members must have names";
        }
        const Expression* type_expression =
            dyn_cast<ExpressionPattern>(binding->Type())->Expression();
        if (type_expression == nullptr) {
          FATAL_COMPILATION_ERROR(binding->SourceLoc())
              << "Struct members must have explicit types";
        }
        auto type = InterpExp(ct_top, type_expression);
        fields.push_back(std::make_pair(*binding->Name(), type));
        break;
      }
    }
  }
  return global_arena->RawNew<ClassType>(sd->name, std::move(fields),
                                         std::move(methods));
}

static auto GetName(const Declaration& d) -> const std::string& {
  switch (d.Tag()) {
    case Declaration::Kind::FunctionDeclaration:
      return cast<FunctionDeclaration>(d).Definition().name;
    case Declaration::Kind::ClassDeclaration:
      return cast<ClassDeclaration>(d).Definition().name;
    case Declaration::Kind::ChoiceDeclaration:
      return cast<ChoiceDeclaration>(d).Name();
    case Declaration::Kind::VariableDeclaration: {
      const BindingPattern* binding = cast<VariableDeclaration>(d).Binding();
      if (!binding->Name().has_value()) {
        FATAL_COMPILATION_ERROR(binding->SourceLoc())
            << "Top-level variable declarations must have names";
      }
      return *binding->Name();
    }
  }
}

auto MakeTypeChecked(const Ptr<const Declaration> d, const TypeEnv& types,
                     const Env& values) -> Ptr<const Declaration> {
  switch (d->Tag()) {
    case Declaration::Kind::FunctionDeclaration:
      return global_arena->New<FunctionDeclaration>(TypeCheckFunDef(
          &cast<FunctionDeclaration>(*d).Definition(), types, values));

    case Declaration::Kind::ClassDeclaration: {
      const ClassDefinition& class_def =
          cast<ClassDeclaration>(*d).Definition();
      std::list<Ptr<Member>> fields;
      for (Ptr<Member> m : class_def.members) {
        switch (m->Tag()) {
          case Member::Kind::FieldMember:
            // TODO: Interpret the type expression and store the result.
            fields.push_back(m);
            break;
        }
      }
      return global_arena->New<ClassDeclaration>(class_def.loc, class_def.name,
                                                 std::move(fields));
    }

    case Declaration::Kind::ChoiceDeclaration:
      // TODO
      return d;

    case Declaration::Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*d);
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      TCExpression type_checked_initializer =
          TypeCheckExp(var.Initializer(), types, values);
      const Expression* type =
          dyn_cast<ExpressionPattern>(var.Binding()->Type())->Expression();
      if (type == nullptr) {
        // TODO: consider adding support for `auto`
        FATAL_COMPILATION_ERROR(var.SourceLoc())
            << "Type of a top-level variable must be an expression.";
      }
      const Value* declared_type = InterpExp(values, type);
      ExpectType(var.SourceLoc(), "initializer of variable", declared_type,
                 type_checked_initializer.type);
      return d;
    }
  }
}

static void TopLevel(const Declaration& d, TypeCheckContext* tops) {
  switch (d.Tag()) {
    case Declaration::Kind::FunctionDeclaration: {
      const FunctionDefinition& func_def =
          cast<FunctionDeclaration>(d).Definition();
      auto t = TypeOfFunDef(tops->types, tops->values, &func_def);
      tops->types.Set(func_def.name, t);
      InitEnv(d, &tops->values);
      break;
    }

    case Declaration::Kind::ClassDeclaration: {
      const ClassDefinition& class_def = cast<ClassDeclaration>(d).Definition();
      auto st = TypeOfClassDef(&class_def, tops->types, tops->values);
      Address a = state->heap.AllocateValue(st);
      tops->values.Set(class_def.name, a);  // Is this obsolete?
      std::vector<TupleElement> field_types;
      for (const auto& [field_name, field_value] :
           cast<ClassType>(*st).Fields()) {
        field_types.push_back({.name = field_name, .value = field_value});
      }
      auto fun_ty = global_arena->RawNew<FunctionType>(
          std::vector<GenericBinding>(),
          global_arena->RawNew<TupleValue>(std::move(field_types)), st);
      tops->types.Set(class_def.name, fun_ty);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      VarValues alts;
      for (const auto& [name, signature] : choice.Alternatives()) {
        auto t = InterpExp(tops->values, signature);
        alts.push_back(std::make_pair(name, t));
      }
      auto ct =
          global_arena->RawNew<ChoiceType>(choice.Name(), std::move(alts));
      Address a = state->heap.AllocateValue(ct);
      tops->values.Set(choice.Name(), a);  // Is this obsolete?
      tops->types.Set(choice.Name(), ct);
      break;
    }

    case Declaration::Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      const Expression* type =
          cast<ExpressionPattern>(var.Binding()->Type())->Expression();
      const Value* declared_type = InterpExp(tops->values, type);
      tops->types.Set(*var.Binding()->Name(), declared_type);
      break;
    }
  }
}

auto TopLevel(const std::list<Ptr<const Declaration>>& fs) -> TypeCheckContext {
  TypeCheckContext tops;
  bool found_main = false;

  for (auto const& d : fs) {
    if (GetName(*d) == "main") {
      found_main = true;
    }
    TopLevel(*d, &tops);
  }

  if (found_main == false) {
    FATAL_COMPILATION_ERROR_NO_LINE()
        << "program must contain a function named `main`";
  }
  return tops;
}

}  // namespace Carbon
