// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/typecheck.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/tracing_flag.h"

namespace Carbon {

void ExpectType(int line_num, const std::string& context, const Value* expected,
                const Value* actual) {
  if (!TypeEqual(expected, actual)) {
    std::cerr << line_num << ": type error in " << context << std::endl;
    std::cerr << "expected: ";
    PrintValue(expected, std::cerr);
    std::cerr << std::endl << "actual: ";
    PrintValue(actual, std::cerr);
    std::cerr << std::endl;
    exit(-1);
  }
}

void PrintErrorString(const std::string& s) { std::cerr << s; }

void PrintTypeEnv(TypeEnv types, std::ostream& out) {
  for (const auto& [name, value] : types) {
    out << name << ": ";
    PrintValue(value, out);
    out << ", ";
  }
}

// Reify type to type expression.
auto ReifyType(const Value* t, int line_num) -> const Expression* {
  switch (t->tag) {
    case ValKind::VarTV:
      return Expression::MakeVar(0, *t->GetVariableType());
    case ValKind::IntTV:
      return Expression::MakeIntType(0);
    case ValKind::BoolTV:
      return Expression::MakeBoolType(0);
    case ValKind::TypeTV:
      return Expression::MakeTypeType(0);
    case ValKind::ContinuationTV:
      return Expression::MakeContinuationType(0);
    case ValKind::FunctionTV:
      return Expression::MakeFunType(
          0, ReifyType(t->GetFunctionType().param, line_num),
          ReifyType(t->GetFunctionType().ret, line_num));
    case ValKind::TupleV: {
      auto args = new std::vector<FieldInitializer>();
      for (const TupleElement& field : *t->GetTuple().elements) {
        args->push_back(
            {.name = field.name,
             .expression = ReifyType(state->heap.Read(field.address, line_num),
                                     line_num)});
      }
      return Expression::MakeTuple(0, args);
    }
    case ValKind::StructTV:
      return Expression::MakeVar(0, *t->GetStructType().name);
    case ValKind::ChoiceTV:
      return Expression::MakeVar(0, *t->GetChoiceType().name);
    default:
      std::cerr << line_num << ": expected a type, not ";
      PrintValue(t, std::cerr);
      std::cerr << std::endl;
      exit(-1);
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
// expected is the type that this expression is expected to have.
//    This parameter is non-null when the expression is in a pattern context
//    and it is used to implement `auto`, otherwise it is null.
// context says what kind of position this expression is nested in,
//    whether it's a position that expects a value, a pattern, or a type.
auto TypeCheckExp(const Expression* e, TypeEnv types, Env values,
                  const Value* expected, TCContext context) -> TCResult {
  if (tracing_output) {
    switch (context) {
      case TCContext::ValueContext:
        std::cout << "checking expression ";
        break;
      case TCContext::PatternContext:
        std::cout << "checking pattern, ";
        if (expected) {
          std::cout << "expecting ";
          PrintValue(expected, std::cerr);
        }
        std::cout << ", ";
        break;
      case TCContext::TypeContext:
        std::cout << "checking type ";
        break;
    }
    PrintExp(e);
    std::cout << std::endl;
  }
  switch (e->tag) {
    case ExpressionKind::PatternVariable: {
      if (context != TCContext::PatternContext) {
        std::cerr
            << e->line_num
            << ": compilation error, pattern variables are only allowed in "
               "pattern context"
            << std::endl;
        exit(-1);
      }
      auto t = InterpExp(values, e->GetPatternVariable().type);
      if (t->tag == ValKind::AutoTV) {
        if (expected == nullptr) {
          std::cerr << e->line_num
                    << ": compilation error, auto not allowed here"
                    << std::endl;
          exit(-1);
        } else {
          t = expected;
        }
      } else if (expected) {
        ExpectType(e->line_num, "pattern variable", t, expected);
      }
      auto new_e =
          Expression::MakeVarPat(e->line_num, *e->GetPatternVariable().name,
                                 ReifyType(t, e->line_num));
      types.Set(*e->GetPatternVariable().name, t);
      return TCResult(new_e, t, types);
    }
    case ExpressionKind::Index: {
      auto res = TypeCheckExp(e->GetFieldAccess().aggregate, types, values,
                              nullptr, TCContext::ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case ValKind::TupleV: {
          auto i = ToInteger(InterpExp(values, e->GetIndex().offset));
          std::string f = std::to_string(i);
          std::optional<Address> field_address = FindTupleField(f, t);
          if (field_address == std::nullopt) {
            std::cerr << e->line_num << ": compilation error, field " << f
                      << " is not in the tuple ";
            PrintValue(t, std::cerr);
            std::cerr << std::endl;
            exit(-1);
          }
          auto field_t = state->heap.Read(*field_address, e->line_num);
          auto new_e = Expression::MakeIndex(
              e->line_num, res.exp, Expression::MakeInt(e->line_num, i));
          return TCResult(new_e, field_t, res.types);
        }
        default:
          std::cerr << e->line_num << ": compilation error, expected a tuple"
                    << std::endl;
          exit(-1);
      }
    }
    case ExpressionKind::Tuple: {
      auto new_args = new std::vector<FieldInitializer>();
      auto arg_types = new std::vector<TupleElement>();
      auto new_types = types;
      if (expected && expected->tag != ValKind::TupleV) {
        std::cerr << e->line_num << ": compilation error, didn't expect a tuple"
                  << std::endl;
        exit(-1);
      }
      if (expected && e->GetTuple().fields->size() !=
                          expected->GetTuple().elements->size()) {
        std::cerr << e->line_num
                  << ": compilation error, tuples of different length"
                  << std::endl;
        exit(-1);
      }
      int i = 0;
      for (auto arg = e->GetTuple().fields->begin();
           arg != e->GetTuple().fields->end(); ++arg, ++i) {
        const Value* arg_expected = nullptr;
        if (expected && expected->tag == ValKind::TupleV) {
          if ((*expected->GetTuple().elements)[i].name != arg->name) {
            std::cerr << e->line_num
                      << ": compilation error, field names do not match, "
                      << "expected " << (*expected->GetTuple().elements)[i].name
                      << " but got " << arg->name << std::endl;
            exit(-1);
          }
          arg_expected = state->heap.Read(
              (*expected->GetTuple().elements)[i].address, e->line_num);
        }
        auto arg_res = TypeCheckExp(arg->expression, new_types, values,
                                    arg_expected, context);
        new_types = arg_res.types;
        new_args->push_back({.name = arg->name, .expression = arg_res.exp});
        arg_types->push_back(
            {.name = arg->name,
             .address = state->heap.AllocateValue(arg_res.type)});
      }
      auto tuple_e = Expression::MakeTuple(e->line_num, new_args);
      auto tuple_t = Value::MakeTupleVal(arg_types);
      return TCResult(tuple_e, tuple_t, new_types);
    }
    case ExpressionKind::GetField: {
      auto res = TypeCheckExp(e->GetFieldAccess().aggregate, types, values,
                              nullptr, TCContext::ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case ValKind::StructTV:
          // Search for a field
          for (auto& field : *t->GetStructType().fields) {
            if (*e->GetFieldAccess().field == field.first) {
              const Expression* new_e = Expression::MakeGetField(
                  e->line_num, res.exp, *e->GetFieldAccess().field);
              return TCResult(new_e, field.second, res.types);
            }
          }
          // Search for a method
          for (auto& method : *t->GetStructType().methods) {
            if (*e->GetFieldAccess().field == method.first) {
              const Expression* new_e = Expression::MakeGetField(
                  e->line_num, res.exp, *e->GetFieldAccess().field);
              return TCResult(new_e, method.second, res.types);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->GetStructType().name
                    << " does not have a field named "
                    << *e->GetFieldAccess().field << std::endl;
          exit(-1);
        case ValKind::TupleV:
          for (const TupleElement& field : *t->GetTuple().elements) {
            if (*e->GetFieldAccess().field == field.name) {
              auto new_e = Expression::MakeGetField(e->line_num, res.exp,
                                                    *e->GetFieldAccess().field);
              return TCResult(new_e,
                              state->heap.Read(field.address, e->line_num),
                              res.types);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->GetStructType().name
                    << " does not have a field named "
                    << *e->GetFieldAccess().field << std::endl;
          exit(-1);
        case ValKind::ChoiceTV:
          for (auto vt = t->GetChoiceType().alternatives->begin();
               vt != t->GetChoiceType().alternatives->end(); ++vt) {
            if (*e->GetFieldAccess().field == vt->first) {
              const Expression* new_e = Expression::MakeGetField(
                  e->line_num, res.exp, *e->GetFieldAccess().field);
              auto fun_ty = Value::MakeFunTypeVal(vt->second, t);
              return TCResult(new_e, fun_ty, res.types);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->GetStructType().name
                    << " does not have a field named "
                    << *e->GetFieldAccess().field << std::endl;
          exit(-1);

        default:
          std::cerr << e->line_num
                    << ": compilation error in field access, expected a struct"
                    << std::endl;
          PrintExp(e);
          std::cerr << std::endl;
          exit(-1);
      }
    }
    case ExpressionKind::Variable: {
      std::optional<const Value*> type = types.Get(*(e->GetVariable().name));
      if (type) {
        return TCResult(e, *type, types);
      } else {
        std::cerr << e->line_num << ": could not find `"
                  << *(e->GetVariable().name) << "`" << std::endl;
        exit(-1);
      }
    }
    case ExpressionKind::Integer:
      return TCResult(e, Value::MakeIntTypeVal(), types);
    case ExpressionKind::Boolean:
      return TCResult(e, Value::MakeBoolTypeVal(), types);
    case ExpressionKind::PrimitiveOp: {
      auto es = new std::vector<const Expression*>();
      std::vector<const Value*> ts;
      auto new_types = types;
      for (auto& argument : *e->GetPrimitiveOperator().arguments) {
        auto res = TypeCheckExp(argument, types, values, nullptr,
                                TCContext::ValueContext);
        new_types = res.types;
        es->push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e =
          Expression::MakeOp(e->line_num, e->GetPrimitiveOperator().op, es);
      switch (e->GetPrimitiveOperator().op) {
        case Operator::Neg:
          ExpectType(e->line_num, "negation", Value::MakeIntTypeVal(), ts[0]);
          return TCResult(new_e, Value::MakeIntTypeVal(), new_types);
        case Operator::Add:
        case Operator::Sub:
          ExpectType(e->line_num, "subtraction(1)", Value::MakeIntTypeVal(),
                     ts[0]);
          ExpectType(e->line_num, "substration(2)", Value::MakeIntTypeVal(),
                     ts[1]);
          return TCResult(new_e, Value::MakeIntTypeVal(), new_types);
        case Operator::And:
          ExpectType(e->line_num, "&&(1)", Value::MakeBoolTypeVal(), ts[0]);
          ExpectType(e->line_num, "&&(2)", Value::MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, Value::MakeBoolTypeVal(), new_types);
        case Operator::Or:
          ExpectType(e->line_num, "||(1)", Value::MakeBoolTypeVal(), ts[0]);
          ExpectType(e->line_num, "||(2)", Value::MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, Value::MakeBoolTypeVal(), new_types);
        case Operator::Not:
          ExpectType(e->line_num, "!", Value::MakeBoolTypeVal(), ts[0]);
          return TCResult(new_e, Value::MakeBoolTypeVal(), new_types);
        case Operator::Eq:
          ExpectType(e->line_num, "==", ts[0], ts[1]);
          return TCResult(new_e, Value::MakeBoolTypeVal(), new_types);
      }
      break;
    }
    case ExpressionKind::Call: {
      auto fun_res = TypeCheckExp(e->GetCall().function, types, values, nullptr,
                                  TCContext::ValueContext);
      switch (fun_res.type->tag) {
        case ValKind::FunctionTV: {
          auto fun_t = fun_res.type;
          auto arg_res =
              TypeCheckExp(e->GetCall().argument, fun_res.types, values,
                           fun_t->GetFunctionType().param, context);
          ExpectType(e->line_num, "call", fun_t->GetFunctionType().param,
                     arg_res.type);
          auto new_e =
              Expression::MakeCall(e->line_num, fun_res.exp, arg_res.exp);
          return TCResult(new_e, fun_t->GetFunctionType().ret, arg_res.types);
        }
        default: {
          std::cerr << e->line_num
                    << ": compilation error in call, expected a function"
                    << std::endl;
          PrintExp(e);
          std::cerr << std::endl;
          exit(-1);
        }
      }
      break;
    }
    case ExpressionKind::FunctionT: {
      switch (context) {
        case TCContext::ValueContext:
        case TCContext::TypeContext: {
          auto pt = InterpExp(values, e->GetFunctionType().parameter);
          auto rt = InterpExp(values, e->GetFunctionType().return_type);
          auto new_e =
              Expression::MakeFunType(e->line_num, ReifyType(pt, e->line_num),
                                      ReifyType(rt, e->line_num));
          return TCResult(new_e, Value::MakeTypeTypeVal(), types);
        }
        case TCContext::PatternContext: {
          auto param_res = TypeCheckExp(e->GetFunctionType().parameter, types,
                                        values, nullptr, context);
          auto ret_res =
              TypeCheckExp(e->GetFunctionType().return_type, param_res.types,
                           values, nullptr, context);
          auto new_e = Expression::MakeFunType(
              e->line_num, ReifyType(param_res.type, e->line_num),
              ReifyType(ret_res.type, e->line_num));
          return TCResult(new_e, Value::MakeTypeTypeVal(), ret_res.types);
        }
      }
    }
    case ExpressionKind::IntT:
      return TCResult(e, Value::MakeTypeTypeVal(), types);
    case ExpressionKind::BoolT:
      return TCResult(e, Value::MakeTypeTypeVal(), types);
    case ExpressionKind::TypeT:
      return TCResult(e, Value::MakeTypeTypeVal(), types);
    case ExpressionKind::AutoT:
      return TCResult(e, Value::MakeTypeTypeVal(), types);
    case ExpressionKind::ContinuationT:
      return TCResult(e, Value::MakeTypeTypeVal(), types);
  }
}

auto TypecheckCase(const Value* expected, const Expression* pat,
                   const Statement* body, TypeEnv types, Env values,
                   const Value*& ret_type)
    -> std::pair<const Expression*, const Statement*> {
  auto pat_res =
      TypeCheckExp(pat, types, values, expected, TCContext::PatternContext);
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
  switch (s->tag) {
    case StatementKind::Match: {
      auto res = TypeCheckExp(s->GetMatch().exp, types, values, nullptr,
                              TCContext::ValueContext);
      auto res_type = res.type;
      auto new_clauses =
          new std::list<std::pair<const Expression*, const Statement*>>();
      for (auto& clause : *s->GetMatch().clauses) {
        new_clauses->push_back(TypecheckCase(
            res_type, clause.first, clause.second, types, values, ret_type));
      }
      const Statement* new_s =
          Statement::MakeMatch(s->line_num, res.exp, new_clauses);
      return TCStatement(new_s, types);
    }
    case StatementKind::While: {
      auto cnd_res = TypeCheckExp(s->GetWhile().cond, types, values, nullptr,
                                  TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `while`", Value::MakeBoolTypeVal(),
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
      auto res = TypeCheckExp(s->GetVariableDefinition().init, types, values,
                              nullptr, TCContext::ValueContext);
      const Value* rhs_ty = res.type;
      auto lhs_res = TypeCheckExp(s->GetVariableDefinition().pat, types, values,
                                  rhs_ty, TCContext::PatternContext);
      const Statement* new_s = Statement::MakeVarDef(
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
          Statement::MakeSeq(s->line_num, stmt_res.stmt, next_res.stmt),
          types3);
    }
    case StatementKind::Assign: {
      auto rhs_res = TypeCheckExp(s->GetAssign().rhs, types, values, nullptr,
                                  TCContext::ValueContext);
      auto rhs_t = rhs_res.type;
      auto lhs_res = TypeCheckExp(s->GetAssign().lhs, types, values, rhs_t,
                                  TCContext::ValueContext);
      auto lhs_t = lhs_res.type;
      ExpectType(s->line_num, "assign", lhs_t, rhs_t);
      auto new_s = Statement::MakeAssign(s->line_num, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case StatementKind::ExpressionStatement: {
      auto res = TypeCheckExp(s->GetExpression(), types, values, nullptr,
                              TCContext::ValueContext);
      auto new_s = Statement::MakeExpStmt(s->line_num, res.exp);
      return TCStatement(new_s, types);
    }
    case StatementKind::If: {
      auto cnd_res = TypeCheckExp(s->GetIf().cond, types, values, nullptr,
                                  TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `if`", Value::MakeBoolTypeVal(),
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
      auto res = TypeCheckExp(s->GetReturn(), types, values, nullptr,
                              TCContext::ValueContext);
      if (ret_type->tag == ValKind::AutoTV) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        ret_type = res.type;
      } else {
        ExpectType(s->line_num, "return", ret_type, res.type);
      }
      return TCStatement(Statement::MakeReturn(s->line_num, res.exp), types);
    }
    case StatementKind::Continuation: {
      TCStatement body_result =
          TypeCheckStmt(s->GetContinuation().body, types, values, ret_type);
      const Statement* new_continuation = Statement::MakeContinuation(
          s->line_num, *s->GetContinuation().continuation_variable,
          body_result.stmt);
      types.Set(*s->GetContinuation().continuation_variable,
                Value::MakeContinuationTypeVal());
      return TCStatement(new_continuation, types);
    }
    case StatementKind::Run: {
      TCResult argument_result =
          TypeCheckExp(s->GetRun().argument, types, values, nullptr,
                       TCContext::ValueContext);
      ExpectType(s->line_num, "argument of `run`",
                 Value::MakeContinuationTypeVal(), argument_result.type);
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
      return Statement::MakeReturn(line_num, Expression::MakeUnit(line_num));
    } else {
      std::cerr
          << "control-flow reaches end of non-void function without a return"
          << std::endl;
      exit(-1);
    }
  }
  switch (stmt->tag) {
    case StatementKind::Match: {
      auto new_clauses =
          new std::list<std::pair<const Expression*, const Statement*>>();
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
        return Statement::MakeSeq(
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
        return Statement::MakeSeq(
            stmt->line_num, stmt,
            Statement::MakeReturn(stmt->line_num,
                                  Expression::MakeUnit(stmt->line_num)));
      } else {
        std::cerr
            << stmt->line_num
            << ": control-flow reaches end of non-void function without a "
               "return"
            << std::endl;
        exit(-1);
      }
  }
}

auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv types, Env values)
    -> struct FunctionDefinition* {
  auto param_res = TypeCheckExp(f->param_pattern, types, values, nullptr,
                                TCContext::PatternContext);
  auto return_type = InterpExp(values, f->return_type);
  if (f->name == "main") {
    ExpectType(f->line_num, "return type of `main`", Value::MakeIntTypeVal(),
               return_type);
    // TODO: Check that main doesn't have any parameters.
  }
  auto res = TypeCheckStmt(f->body, param_res.types, values, return_type);
  bool void_return = TypeEqual(return_type, Value::MakeUnitTypeVal());
  auto body = CheckOrEnsureReturn(res.stmt, void_return, f->line_num);
  return MakeFunDef(f->line_num, f->name, ReifyType(return_type, f->line_num),
                    f->param_pattern, body);
}

auto TypeOfFunDef(TypeEnv types, Env values, const FunctionDefinition* fun_def)
    -> const Value* {
  auto param_res = TypeCheckExp(fun_def->param_pattern, types, values, nullptr,
                                TCContext::PatternContext);
  auto ret = InterpExp(values, fun_def->return_type);
  if (ret->tag == ValKind::AutoTV) {
    auto f = TypeCheckFunDef(fun_def, types, values);
    ret = InterpExp(values, f->return_type);
  }
  return Value::MakeFunTypeVal(param_res.type, ret);
}

auto TypeOfStructDef(const StructDefinition* sd, TypeEnv /*types*/, Env ct_top)
    -> const Value* {
  auto fields = new VarValues();
  auto methods = new VarValues();
  for (auto m = sd->members->begin(); m != sd->members->end(); ++m) {
    if ((*m)->tag == MemberKind::FieldMember) {
      auto t = InterpExp(ct_top, (*m)->u.field.type);
      fields->push_back(std::make_pair(*(*m)->u.field.name, t));
    }
  }
  return Value::MakeStructTypeVal(*sd->name, fields, methods);
}

auto FunctionDeclaration::Name() const -> std::string {
  return definition->name;
}

auto StructDeclaration::Name() const -> std::string { return *definition.name; }

auto ChoiceDeclaration::Name() const -> std::string { return name; }

// Returns the name of the declared variable.
auto VariableDeclaration::Name() const -> std::string { return name; }

auto StructDeclaration::TypeChecked(TypeEnv types, Env values) const
    -> Declaration {
  auto fields = new std::list<Member*>();
  for (auto& m : *definition.members) {
    if (m->tag == MemberKind::FieldMember) {
      // TODO: Interpret the type expression and store the result.
      fields->push_back(m);
    }
  }
  return StructDeclaration(definition.line_num, *definition.name, fields);
}

auto FunctionDeclaration::TypeChecked(TypeEnv types, Env values) const
    -> Declaration {
  return FunctionDeclaration(TypeCheckFunDef(definition, types, values));
}

auto ChoiceDeclaration::TypeChecked(TypeEnv types, Env values) const
    -> Declaration {
  return *this;  // TODO.
}

// Signals a type error if the initializing expression does not have
// the declared type of the variable, otherwise returns this
// declaration with annotated types.
auto VariableDeclaration::TypeChecked(TypeEnv types, Env values) const
    -> Declaration {
  TCResult type_checked_initializer = TypeCheckExp(
      initializer, types, values, nullptr, TCContext::ValueContext);
  const Value* declared_type = InterpExp(values, type);
  ExpectType(source_location, "initializer of variable", declared_type,
             type_checked_initializer.type);
  return *this;
}

auto TopLevel(std::list<Declaration>* fs) -> TypeCheckContext {
  TypeCheckContext tops;
  bool found_main = false;

  for (auto const& d : *fs) {
    if (d.Name() == "main") {
      found_main = true;
    }
    d.TopLevel(tops);
  }

  if (found_main == false) {
    std::cerr << "error, program must contain a function named `main`"
              << std::endl;
    exit(-1);
  }
  return tops;
}

auto FunctionDeclaration::TopLevel(TypeCheckContext& tops) const -> void {
  auto t = TypeOfFunDef(tops.types, tops.values, definition);
  tops.types.Set(Name(), t);
  InitGlobals(tops.values);
}

auto StructDeclaration::TopLevel(TypeCheckContext& tops) const -> void {
  auto st = TypeOfStructDef(&definition, tops.types, tops.values);
  Address a = state->heap.AllocateValue(st);
  tops.values.Set(Name(), a);  // Is this obsolete?
  auto field_types = new std::vector<TupleElement>();
  for (const auto& [field_name, field_value] : *st->GetStructType().fields) {
    field_types->push_back({.name = field_name,
                            .address = state->heap.AllocateValue(field_value)});
  }
  auto fun_ty = Value::MakeFunTypeVal(Value::MakeTupleVal(field_types), st);
  tops.types.Set(Name(), fun_ty);
}

auto ChoiceDeclaration::TopLevel(TypeCheckContext& tops) const -> void {
  auto alts = new VarValues();
  for (auto a : alternatives) {
    auto t = InterpExp(tops.values, a.second);
    alts->push_back(std::make_pair(a.first, t));
  }
  auto ct = Value::MakeChoiceTypeVal(name, alts);
  Address a = state->heap.AllocateValue(ct);
  tops.values.Set(Name(), a);  // Is this obsolete?
  tops.types.Set(Name(), ct);
}

// Associate the variable name with it's declared type in the
// compile-time symbol table.
auto VariableDeclaration::TopLevel(TypeCheckContext& tops) const -> void {
  const Value* declared_type = InterpExp(tops.values, type);
  tops.types.Set(Name(), declared_type);
}

}  // namespace Carbon
