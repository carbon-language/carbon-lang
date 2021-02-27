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
#include "executable_semantics/interpreter/cons_list.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

void ExpectType(int line_num, const std::string& context, Value* expected,
                Value* actual) {
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

void PrintTypeEnv(TypeEnv* env, std::ostream& out) {
  if (env) {
    out << env->key << ": ";
    PrintValue(env->value, out);
    out << ", ";
    PrintTypeEnv(env->next, out);
  }
}

// Convert tuples to tuple types.
auto ToType(int line_num, Value* val) -> Value* {
  switch (val->tag) {
    case ValKind::TupleV: {
      auto fields = new VarValues();
      for (auto& elt : *val->u.tuple.elts) {
        Value* ty = ToType(line_num, state->heap[elt.second]);
        fields->push_back(std::make_pair(elt.first, ty));
      }
      return MakeTupleTypeVal(fields);
    }
    case ValKind::TupleTV: {
      auto fields = new VarValues();
      for (auto& field : *val->u.tuple_type.fields) {
        Value* ty = ToType(line_num, field.second);
        fields->push_back(std::make_pair(field.first, ty));
      }
      return MakeTupleTypeVal(fields);
    }
    case ValKind::PointerTV: {
      return MakePtrTypeVal(ToType(line_num, val->u.ptr_type.type));
    }
    case ValKind::FunctionTV: {
      return MakeFunTypeVal(ToType(line_num, val->u.fun_type.param),
                            ToType(line_num, val->u.fun_type.ret));
    }
    case ValKind::VarPatV: {
      return MakeVarPatVal(*val->u.var_pat.name,
                           ToType(line_num, val->u.var_pat.type));
    }
    case ValKind::ChoiceTV:
    case ValKind::StructTV:
    case ValKind::TypeTV:
    case ValKind::VarTV:
    case ValKind::BoolTV:
    case ValKind::IntTV:
    case ValKind::AutoTV:
      return val;
    default:
      std::cerr << line_num << ": in ToType, expected a type, not ";
      PrintValue(val, std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

// Reify type to type expression.
auto ReifyType(Value* t, int line_num) -> Expression* {
  switch (t->tag) {
    case ValKind::VarTV:
      return MakeVar(0, *t->u.var_type);
    case ValKind::IntTV:
      return MakeIntType(0);
    case ValKind::BoolTV:
      return MakeBoolType(0);
    case ValKind::TypeTV:
      return MakeTypeType(0);
    case ValKind::FunctionTV:
      return MakeFunType(0, ReifyType(t->u.fun_type.param, line_num),
                         ReifyType(t->u.fun_type.ret, line_num));
    case ValKind::TupleTV: {
      auto args = new std::vector<std::pair<std::string, Expression*>>();
      for (auto& field : *t->u.tuple_type.fields) {
        args->push_back(
            make_pair(field.first, ReifyType(field.second, line_num)));
      }
      return MakeTuple(0, args);
    }
    case ValKind::StructTV:
      return MakeVar(0, *t->u.struct_type.name);
    case ValKind::ChoiceTV:
      return MakeVar(0, *t->u.choice_type.name);
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
// env maps variable names to the type of their run-time value.
// ct_env maps variable names to their compile-time values. It is not
//    directly used in this function but is passed to InterExp.
// expected is the type that this expression is expected to have.
//    This parameter is non-null when the expression is in a pattern context
//    and it is used to implement `auto`, otherwise it is null.
// context says what kind of position this expression is nested in,
//    whether it's a position that expects a value, a pattern, or a type.
auto TypeCheckExp(Expression* e, TypeEnv* env, Env* ct_env, Value* expected,
                  TCContext context) -> TCResult {
  switch (e->tag) {
    case ExpressionKind::PatternVariable: {
      if (context != TCContext::PatternContext) {
        std::cerr
            << e->line_num
            << ": compilation error, pattern variables are only allowed in "
               "pattern context"
            << std::endl;
      }
      auto t =
          ToType(e->line_num, InterpExp(ct_env, e->u.pattern_variable.type));
      if (t->tag == ValKind::AutoTV) {
        if (expected == nullptr) {
          std::cerr << e->line_num
                    << ": compilation error, auto not allowed here"
                    << std::endl;
          exit(-1);
        } else {
          t = expected;
        }
      }
      auto new_e = MakeVarPat(e->line_num, *e->u.pattern_variable.name,
                              ReifyType(t, e->line_num));
      return TCResult(new_e, t,
                      new TypeEnv(*e->u.pattern_variable.name, t, env));
    }
    case ExpressionKind::Index: {
      auto res = TypeCheckExp(e->u.get_field.aggregate, env, ct_env, nullptr,
                              TCContext::ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case ValKind::TupleTV: {
          auto i = ToInteger(InterpExp(ct_env, e->u.index.offset));
          std::string f = std::to_string(i);
          auto field_t = FindInVarValues(f, t->u.tuple_type.fields);
          if (field_t == nullptr) {
            std::cerr << e->line_num << ": compilation error, field " << f
                      << " is not in the tuple ";
            PrintValue(t, std::cerr);
            std::cerr << std::endl;
            exit(-1);
          }
          auto new_e = MakeIndex(e->line_num, res.exp, MakeInt(e->line_num, i));
          return TCResult(new_e, field_t, res.env);
        }
        default:
          std::cerr << e->line_num << ": compilation error, expected a tuple"
                    << std::endl;
          exit(-1);
      }
    }
    case ExpressionKind::Tuple: {
      auto new_args = new std::vector<std::pair<std::string, Expression*>>();
      auto arg_types = new VarValues();
      auto new_env = env;
      int i = 0;
      for (auto arg = e->u.tuple.fields->begin();
           arg != e->u.tuple.fields->end(); ++arg, ++i) {
        Value* arg_expected = nullptr;
        if (expected && expected->tag == ValKind::TupleTV) {
          arg_expected =
              FindInVarValues(arg->first, expected->u.tuple_type.fields);
          if (arg_expected == nullptr) {
            std::cerr << e->line_num << ": compilation error, missing field "
                      << arg->first << std::endl;
            exit(-1);
          }
        }
        auto arg_res =
            TypeCheckExp(arg->second, new_env, ct_env, arg_expected, context);
        new_env = arg_res.env;
        new_args->push_back(std::make_pair(arg->first, arg_res.exp));
        arg_types->push_back(std::make_pair(arg->first, arg_res.type));
      }
      auto tuple_e = MakeTuple(e->line_num, new_args);
      auto tuple_t = MakeTupleTypeVal(arg_types);
      return TCResult(tuple_e, tuple_t, new_env);
    }
    case ExpressionKind::GetField: {
      auto res = TypeCheckExp(e->u.get_field.aggregate, env, ct_env, nullptr,
                              TCContext::ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case ValKind::StructTV:
          // Search for a field
          for (auto& field : *t->u.struct_type.fields) {
            if (*e->u.get_field.field == field.first) {
              Expression* new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              return TCResult(new_e, field.second, res.env);
            }
          }
          // Search for a method
          for (auto& method : *t->u.struct_type.methods) {
            if (*e->u.get_field.field == method.first) {
              Expression* new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              return TCResult(new_e, method.second, res.env);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
          exit(-1);
        case ValKind::TupleTV:
          for (auto& field : *t->u.tuple_type.fields) {
            if (*e->u.get_field.field == field.first) {
              auto new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              return TCResult(new_e, field.second, res.env);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
          exit(-1);
        case ValKind::ChoiceTV:
          for (auto vt = t->u.choice_type.alternatives->begin();
               vt != t->u.choice_type.alternatives->end(); ++vt) {
            if (*e->u.get_field.field == vt->first) {
              Expression* new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              auto fun_ty = MakeFunTypeVal(vt->second, t);
              return TCResult(new_e, fun_ty, res.env);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
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
      auto t =
          Lookup(e->line_num, env, *(e->u.variable.name), PrintErrorString);
      return TCResult(e, t, env);
    }
    case ExpressionKind::Integer:
      return TCResult(e, MakeIntTypeVal(), env);
    case ExpressionKind::Boolean:
      return TCResult(e, MakeBoolTypeVal(), env);
    case ExpressionKind::PrimitiveOp: {
      auto es = new std::vector<Expression*>();
      std::vector<Value*> ts;
      auto new_env = env;
      for (auto& argument : *e->u.primitive_op.arguments) {
        auto res = TypeCheckExp(argument, env, ct_env, nullptr,
                                TCContext::ValueContext);
        new_env = res.env;
        es->push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e = MakeOp(e->line_num, e->u.primitive_op.op, es);
      switch (e->u.primitive_op.op) {
        case Operator::Neg:
          ExpectType(e->line_num, "negation", MakeIntTypeVal(), ts[0]);
          return TCResult(new_e, MakeIntTypeVal(), new_env);
        case Operator::Add:
        case Operator::Sub:
          ExpectType(e->line_num, "subtraction(1)", MakeIntTypeVal(), ts[0]);
          ExpectType(e->line_num, "substration(2)", MakeIntTypeVal(), ts[1]);
          return TCResult(new_e, MakeIntTypeVal(), new_env);
        case Operator::And:
          ExpectType(e->line_num, "&&(1)", MakeBoolTypeVal(), ts[0]);
          ExpectType(e->line_num, "&&(2)", MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
        case Operator::Or:
          ExpectType(e->line_num, "||(1)", MakeBoolTypeVal(), ts[0]);
          ExpectType(e->line_num, "||(2)", MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
        case Operator::Not:
          ExpectType(e->line_num, "!", MakeBoolTypeVal(), ts[0]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
        case Operator::Eq:
          ExpectType(e->line_num, "==(1)", MakeIntTypeVal(), ts[0]);
          ExpectType(e->line_num, "==(2)", MakeIntTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
      }
      break;
    }
    case ExpressionKind::Call: {
      auto fun_res = TypeCheckExp(e->u.call.function, env, ct_env, nullptr,
                                  TCContext::ValueContext);
      switch (fun_res.type->tag) {
        case ValKind::FunctionTV: {
          auto fun_t = fun_res.type;
          auto arg_res =
              TypeCheckExp(e->u.call.argument, fun_res.env, ct_env,
                           fun_t->u.fun_type.param, TCContext::ValueContext);
          ExpectType(e->line_num, "call", fun_t->u.fun_type.param,
                     arg_res.type);
          auto new_e = MakeCall(e->line_num, fun_res.exp, arg_res.exp);
          return TCResult(new_e, fun_t->u.fun_type.ret, arg_res.env);
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
          auto pt = ToType(e->line_num,
                           InterpExp(ct_env, e->u.function_type.parameter));
          auto rt = ToType(e->line_num,
                           InterpExp(ct_env, e->u.function_type.return_type));
          auto new_e = MakeFunType(e->line_num, ReifyType(pt, e->line_num),
                                   ReifyType(rt, e->line_num));
          return TCResult(new_e, MakeTypeTypeVal(), env);
        }
        case TCContext::PatternContext: {
          auto param_res = TypeCheckExp(e->u.function_type.parameter, env,
                                        ct_env, nullptr, context);
          auto ret_res = TypeCheckExp(e->u.function_type.return_type,
                                      param_res.env, ct_env, nullptr, context);
          auto new_e =
              MakeFunType(e->line_num, ReifyType(param_res.type, e->line_num),
                          ReifyType(ret_res.type, e->line_num));
          return TCResult(new_e, MakeTypeTypeVal(), ret_res.env);
        }
      }
    }
    case ExpressionKind::IntT:
    case ExpressionKind::BoolT:
    case ExpressionKind::TypeT:
    case ExpressionKind::AutoT:
      return TCResult(e, MakeTypeTypeVal(), env);
  }
}

auto TypecheckCase(Value* expected, Expression* pat, Statement* body,
                   TypeEnv* env, Env* ct_env, Value* ret_type)
    -> std::pair<Expression*, Statement*> {
  auto pat_res =
      TypeCheckExp(pat, env, ct_env, expected, TCContext::PatternContext);
  auto res = TypeCheckStmt(body, pat_res.env, ct_env, ret_type);
  return std::make_pair(pat, res.stmt);
}

// The TypeCheckStmt function performs semantic analysis on a statement.
// It returns a new version of the statement and a new type environment.
//
// The ret_type parameter is used for analyzing return statements.
// It is the declared return type of the enclosing function definition.
// If the return type is "auto", then the return type is inferred from
// the first return statement.
auto TypeCheckStmt(Statement* s, TypeEnv* env, Env* ct_env, Value* ret_type)
    -> TCStatement {
  if (!s) {
    return TCStatement(s, env);
  }
  switch (s->tag) {
    case StatementKind::Match: {
      auto res = TypeCheckExp(s->u.match_stmt.exp, env, ct_env, nullptr,
                              TCContext::ValueContext);
      auto res_type = res.type;
      auto new_clauses = new std::list<std::pair<Expression*, Statement*>>();
      for (auto& clause : *s->u.match_stmt.clauses) {
        new_clauses->push_back(TypecheckCase(
            res_type, clause.first, clause.second, env, ct_env, ret_type));
      }
      Statement* new_s = MakeMatch(s->line_num, res.exp, new_clauses);
      return TCStatement(new_s, env);
    }
    case StatementKind::While: {
      auto cnd_res = TypeCheckExp(s->u.while_stmt.cond, env, ct_env, nullptr,
                                  TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `while`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto body_res =
          TypeCheckStmt(s->u.while_stmt.body, env, ct_env, ret_type);
      auto new_s = MakeWhile(s->line_num, cnd_res.exp, body_res.stmt);
      return TCStatement(new_s, env);
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return TCStatement(s, env);
    case StatementKind::Block: {
      auto stmt_res = TypeCheckStmt(s->u.block.stmt, env, ct_env, ret_type);
      return TCStatement(MakeBlock(s->line_num, stmt_res.stmt), env);
    }
    case StatementKind::VariableDefinition: {
      auto res = TypeCheckExp(s->u.variable_definition.init, env, ct_env,
                              nullptr, TCContext::ValueContext);
      Value* rhs_ty = res.type;
      auto lhs_res = TypeCheckExp(s->u.variable_definition.pat, env, ct_env,
                                  rhs_ty, TCContext::PatternContext);
      Statement* new_s =
          MakeVarDef(s->line_num, s->u.variable_definition.pat, res.exp);
      return TCStatement(new_s, lhs_res.env);
    }
    case StatementKind::Sequence: {
      auto stmt_res = TypeCheckStmt(s->u.sequence.stmt, env, ct_env, ret_type);
      auto env2 = stmt_res.env;
      auto next_res = TypeCheckStmt(s->u.sequence.next, env2, ct_env, ret_type);
      auto env3 = next_res.env;
      return TCStatement(MakeSeq(s->line_num, stmt_res.stmt, next_res.stmt),
                         env3);
    }
    case StatementKind::Assign: {
      auto rhs_res = TypeCheckExp(s->u.assign.rhs, env, ct_env, nullptr,
                                  TCContext::ValueContext);
      auto rhs_t = rhs_res.type;
      auto lhs_res = TypeCheckExp(s->u.assign.lhs, env, ct_env, rhs_t,
                                  TCContext::ValueContext);
      auto lhs_t = lhs_res.type;
      ExpectType(s->line_num, "assign", lhs_t, rhs_t);
      auto new_s = MakeAssign(s->line_num, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.env);
    }
    case StatementKind::ExpressionStatement: {
      auto res =
          TypeCheckExp(s->u.exp, env, ct_env, nullptr, TCContext::ValueContext);
      auto new_s = MakeExpStmt(s->line_num, res.exp);
      return TCStatement(new_s, env);
    }
    case StatementKind::If: {
      auto cnd_res = TypeCheckExp(s->u.if_stmt.cond, env, ct_env, nullptr,
                                  TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `if`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto thn_res =
          TypeCheckStmt(s->u.if_stmt.then_stmt, env, ct_env, ret_type);
      auto els_res =
          TypeCheckStmt(s->u.if_stmt.else_stmt, env, ct_env, ret_type);
      auto new_s = MakeIf(s->line_num, cnd_res.exp, thn_res.stmt, els_res.stmt);
      return TCStatement(new_s, env);
    }
    case StatementKind::Return: {
      auto res = TypeCheckExp(s->u.return_stmt, env, ct_env, nullptr,
                              TCContext::ValueContext);
      if (ret_type->tag == ValKind::AutoTV) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        *ret_type = *res.type;
      } else {
        ExpectType(s->line_num, "return", ret_type, res.type);
      }
      return TCStatement(MakeReturn(s->line_num, res.exp), env);
    }
  }
}

auto CheckOrEnsureReturn(Statement* stmt, bool void_return, int line_num)
    -> Statement* {
  if (!stmt) {
    if (void_return) {
      auto args = new std::vector<std::pair<std::string, Expression*>>();
      return MakeReturn(line_num, MakeTuple(line_num, args));
    } else {
      std::cerr
          << "control-flow reaches end of non-void function without a return"
          << std::endl;
      exit(-1);
    }
  }
  switch (stmt->tag) {
    case StatementKind::Match: {
      auto new_clauses = new std::list<std::pair<Expression*, Statement*>>();
      for (auto i = stmt->u.match_stmt.clauses->begin();
           i != stmt->u.match_stmt.clauses->end(); ++i) {
        auto s = CheckOrEnsureReturn(i->second, void_return, stmt->line_num);
        new_clauses->push_back(std::make_pair(i->first, s));
      }
      return MakeMatch(stmt->line_num, stmt->u.match_stmt.exp, new_clauses);
    }
    case StatementKind::Block:
      return MakeBlock(
          stmt->line_num,
          CheckOrEnsureReturn(stmt->u.block.stmt, void_return, stmt->line_num));
    case StatementKind::If:
      return MakeIf(stmt->line_num, stmt->u.if_stmt.cond,
                    CheckOrEnsureReturn(stmt->u.if_stmt.then_stmt, void_return,
                                        stmt->line_num),
                    CheckOrEnsureReturn(stmt->u.if_stmt.else_stmt, void_return,
                                        stmt->line_num));
    case StatementKind::Return:
      return stmt;
    case StatementKind::Sequence:
      if (stmt->u.sequence.next) {
        return MakeSeq(stmt->line_num, stmt->u.sequence.stmt,
                       CheckOrEnsureReturn(stmt->u.sequence.next, void_return,
                                           stmt->line_num));
      } else {
        return CheckOrEnsureReturn(stmt->u.sequence.stmt, void_return,
                                   stmt->line_num);
      }
    case StatementKind::Assign:
    case StatementKind::ExpressionStatement:
    case StatementKind::While:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::VariableDefinition:
      if (void_return) {
        auto args = new std::vector<std::pair<std::string, Expression*>>();
        return MakeSeq(
            stmt->line_num, stmt,
            MakeReturn(stmt->line_num, MakeTuple(stmt->line_num, args)));
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

auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv* env, Env* ct_env)
    -> struct FunctionDefinition* {
  auto param_res = TypeCheckExp(f->param_pattern, env, ct_env, nullptr,
                                TCContext::PatternContext);
  auto return_type = ToType(f->line_num, InterpExp(ct_env, f->return_type));
  if (f->name == "main") {
    ExpectType(f->line_num, "return type of `main`", MakeIntTypeVal(),
               return_type);
    // TODO: Check that main doesn't have any parameters.
  }
  auto res = TypeCheckStmt(f->body, param_res.env, ct_env, return_type);
  bool void_return = TypeEqual(return_type, MakeVoidTypeVal());
  auto body = CheckOrEnsureReturn(res.stmt, void_return, f->line_num);
  return MakeFunDef(f->line_num, f->name, ReifyType(return_type, f->line_num),
                    f->param_pattern, body);
}

auto TypeOfFunDef(TypeEnv* env, Env* ct_env, const FunctionDefinition* fun_def)
    -> Value* {
  auto param_res = TypeCheckExp(fun_def->param_pattern, env, ct_env, nullptr,
                                TCContext::PatternContext);
  auto param_type = ToType(fun_def->line_num, param_res.type);
  auto ret = InterpExp(ct_env, fun_def->return_type);
  if (ret->tag == ValKind::AutoTV) {
    auto f = TypeCheckFunDef(fun_def, env, ct_env);
    ret = InterpExp(ct_env, f->return_type);
  }
  return MakeFunTypeVal(param_type, ret);
}

auto TypeOfStructDef(const StructDefinition* sd, TypeEnv* /*env*/, Env* ct_top)
    -> Value* {
  auto fields = new VarValues();
  auto methods = new VarValues();
  for (auto m = sd->members->begin(); m != sd->members->end(); ++m) {
    if ((*m)->tag == MemberKind::FieldMember) {
      auto t = ToType(sd->line_num, InterpExp(ct_top, (*m)->u.field.type));
      fields->push_back(std::make_pair(*(*m)->u.field.name, t));
    }
  }
  return MakeStructTypeVal(*sd->name, fields, methods);
}

auto FunctionDeclaration::Name() const -> std::string {
  return definition->name;
}

auto StructDeclaration::Name() const -> std::string { return *definition.name; }

auto ChoiceDeclaration::Name() const -> std::string { return name; }

auto StructDeclaration::TypeChecked(TypeEnv* env, Env* ct_env) const
    -> const Declaration* {
  auto fields = new std::list<Member*>();
  for (auto& m : *definition.members) {
    if (m->tag == MemberKind::FieldMember) {
      // TODO: Interpret the type expression and store the result.
      fields->push_back(m);
    }
  }
  return new StructDeclaration(definition.line_num, *definition.name, fields);
}

auto FunctionDeclaration::TypeChecked(TypeEnv* env, Env* ct_env) const
    -> const Declaration* {
  return new FunctionDeclaration(TypeCheckFunDef(definition, env, ct_env));
}

auto ChoiceDeclaration::TypeChecked(TypeEnv* env, Env* ct_env) const
    -> const Declaration* {
  return this;  // TODO.
}

auto TopLevel(std::list<Declaration*>* fs) -> std::pair<TypeEnv*, Env*> {
  ExecutionEnvironment tops = {nullptr, nullptr};
  bool found_main = false;

  for (auto d : *fs) {
    if (d->Name() == "main") {
      found_main = true;
    }
    d->TopLevel(tops);
  }

  if (found_main == false) {
    std::cerr << "error, program must contain a function named `main`"
              << std::endl;
    exit(-1);
  }
  return tops;
}

auto FunctionDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  auto t = TypeOfFunDef(tops.first, tops.second, definition);
  tops.first = new TypeEnv(Name(), t, tops.first);
}

auto StructDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  auto st = TypeOfStructDef(&definition, tops.first, tops.second);
  Address a = AllocateValue(st);
  tops.second = new Env(Name(), a, tops.second);  // Is this obsolete?
  auto params = MakeTupleTypeVal(st->u.struct_type.fields);
  auto fun_ty = MakeFunTypeVal(params, st);
  tops.first = new TypeEnv(Name(), fun_ty, tops.first);
}

auto ChoiceDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  auto alts = new VarValues();
  for (auto a : alternatives) {
    auto t = ToType(line_num, InterpExp(tops.second, a.second));
    alts->push_back(std::make_pair(a.first, t));
  }
  auto ct = MakeChoiceTypeVal(name, alts);
  Address a = AllocateValue(ct);
  tops.second = new Env(Name(), a, tops.second);  // Is this obsolete?
  tops.first = new TypeEnv(Name(), ct, tops.first);
}

}  // namespace Carbon
