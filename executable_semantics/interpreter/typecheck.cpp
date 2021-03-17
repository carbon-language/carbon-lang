// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/typecheck.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/utility/fatal.h"

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

void PrintTypeEnv(TypeEnv env, std::ostream& out) {
  for (const auto& [name, value] : env) {
    out << name << ": ";
    PrintValue(value, out);
    out << ", ";
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
auto ReifyType(Value* t, int line_num) -> Expression {
  ExpressionSource::Location bogus(0);
  switch (t->tag) {
    case ValKind::VarTV:
      return VariableExpression(bogus, *t->u.var_type);
    case ValKind::IntTV:
      return IntTypeExpression(bogus);
    case ValKind::BoolTV:
      return BoolTypeExpression(bogus);
    case ValKind::TypeTV:
      return TypeTypeExpression(bogus);
    case ValKind::FunctionTV:
      return FunctionTypeExpression(bogus,
                                    ReifyType(t->u.fun_type.param, line_num),
                                    ReifyType(t->u.fun_type.ret, line_num));
    case ValKind::TupleTV: {
      std::vector<std::pair<std::string, Expression>> args;
      for (auto& field : *t->u.tuple_type.fields) {
        args.push_back(
            make_pair(field.first, ReifyType(field.second, line_num)));
      }
      return TupleExpression(bogus, args);
    }
    case ValKind::StructTV:
      return VariableExpression(bogus, *t->u.struct_type.name);
    case ValKind::ChoiceTV:
      return VariableExpression(bogus, *t->u.choice_type.name);
    default:
      std::cerr << line_num << ": expected a type, not ";
      PrintValue(t, std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

// Resolves types in this expression, given the environments and type checking
// context in which *this, appears and any expected type, returning a result
// object bundling 1) a new version of *this containing deduced type
// information, 2) the computed type of *this, and 3) an updated environment
// binding any pattern variables.
//
// env maps variable names to the type of their run-time value.
// ct_env maps variable names to their compile-time values.
// expectedType may be nullopt outside of a pattern context.
// context says what kind of position this expression is nested in,
//    whether it's a position that expects a value, a pattern, or a type.
auto Expression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                           TCContext_ context) const -> TCResult {
  return box->TypeCheck(env, ct_env, expectedType, context);
}

template <class Content>
auto Expression::Boxed<Content>::TypeCheck(TypeEnv env, Env ct_env,
                                           Value* expected,
                                           TCContext_ context) const
    -> TCResult {
  return content.TypeCheck(env, ct_env, expected, context);
}

auto PatternVariableExpression::TypeCheck(TypeEnv env, Env ct_env,
                                          Value* expectedType,
                                          TCContext_ context) const
    -> TCResult {
  auto line_number = location.lineNumber;

  if (context.value != TCContext::PatternContext) {
    fatal(line_number,
          ": compilation error, pattern variables are only allowed in "
          "pattern context");
  }

  auto t = ToType(line_number, InterpExp(ct_env, type));

  if (t->tag == ValKind::AutoTV) {
    if (expectedType == nullptr) {
      fatal(line_number, ": compilation error, auto not allowed here");
    } else {
      t = expectedType;
    }
  }

  auto new_e =
      PatternVariableExpression(ExpressionSource::Location(line_number),
                                this->name, ReifyType(t, line_number));

  env.Set(this->name, t);
  return TCResult(new_e, t, env);
}

auto IndexExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                TCContext_ context) const -> TCResult {
  auto res = aggregate.TypeCheck(env, ct_env, nullptr, TCContext::ValueContext);
  auto t = res.type;
  auto line_number = location.lineNumber;
  switch (t->tag) {
    case ValKind::TupleTV: {
      auto line_number = location.lineNumber;
      auto i = ToInteger(InterpExp(ct_env, offset));
      std::string f = std::to_string(i);
      auto field_t = FindInVarValues(f, t->u.tuple_type.fields);
      if (field_t == nullptr) {
        std::ostringstream actualValueText;
        PrintValue(t, actualValueText);
        fatal(line_number, ": compilation error, field ", f,
              " is not in the tuple ", actualValueText.str());
      }
      auto new_e =
          IndexExpression(location, res.exp, IntegerExpression(location, i));
      return TCResult(new_e, field_t, res.env);
    }
    default:
      fatal(line_number, ": compilation error, expected a tuple");
  }
}

auto TupleExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                TCContext_ context) const -> TCResult {
  std::vector<std::pair<std::string, Expression>> new_args;
  auto arg_types = new VarValues();
  auto new_env = env;
  for (auto const& arg : elements) {
    Value* arg_expected = nullptr;
    if (expectedType && expectedType->tag == ValKind::TupleTV) {
      arg_expected =
          FindInVarValues(arg.first, expectedType->u.tuple_type.fields);
      if (arg_expected == nullptr) {
        fatal(location.lineNumber, ": compilation error, missing field ",
              arg.first);
      }
    }
    auto arg_res = arg.second.TypeCheck(new_env, ct_env, arg_expected, context);
    new_env = arg_res.env;
    new_args.push_back(std::make_pair(arg.first, arg_res.exp));
    arg_types->push_back(std::make_pair(arg.first, arg_res.type));
  }
  auto tuple_e = TupleExpression(location, new_args);
  auto tuple_t = MakeTupleTypeVal(arg_types);
  return TCResult(tuple_e, tuple_t, new_env);
}

auto GetFieldExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                   TCContext_ context) const -> TCResult {
  auto res = aggregate.TypeCheck(env, ct_env, nullptr, TCContext::ValueContext);
  auto t = res.type;
  switch (t->tag) {
    case ValKind::StructTV:
      // Search for a field
      for (auto& field : *t->u.struct_type.fields) {
        if (fieldName == field.first) {
          Expression new_e = GetFieldExpression(location, res.exp, fieldName);
          return TCResult(new_e, field.second, res.env);
        }
      }
      // Search for a method
      for (auto& method : *t->u.struct_type.methods) {
        if (fieldName == method.first) {
          Expression new_e = GetFieldExpression(location, res.exp, fieldName);
          return TCResult(new_e, method.second, res.env);
        }
      }
      fatal(location.lineNumber, ": compilation error, struct ",
            *t->u.struct_type.name, " does not have a field named ", fieldName);
    case ValKind::TupleTV:
      for (auto& field : *t->u.tuple_type.fields) {
        if (fieldName == field.first) {
          auto new_e = GetFieldExpression(location, res.exp, fieldName);
          return TCResult(new_e, field.second, res.env);
        }
      }
      fatal(location.lineNumber, ": compilation error, struct ",
            *t->u.struct_type.name, " does not have a field named ", fieldName);
    case ValKind::ChoiceTV:
      for (auto vt = t->u.choice_type.alternatives->begin();
           vt != t->u.choice_type.alternatives->end(); ++vt) {
        if (fieldName == vt->first) {
          Expression new_e = GetFieldExpression(location, res.exp, fieldName);
          auto fun_ty = MakeFunTypeVal(vt->second, t);
          return TCResult(new_e, fun_ty, res.env);
        }
      }
      fatal(location.lineNumber, ": compilation error, struct ",
            *t->u.struct_type.name, " does not have a field named ", fieldName);

    default:
      std::cerr << location.lineNumber
                << ": compilation error in field access, expected a struct"
                << std::endl;
      this->Print();
      fatal("");
  }
}

auto VariableExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                   TCContext_ context) const -> TCResult {
  std::optional<Value*> type = env.Get(name);
  if (type) {
    return TCResult(*this, *type, env);
  } else {
    fatal(location.lineNumber, ": could not find `", name, "`");
  }
}

auto IntegerExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                  TCContext_ context) const -> TCResult {
  return TCResult(*this, MakeIntTypeVal(), env);
}

auto BooleanExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                  TCContext_ context) const -> TCResult {
  return TCResult(*this, MakeBoolTypeVal(), env);
}

auto PrimitiveOperatorExpression::TypeCheck(TypeEnv env, Env ct_env,
                                            Value* expectedType,
                                            TCContext_ context) const
    -> TCResult {
  std::vector<Expression> es;
  std::vector<Value*> ts;
  auto new_env = env;
  for (auto& argument : arguments) {
    auto res =
        argument.TypeCheck(env, ct_env, nullptr, TCContext::ValueContext);
    new_env = res.env;
    es.push_back(res.exp);
    ts.push_back(res.type);
  }
  auto new_e = PrimitiveOperatorExpression(location, operation, es);
  auto line_number = location.lineNumber;
  switch (operation) {
    case Operation::Neg:
      ExpectType(line_number, "negation", MakeIntTypeVal(), ts[0]);
      return TCResult(new_e, MakeIntTypeVal(), new_env);
    case Operation::Add:
    case Operation::Sub:
      ExpectType(line_number, "subtraction(1)", MakeIntTypeVal(), ts[0]);
      ExpectType(line_number, "substration(2)", MakeIntTypeVal(), ts[1]);
      return TCResult(new_e, MakeIntTypeVal(), new_env);
    case Operation::And:
      ExpectType(line_number, "&&(1)", MakeBoolTypeVal(), ts[0]);
      ExpectType(line_number, "&&(2)", MakeBoolTypeVal(), ts[1]);
      return TCResult(new_e, MakeBoolTypeVal(), new_env);
    case Operation::Or:
      ExpectType(line_number, "||(1)", MakeBoolTypeVal(), ts[0]);
      ExpectType(line_number, "||(2)", MakeBoolTypeVal(), ts[1]);
      return TCResult(new_e, MakeBoolTypeVal(), new_env);
    case Operation::Not:
      ExpectType(line_number, "!", MakeBoolTypeVal(), ts[0]);
      return TCResult(new_e, MakeBoolTypeVal(), new_env);
    case Operation::Eq:
      ExpectType(line_number, "==(1)", MakeIntTypeVal(), ts[0]);
      ExpectType(line_number, "==(2)", MakeIntTypeVal(), ts[1]);
      return TCResult(new_e, MakeBoolTypeVal(), new_env);
  }
}

auto CallExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                               TCContext_ context) const -> TCResult {
  auto fun_res =
      function.TypeCheck(env, ct_env, nullptr, TCContext::ValueContext);
  switch (fun_res.type->tag) {
    case ValKind::FunctionTV: {
      auto fun_t = fun_res.type;
      auto arg_res = argumentTuple.TypeCheck(fun_res.env, ct_env,
                                             fun_t->u.fun_type.param, context);
      ExpectType(location.lineNumber, "call", fun_t->u.fun_type.param,
                 arg_res.type);
      auto new_e = CallExpression(location, fun_res.exp, arg_res.exp);
      return TCResult(new_e, fun_t->u.fun_type.ret, arg_res.env);
    }
    default: {
      std::cerr << location.lineNumber
                << ": compilation error in call, expected a function"
                << std::endl;
      this->Print();
      fatal("");
    }
  }
}

auto FunctionTypeExpression::TypeCheck(TypeEnv env, Env ct_env,
                                       Value* expectedType,
                                       TCContext_ context) const -> TCResult {
  auto line_number = location.lineNumber;
  switch (context.value) {
    case TCContext::ValueContext:
    case TCContext::TypeContext: {
      auto pt = ToType(line_number, InterpExp(ct_env, parameterTupleType));
      auto rt = ToType(line_number, InterpExp(ct_env, returnType));
      auto new_e = FunctionTypeExpression(location, ReifyType(pt, line_number),
                                          ReifyType(rt, line_number));
      return TCResult(new_e, MakeTypeTypeVal(), env);
    }
    case TCContext::PatternContext: {
      auto param_res =
          parameterTupleType.TypeCheck(env, ct_env, nullptr, context);
      auto ret_res =
          returnType.TypeCheck(param_res.env, ct_env, nullptr, context);
      auto new_e = FunctionTypeExpression(
          location, ReifyType(param_res.type, line_number),
          ReifyType(ret_res.type, line_number));
      return TCResult(new_e, MakeTypeTypeVal(), ret_res.env);
    }
  }
}

auto IntTypeExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                  TCContext_ context) const -> TCResult {
  return TCResult(*this, MakeTypeTypeVal(), env);
}

auto BoolTypeExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                   TCContext_ context) const -> TCResult {
  return TCResult(*this, MakeTypeTypeVal(), env);
}

auto TypeTypeExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                   TCContext_ context) const -> TCResult {
  return TCResult(*this, MakeTypeTypeVal(), env);
}

auto AutoTypeExpression::TypeCheck(TypeEnv env, Env ct_env, Value* expectedType,
                                   TCContext_ context) const -> TCResult {
  return TCResult(*this, MakeTypeTypeVal(), env);
}

auto TypecheckCase(Value* expected, Expression* pat, Statement* body,
                   TypeEnv env, Env ct_env, Value* ret_type)
    -> std::pair<Expression*, Statement*> {
  auto pat_res =
      pat->TypeCheck(env, ct_env, expected, TCContext::PatternContext);
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
auto TypeCheckStmt(Statement* s, TypeEnv env, Env ct_env, Value* ret_type)
    -> TCStatement {
  if (!s) {
    return TCStatement(s, env);
  }
  switch (s->tag) {
    case StatementKind::Match: {
      auto res = s->u.match_stmt.exp->TypeCheck(env, ct_env, nullptr,
                                                TCContext::ValueContext);
      auto res_type = res.type;
      auto new_clauses = new std::list<std::pair<Expression*, Statement*>>();
      for (auto& clause : *s->u.match_stmt.clauses) {
        new_clauses->push_back(TypecheckCase(
            res_type, clause.first, clause.second, env, ct_env, ret_type));
      }
      Statement* new_s =
          MakeMatch(s->line_num, new Expression(res.exp), new_clauses);
      return TCStatement(new_s, env);
    }
    case StatementKind::While: {
      auto cnd_res = s->u.while_stmt.cond->TypeCheck(env, ct_env, nullptr,
                                                     TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `while`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto body_res =
          TypeCheckStmt(s->u.while_stmt.body, env, ct_env, ret_type);
      auto new_s =
          MakeWhile(s->line_num, new Expression(cnd_res.exp), body_res.stmt);
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
      auto res = s->u.variable_definition.init->TypeCheck(
          env, ct_env, nullptr, TCContext::ValueContext);
      Value* rhs_ty = res.type;
      auto lhs_res = s->u.variable_definition.pat->TypeCheck(
          env, ct_env, rhs_ty, TCContext::PatternContext);
      Statement* new_s = MakeVarDef(s->line_num, s->u.variable_definition.pat,
                                    new Expression(res.exp));
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
      auto rhs_res = s->u.assign.rhs->TypeCheck(env, ct_env, nullptr,
                                                TCContext::ValueContext);
      auto rhs_t = rhs_res.type;
      auto lhs_res = s->u.assign.lhs->TypeCheck(env, ct_env, rhs_t,
                                                TCContext::ValueContext);
      auto lhs_t = lhs_res.type;
      ExpectType(s->line_num, "assign", lhs_t, rhs_t);
      auto new_s = MakeAssign(s->line_num, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.env);
    }
    case StatementKind::ExpressionStatement: {
      auto res =
          s->u.exp->TypeCheck(env, ct_env, nullptr, TCContext::ValueContext);
      auto new_s = MakeExpStmt(s->line_num, res.exp);
      return TCStatement(new_s, env);
    }
    case StatementKind::If: {
      auto cnd_res = s->u.if_stmt.cond->TypeCheck(env, ct_env, nullptr,
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
      auto res = s->u.return_stmt->TypeCheck(env, ct_env, nullptr,
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
      return MakeReturn(
          line_num,
          TupleExpression(ExpressionSource::Location(line_num), args));
    } else {
      fatal("control-flow reaches end of non-void function without a return");
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
            MakeReturn(stmt->line_num,
                       TupleExpression(
                           ExpressionSource::Location(stmt->line_num), args)));
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

auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv env, Env ct_env)
    -> struct FunctionDefinition* {
  auto param_res = f->param_pattern->TypeCheck(env, ct_env, nullptr,
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

auto TypeOfFunDef(TypeEnv env, Env ct_env, const FunctionDefinition* fun_def)
    -> Value* {
  auto param_res = fun_def->param_pattern->TypeCheck(env, ct_env, nullptr,
                                                     TCContext::PatternContext);
  auto param_type = ToType(fun_def->line_num, param_res.type);
  auto ret = InterpExp(ct_env, fun_def->return_type);
  if (ret->tag == ValKind::AutoTV) {
    auto f = TypeCheckFunDef(fun_def, env, ct_env);
    ret = InterpExp(ct_env, f->return_type);
  }
  return MakeFunTypeVal(param_type, ret);
}

auto TypeOfStructDef(const StructDefinition* sd, TypeEnv /*env*/, Env ct_top)
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

// Returns the name of the declared variable.
auto VariableDeclaration::Name() const -> std::string { return name; }

auto StructDeclaration::TypeChecked(TypeEnv env, Env ct_env) const
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

auto FunctionDeclaration::TypeChecked(TypeEnv env, Env ct_env) const
    -> Declaration {
  return FunctionDeclaration(TypeCheckFunDef(definition, env, ct_env));
}

auto ChoiceDeclaration::TypeChecked(TypeEnv env, Env ct_env) const
    -> Declaration {
  return *this;  // TODO.
}

// Signals a type error if the initializing expression does not have
// the declared type of the variable, otherwise returns this
// declaration with annotated types.
auto VariableDeclaration::TypeChecked(TypeEnv env, Env ct_env) const
    -> Declaration {
  TCResult type_checked_initializer =
      TypeCheckExp(initializer, env, ct_env, nullptr, TCContext::ValueContext);
  Value* declared_type = ToType(source_location, InterpExp(ct_env, type));
  ExpectType(source_location, "initializer of variable", declared_type,
             type_checked_initializer.type);
  return *this;
}

auto TopLevel(std::list<Declaration>* fs) -> std::pair<TypeEnv, Env> {
  ExecutionEnvironment tops;
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

auto FunctionDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  auto t = TypeOfFunDef(tops.first, tops.second, definition);
  tops.first.Set(Name(), t);
}

auto StructDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  auto st = TypeOfStructDef(&definition, tops.first, tops.second);
  Address a = AllocateValue(st);
  tops.second.Set(Name(), a);  // Is this obsolete?
  auto params = MakeTupleTypeVal(st->u.struct_type.fields);
  auto fun_ty = MakeFunTypeVal(params, st);
  tops.first.Set(Name(), fun_ty);
}

auto ChoiceDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  auto alts = new VarValues();
  for (auto a : alternatives) {
    auto t = ToType(line_num, InterpExp(tops.second, a.second));
    alts->push_back(std::make_pair(a.first, t));
  }
  auto ct = MakeChoiceTypeVal(name, alts);
  Address a = AllocateValue(ct);
  tops.second.Set(Name(), a);  // Is this obsolete?
  tops.first.Set(Name(), ct);
}

// Associate the variable name with it's declared type in the
// compile-time symbol table.
auto VariableDeclaration::TopLevel(ExecutionEnvironment& tops) const -> void {
  Value* declared_type = ToType(source_location, InterpExp(tops.second, type));
  tops.first.Set(Name(), declared_type);
}

}  // namespace Carbon
