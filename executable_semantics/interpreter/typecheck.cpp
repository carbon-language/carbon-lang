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
      return MakeVar(0, *t->u.var_type);
    case ValKind::IntTV:
      return MakeIntType(0);
    case ValKind::BoolTV:
      return MakeBoolType(0);
    case ValKind::TypeTV:
      return MakeTypeType(0);
    case ValKind::ContinuationTV:
      return MakeContinuationType(0);
    case ValKind::FunctionTV:
      return MakeFunType(0, ReifyType(t->u.fun_type.param, line_num),
                         ReifyType(t->u.fun_type.ret, line_num));
    case ValKind::TupleV: {
      auto args = new std::vector<FieldInitializer>();
      for (const TupleElement& field : *t->u.tuple.elements) {
        args->push_back(
            {.name = field.name,
             .expression = ReifyType(
                 state->ReadFromMemory(field.address, line_num), line_num)});
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
      auto t = InterpExp(values, e->u.pattern_variable.type);
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
      types.Set(*e->u.pattern_variable.name, t);
      return TCResult(new_e, t, types);
    }
    case ExpressionKind::Index: {
      auto res = TypeCheckExp(e->u.get_field.aggregate, types, values, nullptr,
                              TCContext::ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case ValKind::TupleV: {
          auto i = ToInteger(InterpExp(values, e->u.index.offset));
          std::string f = std::to_string(i);
          std::optional<Address> field_address = FindTupleField(f, t);
          if (field_address == std::nullopt) {
            std::cerr << e->line_num << ": compilation error, field " << f
                      << " is not in the tuple ";
            PrintValue(t, std::cerr);
            std::cerr << std::endl;
            exit(-1);
          }
          auto field_t = state->ReadFromMemory(*field_address, e->line_num);
          auto new_e = MakeIndex(e->line_num, res.exp, MakeInt(e->line_num, i));
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
      int i = 0;
      for (auto arg = e->u.tuple.fields->begin();
           arg != e->u.tuple.fields->end(); ++arg, ++i) {
        const Value* arg_expected = nullptr;
        if (expected && expected->tag == ValKind::TupleV) {
          std::optional<Address> expected_field =
              FindTupleField(arg->name, expected);
          if (expected_field == std::nullopt) {
            std::cerr << e->line_num << ": compilation error, missing field "
                      << arg->name << std::endl;
            exit(-1);
          }
          arg_expected = state->ReadFromMemory(*expected_field, e->line_num);
        }
        auto arg_res = TypeCheckExp(arg->expression, new_types, values,
                                    arg_expected, context);
        new_types = arg_res.types;
        new_args->push_back({.name = arg->name, .expression = arg_res.exp});
        arg_types->push_back(
            {.name = arg->name, .address = state->AllocateValue(arg_res.type)});
      }
      auto tuple_e = MakeTuple(e->line_num, new_args);
      auto tuple_t = MakeTupleVal(arg_types);
      return TCResult(tuple_e, tuple_t, new_types);
    }
    case ExpressionKind::GetField: {
      auto res = TypeCheckExp(e->u.get_field.aggregate, types, values, nullptr,
                              TCContext::ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case ValKind::StructTV:
          // Search for a field
          for (auto& field : *t->u.struct_type.fields) {
            if (*e->u.get_field.field == field.first) {
              const Expression* new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              return TCResult(new_e, field.second, res.types);
            }
          }
          // Search for a method
          for (auto& method : *t->u.struct_type.methods) {
            if (*e->u.get_field.field == method.first) {
              const Expression* new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              return TCResult(new_e, method.second, res.types);
            }
          }
          std::cerr << e->line_num << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
          exit(-1);
        case ValKind::TupleV:
          for (const TupleElement& field : *t->u.tuple.elements) {
            if (*e->u.get_field.field == field.name) {
              auto new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              return TCResult(new_e,
                              state->ReadFromMemory(field.address, e->line_num),
                              res.types);
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
              const Expression* new_e =
                  MakeGetField(e->line_num, res.exp, *e->u.get_field.field);
              auto fun_ty = MakeFunTypeVal(vt->second, t);
              return TCResult(new_e, fun_ty, res.types);
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
      std::optional<const Value*> type = types.Get(*(e->u.variable.name));
      if (type) {
        return TCResult(e, *type, types);
      } else {
        std::cerr << e->line_num << ": could not find `"
                  << *(e->u.variable.name) << "`" << std::endl;
        exit(-1);
      }
    }
    case ExpressionKind::Integer:
      return TCResult(e, MakeIntTypeVal(), types);
    case ExpressionKind::Boolean:
      return TCResult(e, MakeBoolTypeVal(), types);
    case ExpressionKind::PrimitiveOp: {
      auto es = new std::vector<const Expression*>();
      std::vector<const Value*> ts;
      auto new_types = types;
      for (auto& argument : *e->u.primitive_op.arguments) {
        auto res = TypeCheckExp(argument, types, values, nullptr,
                                TCContext::ValueContext);
        new_types = res.types;
        es->push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e = MakeOp(e->line_num, e->u.primitive_op.op, es);
      switch (e->u.primitive_op.op) {
        case Operator::Neg:
          ExpectType(e->line_num, "negation", MakeIntTypeVal(), ts[0]);
          return TCResult(new_e, MakeIntTypeVal(), new_types);
        case Operator::Add:
        case Operator::Sub:
          ExpectType(e->line_num, "subtraction(1)", MakeIntTypeVal(), ts[0]);
          ExpectType(e->line_num, "substration(2)", MakeIntTypeVal(), ts[1]);
          return TCResult(new_e, MakeIntTypeVal(), new_types);
        case Operator::And:
          ExpectType(e->line_num, "&&(1)", MakeBoolTypeVal(), ts[0]);
          ExpectType(e->line_num, "&&(2)", MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_types);
        case Operator::Or:
          ExpectType(e->line_num, "||(1)", MakeBoolTypeVal(), ts[0]);
          ExpectType(e->line_num, "||(2)", MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_types);
        case Operator::Not:
          ExpectType(e->line_num, "!", MakeBoolTypeVal(), ts[0]);
          return TCResult(new_e, MakeBoolTypeVal(), new_types);
        case Operator::Eq:
          ExpectType(e->line_num, "==", ts[0], ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_types);
      }
      break;
    }
    case ExpressionKind::Call: {
      auto fun_res = TypeCheckExp(e->u.call.function, types, values, nullptr,
                                  TCContext::ValueContext);
      switch (fun_res.type->tag) {
        case ValKind::FunctionTV: {
          auto fun_t = fun_res.type;
          auto arg_res = TypeCheckExp(e->u.call.argument, fun_res.types, values,
                                      fun_t->u.fun_type.param, context);
          ExpectType(e->line_num, "call", fun_t->u.fun_type.param,
                     arg_res.type);
          auto new_e = MakeCall(e->line_num, fun_res.exp, arg_res.exp);
          return TCResult(new_e, fun_t->u.fun_type.ret, arg_res.types);
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
          auto pt = InterpExp(values, e->u.function_type.parameter);
          auto rt = InterpExp(values, e->u.function_type.return_type);
          auto new_e = MakeFunType(e->line_num, ReifyType(pt, e->line_num),
                                   ReifyType(rt, e->line_num));
          return TCResult(new_e, MakeTypeTypeVal(), types);
        }
        case TCContext::PatternContext: {
          auto param_res = TypeCheckExp(e->u.function_type.parameter, types,
                                        values, nullptr, context);
          auto ret_res =
              TypeCheckExp(e->u.function_type.return_type, param_res.types,
                           values, nullptr, context);
          auto new_e =
              MakeFunType(e->line_num, ReifyType(param_res.type, e->line_num),
                          ReifyType(ret_res.type, e->line_num));
          return TCResult(new_e, MakeTypeTypeVal(), ret_res.types);
        }
      }
    }
    case ExpressionKind::IntT:
      return TCResult(e, MakeIntTypeVal(), types);
    case ExpressionKind::BoolT:
      return TCResult(e, MakeBoolTypeVal(), types);
    case ExpressionKind::TypeT:
      return TCResult(e, MakeTypeTypeVal(), types);
    case ExpressionKind::AutoT:
      return TCResult(e, MakeAutoTypeVal(), types);
    case ExpressionKind::ContinuationT:
      return TCResult(e, MakeContinuationTypeVal(), types);
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
      auto res = TypeCheckExp(s->u.match_stmt.exp, types, values, nullptr,
                              TCContext::ValueContext);
      auto res_type = res.type;
      auto new_clauses =
          new std::list<std::pair<const Expression*, const Statement*>>();
      for (auto& clause : *s->u.match_stmt.clauses) {
        new_clauses->push_back(TypecheckCase(
            res_type, clause.first, clause.second, types, values, ret_type));
      }
      const Statement* new_s = MakeMatch(s->line_num, res.exp, new_clauses);
      return TCStatement(new_s, types);
    }
    case StatementKind::While: {
      auto cnd_res = TypeCheckExp(s->u.while_stmt.cond, types, values, nullptr,
                                  TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `while`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto body_res =
          TypeCheckStmt(s->u.while_stmt.body, types, values, ret_type);
      auto new_s = MakeWhile(s->line_num, cnd_res.exp, body_res.stmt);
      return TCStatement(new_s, types);
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return TCStatement(s, types);
    case StatementKind::Block: {
      auto stmt_res = TypeCheckStmt(s->u.block.stmt, types, values, ret_type);
      return TCStatement(MakeBlock(s->line_num, stmt_res.stmt), types);
    }
    case StatementKind::VariableDefinition: {
      auto res = TypeCheckExp(s->u.variable_definition.init, types, values,
                              nullptr, TCContext::ValueContext);
      const Value* rhs_ty = res.type;
      auto lhs_res = TypeCheckExp(s->u.variable_definition.pat, types, values,
                                  rhs_ty, TCContext::PatternContext);
      const Statement* new_s =
          MakeVarDef(s->line_num, s->u.variable_definition.pat, res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case StatementKind::Sequence: {
      auto stmt_res =
          TypeCheckStmt(s->u.sequence.stmt, types, values, ret_type);
      auto types2 = stmt_res.types;
      auto next_res =
          TypeCheckStmt(s->u.sequence.next, types2, values, ret_type);
      auto types3 = next_res.types;
      return TCStatement(MakeSeq(s->line_num, stmt_res.stmt, next_res.stmt),
                         types3);
    }
    case StatementKind::Assign: {
      auto rhs_res = TypeCheckExp(s->u.assign.rhs, types, values, nullptr,
                                  TCContext::ValueContext);
      auto rhs_t = rhs_res.type;
      auto lhs_res = TypeCheckExp(s->u.assign.lhs, types, values, rhs_t,
                                  TCContext::ValueContext);
      auto lhs_t = lhs_res.type;
      ExpectType(s->line_num, "assign", lhs_t, rhs_t);
      auto new_s = MakeAssign(s->line_num, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.types);
    }
    case StatementKind::ExpressionStatement: {
      auto res = TypeCheckExp(s->u.exp, types, values, nullptr,
                              TCContext::ValueContext);
      auto new_s = MakeExpStmt(s->line_num, res.exp);
      return TCStatement(new_s, types);
    }
    case StatementKind::If: {
      auto cnd_res = TypeCheckExp(s->u.if_stmt.cond, types, values, nullptr,
                                  TCContext::ValueContext);
      ExpectType(s->line_num, "condition of `if`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto thn_res =
          TypeCheckStmt(s->u.if_stmt.then_stmt, types, values, ret_type);
      auto els_res =
          TypeCheckStmt(s->u.if_stmt.else_stmt, types, values, ret_type);
      auto new_s = MakeIf(s->line_num, cnd_res.exp, thn_res.stmt, els_res.stmt);
      return TCStatement(new_s, types);
    }
    case StatementKind::Return: {
      auto res = TypeCheckExp(s->u.return_stmt, types, values, nullptr,
                              TCContext::ValueContext);
      if (ret_type->tag == ValKind::AutoTV) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        ret_type = res.type;
      } else {
        ExpectType(s->line_num, "return", ret_type, res.type);
      }
      return TCStatement(MakeReturn(s->line_num, res.exp), types);
    }
    case StatementKind::Continuation: {
      TCStatement body_result =
          TypeCheckStmt(s->u.continuation.body, types, values, ret_type);
      const Statement* new_continuation = MakeContinuationStatement(
          s->line_num, *s->u.continuation.continuation_variable,
          body_result.stmt);
      types.Set(*s->u.continuation.continuation_variable,
                MakeContinuationTypeVal());
      return TCStatement(new_continuation, types);
    }
    case StatementKind::Run: {
      TCResult argument_result = TypeCheckExp(s->u.run.argument, types, values,
                                              nullptr, TCContext::ValueContext);
      ExpectType(s->line_num, "argument of `run`", MakeContinuationTypeVal(),
                 argument_result.type);
      const Statement* new_run = MakeRun(s->line_num, argument_result.exp);
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
      return MakeReturn(line_num, MakeUnit(line_num));
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
        return MakeSeq(stmt->line_num, stmt,
                       MakeReturn(stmt->line_num, MakeUnit(stmt->line_num)));
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
    ExpectType(f->line_num, "return type of `main`", MakeIntTypeVal(),
               return_type);
    // TODO: Check that main doesn't have any parameters.
  }
  auto res = TypeCheckStmt(f->body, param_res.types, values, return_type);
  bool void_return = TypeEqual(return_type, MakeVoidTypeVal());
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
  return MakeFunTypeVal(param_res.type, ret);
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
  return MakeStructTypeVal(*sd->name, fields, methods);
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
}

auto StructDeclaration::TopLevel(TypeCheckContext& tops) const -> void {
  auto st = TypeOfStructDef(&definition, tops.types, tops.values);
  Address a = state->AllocateValue(st);
  tops.values.Set(Name(), a);  // Is this obsolete?
  auto field_types = new std::vector<TupleElement>();
  for (const auto& [field_name, field_value] : *st->u.struct_type.fields) {
    field_types->push_back(
        {.name = field_name, .address = state->AllocateValue(field_value)});
  }
  auto fun_ty = MakeFunTypeVal(MakeTupleVal(field_types), st);
  tops.types.Set(Name(), fun_ty);
}

auto ChoiceDeclaration::TopLevel(TypeCheckContext& tops) const -> void {
  auto alts = new VarValues();
  for (auto a : alternatives) {
    auto t = InterpExp(tops.values, a.second);
    alts->push_back(std::make_pair(a.first, t));
  }
  auto ct = MakeChoiceTypeVal(name, alts);
  Address a = state->AllocateValue(ct);
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
