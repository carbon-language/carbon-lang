// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/typecheck.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "executable_semantics/cons_list.h"
#include "executable_semantics/interp.h"

template <class T>
auto ListEqual(std::list<T*>* ts1, std::list<T*>* ts2, bool (*eq)(T*, T*))
    -> bool {
  if (ts1->size() == ts2->size()) {
    auto iter2 = ts2->begin();
    for (auto iter1 = ts1->begin(); iter1 != ts1->end(); ++iter1, ++iter2) {
      if (!eq(*iter1, *iter2)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

template <class T>
auto VectorEqual(std::vector<T*>* ts1, std::vector<T*>* ts2, bool (*eq)(T*, T*))
    -> bool {
  if (ts1->size() == ts2->size()) {
    auto iter2 = ts2->begin();
    for (auto iter1 = ts1->begin(); iter1 != ts1->end(); ++iter1, ++iter2) {
      if (!eq(*iter1, *iter2)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

auto FieldsEqual(VarValues* ts1, VarValues* ts2) -> bool {
  if (ts1->size() == ts2->size()) {
    for (auto& iter1 : *ts1) {
      try {
        auto t2 = FindAlist(iter1.first, ts2);
        if (!TypeEqual(iter1.second, t2)) {
          return false;
        }
      } catch (std::domain_error de) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

auto Find(const std::string& s, Cons<std::string>* ls, int n) -> int {
  if (ls) {
    if (ls->curr == s) {
      return n;
    } else {
      return Find(s, ls->next, n + 1);
    }
  } else {
    std::cerr << "could not find " << s << std::endl;
    exit(-1);
  }
}

auto TypeEqual(Value* t1, Value* t2) -> bool {
  return (t1->tag == VarTV && t2->tag == VarTV &&
          *t1->u.var_type == *t2->u.var_type) ||
         (t1->tag == IntTV && t2->tag == IntTV) ||
         (t1->tag == BoolTV && t2->tag == BoolTV) ||
         (t1->tag == PointerTV && t2->tag == PointerTV &&
          TypeEqual(t1->u.ptr_type.type, t2->u.ptr_type.type)) ||
         (t1->tag == FunctionTV && t2->tag == FunctionTV &&
          TypeEqual(t1->u.fun_type.param, t2->u.fun_type.param) &&
          TypeEqual(t1->u.fun_type.ret, t2->u.fun_type.ret)) ||
         (t1->tag == StructTV && t2->tag == StructTV &&
          *t1->u.struct_type.name == *t2->u.struct_type.name) ||
         (t1->tag == ChoiceTV && t2->tag == ChoiceTV &&
          *t1->u.choice_type.name == *t2->u.choice_type.name) ||
         (t1->tag == TupleTV && t2->tag == TupleTV &&
          FieldsEqual(t1->u.tuple_type.fields, t2->u.tuple_type.fields));
}

void ExpectType(int lineno, const std::string& context, Value* expected,
                Value* actual) {
  if (!TypeEqual(expected, actual)) {
    std::cerr << lineno << ": type error in " << context << std::endl;
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
auto ToType(int lineno, Value* val) -> Value* {
  switch (val->tag) {
    case TupleV: {
      auto fields = new VarValues();
      for (auto& elt : *val->u.tuple.elts) {
        Value* ty = ToType(lineno, state->heap[elt.second]);
        fields->push_back(std::make_pair(elt.first, ty));
      }
      return MakeTupleTypeVal(fields);
    }
    case TupleTV: {
      auto fields = new VarValues();
      for (auto& field : *val->u.tuple_type.fields) {
        Value* ty = ToType(lineno, field.second);
        fields->push_back(std::make_pair(field.first, ty));
      }
      return MakeTupleTypeVal(fields);
    }
    case PointerTV: {
      return MakePtrTypeVal(ToType(lineno, val->u.ptr_type.type));
    }
    case FunctionTV: {
      return MakeFunTypeVal(ToType(lineno, val->u.fun_type.param),
                            ToType(lineno, val->u.fun_type.ret));
    }
    case VarPatV: {
      return MakeVarPatVal(*val->u.var_pat.name,
                           ToType(lineno, val->u.var_pat.type));
    }
    case ChoiceTV:
    case StructTV:
    case TypeTV:
    case VarTV:
    case BoolTV:
    case IntTV:
    case AutoTV:
      return val;
    default:
      std::cerr << lineno << ": in ToType, expected a type, not ";
      PrintValue(val, std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

// Reify type to type expression.
auto ReifyType(Value* t, int lineno) -> Expression* {
  switch (t->tag) {
    case VarTV:
      return MakeVar(0, *t->u.var_type);
    case IntTV:
      return MakeIntType(0);
    case BoolTV:
      return MakeBoolType(0);
    case TypeTV:
      return MakeTypeType(0);
    case FunctionTV:
      return MakeFunType(0, ReifyType(t->u.fun_type.param, lineno),
                         ReifyType(t->u.fun_type.ret, lineno));
    case TupleTV: {
      auto args = new std::vector<std::pair<std::string, Expression*> >();
      for (auto& field : *t->u.tuple_type.fields) {
        args->push_back(
            make_pair(field.first, ReifyType(field.second, lineno)));
      }
      return MakeTuple(0, args);
    }
    case StructTV:
      return MakeVar(0, *t->u.struct_type.name);
    case ChoiceTV:
      return MakeVar(0, *t->u.choice_type.name);
    default:
      std::cerr << lineno << ": expected a type, not ";
      PrintValue(t, std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

auto TypeCheckExp(Expression* e, TypeEnv* env, Env* ct_env, Value* expected,
                  TCContext context)
    -> TCResult {  //                   expected can be null
  switch (e->tag) {
    case PatternVariable: {
      if (context != PatternContext) {
        std::cerr
            << e->lineno
            << ": compilation error, pattern variables are only allowed in "
               "pattern context"
            << std::endl;
      }
      auto t = ToType(e->lineno, InterpExp(ct_env, e->u.pattern_variable.type));
      if (t->tag == AutoTV) {
        if (expected == nullptr) {
          std::cerr << e->lineno << ": compilation error, auto not allowed here"
                    << std::endl;
          exit(-1);
        } else {
          t = expected;
        }
      }
      auto new_e = MakeVarPat(e->lineno, *e->u.pattern_variable.name,
                              ReifyType(t, e->lineno));
      return TCResult(new_e, t,
                      new TypeEnv(*e->u.pattern_variable.name, t, env));
    }
    case Index: {
      auto res = TypeCheckExp(e->u.get_field.aggregate, env, ct_env, nullptr,
                              ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case TupleTV: {
          auto i = ToInteger(InterpExp(ct_env, e->u.index.offset));
          std::string f = std::to_string(i);
          try {
            auto field_t = FindAlist(f, t->u.tuple_type.fields);
            auto new_e = MakeIndex(e->lineno, res.exp, MakeInt(e->lineno, i));
            return TCResult(new_e, field_t, res.env);
          } catch (std::domain_error de) {
            std::cerr << e->lineno << ": compilation error, field " << f
                      << " is not in the tuple ";
            PrintValue(t, std::cerr);
            std::cerr << std::endl;
          }
        }
        default:
          std::cerr << e->lineno << ": compilation error, expected a tuple"
                    << std::endl;
          exit(-1);
      }
    }
    case Tuple: {
      auto new_args = new std::vector<std::pair<std::string, Expression*> >();
      auto arg_types = new VarValues();
      auto new_env = env;
      int i = 0;
      for (auto arg = e->u.tuple.fields->begin();
           arg != e->u.tuple.fields->end(); ++arg, ++i) {
        Value* arg_expected = nullptr;
        if (expected && expected->tag == TupleTV) {
          try {
            arg_expected = FindAlist(arg->first, expected->u.tuple_type.fields);
          } catch (std::domain_error de) {
            std::cerr << e->lineno << ": compilation error, missing field "
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
      auto tuple_e = MakeTuple(e->lineno, new_args);
      auto tuple_t = MakeTupleTypeVal(arg_types);
      return TCResult(tuple_e, tuple_t, new_env);
    }
    case GetField: {
      auto res = TypeCheckExp(e->u.get_field.aggregate, env, ct_env, nullptr,
                              ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case StructTV:
          // Search for a field
          for (auto& field : *t->u.struct_type.fields) {
            if (*e->u.get_field.field == field.first) {
              Expression* new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              return TCResult(new_e, field.second, res.env);
            }
          }
          // Search for a method
          for (auto& method : *t->u.struct_type.methods) {
            if (*e->u.get_field.field == method.first) {
              Expression* new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              return TCResult(new_e, method.second, res.env);
            }
          }
          std::cerr << e->lineno << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
          exit(-1);
        case TupleTV:
          for (auto& field : *t->u.tuple_type.fields) {
            if (*e->u.get_field.field == field.first) {
              auto new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              return TCResult(new_e, field.second, res.env);
            }
          }
          std::cerr << e->lineno << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
          exit(-1);
        case ChoiceTV:
          for (auto vt = t->u.choice_type.alternatives->begin();
               vt != t->u.choice_type.alternatives->end(); ++vt) {
            if (*e->u.get_field.field == vt->first) {
              Expression* new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              auto fun_ty = MakeFunTypeVal(vt->second, t);
              return TCResult(new_e, fun_ty, res.env);
            }
          }
          std::cerr << e->lineno << ": compilation error, struct "
                    << *t->u.struct_type.name << " does not have a field named "
                    << *e->u.get_field.field << std::endl;
          exit(-1);

        default:
          std::cerr << e->lineno
                    << ": compilation error in field access, expected a struct"
                    << std::endl;
          PrintExp(e);
          std::cerr << std::endl;
          exit(-1);
      }
    }
    case Variable: {
      auto t = Lookup(e->lineno, env, *(e->u.variable.name), PrintErrorString);
      return TCResult(e, t, env);
    }
    case Integer:
      return TCResult(e, MakeIntTypeVal(), env);
      break;
    case Boolean:
      return TCResult(e, MakeBoolTypeVal(), env);
      break;
    case PrimitiveOp: {
      auto es = new std::vector<Expression*>();
      std::vector<Value*> ts;
      auto new_env = env;
      for (auto& argument : *e->u.primitive_op.arguments) {
        auto res = TypeCheckExp(argument, env, ct_env, nullptr, ValueContext);
        new_env = res.env;
        es->push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e = MakeOp(e->lineno, e->u.primitive_op.operator_, es);
      switch (e->u.primitive_op.operator_) {
        case Neg:
          ExpectType(e->lineno, "negation", MakeIntTypeVal(), ts[0]);
          return TCResult(new_e, MakeIntTypeVal(), new_env);
        case Add:
        case Sub:
          ExpectType(e->lineno, "subtraction(1)", MakeIntTypeVal(), ts[0]);
          ExpectType(e->lineno, "substration(2)", MakeIntTypeVal(), ts[1]);
          return TCResult(new_e, MakeIntTypeVal(), new_env);
        case And:
          ExpectType(e->lineno, "&&(1)", MakeBoolTypeVal(), ts[0]);
          ExpectType(e->lineno, "&&(2)", MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
        case Or:
          ExpectType(e->lineno, "||(1)", MakeBoolTypeVal(), ts[0]);
          ExpectType(e->lineno, "||(2)", MakeBoolTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
        case Not:
          ExpectType(e->lineno, "!", MakeBoolTypeVal(), ts[0]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
        case Eq:
          ExpectType(e->lineno, "==(1)", MakeIntTypeVal(), ts[0]);
          ExpectType(e->lineno, "==(2)", MakeIntTypeVal(), ts[1]);
          return TCResult(new_e, MakeBoolTypeVal(), new_env);
      }
      break;
    }
    case Call: {
      auto fun_res =
          TypeCheckExp(e->u.call.function, env, ct_env, nullptr, ValueContext);
      switch (fun_res.type->tag) {
        case FunctionTV: {
          auto fun_t = fun_res.type;
          auto arg_res = TypeCheckExp(e->u.call.argument, fun_res.env, ct_env,
                                      fun_t->u.fun_type.param, ValueContext);
          ExpectType(e->lineno, "call", fun_t->u.fun_type.param, arg_res.type);
          auto new_e = MakeCall(e->lineno, fun_res.exp, arg_res.exp);
          return TCResult(new_e, fun_t->u.fun_type.ret, arg_res.env);
        }
        default: {
          std::cerr << e->lineno
                    << ": compilation error in call, expected a function"
                    << std::endl;
          PrintExp(e);
          std::cerr << std::endl;
          exit(-1);
        }
      }
      break;
    }
    case FunctionT: {
      switch (context) {
        case ValueContext:
        case TypeContext: {
          auto pt = ToType(e->lineno,
                           InterpExp(ct_env, e->u.function_type.parameter));
          auto rt = ToType(e->lineno,
                           InterpExp(ct_env, e->u.function_type.return_type));
          auto new_e = MakeFunType(e->lineno, ReifyType(pt, e->lineno),
                                   ReifyType(rt, e->lineno));
          return TCResult(new_e, MakeTypeTypeVal(), env);
        }
        case PatternContext: {
          auto param_res = TypeCheckExp(e->u.function_type.parameter, env,
                                        ct_env, nullptr, context);
          auto ret_res = TypeCheckExp(e->u.function_type.return_type,
                                      param_res.env, ct_env, nullptr, context);
          auto new_e =
              MakeFunType(e->lineno, ReifyType(param_res.type, e->lineno),
                          ReifyType(ret_res.type, e->lineno));
          return TCResult(new_e, MakeTypeTypeVal(), ret_res.env);
        }
      }
    }
    case IntT:
    case BoolT:
    case TypeT:
    case AutoT:
      return TCResult(e, MakeTypeTypeVal(), env);
  }
}

auto TypecheckCase(Value* expected, Expression* pat, Statement* body,
                   TypeEnv* env, Env* ct_env, Value* ret_type)
    -> std::pair<Expression*, Statement*> {
  auto pat_res = TypeCheckExp(pat, env, ct_env, expected, PatternContext);
  auto res = TypeCheckStmt(body, pat_res.env, ct_env, ret_type);
  return std::make_pair(pat, res.stmt);
}

auto TypeCheckStmt(Statement* s, TypeEnv* env, Env* ct_env, Value* ret_type)
    -> TCStatement {
  if (!s) {
    return TCStatement(s, env);
  }
  switch (s->tag) {
    case Match: {
      auto res =
          TypeCheckExp(s->u.match_stmt.exp, env, ct_env, nullptr, ValueContext);
      auto res_type = res.type;
      auto new_clauses = new std::list<std::pair<Expression*, Statement*> >();
      for (auto& clause : *s->u.match_stmt.clauses) {
        new_clauses->push_back(TypecheckCase(
            res_type, clause.first, clause.second, env, ct_env, ret_type));
      }
      Statement* new_s = MakeMatch(s->lineno, res.exp, new_clauses);
      return TCStatement(new_s, env);
    }
    case While: {
      auto cnd_res = TypeCheckExp(s->u.while_stmt.cond, env, ct_env, nullptr,
                                  ValueContext);
      ExpectType(s->lineno, "condition of `while`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto body_res =
          TypeCheckStmt(s->u.while_stmt.body, env, ct_env, ret_type);
      auto new_s = MakeWhile(s->lineno, cnd_res.exp, body_res.stmt);
      return TCStatement(new_s, env);
    }
    case Break:
    case Continue:
      return TCStatement(s, env);
    case Block: {
      auto stmt_res = TypeCheckStmt(s->u.block.stmt, env, ct_env, ret_type);
      return TCStatement(MakeBlock(s->lineno, stmt_res.stmt), env);
    }
    case VariableDefinition: {
      auto res = TypeCheckExp(s->u.variable_definition.init, env, ct_env,
                              nullptr, ValueContext);
      Value* rhs_ty = res.type;
      auto lhs_res = TypeCheckExp(s->u.variable_definition.pat, env, ct_env,
                                  rhs_ty, PatternContext);
      Statement* new_s =
          MakeVarDef(s->lineno, s->u.variable_definition.pat, res.exp);
      return TCStatement(new_s, lhs_res.env);
    }
    case Sequence: {
      auto stmt_res = TypeCheckStmt(s->u.sequence.stmt, env, ct_env, ret_type);
      auto env2 = stmt_res.env;
      auto next_res = TypeCheckStmt(s->u.sequence.next, env2, ct_env, ret_type);
      auto env3 = next_res.env;
      return TCStatement(MakeSeq(s->lineno, stmt_res.stmt, next_res.stmt),
                         env3);
    }
    case Assign: {
      auto rhs_res =
          TypeCheckExp(s->u.assign.rhs, env, ct_env, nullptr, ValueContext);
      auto rhs_t = rhs_res.type;
      auto lhs_res =
          TypeCheckExp(s->u.assign.lhs, env, ct_env, rhs_t, ValueContext);
      auto lhs_t = lhs_res.type;
      ExpectType(s->lineno, "assign", lhs_t, rhs_t);
      auto new_s = MakeAssign(s->lineno, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.env);
    }
    case ExpressionStatement: {
      auto res = TypeCheckExp(s->u.exp, env, ct_env, nullptr, ValueContext);
      auto new_s = MakeExpStmt(s->lineno, res.exp);
      return TCStatement(new_s, env);
    }
    case If: {
      auto cnd_res =
          TypeCheckExp(s->u.if_stmt.cond, env, ct_env, nullptr, ValueContext);
      ExpectType(s->lineno, "condition of `if`", MakeBoolTypeVal(),
                 cnd_res.type);
      auto thn_res =
          TypeCheckStmt(s->u.if_stmt.then_stmt, env, ct_env, ret_type);
      auto els_res =
          TypeCheckStmt(s->u.if_stmt.else_stmt, env, ct_env, ret_type);
      auto new_s = MakeIf(s->lineno, cnd_res.exp, thn_res.stmt, els_res.stmt);
      return TCStatement(new_s, env);
    }
    case Return: {
      auto res =
          TypeCheckExp(s->u.return_stmt, env, ct_env, nullptr, ValueContext);
      if (ret_type->tag == AutoTV) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        *ret_type = *res.type;
      } else {
        ExpectType(s->lineno, "return", ret_type, res.type);
      }
      return TCStatement(MakeReturn(s->lineno, res.exp), env);
    }
  }
}

auto CheckOrEnsureReturn(Statement* stmt, bool void_return, int lineno)
    -> Statement* {
  if (!stmt) {
    if (void_return) {
      auto args = new std::vector<std::pair<std::string, Expression*> >();
      return MakeReturn(lineno, MakeTuple(lineno, args));
    } else {
      std::cerr
          << "control-flow reaches end of non-void function without a return"
          << std::endl;
      exit(-1);
    }
  }
  switch (stmt->tag) {
    case Match: {
      auto new_clauses = new std::list<std::pair<Expression*, Statement*> >();
      for (auto i = stmt->u.match_stmt.clauses->begin();
           i != stmt->u.match_stmt.clauses->end(); ++i) {
        auto s = CheckOrEnsureReturn(i->second, void_return, stmt->lineno);
        new_clauses->push_back(std::make_pair(i->first, s));
      }
      return MakeMatch(stmt->lineno, stmt->u.match_stmt.exp, new_clauses);
    }
    case Block:
      return MakeBlock(
          stmt->lineno,
          CheckOrEnsureReturn(stmt->u.block.stmt, void_return, stmt->lineno));
    case If:
      return MakeIf(stmt->lineno, stmt->u.if_stmt.cond,
                    CheckOrEnsureReturn(stmt->u.if_stmt.then_stmt, void_return,
                                        stmt->lineno),
                    CheckOrEnsureReturn(stmt->u.if_stmt.else_stmt, void_return,
                                        stmt->lineno));
    case Return:
      return stmt;
    case Sequence:
      if (stmt->u.sequence.next) {
        return MakeSeq(stmt->lineno, stmt->u.sequence.stmt,
                       CheckOrEnsureReturn(stmt->u.sequence.next, void_return,
                                           stmt->lineno));
      } else {
        return CheckOrEnsureReturn(stmt->u.sequence.stmt, void_return,
                                   stmt->lineno);
      }
    case Assign:
    case ExpressionStatement:
    case While:
    case Break:
    case Continue:
    case VariableDefinition:
      if (void_return) {
        auto args = new std::vector<std::pair<std::string, Expression*> >();
        return MakeSeq(stmt->lineno, stmt,
                       MakeReturn(stmt->lineno, MakeTuple(stmt->lineno, args)));
      } else {
        std::cerr
            << stmt->lineno
            << ": control-flow reaches end of non-void function without a "
               "return"
            << std::endl;
        exit(-1);
      }
  }
}

auto TypeCheckFunDef(struct FunctionDefinition* f, TypeEnv* env, Env* ct_env)
    -> struct FunctionDefinition* {
  auto param_res =
      TypeCheckExp(f->param_pattern, env, ct_env, nullptr, PatternContext);
  auto return_type = ToType(f->lineno, InterpExp(ct_env, f->return_type));
  if (f->name == "main") {
    ExpectType(f->lineno, "return type of `main`", MakeIntTypeVal(),
               return_type);
    // todo: check that main doesn't have any parameters
  }
  auto res = TypeCheckStmt(f->body, param_res.env, ct_env, return_type);
  bool void_return = TypeEqual(return_type, MakeVoidTypeVal());
  auto body = CheckOrEnsureReturn(res.stmt, void_return, f->lineno);
  return MakeFunDef(f->lineno, f->name, ReifyType(return_type, f->lineno),
                    f->param_pattern, body);
}

auto TypeOfFunDef(TypeEnv* env, Env* ct_env, struct FunctionDefinition* fun_def)
    -> Value* {
  auto param_res = TypeCheckExp(fun_def->param_pattern, env, ct_env, nullptr,
                                PatternContext);
  auto param_type = ToType(fun_def->lineno, param_res.type);
  auto ret = InterpExp(ct_env, fun_def->return_type);
  if (ret->tag == AutoTV) {
    auto f = TypeCheckFunDef(fun_def, env, ct_env);
    ret = InterpExp(ct_env, f->return_type);
  }
  return MakeFunTypeVal(param_type, ret);
}

auto TypeOfStructDef(struct StructDefinition* sd, TypeEnv* /*env*/, Env* ct_top)
    -> Value* {
  auto fields = new VarValues();
  auto methods = new VarValues();
  for (auto m = sd->members->begin(); m != sd->members->end(); ++m) {
    if ((*m)->tag == FieldMember) {
      auto t = ToType(sd->lineno, InterpExp(ct_top, (*m)->u.field.type));
      fields->push_back(std::make_pair(*(*m)->u.field.name, t));
    }
  }
  return MakeStructTypeVal(*sd->name, fields, methods);
}

auto NameOfDecl(Declaration* d) -> std::string {
  switch (d->tag) {
    case FunctionDeclaration:
      return d->u.fun_def->name;
    case StructDeclaration:
      return *d->u.struct_def->name;
    case ChoiceDeclaration:
      return *d->u.choice_def.name;
  }
}

auto TypeCheckDecl(Declaration* d, TypeEnv* env, Env* ct_env) -> Declaration* {
  switch (d->tag) {
    case StructDeclaration: {
      auto members = new std::list<Member*>();
      for (auto& member : *d->u.struct_def->members) {
        switch (member->tag) {
          case FieldMember: {
            // TODO interpret the type expression and store the result.
            members->push_back(member);
            break;
          }
        }
      }
      return MakeStructDecl(d->u.struct_def->lineno, *d->u.struct_def->name,
                            members);
    }
    case FunctionDeclaration:
      return MakeFunDecl(TypeCheckFunDef(d->u.fun_def, env, ct_env));
    case ChoiceDeclaration:
      return d;  // TODO
  }
}

auto TopLevel(std::list<Declaration*>* fs) -> std::pair<TypeEnv*, Env*> {
  TypeEnv* top = nullptr;
  Env* ct_top = nullptr;
  bool found_main = false;
  for (auto d : *fs) {
    if (NameOfDecl(d) == "main") {
      found_main = true;
    }
    switch (d->tag) {
      case FunctionDeclaration: {
        auto t = TypeOfFunDef(top, ct_top, d->u.fun_def);
        top = new TypeEnv(NameOfDecl(d), t, top);
        break;
      }
      case StructDeclaration: {
        auto st = TypeOfStructDef(d->u.struct_def, top, ct_top);
        Address a = AllocateValue(st);
        ct_top = new Env(NameOfDecl(d), a, ct_top);  // is this obsolete?
        auto params = MakeTupleTypeVal(st->u.struct_type.fields);
        auto fun_ty = MakeFunTypeVal(params, st);
        top = new TypeEnv(NameOfDecl(d), fun_ty, top);
        break;
      }
      case ChoiceDeclaration: {
        auto alts = new VarValues();
        for (auto i = d->u.choice_def.alternatives->begin();
             i != d->u.choice_def.alternatives->end(); ++i) {
          auto t = ToType(d->u.choice_def.lineno, InterpExp(ct_top, i->second));
          alts->push_back(std::make_pair(i->first, t));
        }
        auto ct = MakeChoiceTypeVal(d->u.choice_def.name, alts);
        Address a = AllocateValue(ct);
        ct_top = new Env(NameOfDecl(d), a, ct_top);  // is this obsolete?
        top = new TypeEnv(NameOfDecl(d), ct, top);
        break;
      }
    }  // switch (d->tag)
  }    // for
  if (found_main == false) {
    std::cerr << "error, program must contain a function named `main`"
              << std::endl;
    exit(-1);
  }
  return make_pair(top, ct_top);
}
