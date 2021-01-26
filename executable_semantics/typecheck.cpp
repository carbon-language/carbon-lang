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
using std::cerr;
using std::cout;
using std::endl;
using std::make_pair;
using std::map;
using std::set;
using std::vector;

template <class T>
bool list_equal(std::list<T*>* ts1, std::list<T*>* ts2, bool (*eq)(T*, T*)) {
  if (ts1->size() == ts2->size()) {
    auto iter2 = ts2->begin();
    for (auto iter1 = ts1->begin(); iter1 != ts1->end(); ++iter1, ++iter2) {
      if (!eq(*iter1, *iter2))
        return false;
    }
    return true;
  } else {
    return false;
  }
}

template <class T>
bool vector_equal(std::vector<T*>* ts1, std::vector<T*>* ts2,
                  bool (*eq)(T*, T*)) {
  if (ts1->size() == ts2->size()) {
    auto iter2 = ts2->begin();
    for (auto iter1 = ts1->begin(); iter1 != ts1->end(); ++iter1, ++iter2) {
      if (!eq(*iter1, *iter2))
        return false;
    }
    return true;
  } else {
    return false;
  }
}

bool FieldsEqual(VarValues* ts1, VarValues* ts2) {
  if (ts1->size() == ts2->size()) {
    for (auto iter1 = ts1->begin(); iter1 != ts1->end(); ++iter1) {
      try {
        auto t2 = FindAlist(iter1->first, ts2);
        if (!TypeEqual(iter1->second, t2))
          return false;
      } catch (std::domain_error de) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

int find(const std::string& s, Cons<std::string>* ls, int n) {
  if (ls) {
    if (ls->curr == s)
      return n;
    else
      return find(s, ls->next, n + 1);
  } else {
    cerr << "could not find " << s << endl;
    exit(-1);
  }
}

bool TypeEqual(Value* t1, Value* t2) {
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

void expect_type(int lineno, std::string context, Value* expected,
                 Value* actual) {
  if (!TypeEqual(expected, actual)) {
    cerr << lineno << ": type error in " << context << endl;
    cerr << "expected: ";
    print_value(expected, cerr);
    cerr << endl << "actual: ";
    print_value(actual, cerr);
    cerr << endl;
    exit(-1);
  }
}

void PrintErrorString(std::string s) { cerr << s; }

void print_type_env(TypeEnv* env, std::ostream& out) {
  if (env) {
    out << env->key << ": ";
    print_value(env->value, out);
    out << ", ";
    print_type_env(env->next, out);
  }
}

void match_types(int lineno, Value* param, Value* arg,
                 map<std::string, Value*>& tyvar_map) {
  switch (param->tag) {
    case TupleTV:
      if (param->u.tuple_type.fields->size() !=
          arg->u.tuple_type.fields->size()) {
        expect_type(lineno, "arity mismatch", param, arg);
        exit(-1);
      }
      for (auto i = param->u.tuple_type.fields->begin();
           i != param->u.tuple_type.fields->end(); ++i) {
        try {
          auto argT = FindAlist(i->first, arg->u.tuple_type.fields);
          match_types(lineno, i->second, argT, tyvar_map);
        } catch (std::domain_error de) {
          cerr << "missing field " << i->first << endl;
          exit(-1);
        }
      }
      break;
    case ChoiceTV:
      expect_type(lineno, "function call", param, arg);
      break;
    case StructTV:
      expect_type(lineno, "function call", param, arg);
      break;
    case TypeTV:
      expect_type(lineno, "function call", param, arg);
      break;
    case VarTV:
      if (tyvar_map.count(*param->u.var_type) == 0) {
        tyvar_map[*param->u.var_type] = arg;
      } else {
        expect_type(lineno, "call to generic function",
                    tyvar_map[*param->u.var_type], arg);
      }
      break;
    case BoolTV:
      expect_type(lineno, "function call", param, arg);
      break;
    case IntTV:
      expect_type(lineno, "function call", param, arg);
      break;
    case PointerTV:
      switch (arg->tag) {
        case PointerTV:
          match_types(lineno, param->u.ptr_type.type, arg->u.ptr_type.type,
                      tyvar_map);
          break;
        default:
          cerr << "expected argument to be a pointer" << endl;
          expect_type(lineno, "function call", param, arg);
          exit(-1);
      }
      break;
    case FunctionTV:
      switch (arg->tag) {
        case FunctionTV:
          match_types(lineno, param->u.fun_type.param, arg->u.fun_type.param,
                      tyvar_map);
          match_types(lineno, param->u.fun_type.ret, arg->u.fun_type.ret,
                      tyvar_map);
          break;
        default:
          cerr << "expected argument to be a function" << endl;
          expect_type(lineno, "function call", param, arg);
          exit(-1);
      }
      break;
    default:
      cerr << "in match_type, expected a type, not ";
      print_value(param, cerr);
      cerr << endl;
      exit(-1);
  }
}

Value* subst_type(Value* t, map<std::string, Value*>& tyvar_map) {
  switch (t->tag) {
    case ChoiceTV:
      return t;  // update when choices get type parameters
    case StructTV:
      return t;  // update when structs get type parameters
    case TypeTV:
      return t;
    case VarTV:
      return tyvar_map[*t->u.var_type];
    case BoolTV:
      return t;
    case IntTV:
      return t;
    case PointerTV:
      return MakePtr_type_val(subst_type(t->u.ptr_type.type, tyvar_map));
    case FunctionTV: {
      return MakeFunType_val(subst_type(t->u.fun_type.param, tyvar_map),
                             subst_type(t->u.fun_type.ret, tyvar_map));
    }
    case TupleTV: {
      auto fields = new VarValues();
      for (auto i = t->u.tuple_type.fields->begin();
           i != t->u.tuple_type.fields->end(); ++i) {
        fields->push_back(
            make_pair(i->first, subst_type(i->second, tyvar_map)));
      }
      return MakeTuple_type_val(fields);
    }
    default:
      cerr << "in subst_type, expected a type " << endl;
      exit(-1);
  }
}

// Convert tuples to tuple types.
Value* ToType(int lineno, Value* val) {
  switch (val->tag) {
    case TupleV: {
      auto fields = new VarValues();
      for (auto i = val->u.tuple.elts->begin(); i != val->u.tuple.elts->end();
           ++i) {
        Value* ty = ToType(lineno, state->heap[i->second]);
        fields->push_back(make_pair(i->first, ty));
      }
      return MakeTuple_type_val(fields);
    }
    case TupleTV: {
      auto fields = new VarValues();
      for (auto i = val->u.tuple_type.fields->begin();
           i != val->u.tuple_type.fields->end(); ++i) {
        Value* ty = ToType(lineno, i->second);
        fields->push_back(make_pair(i->first, ty));
      }
      return MakeTuple_type_val(fields);
    }
    case PointerTV: {
      return MakePtr_type_val(ToType(lineno, val->u.ptr_type.type));
    }
    case FunctionTV: {
      return MakeFunType_val(ToType(lineno, val->u.fun_type.param),
                             ToType(lineno, val->u.fun_type.ret));
    }
    case VarPatV: {
      return MakeVarPat_val(*val->u.var_pat.name,
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
      cerr << lineno << ": in ToType, expected a type, not ";
      print_value(val, cerr);
      cerr << endl;
      exit(-1);
  }
}

// Reify type to type expression.
Expression* reify_type(Value* t, int lineno) {
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
      return MakeFunType(0, reify_type(t->u.fun_type.param, lineno),
                         reify_type(t->u.fun_type.ret, lineno));
    case TupleTV: {
      auto args = new std::vector<std::pair<std::string, Expression*> >();
      for (auto i = t->u.tuple_type.fields->begin();
           i != t->u.tuple_type.fields->end(); ++i) {
        args->push_back(make_pair(i->first, reify_type(i->second, lineno)));
      }
      return MakeTuple(0, args);
    }
    case StructTV:
      return MakeVar(0, *t->u.struct_type.name);
    case ChoiceTV:
      return MakeVar(0, *t->u.choice_type.name);
    default:
      cerr << lineno << ": expected a type, not ";
      print_value(t, cerr);
      cerr << endl;
      exit(-1);
  }
}

TCResult TypeCheckExp(
    Expression* e, TypeEnv* env, Env* ct_env, Value* expected,
    TCContext context) {  //                   expected can be null
  switch (e->tag) {
    case PatternVariable: {
      if (context != PatternContext) {
        cerr << e->lineno
             << ": compilation error, pattern variables are only allowed in "
                "pattern context"
             << endl;
      }
      auto t =
          ToType(e->lineno, interp_exp(ct_env, e->u.pattern_variable.type));
      if (t->tag == AutoTV) {
        if (expected == 0) {
          cerr << e->lineno << ": compilation error, auto not allowed here"
               << endl;
          exit(-1);
        } else {
          t = expected;
        }
      }
      auto new_e = MakeVarPat(e->lineno, *e->u.pattern_variable.name,
                              reify_type(t, e->lineno));
      return TCResult(new_e, t,
                      new TypeEnv(*e->u.pattern_variable.name, t, env));
    }
    case Index: {
      auto res =
          TypeCheckExp(e->u.get_field.aggregate, env, ct_env, 0, ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case TupleTV: {
          auto i = to_integer(interp_exp(ct_env, e->u.index.offset));
          std::string f = std::to_string(i);
          try {
            auto fieldT = FindAlist(f, t->u.tuple_type.fields);
            auto new_e = MakeIndex(e->lineno, res.exp, MakeInt(e->lineno, i));
            return TCResult(new_e, fieldT, res.env);
          } catch (std::domain_error de) {
            cerr << e->lineno << ": compilation error, field " << f
                 << " is not in the tuple ";
            print_value(t, cerr);
            cerr << endl;
          }
        }
        default:
          cerr << e->lineno << ": compilation error, expected a tuple" << endl;
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
        Value* arg_expected = 0;
        if (expected && expected->tag == TupleTV) {
          try {
            arg_expected = FindAlist(arg->first, expected->u.tuple_type.fields);
          } catch (std::domain_error de) {
            cerr << e->lineno << ": compilation error, missing field "
                 << arg->first << endl;
            exit(-1);
          }
        }
        auto arg_res =
            TypeCheckExp(arg->second, new_env, ct_env, arg_expected, context);
        new_env = arg_res.env;
        new_args->push_back(make_pair(arg->first, arg_res.exp));
        arg_types->push_back(make_pair(arg->first, arg_res.type));
      }
      auto tupleE = MakeTuple(e->lineno, new_args);
      auto tupleT = MakeTuple_type_val(arg_types);
      return TCResult(tupleE, tupleT, new_env);
    }
    case GetField: {
      auto res =
          TypeCheckExp(e->u.get_field.aggregate, env, ct_env, 0, ValueContext);
      auto t = res.type;
      switch (t->tag) {
        case StructTV:
          // Search for a field
          for (auto vt = t->u.struct_type.fields->begin();
               vt != t->u.struct_type.fields->end(); ++vt) {
            if (*e->u.get_field.field == vt->first) {
              Expression* new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              return TCResult(new_e, vt->second, res.env);
            }
          }
          // Search for a method
          for (auto vt = t->u.struct_type.methods->begin();
               vt != t->u.struct_type.methods->end(); ++vt) {
            if (*e->u.get_field.field == vt->first) {
              Expression* new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              return TCResult(new_e, vt->second, res.env);
            }
          }
          cerr << e->lineno << ": compilation error, struct "
               << *t->u.struct_type.name << " does not have a field named "
               << *e->u.get_field.field << endl;
          exit(-1);
        case TupleTV:
          for (auto vt = t->u.tuple_type.fields->begin();
               vt != t->u.tuple_type.fields->end(); ++vt) {
            if (*e->u.get_field.field == vt->first) {
              auto new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              return TCResult(new_e, vt->second, res.env);
            }
          }
          cerr << e->lineno << ": compilation error, struct "
               << *t->u.struct_type.name << " does not have a field named "
               << *e->u.get_field.field << endl;
          exit(-1);
        case ChoiceTV:
          for (auto vt = t->u.choice_type.alternatives->begin();
               vt != t->u.choice_type.alternatives->end(); ++vt) {
            if (*e->u.get_field.field == vt->first) {
              Expression* new_e =
                  MakeGetField(e->lineno, res.exp, *e->u.get_field.field);
              auto fun_ty = MakeFunType_val(vt->second, t);
              return TCResult(new_e, fun_ty, res.env);
            }
          }
          cerr << e->lineno << ": compilation error, struct "
               << *t->u.struct_type.name << " does not have a field named "
               << *e->u.get_field.field << endl;
          exit(-1);

        default:
          cerr << e->lineno
               << ": compilation error in field access, expected a struct"
               << endl;
          PrintExp(e);
          cerr << endl;
          exit(-1);
      }
    }
    case Variable: {
      auto t = Lookup(e->lineno, env, *(e->u.variable.name), PrintErrorString);
      return TCResult(e, t, env);
    }
    case Integer:
      return TCResult(e, MakeIntType_val(), env);
      break;
    case Boolean:
      return TCResult(e, MakeBoolType_val(), env);
      break;
    case PrimitiveOp: {
      auto es = new std::vector<Expression*>();
      std::vector<Value*> ts;
      auto new_env = env;
      for (auto iter = e->u.primitive_op.arguments->begin();
           iter != e->u.primitive_op.arguments->end(); ++iter) {
        auto res = TypeCheckExp(*iter, env, ct_env, 0, ValueContext);
        new_env = res.env;
        es->push_back(res.exp);
        ts.push_back(res.type);
      }
      auto new_e = MakeOp(e->lineno, e->u.primitive_op.operator_, es);
      switch (e->u.primitive_op.operator_) {
        case Neg:
          expect_type(e->lineno, "negation", MakeIntType_val(), ts[0]);
          return TCResult(new_e, MakeIntType_val(), new_env);
        case Add:
        case Sub:
          expect_type(e->lineno, "subtraction(1)", MakeIntType_val(), ts[0]);
          expect_type(e->lineno, "substration(2)", MakeIntType_val(), ts[1]);
          return TCResult(new_e, MakeIntType_val(), new_env);
        case And:
          expect_type(e->lineno, "&&(1)", MakeBoolType_val(), ts[0]);
          expect_type(e->lineno, "&&(2)", MakeBoolType_val(), ts[1]);
          return TCResult(new_e, MakeBoolType_val(), new_env);
        case Or:
          expect_type(e->lineno, "||(1)", MakeBoolType_val(), ts[0]);
          expect_type(e->lineno, "||(2)", MakeBoolType_val(), ts[1]);
          return TCResult(new_e, MakeBoolType_val(), new_env);
        case Not:
          expect_type(e->lineno, "!", MakeBoolType_val(), ts[0]);
          return TCResult(new_e, MakeBoolType_val(), new_env);
        case Eq:
          expect_type(e->lineno, "==(1)", MakeIntType_val(), ts[0]);
          expect_type(e->lineno, "==(2)", MakeIntType_val(), ts[1]);
          return TCResult(new_e, MakeBoolType_val(), new_env);
      }
      break;
    }
    case Call: {
      auto fun_res =
          TypeCheckExp(e->u.call.function, env, ct_env, 0, ValueContext);
      switch (fun_res.type->tag) {
        case FunctionTV: {
          auto funT = fun_res.type;
          auto arg_res = TypeCheckExp(e->u.call.argument, fun_res.env, ct_env,
                                      funT->u.fun_type.param, ValueContext);
          expect_type(e->lineno, "call", funT->u.fun_type.param, arg_res.type);
          auto new_e = MakeCall(e->lineno, fun_res.exp, arg_res.exp);
          return TCResult(new_e, funT->u.fun_type.ret, arg_res.env);
        }
        default: {
          cerr << e->lineno
               << ": compilation error in call, expected a function" << endl;
          PrintExp(e);
          cerr << endl;
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
                           interp_exp(ct_env, e->u.function_type.parameter));
          auto rt = ToType(e->lineno,
                           interp_exp(ct_env, e->u.function_type.return_type));
          auto new_e = MakeFunType(e->lineno, reify_type(pt, e->lineno),
                                   reify_type(rt, e->lineno));
          return TCResult(new_e, MakeTypeType_val(), env);
        }
        case PatternContext: {
          auto param_res = TypeCheckExp(e->u.function_type.parameter, env,
                                        ct_env, 0, context);
          auto ret_res = TypeCheckExp(e->u.function_type.return_type,
                                      param_res.env, ct_env, 0, context);
          auto new_e =
              MakeFunType(e->lineno, reify_type(param_res.type, e->lineno),
                          reify_type(ret_res.type, e->lineno));
          return TCResult(new_e, MakeTypeType_val(), ret_res.env);
        }
      }
    }
    case IntT:
    case BoolT:
    case TypeT:
    case AutoT:
      return TCResult(e, MakeTypeType_val(), env);
  }
}

std::pair<Expression*, Statement*> typecheck_case(Value* expected,
                                                  Expression* pat,
                                                  Statement* body, TypeEnv* env,
                                                  Env* ct_env,
                                                  Value* ret_type) {
  auto pat_res = TypeCheckExp(pat, env, ct_env, expected, PatternContext);
  auto res = TypeCheckStmt(body, pat_res.env, ct_env, ret_type);
  return make_pair(pat, res.stmt);
}

TCStatement TypeCheckStmt(Statement* s, TypeEnv* env, Env* ct_env,
                          Value* ret_type) {
  if (!s) {
    return TCStatement(s, env);
  }
  switch (s->tag) {
    case Match: {
      auto res =
          TypeCheckExp(s->u.match_stmt.exp, env, ct_env, 0, ValueContext);
      auto res_type = res.type;
      auto new_clauses = new std::list<std::pair<Expression*, Statement*> >();
      for (auto i = s->u.match_stmt.clauses->begin();
           i != s->u.match_stmt.clauses->end(); ++i) {
        new_clauses->push_back(typecheck_case(res_type, i->first, i->second,
                                              env, ct_env, ret_type));
      }
      Statement* new_s = MakeMatch(s->lineno, res.exp, new_clauses);
      return TCStatement(new_s, env);
    }
    case While: {
      auto cnd_res =
          TypeCheckExp(s->u.while_stmt.cond, env, ct_env, 0, ValueContext);
      expect_type(s->lineno, "condition of `while`", MakeBoolType_val(),
                  cnd_res.type);
      auto body_res =
          TypeCheckStmt(s->u.while_stmt.body, env, ct_env, ret_type);
      auto new_s = MakeWhile(s->lineno, cnd_res.exp, body_res.stmt);
      return TCStatement(new_s, env);
    }
    case Break:
      return TCStatement(s, env);
    case Continue:
      return TCStatement(s, env);
    case Block: {
      auto stmt_res = TypeCheckStmt(s->u.block.stmt, env, ct_env, ret_type);
      return TCStatement(MakeBlock(s->lineno, stmt_res.stmt), env);
    }
    case VariableDefinition: {
      auto res = TypeCheckExp(s->u.variable_definition.init, env, ct_env, 0,
                              ValueContext);
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
          TypeCheckExp(s->u.assign.rhs, env, ct_env, 0, ValueContext);
      auto rhsT = rhs_res.type;
      auto lhs_res =
          TypeCheckExp(s->u.assign.lhs, env, ct_env, rhsT, ValueContext);
      auto lhsT = lhs_res.type;
      expect_type(s->lineno, "assign", lhsT, rhsT);
      auto new_s = MakeAssign(s->lineno, lhs_res.exp, rhs_res.exp);
      return TCStatement(new_s, lhs_res.env);
    }
    case ExpressionStatement: {
      auto res = TypeCheckExp(s->u.exp, env, ct_env, 0, ValueContext);
      auto new_s = MakeExpStmt(s->lineno, res.exp);
      return TCStatement(new_s, env);
    }
    case If: {
      auto cnd_res =
          TypeCheckExp(s->u.if_stmt.cond, env, ct_env, 0, ValueContext);
      expect_type(s->lineno, "condition of `if`", MakeBoolType_val(),
                  cnd_res.type);
      auto thn_res =
          TypeCheckStmt(s->u.if_stmt.then_stmt, env, ct_env, ret_type);
      auto els_res =
          TypeCheckStmt(s->u.if_stmt.else_stmt, env, ct_env, ret_type);
      auto new_s = MakeIf(s->lineno, cnd_res.exp, thn_res.stmt, els_res.stmt);
      return TCStatement(new_s, env);
    }
    case Return: {
      auto res = TypeCheckExp(s->u.return_stmt, env, ct_env, 0, ValueContext);
      if (ret_type->tag == AutoTV) {
        // The following infers the return type from the first 'return'
        // statement. This will get more difficult with subtyping, when we
        // should infer the least-upper bound of all the 'return' statements.
        *ret_type = *res.type;
      } else {
        expect_type(s->lineno, "return", ret_type, res.type);
      }
      return TCStatement(MakeReturn(s->lineno, res.exp), env);
    }
  }
}

Statement* check_or_ensure_return(Statement* stmt, bool void_return,
                                  int lineno) {
  if (!stmt) {
    if (void_return) {
      auto args = new std::vector<std::pair<std::string, Expression*> >();
      return MakeReturn(lineno, MakeTuple(lineno, args));
    } else {
      cerr << "control-flow reaches end of non-void function without a return"
           << endl;
      exit(-1);
    }
  }
  switch (stmt->tag) {
    case Match: {
      auto new_clauses = new std::list<std::pair<Expression*, Statement*> >();
      for (auto i = stmt->u.match_stmt.clauses->begin();
           i != stmt->u.match_stmt.clauses->end(); ++i) {
        auto s = check_or_ensure_return(i->second, void_return, stmt->lineno);
        new_clauses->push_back(make_pair(i->first, s));
      }
      return MakeMatch(stmt->lineno, stmt->u.match_stmt.exp, new_clauses);
    }
    case Block:
      return MakeBlock(stmt->lineno,
                       check_or_ensure_return(stmt->u.block.stmt, void_return,
                                              stmt->lineno));
    case If:
      return MakeIf(stmt->lineno, stmt->u.if_stmt.cond,
                    check_or_ensure_return(stmt->u.if_stmt.then_stmt,
                                           void_return, stmt->lineno),
                    check_or_ensure_return(stmt->u.if_stmt.else_stmt,
                                           void_return, stmt->lineno));
    case Return:
      return stmt;
    case Sequence:
      if (stmt->u.sequence.next) {
        return MakeSeq(stmt->lineno, stmt->u.sequence.stmt,
                       check_or_ensure_return(stmt->u.sequence.next,
                                              void_return, stmt->lineno));
      } else {
        return check_or_ensure_return(stmt->u.sequence.stmt, void_return,
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
        cerr << stmt->lineno
             << ": control-flow reaches end of non-void function without a "
                "return"
             << endl;
        exit(-1);
      }
  }
}

struct FunctionDefinition* TypeCheckFunDef(struct FunctionDefinition* f,
                                           TypeEnv* env, Env* ct_env) {
  auto param_res =
      TypeCheckExp(f->param_pattern, env, ct_env, 0, PatternContext);
  auto return_type = ToType(f->lineno, interp_exp(ct_env, f->return_type));
  if (f->name == "main") {
    expect_type(f->lineno, "return type of `main`", MakeIntType_val(),
                return_type);
    // todo: check that main doesn't have any parameters
  }
  auto res = TypeCheckStmt(f->body, param_res.env, ct_env, return_type);
  bool void_return = TypeEqual(return_type, MakeVoid_type_val());
  auto body = check_or_ensure_return(res.stmt, void_return, f->lineno);
  return MakeFunDef(f->lineno, f->name, reify_type(return_type, f->lineno),
                    f->param_pattern, body);
}

Value* type_of_fun_def(TypeEnv* env, Env* ct_env,
                       struct FunctionDefinition* fun_def) {
  auto param_res =
      TypeCheckExp(fun_def->param_pattern, env, ct_env, 0, PatternContext);
  auto param_type = ToType(fun_def->lineno, param_res.type);
  auto ret = interp_exp(ct_env, fun_def->return_type);
  if (ret->tag == AutoTV) {
    auto f = TypeCheckFunDef(fun_def, env, ct_env);
    ret = interp_exp(ct_env, f->return_type);
  }
  return MakeFunType_val(param_type, ret);
}

Value* type_of_struct_def(struct StructDefinition* sd, TypeEnv* env,
                          Env* ct_top) {
  auto fields = new VarValues();
  auto methods = new VarValues();
  for (auto m = sd->members->begin(); m != sd->members->end(); ++m) {
    if ((*m)->tag == FieldMember) {
      auto t = ToType(sd->lineno, interp_exp(ct_top, (*m)->u.field.type));
      fields->push_back(make_pair(*(*m)->u.field.name, t));
    }
  }
  return MakeStruct_type_val(*sd->name, fields, methods);
}

std::string name_of_decl(Declaration* d) {
  switch (d->tag) {
    case FunctionDeclaration:
      return d->u.fun_def->name;
    case StructDeclaration:
      return *d->u.struct_def->name;
    case ChoiceDeclaration:
      return *d->u.choice_def.name;
  }
}

Declaration* TypeCheckDecl(Declaration* d, TypeEnv* env, Env* ct_env) {
  switch (d->tag) {
    case StructDeclaration: {
      auto members = new std::list<Member*>();
      for (auto m = d->u.struct_def->members->begin();
           m != d->u.struct_def->members->end(); ++m) {
        switch ((*m)->tag) {
          case FieldMember: {
            // TODO interpret the type expression and store the result.
            members->push_back(*m);
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

std::pair<TypeEnv*, Env*> TopLevel(std::list<Declaration*>* fs) {
  TypeEnv* top = 0;
  Env* ct_top = 0;
  bool found_main = false;
  for (auto i = fs->begin(); i != fs->end(); ++i) {
    auto d = *i;
    if (name_of_decl(d) == "main") {
      found_main = true;
    }
    switch (d->tag) {
      case FunctionDeclaration: {
        auto t = type_of_fun_def(top, ct_top, d->u.fun_def);
        top = new TypeEnv(name_of_decl(d), t, top);
        break;
      }
      case StructDeclaration: {
        auto st = type_of_struct_def(d->u.struct_def, top, ct_top);
        Address a = allocate_value(st);
        ct_top = new Env(name_of_decl(d), a, ct_top);  // is this obsolete?
        auto params = MakeTuple_type_val(st->u.struct_type.fields);
        auto fun_ty = MakeFunType_val(params, st);
        top = new TypeEnv(name_of_decl(d), fun_ty, top);
        break;
      }
      case ChoiceDeclaration: {
        auto alts = new VarValues();
        for (auto i = d->u.choice_def.alternatives->begin();
             i != d->u.choice_def.alternatives->end(); ++i) {
          auto t =
              ToType(d->u.choice_def.lineno, interp_exp(ct_top, i->second));
          alts->push_back(make_pair(i->first, t));
        }
        auto ct = MakeChoice_type_val(d->u.choice_def.name, alts);
        Address a = allocate_value(ct);
        ct_top = new Env(name_of_decl(d), a, ct_top);  // is this obsolete?
        top = new TypeEnv(name_of_decl(d), ct, top);
        break;
      }
    }  // switch (d->tag)
  }    // for
  if (found_main == false) {
    cerr << "error, program must contain a function named `main`" << endl;
    exit(-1);
  }
  return make_pair(top, ct_top);
}
