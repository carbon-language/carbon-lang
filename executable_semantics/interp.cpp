// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "interp.h"

#include <iostream>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

#include "typecheck.h"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

State* state;

auto PatternMatch(Value* pat, Value* val, Env*, std::list<std::string>&, int)
    -> Env*;
void HandleValue();

/***** Value Operations *****/

auto ToInteger(Value* v) -> int {
  switch (v->tag) {
    case IntV:
      return v->u.integer;
    default:
      std::cerr << "expected an integer, not ";
      PrintValue(v, cerr);
      exit(-1);
  }
}

void CheckAlive(Value* v, int lineno) {
  if (!v->alive) {
    std::cerr << lineno << ": undefined behavior: access to dead value ";
    PrintValue(v, cerr);
    std::cerr << std::endl;
    exit(-1);
  }
}

auto MakeInt_val(int i) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = IntV;
  v->u.integer = i;
  return v;
}

auto MakeBool_val(bool b) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = BoolV;
  v->u.boolean = b;
  return v;
}

auto MakeFunVal(std::string name, Value* param, Statement* body) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = FunV;
  v->u.fun.name = new std::string(std::move(name));
  v->u.fun.param = param;
  v->u.fun.body = body;
  return v;
}

auto MakePtr_val(Address addr) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = PtrV;
  v->u.ptr = addr;
  return v;
}

auto MakeStructVal(Value* type, Value* inits) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = StructV;
  v->u.struct_val.type = type;
  v->u.struct_val.inits = inits;
  return v;
}

auto MakeTuple_val(std::vector<std::pair<std::string, Address>>* elts)
    -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = TupleV;
  v->u.tuple.elts = elts;
  return v;
}

auto MakeAlt_val(std::string alt_name, std::string choice_name, Value* arg)
    -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = AltV;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  v->u.alt.arg = arg;
  return v;
}

auto MakeAlt_cons(std::string alt_name, std::string choice_name) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = AltConsV;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  return v;
}

auto MakeVarPatVal(std::string name, Value* type) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = VarPatV;
  v->u.var_pat.name = new std::string(std::move(name));
  v->u.var_pat.type = type;
  return v;
}

auto MakeVarTypeVal(std::string name) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = VarTV;
  v->u.var_type = new std::string(std::move(name));
  return v;
}

auto MakeIntTypeVal() -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = IntTV;
  return v;
}

auto MakeBoolTypeVal() -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = BoolTV;
  return v;
}

auto MakeTypeTypeVal() -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = TypeTV;
  return v;
}

auto MakeAutoType_val() -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = AutoTV;
  return v;
}

auto MakeFunTypeVal(Value* param, Value* ret) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = FunctionTV;
  v->u.fun_type.param = param;
  v->u.fun_type.ret = ret;
  return v;
}

auto MakePtrTypeVal(Value* type) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = PointerTV;
  v->u.ptr_type.type = type;
  return v;
}

auto MakeStructTypeVal(std::string name, VarValues* fields,
                         VarValues* methods) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = StructTV;
  v->u.struct_type.name = new std::string(std::move(name));
  v->u.struct_type.fields = fields;
  v->u.struct_type.methods = methods;
  return v;
}

auto MakeTupleTypeVal(VarValues* fields) -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = TupleTV;
  v->u.tuple_type.fields = fields;
  return v;
}

auto MakeVoidTypeVal() -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = TupleTV;
  v->u.tuple_type.fields = new VarValues();
  return v;
}

auto MakeChoiceTypeVal(std::string* name,
                         std::list<std::pair<std::string, Value*>>* alts)
    -> Value* {
  auto* v = new Value();
  v->alive = true;
  v->tag = ChoiceTV;
  v->u.choice_type.name = name;
  v->u.choice_type.alternatives = alts;
  return v;
}

/**** Auxiliary Functions ****/

auto AllocateValue(Value* v) -> Address {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  Address a = state->heap.size();
  state->heap.push_back(v);
  return a;
}

auto CopyVal(Value* val, int lineno) -> Value* {
  CheckAlive(val, lineno);
  switch (val->tag) {
    case TupleV: {
      auto elts = new std::vector<std::pair<std::string, Address>>();
      for (auto& i : *val->u.tuple.elts) {
        Value* elt = CopyVal(state->heap[i.second], lineno);
        elts->push_back(make_pair(i.first, AllocateValue(elt)));
      }
      return MakeTuple_val(elts);
    }
    case AltV: {
      Value* arg = CopyVal(val->u.alt.arg, lineno);
      return MakeAlt_val(*val->u.alt.alt_name, *val->u.alt.choice_name, arg);
    }
    case StructV: {
      Value* inits = CopyVal(val->u.struct_val.inits, lineno);
      return MakeStructVal(val->u.struct_val.type, inits);
    }
    case IntV:
      return MakeInt_val(val->u.integer);
    case BoolV:
      return MakeBool_val(val->u.boolean);
    case FunV:
      return MakeFunVal(*val->u.fun.name, val->u.fun.param, val->u.fun.body);
    case PtrV:
      return MakePtr_val(val->u.ptr);
    case FunctionTV:
      return MakeFunTypeVal(CopyVal(val->u.fun_type.param, lineno),
                             CopyVal(val->u.fun_type.ret, lineno));

    case PointerTV:
      return MakePtrTypeVal(CopyVal(val->u.ptr_type.type, lineno));
    case IntTV:
      return MakeIntTypeVal();
    case BoolTV:
      return MakeBoolTypeVal();
    case TypeTV:
      return MakeTypeTypeVal();
    case VarTV:
      return MakeVarTypeVal(*val->u.var_type);
    case AutoTV:
      return MakeAutoType_val();
    case TupleTV: {
      auto new_fields = new VarValues();
      for (auto& field : *val->u.tuple_type.fields) {
        auto v = CopyVal(field.second, lineno);
        new_fields->push_back(make_pair(field.first, v));
      }
      return MakeTupleTypeVal(new_fields);
    }
    case StructTV:
    case ChoiceTV:
    case VarPatV:
    case AltConsV:
      return val;  // no need to copy these because they are immutable?
      // No, they need to be copied so they don't get killed. -Jeremy
  }
}

void KillValue(Value* val) {
  val->alive = false;
  switch (val->tag) {
    case AltV:
      KillValue(val->u.alt.arg);
      break;
    case StructV:
      KillValue(val->u.struct_val.inits);
      break;
    case TupleV:
      for (auto& elt : *val->u.tuple.elts) {
        if (state->heap[elt.second]->alive) {
          KillValue(state->heap[elt.second]);
        } else {
          std::cerr << "runtime error, killing an already dead value" << std::endl;
          exit(-1);
        }
      }
      break;
    default:
      break;
  }
}

void PrintEnv(Env* env, std::ostream& out) {
  if (env) {
    std::cout << env->key << ": ";
    PrintValue(state->heap[env->value], out);
    std::cout << ", ";
    PrintEnv(env->next, out);
  }
}

void PrintValue(Value* val, std::ostream& out) {
  if (!val->alive) {
    out << "!!";
  }
  switch (val->tag) {
    case AltConsV: {
      out << *val->u.alt_cons.choice_name << "." << *val->u.alt_cons.alt_name;
      break;
    }
    case VarPatV: {
      PrintValue(val->u.var_pat.type, out);
      out << ": " << *val->u.var_pat.name;
      break;
    }
    case AltV: {
      out << "alt " << *val->u.alt.choice_name << "." << *val->u.alt.alt_name
          << " ";
      PrintValue(val->u.alt.arg, out);
      break;
    }
    case StructV: {
      out << *val->u.struct_val.type->u.struct_type.name;
      PrintValue(val->u.struct_val.inits, out);
      break;
    }
    case TupleV: {
      out << "(";
      int i = 0;
      for (auto elt = val->u.tuple.elts->begin();
           elt != val->u.tuple.elts->end(); ++elt, ++i) {
        if (i != 0) {
          out << ", ";
        }
        out << elt->first << " = ";
        PrintValue(state->heap[elt->second], out);
        out << "@" << elt->second;
      }
      out << ")";
      break;
    }
    case IntV:
      out << val->u.integer;
      break;
    case BoolV:
      out << std::boolalpha;
      out << val->u.boolean;
      break;
    case FunV:
      out << "fun<" << *val->u.fun.name << ">";
      break;
    case PtrV:
      out << "ptr<" << val->u.ptr << ">";
      break;
    case BoolTV:
      out << "Bool";
      break;
    case IntTV:
      out << "Int";
      break;
    case TypeTV:
      out << "Type";
      break;
    case AutoTV:
      out << "auto";
      break;
    case PointerTV:
      out << "Ptr(";
      PrintValue(val->u.ptr_type.type, out);
      out << ")";
      break;
    case FunctionTV:
      out << "fn ";
      PrintValue(val->u.fun_type.param, out);
      out << " -> ";
      PrintValue(val->u.fun_type.ret, out);
      break;
    case VarTV:
      out << *val->u.var_type;
      break;
    case TupleTV: {
      out << "Tuple(";
      int i = 0;
      for (auto elt = val->u.tuple_type.fields->begin();
           elt != val->u.tuple_type.fields->end(); ++elt, ++i) {
        if (i != 0) {
          out << ", ";
        }
        out << elt->first << " = ";
        PrintValue(elt->second, out);
      }
      out << ")";
      break;
    }
    case StructTV:
      out << "struct " << *val->u.struct_type.name;
      break;
    case ChoiceTV:
      out << "choice " << *val->u.choice_type.name;
      break;
  }
}

/***** Action Operations *****/

void PrintAct(Action* act, std::ostream& out) {
  switch (act->tag) {
    case DeleteTmpAction:
      std::cout << "delete_tmp(" << act->u.delete_ << ")";
      break;
    case ExpToLValAction:
      out << "exp=>lval";
      break;
    case LValAction:
    case ExpressionAction:
      PrintExp(act->u.exp);
      break;
    case StatementAction:
      PrintStatement(act->u.stmt, 1);
      break;
    case ValAction:
      PrintValue(act->u.val, out);
      break;
  }
  out << "<" << act->pos << ">";
  if (act->results.size() > 0) {
    out << "(";
    for (auto& result : act->results) {
      if (result) {
        PrintValue(result, out);
      }
      out << ",";
    }
    out << ")";
  }
}

void PrintActList(Cons<Action*>* ls, std::ostream& out) {
  if (ls) {
    PrintAct(ls->curr, out);
    if (ls->next) {
      out << " :: ";
      PrintActList(ls->next, out);
    }
  }
}

auto MakeExpAct(Expression* e) -> Action* {
  auto* act = new Action();
  act->tag = ExpressionAction;
  act->u.exp = e;
  act->pos = -1;
  return act;
}

auto MakeLvalAct(Expression* e) -> Action* {
  auto* act = new Action();
  act->tag = LValAction;
  act->u.exp = e;
  act->pos = -1;
  return act;
}

auto MakeStmtAct(Statement* s) -> Action* {
  auto* act = new Action();
  act->tag = StatementAction;
  act->u.stmt = s;
  act->pos = -1;
  return act;
}

auto MakeValAct(Value* v) -> Action* {
  auto* act = new Action();
  act->tag = ValAction;
  act->u.val = v;
  act->pos = -1;
  return act;
}

auto MakeExpToLvalAct() -> Action* {
  auto* act = new Action();
  act->tag = ExpToLValAction;
  act->pos = -1;
  return act;
}

auto MakeDeleteAct(Address a) -> Action* {
  auto* act = new Action();
  act->tag = DeleteTmpAction;
  act->pos = -1;
  act->u.delete_ = a;
  return act;
}

/***** Frame and State Operations *****/

void PrintFrame(Frame* frame, std::ostream& out) {
  out << frame->name;
  out << "{";
  PrintActList(frame->todo, out);
  out << "}";
}

void PrintStack(Cons<Frame*>* ls, std::ostream& out) {
  if (ls) {
    PrintFrame(ls->curr, out);
    if (ls->next) {
      out << " :: ";
      PrintStack(ls->next, out);
    }
  }
}

void PrintHeap(std::vector<Value*>& heap, std::ostream& out) {
  for (auto& iter : heap) {
    if (iter) {
      PrintValue(iter, out);
    } else {
      out << "_";
    }
    out << ", ";
  }
}

auto CurrentEnv(State* state) -> Env* {
  Frame* frame = state->stack->curr;
  return frame->scopes->curr->env;
}

void PrintState(std::ostream& out) {
  out << "{" << std::endl;
  out << "stack: ";
  PrintStack(state->stack, out);
  out << std::endl << "heap: ";
  PrintHeap(state->heap, out);
  out << std::endl << "env: ";
  PrintEnv(CurrentEnv(state), out);
  out << std::endl << "}" << std::endl;
}

/***** Auxiliary Functions *****/

auto ValToInt(Value* v, int lineno) -> int {
  CheckAlive(v, lineno);
  switch (v->tag) {
    case IntV:
      return v->u.integer;
    default:
      std::cerr << lineno << ": runtime error: expected an integer" << std::endl;
      exit(-1);
  }
}

auto ValToBool(Value* v, int lineno) -> int {
  CheckAlive(v, lineno);
  switch (v->tag) {
    case BoolV:
      return v->u.boolean;
    default:
      std::cerr << "runtime type error: expected a Boolean" << std::endl;
      exit(-1);
  }
}

auto ValToPtr(Value* v, int lineno) -> Address {
  CheckAlive(v, lineno);
  switch (v->tag) {
    case PtrV:
      return v->u.ptr;
    default:
      std::cerr << "runtime type error: expected a pointer, not ";
      PrintValue(v, cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

auto FieldsValueEqual(VarValues* ts1, VarValues* ts2, int lineno) -> bool {
  if (ts1->size() == ts2->size()) {
    for (auto& iter1 : *ts1) {
      try {
        auto t2 = FindAlist(iter1.first, ts2);
        if (!ValueEqual(iter1.second, t2, lineno)) {
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

auto ValueEqual(Value* v1, Value* v2, int lineno) -> bool {
  CheckAlive(v1, lineno);
  CheckAlive(v2, lineno);
  return (v1->tag == IntV && v2->tag == IntV &&
          v1->u.integer == v2->u.integer) ||
         (v1->tag == BoolV && v2->tag == BoolV &&
          v1->u.boolean == v2->u.boolean) ||
         (v1->tag == PtrV && v2->tag == PtrV && v1->u.ptr == v2->u.ptr) ||
         (v1->tag == FunV && v2->tag == FunV &&
          v1->u.fun.body == v2->u.fun.body) ||
         (v1->tag == TupleV && v2->tag == TupleV &&
          FieldsValueEqual(v1->u.tuple_type.fields, v2->u.tuple_type.fields,
                           lineno))
         // TODO: struct and alternative values
         || TypeEqual(v1, v2);
}

auto EvalPrim(Operator op, const std::vector<Value*>& args, int lineno)
    -> Value* {
  switch (op) {
    case Neg:
      return MakeInt_val(-ValToInt(args[0], lineno));
    case Add:
      return MakeInt_val(ValToInt(args[0], lineno) + ValToInt(args[1], lineno));
    case Sub:
      return MakeInt_val(ValToInt(args[0], lineno) - ValToInt(args[1], lineno));
    case Not:
      return MakeBool_val(!ValToBool(args[0], lineno));
    case And:
      return MakeBool_val(ValToBool(args[0], lineno) &&
                          ValToBool(args[1], lineno));
    case Or:
      return MakeBool_val(ValToBool(args[0], lineno) ||
                          ValToBool(args[1], lineno));
    case Eq:
      return MakeBool_val(ValueEqual(args[0], args[1], lineno));
  }
}

Env* globals;

void InitGlobals(std::list<Declaration*>* fs) {
  globals = nullptr;
  for (auto& iter : *fs) {
    switch (iter->tag) {
      case ChoiceDeclaration: {
        auto d = iter;
        auto alts = new VarValues();
        for (auto i = d->u.choice_def.alternatives->begin();
             i != d->u.choice_def.alternatives->end(); ++i) {
          auto t =
              ToType(d->u.choice_def.lineno, InterpExp(nullptr, i->second));
          alts->push_back(make_pair(i->first, t));
        }
        auto ct = MakeChoiceTypeVal(d->u.choice_def.name, alts);
        auto a = AllocateValue(ct);
        globals = new Env(*d->u.choice_def.name, a, globals);
        break;
      }
      case StructDeclaration: {
        auto d = iter;
        auto fields = new VarValues();
        auto methods = new VarValues();
        for (auto i = d->u.struct_def->members->begin();
             i != d->u.struct_def->members->end(); ++i) {
          switch ((*i)->tag) {
            case FieldMember: {
              auto t = ToType(d->u.struct_def->lineno,
                              InterpExp(nullptr, (*i)->u.field.type));
              fields->push_back(make_pair(*(*i)->u.field.name, t));
              break;
            }
          }
        }
        auto st = MakeStructTypeVal(*d->u.struct_def->name, fields, methods);
        auto a = AllocateValue(st);
        globals = new Env(*d->u.struct_def->name, a, globals);
        break;
      }
      case FunctionDeclaration: {
        struct FunctionDefinition* fun = iter->u.fun_def;
        Env* env = nullptr;
        auto pt = InterpExp(env, fun->param_pattern);
        auto f = MakeFunVal(fun->name, pt, fun->body);
        Address a = AllocateValue(f);
        globals = new Env(fun->name, a, globals);
        break;
      }
    }
  }
}

//    { S, H} -> { { C, E, F} :: S, H}
// where C is the body of the function,
//       E is the environment (functions + parameters + locals)
//       F is the function
void CallFunction(int lineno, std::vector<Value*> operas, State* state) {
  CheckAlive(operas[0], lineno);
  switch (operas[0]->tag) {
    case FunV: {
      Env* env = globals;
      // Bind arguments to parameters
      std::list<std::string> params;
      env =
          PatternMatch(operas[0]->u.fun.param, operas[1], env, params, lineno);
      if (!env) {
        std::cerr << "internal error in call_function, pattern match failed" << std::endl;
        exit(-1);
      }
      // Create the new frame and push it on the stack
      auto* scope = new Scope(env, params);
      auto* frame = new Frame(*operas[0]->u.fun.name,
                              MakeCons(scope, (Cons<Scope*>*)nullptr),
                              MakeCons(MakeStmtAct(operas[0]->u.fun.body),
                                       (Cons<Action*>*)nullptr));
      state->stack = MakeCons(frame, state->stack);
      break;
    }
    case StructTV: {
      Value* arg = CopyVal(operas[1], lineno);
      Value* sv = MakeStructVal(operas[0], arg);
      Frame* frame = state->stack->curr;
      frame->todo = MakeCons(MakeValAct(sv), frame->todo);
      break;
    }
    case AltConsV: {
      Value* arg = CopyVal(operas[1], lineno);
      Value* av = MakeAlt_val(*operas[0]->u.alt_cons.alt_name,
                              *operas[0]->u.alt_cons.choice_name, arg);
      Frame* frame = state->stack->curr;
      frame->todo = MakeCons(MakeValAct(av), frame->todo);
      break;
    }
    default:
      std::cerr << lineno << ": in call, expected a function, not ";
      PrintValue(operas[0], cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

void KillScope(int lineno, Scope* scope) {
  for (const auto& l : scope->locals) {
    Address a = Lookup(lineno, scope->env, l, PrintErrorString);
    KillValue(state->heap[a]);
  }
}

void KillLocals(int lineno, Frame* frame) {
  Cons<Scope*>* scopes = frame->scopes;
  for (Scope* scope = scopes->curr; scopes; scopes = scopes->next) {
    KillScope(lineno, scope);
  }
}

void CreateTuple(Frame* frame, Action* act, Expression* /*exp*/) {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  auto elts = new std::vector<std::pair<std::string, Address>>();
  auto f = act->u.exp->u.tuple.fields->begin();
  for (auto i = act->results.begin(); i != act->results.end(); ++i, ++f) {
    Address a = AllocateValue(*i);  // copy?
    elts->push_back(make_pair(f->first, a));
  }
  Value* tv = MakeTuple_val(elts);
  frame->todo = MakeCons(MakeValAct(tv), frame->todo->next);
}

auto ToValue(Expression* value) -> Value* {
  switch (value->tag) {
    case Integer:
      return MakeInt_val(value->u.integer);
    case Boolean:
      return MakeBool_val(value->u.boolean);
    case IntT:
      return MakeIntTypeVal();
    case BoolT:
      return MakeBoolTypeVal();
    case TypeT:
      return MakeTypeTypeVal();
    case FunctionT:
      // instead add to patterns?
    default:
      std::cerr << "internal error in to_value, didn't expect ";
      PrintExp(value);
      std::cerr << std::endl;
      exit(-1);
  }
}

//
// Returns 0 if the value doesn't match the pattern
//
auto PatternMatch(Value* p, Value* v, Env* env, std::list<std::string>& vars,
                  int lineno) -> Env* {
  std::cout << "pattern_match(";
  PrintValue(p, cout);
  std::cout << ", ";
  PrintValue(v, cout);
  std::cout << ")" << std::endl;
  switch (p->tag) {
    case VarPatV: {
      Address a = AllocateValue(CopyVal(v, lineno));
      vars.push_back(*p->u.var_pat.name);
      return new Env(*p->u.var_pat.name, a, env);
    }
    case TupleV:
      switch (v->tag) {
        case TupleV: {
          if (p->u.tuple.elts->size() != v->u.tuple.elts->size()) {
            std::cerr << "runtime error: arity mismatch in tuple pattern match"
                 << std::endl;
            exit(-1);
          }
          for (auto& elt : *p->u.tuple.elts) {
            Address a = FindField(elt.first, v->u.tuple.elts);
            env = PatternMatch(state->heap[elt.second], state->heap[a], env,
                               vars, lineno);
          }
          return env;
        }
        default:
          std::cerr << "internal error, expected a tuple value in pattern, not ";
          PrintValue(v, cerr);
          std::cerr << std::endl;
          exit(-1);
      }
    case AltV:
      switch (v->tag) {
        case AltV: {
          if (*p->u.alt.choice_name != *v->u.alt.choice_name ||
              *p->u.alt.alt_name != *v->u.alt.alt_name) {
            return nullptr;
          }
          env = PatternMatch(p->u.alt.arg, v->u.alt.arg, env, vars, lineno);
          return env;
        }
        default:
          std::cerr << "internal error, expected a choice alternative in pattern, "
                  "not ";
          PrintValue(v, cerr);
          std::cerr << std::endl;
          exit(-1);
      }
    case FunctionTV:
      switch (v->tag) {
        case FunctionTV:
          env = PatternMatch(p->u.fun_type.param, v->u.fun_type.param, env,
                             vars, lineno);
          env = PatternMatch(p->u.fun_type.ret, v->u.fun_type.ret, env, vars,
                             lineno);
          return env;
        default:
          return nullptr;
      }
    default:
      if (ValueEqual(p, v, lineno)) {
        return env;
      } else {
        return nullptr;
      }
  }
}

void PatternAssignment(Value* pat, Value* val, int lineno) {
  switch (pat->tag) {
    case PtrV:
      state->heap[ValToPtr(pat, lineno)] = val;
      break;
    case TupleV: {
      switch (val->tag) {
        case TupleV: {
          if (pat->u.tuple.elts->size() != val->u.tuple.elts->size()) {
            std::cerr << "runtime error: arity mismatch in tuple pattern match"
                 << std::endl;
            exit(-1);
          }
          for (auto& elt : *pat->u.tuple.elts) {
            Address a = FindField(elt.first, val->u.tuple.elts);
            PatternAssignment(state->heap[elt.second], state->heap[a], lineno);
          }
          break;
        }
        default:
          std::cerr << "internal error, expected a tuple value on right-hand-side, "
                  "not ";
          PrintValue(val, cerr);
          std::cerr << std::endl;
          exit(-1);
      }
      break;
    }
    case AltV: {
      switch (val->tag) {
        case AltV: {
          if (*pat->u.alt.choice_name != *val->u.alt.choice_name ||
              *pat->u.alt.alt_name != *val->u.alt.alt_name) {
            std::cerr << "internal error in pattern assignment" << std::endl;
            exit(-1);
          }
          PatternAssignment(pat->u.alt.arg, val->u.alt.arg, lineno);
          break;
        }
        default:
          std::cerr << "internal error, expected an alternative in left-hand-side, "
                  "not ";
          PrintValue(val, cerr);
          std::cerr << std::endl;
          exit(-1);
      }
      break;
    }
    default:
      if (!ValueEqual(pat, val, lineno)) {
        std::cerr << "internal error in pattern assignment" << std::endl;
        exit(-1);
      }
  }
}

/***** state transitions for lvalues *****/

void StepLvalue() {
  Frame* frame = state->stack->curr;
  Action* act = frame->todo->curr;
  Expression* exp = act->u.exp;
  std::cout << "--- step lvalue ";
  PrintExp(exp);
  std::cout << " --->" << std::endl;
  switch (exp->tag) {
    case Variable: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      Address a = Lookup(exp->lineno, CurrentEnv(state),
                         *(exp->u.variable.name), PrintErrorString);
      Value* v = MakePtr_val(a);
      CheckAlive(v, exp->lineno);
      frame->todo = MakeCons(MakeValAct(v), frame->todo->next);
      break;
    }
    case GetField: {
      //    { {e.f :: C, E, F} :: S, H}
      // -> { e :: [].f :: C, E, F} :: S, H}
      frame->todo =
          MakeCons(MakeLvalAct(exp->u.get_field.aggregate), frame->todo);
      act->pos++;
      break;
    }
    case Index: {
      //    { {e[i] :: C, E, F} :: S, H}
      // -> { e :: [][i] :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(exp->u.index.aggregate), frame->todo);
      act->pos++;
      break;
    }
    case Tuple: {
      //    { {(f1=e1,...) :: C, E, F} :: S, H}
      // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
      Expression* e1 = (*exp->u.tuple.fields)[0].second;
      frame->todo = MakeCons(MakeLvalAct(e1), frame->todo);
      act->pos++;
      break;
    }
    case Integer:
    case Boolean:
    case Call:
    case PrimitiveOp:
    case IntT:
    case BoolT:
    case TypeT:
    case FunctionT:
    case AutoT:
    case PatternVariable: {
      frame->todo = MakeCons(MakeExpAct(exp),
                             MakeCons(MakeExpToLvalAct(), frame->todo->next));
    }
  }
}

/***** state transitions for expressions *****/

void StepExp() {
  Frame* frame = state->stack->curr;
  Action* act = frame->todo->curr;
  Expression* exp = act->u.exp;
  std::cout << "--- step exp ";
  PrintExp(exp);
  std::cout << " --->" << std::endl;
  switch (exp->tag) {
    case PatternVariable: {
      frame->todo =
          MakeCons(MakeExpAct(exp->u.pattern_variable.type), frame->todo);
      act->pos++;
      break;
    }
    case Index: {
      //    { { e[i] :: C, E, F} :: S, H}
      // -> { { e :: [][i] :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(exp->u.index.aggregate), frame->todo);
      act->pos++;
      break;
    }
    case Tuple: {
      if (exp->u.tuple.fields->size() > 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        Expression* e1 = (*exp->u.tuple.fields)[0].second;
        frame->todo = MakeCons(MakeExpAct(e1), frame->todo);
        act->pos++;
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
    case GetField: {
      //    { { e.f :: C, E, F} :: S, H}
      // -> { { e :: [].f :: C, E, F} :: S, H}
      frame->todo =
          MakeCons(MakeLvalAct(exp->u.get_field.aggregate), frame->todo);
      act->pos++;
      break;
    }
    case Variable: {
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address a = Lookup(exp->lineno, CurrentEnv(state),
                         *(exp->u.variable.name), PrintErrorString);
      Value* v = state->heap[a];
      frame->todo = MakeCons(MakeValAct(v), frame->todo->next);
      break;
    }
    case Integer:
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo =
          MakeCons(MakeValAct(MakeInt_val(exp->u.integer)), frame->todo->next);
      break;
    case Boolean:
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo =
          MakeCons(MakeValAct(MakeBool_val(exp->u.boolean)), frame->todo->next);
      break;
    case PrimitiveOp:
      if (exp->u.primitive_op.arguments->size() > 0) {
        //    { {op(e :: es) :: C, E, F} :: S, H}
        // -> { e :: op([] :: es) :: C, E, F} :: S, H}
        frame->todo = MakeCons(
            MakeExpAct(exp->u.primitive_op.arguments->front()), frame->todo);
        act->pos++;
      } else {
        //    { {v :: op(]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, ()) :: C, E, F} :: S, H}
        Value* v =
            EvalPrim(exp->u.primitive_op.operator_, act->results, exp->lineno);
        frame->todo = MakeCons(MakeValAct(v), frame->todo->next->next);
      }
      break;
    case Call:
      //    { {e1(e2) :: C, E, F} :: S, H}
      // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(exp->u.call.function), frame->todo);
      act->pos++;
      break;
    case IntT: {
      Value* v = MakeIntTypeVal();
      frame->todo = MakeCons(MakeValAct(v), frame->todo->next);
      break;
    }
    case BoolT: {
      Value* v = MakeBoolTypeVal();
      frame->todo = MakeCons(MakeValAct(v), frame->todo->next);
      break;
    }
    case AutoT: {
      Value* v = MakeAutoType_val();
      frame->todo = MakeCons(MakeValAct(v), frame->todo->next);
      break;
    }
    case TypeT: {
      Value* v = MakeTypeTypeVal();
      frame->todo = MakeCons(MakeValAct(v), frame->todo->next);
      break;
    }
    case FunctionT: {
      frame->todo =
          MakeCons(MakeExpAct(exp->u.function_type.parameter), frame->todo);
      act->pos++;
      break;
    }
  }  // switch (exp->tag)
}

/***** state transitions for statements *****/

auto IsWhileAct(Action* act) -> bool {
  switch (act->tag) {
    case StatementAction:
      switch (act->u.stmt->tag) {
        case While:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

auto IsBlockAct(Action* act) -> bool {
  switch (act->tag) {
    case StatementAction:
      switch (act->u.stmt->tag) {
        case Block:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

void StepStmt() {
  Frame* frame = state->stack->curr;
  Action* act = frame->todo->curr;
  Statement* stmt = act->u.stmt;
  std::cout << "--- step stmt ";
  PrintStatement(stmt, 1);
  std::cout << " --->" << std::endl;
  switch (stmt->tag) {
    case Match:
      //    { { (match (e) ...) :: C, E, F} :: S, H}
      // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(stmt->u.match_stmt.exp), frame->todo);
      act->pos++;
      break;
    case While:
      //    { { (while (e) s) :: C, E, F} :: S, H}
      // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(stmt->u.while_stmt.cond), frame->todo);
      act->pos++;
      break;
    case Break:
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      frame->todo = frame->todo->next;
      while (frame->todo && !IsWhileAct(frame->todo->curr)) {
        if (IsBlockAct(frame->todo->curr)) {
          KillScope(stmt->lineno, frame->scopes->curr);
          frame->scopes = frame->scopes->next;
        }
        frame->todo = frame->todo->next;
      }
      frame->todo = frame->todo->next;
      break;
    case Continue:
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      frame->todo = frame->todo->next;
      while (frame->todo && !IsWhileAct(frame->todo->curr)) {
        if (IsBlockAct(frame->todo->curr)) {
          KillScope(stmt->lineno, frame->scopes->curr);
          frame->scopes = frame->scopes->next;
        }
        frame->todo = frame->todo->next;
      }
      break;
    case Block: {
      if (act->pos == -1) {
        auto* scope = new Scope(CurrentEnv(state), std::list<std::string>());
        frame->scopes = MakeCons(scope, frame->scopes);
        frame->todo = MakeCons(MakeStmtAct(stmt->u.block.stmt), frame->todo);
        act->pos++;
      } else {
        Scope* scope = frame->scopes->curr;
        KillScope(stmt->lineno, scope);
        frame->scopes = frame->scopes->next;
        frame->todo = frame->todo->next;
      }
      break;
    }
    case VariableDefinition:
      //    { {(var x = e) :: C, E, F} :: S, H}
      // -> { {e :: (var x = []) :: C, E, F} :: S, H}
      frame->todo =
          MakeCons(MakeExpAct(stmt->u.variable_definition.init), frame->todo);
      act->pos++;
      break;
    case ExpressionStatement:
      //    { {e :: C, E, F} :: S, H}
      // -> { {e :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(stmt->u.exp), frame->todo);
      break;
    case Assign:
      //    { {(lv = e) :: C, E, F} :: S, H}
      // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeLvalAct(stmt->u.assign.lhs), frame->todo);
      act->pos++;
      break;
    case If:
      //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
      // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(stmt->u.if_stmt.cond), frame->todo);
      act->pos++;
      break;
    case Return:
      //    { {return e :: C, E, F} :: S, H}
      // -> { {e :: return [] :: C, E, F} :: S, H}
      frame->todo = MakeCons(MakeExpAct(stmt->u.return_stmt), frame->todo);
      act->pos++;
      break;
    case Sequence:
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      Cons<Action*>* todo = frame->todo->next;
      if (stmt->u.sequence.next) {
        todo = MakeCons(MakeStmtAct(stmt->u.sequence.next), todo);
      }
      frame->todo = MakeCons(MakeStmtAct(stmt->u.sequence.stmt), todo);
      break;
  }
}

auto GetMember(Address a, const std::string& f) -> Address {
  std::vector<std::pair<std::string, Address>>* fields;
  Value* v = state->heap[a];
  switch (v->tag) {
    case StructV:
      fields = v->u.struct_val.inits->u.tuple.elts;
      try {
        return FindField(f, fields);
      } catch (std::domain_error de) {
        std::cerr << "runtime error, member " << f << " not in ";
        PrintValue(v, cerr);
        std::cerr << std::endl;
        exit(-1);
      }
      break;
    case TupleV:
      fields = v->u.tuple.elts;
      try {
        return FindField(f, fields);
      } catch (std::domain_error de) {
        std::cerr << "field " << f << " not in ";
        PrintValue(v, cerr);
        std::cerr << std::endl;
        exit(-1);
      }
      break;
    case ChoiceTV: {
      try {
        FindAlist(f, v->u.choice_type.alternatives);
        auto ac = MakeAlt_cons(f, *v->u.choice_type.name);
        return AllocateValue(ac);
      } catch (std::domain_error de) {
        std::cerr << "alternative " << f << " not in ";
        PrintValue(v, cerr);
        std::cerr << std::endl;
        exit(-1);
      }
      break;
    }
    default:
      std::cerr << "field access not allowed for value ";
      PrintValue(v, cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

auto InsertDelete(Action* del, Cons<Action*>* todo) -> Cons<Action*>* {
  if (todo) {
    switch (todo->curr->tag) {
      case StatementAction: {
        // This places the delete before the enclosing statement.
        // Not sure if that is OK. Conceptually it should go after
        // but that is tricky for some statements, like 'return'. -Jeremy
        return MakeCons(del, todo);
      }
      case LValAction:
      case ExpressionAction:
      case ValAction:
      case ExpToLValAction:
      case DeleteTmpAction:
        return MakeCons(todo->curr, InsertDelete(del, todo->next));
    }
  } else {
    return MakeCons(del, todo);
  }
}

/***** State transition for handling a value *****/

void HandleValue() {
  Frame* frame = state->stack->curr;
  Action* val_act = frame->todo->curr;
  Action* act = frame->todo->next->curr;
  act->results.push_back(val_act->u.val);
  act->pos++;

  std::cout << "--- handle value ";
  PrintValue(val_act->u.val, cout);
  std::cout << " with ";
  PrintAct(act, cout);
  std::cout << " --->" << std::endl;

  switch (act->tag) {
    case DeleteTmpAction: {
      KillValue(state->heap[act->u.delete_]);
      frame->todo = MakeCons(val_act, frame->todo->next->next);
      break;
    }
    case ExpToLValAction: {
      Address a = AllocateValue(act->results[0]);
      auto del = MakeDeleteAct(a);
      frame->todo = MakeCons(MakeValAct(MakePtr_val(a)),
                             InsertDelete(del, frame->todo->next->next));
      break;
    }
    case LValAction: {
      Expression* exp = act->u.exp;
      switch (exp->tag) {
        case GetField: {
          //    { v :: [].f :: C, E, F} :: S, H}
          // -> { { &v.f :: C, E, F} :: S, H }
          Value* str = act->results[0];
          try {
            Address a =
                GetMember(ValToPtr(str, exp->lineno), *exp->u.get_field.field);
            frame->todo =
                MakeCons(MakeValAct(MakePtr_val(a)), frame->todo->next->next);
          } catch (std::domain_error de) {
            std::cerr << "field " << *exp->u.get_field.field << " not in ";
            PrintValue(str, cerr);
            std::cerr << std::endl;
            exit(-1);
          }
          break;
        }
        case Index: {
          if (act->pos == 1) {
            frame->todo =
                MakeCons(MakeExpAct(exp->u.index.offset), frame->todo->next);
          } else if (act->pos == 2) {
            //    { v :: [][i] :: C, E, F} :: S, H}
            // -> { { &v[i] :: C, E, F} :: S, H }
            Value* tuple = act->results[0];
            std::string f = std::to_string(ToInteger(act->results[1]));
            try {
              Address a = FindField(f, tuple->u.tuple.elts);
              frame->todo =
                  MakeCons(MakeValAct(MakePtr_val(a)), frame->todo->next->next);
            } catch (std::domain_error de) {
              std::cerr << "runtime error: field " << f << "not in ";
              PrintValue(tuple, cerr);
              std::cerr << std::endl;
              exit(-1);
            }
          }
          break;
        }
        case Tuple: {
          if (act->pos != static_cast<int>(exp->u.tuple.fields->size())) {
            //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
            //    H}
            // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
            // H}
            Expression* elt = (*exp->u.tuple.fields)[act->pos].second;
            frame->todo = MakeCons(MakeLvalAct(elt), frame->todo->next);
          } else {
            frame->todo = frame->todo->next;
            CreateTuple(frame, act, exp);
          }
          break;
        }
        default:
          std::cerr << "internal error in handle_value, LValAction" << std::endl;
          exit(-1);
      }
      break;
    }
    case ExpressionAction: {
      Expression* exp = act->u.exp;
      switch (exp->tag) {
        case PatternVariable: {
          auto v =
              MakeVarPatVal(*exp->u.pattern_variable.name, act->results[0]);
          frame->todo = MakeCons(MakeValAct(v), frame->todo->next->next);
          break;
        }
        case Tuple: {
          if (act->pos != static_cast<int>(exp->u.tuple.fields->size())) {
            //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
            //    H}
            // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
            // H}
            Expression* elt = (*exp->u.tuple.fields)[act->pos].second;
            frame->todo = MakeCons(MakeExpAct(elt), frame->todo->next);
          } else {
            frame->todo = frame->todo->next;
            CreateTuple(frame, act, exp);
          }
          break;
        }
        case Index: {
          if (act->pos == 1) {
            frame->todo =
                MakeCons(MakeExpAct(exp->u.index.offset), frame->todo->next);
          } else if (act->pos == 2) {
            auto tuple = act->results[0];
            ;
            switch (tuple->tag) {
              case TupleV: {
                //    { { v :: [][i] :: C, E, F} :: S, H}
                // -> { { v_i :: C, E, F} : S, H}
                std::string f = std::to_string(ToInteger(act->results[1]));
                try {
                  auto a = FindField(f, tuple->u.tuple.elts);
                  frame->todo = MakeCons(MakeValAct(state->heap[a]),
                                         frame->todo->next->next);
                } catch (std::domain_error de) {
                  std::cerr << "runtime error, field " << f << " not in ";
                  PrintValue(tuple, cerr);
                  std::cerr << std::endl;
                  exit(-1);
                }
                break;
              }
              default:
                std::cerr << "runtime type error, expected a tuple in field access, "
                        "not ";
                PrintValue(tuple, cerr);
                exit(-1);
            }
          }
          break;
        }
        case GetField: {
          //    { { v :: [].f :: C, E, F} :: S, H}
          // -> { { v_f :: C, E, F} : S, H}
          auto a = GetMember(ValToPtr(act->results[0], exp->lineno),
                             *exp->u.get_field.field);
          frame->todo =
              MakeCons(MakeValAct(state->heap[a]), frame->todo->next->next);
          break;
        }
        case PrimitiveOp: {
          if (act->pos !=
              static_cast<int>(exp->u.primitive_op.arguments->size())) {
            //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
            // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
            Expression* arg = (*exp->u.primitive_op.arguments)[act->pos];
            frame->todo = MakeCons(MakeExpAct(arg), frame->todo->next);
          } else {
            //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
            // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
            Value* v = EvalPrim(exp->u.primitive_op.operator_, act->results,
                                exp->lineno);
            frame->todo = MakeCons(MakeValAct(v), frame->todo->next->next);
          }
          break;
        }
        case Call: {
          if (act->pos == 1) {
            //    { { v :: [](e) :: C, E, F} :: S, H}
            // -> { { e :: v([]) :: C, E, F} :: S, H}
            frame->todo =
                MakeCons(MakeExpAct(exp->u.call.argument), frame->todo->next);
          } else if (act->pos == 2) {
            //    { { v2 :: v1([]) :: C, E, F} :: S, H}
            // -> { {C',E',F'} :: {C, E, F} :: S, H}
            frame->todo = frame->todo->next->next;
            CallFunction(exp->lineno, act->results, state);
          } else {
            std::cerr << "internal error in handle_value with Call" << std::endl;
            exit(-1);
          }
          break;
        }
        case FunctionT: {
          if (act->pos == 2) {
            //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
            // -> { fn pt -> rt :: {C, E, F} :: S, H}
            Value* v = MakeFunTypeVal(act->results[0], act->results[1]);
            frame->todo = MakeCons(MakeValAct(v), frame->todo->next->next);
          } else {
            //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
            // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
            frame->todo = MakeCons(MakeExpAct(exp->u.function_type.return_type),
                                   frame->todo->next);
          }
          break;
        }
        case Variable:
        case Integer:
        case Boolean:
        case IntT:
        case BoolT:
        case TypeT:
        case AutoT:
          std::cerr << "internal error, bad expression context in handle_value"
               << std::endl;
          exit(-1);
      }
      break;
    }
    case StatementAction: {
      Statement* stmt = act->u.stmt;
      switch (stmt->tag) {
        case ExpressionStatement:
          frame->todo = frame->todo->next->next;
          break;
        case VariableDefinition: {
          if (act->pos == 1) {
            frame->todo = MakeCons(MakeExpAct(stmt->u.variable_definition.pat),
                                   frame->todo->next);
          } else if (act->pos == 2) {
            //    { { v :: (x = []) :: C, E, F} :: S, H}
            // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
            Value* v = act->results[0];
            Value* p = act->results[1];
            // Address a = AllocateValue(CopyVal(v));
            frame->scopes->curr->env =
                PatternMatch(p, v, frame->scopes->curr->env,
                             frame->scopes->curr->locals, stmt->lineno);
            if (!frame->scopes->curr->env) {
              std::cerr << stmt->lineno
                   << ": internal error in variable definition, match failed"
                   << std::endl;
              exit(-1);
            }
            frame->todo = frame->todo->next->next;
          }
          break;
        }
        case Assign:
          if (act->pos == 1) {
            //    { { a :: ([] = e) :: C, E, F} :: S, H}
            // -> { { e :: (a = []) :: C, E, F} :: S, H}
            frame->todo =
                MakeCons(MakeExpAct(stmt->u.assign.rhs), frame->todo->next);
          } else if (act->pos == 2) {
            //    { { v :: (a = []) :: C, E, F} :: S, H}
            // -> { { C, E, F} :: S, H(a := v)}
            auto pat = act->results[0];
            auto val = act->results[1];
            PatternAssignment(pat, val, stmt->lineno);
            frame->todo = frame->todo->next->next;
          }
          break;
        case If:
          if (ValToBool(act->results[0], stmt->lineno)) {
            //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
            //      S, H}
            // -> { { then_stmt :: C, E, F } :: S, H}
            frame->todo = MakeCons(MakeStmtAct(stmt->u.if_stmt.then_stmt),
                                   frame->todo->next->next);
          } else {
            //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
            //      S, H}
            // -> { { else_stmt :: C, E, F } :: S, H}
            frame->todo = MakeCons(MakeStmtAct(stmt->u.if_stmt.else_stmt),
                                   frame->todo->next->next);
          }
          break;
        case While:
          if (ValToBool(act->results[0], stmt->lineno)) {
            //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
            // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
            frame->todo->next->curr->pos = -1;
            frame->todo->next->curr->results.clear();
            frame->todo = MakeCons(MakeStmtAct(stmt->u.while_stmt.body),
                                   frame->todo->next);
          } else {
            //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
            // -> { { C, E, F } :: S, H}
            frame->todo->next->curr->pos = -1;
            frame->todo->next->curr->results.clear();
            frame->todo = frame->todo->next->next;
          }
          break;
        case Match: {
          /*
            Regarding act->pos:
            * odd: start interpreting the pattern of a clause
            * even: finished interpreting the pattern, now try to match

            Regarding act->results:
            * 0: the value that we're matching
            * 1: the pattern for clause 0
            * 2: the pattern for clause 1
            * ...
          */
          auto clause_num = (act->pos - 1) / 2;
          if (clause_num >=
              static_cast<int>(stmt->u.match_stmt.clauses->size())) {
            frame->todo = frame->todo->next->next;
            break;
          }
          auto c = stmt->u.match_stmt.clauses->begin();
          std::advance(c, clause_num);

          if (act->pos % 2 == 1) {
            // start interpreting the pattern of the clause
            //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
            // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
            frame->todo = MakeCons(MakeExpAct(c->first), frame->todo->next);
          } else {  // try to match
            auto v = act->results[0];
            auto pat = act->results[clause_num + 1];
            auto env = CurrentEnv(state);
            std::list<std::string> vars;
            Env* new_env = PatternMatch(pat, v, env, vars, stmt->lineno);
            if (new_env) {  // we have a match, start the body
              auto* new_scope = new Scope(new_env, vars);
              frame->scopes = MakeCons(new_scope, frame->scopes);
              Statement* body_block = MakeBlock(stmt->lineno, c->second);
              Action* body_act = MakeStmtAct(body_block);
              body_act->pos = 0;
              frame->todo =
                  MakeCons(MakeStmtAct(c->second),
                           MakeCons(body_act, frame->todo->next->next));
            } else {
              act->pos++;
              clause_num = (act->pos - 1) / 2;
              if (clause_num <
                  static_cast<int>(stmt->u.match_stmt.clauses->size())) {
                // move on to the next clause
                c = stmt->u.match_stmt.clauses->begin();
                std::advance(c, clause_num);
                frame->todo = MakeCons(MakeExpAct(c->first), frame->todo->next);
              } else {  // No more clauses in match
                frame->todo = frame->todo->next->next;
              }
            }
          }
          break;
        }
        case Return: {
          //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
          // -> { {v :: C', E', F'} :: S, H}
          Value* ret_val = CopyVal(val_act->u.val, stmt->lineno);
          KillLocals(stmt->lineno, frame);
          state->stack = state->stack->next;
          frame = state->stack->curr;
          frame->todo = MakeCons(MakeValAct(ret_val), frame->todo);
          break;
        }
        case Block:
        case Sequence:
        case Break:
        case Continue:
          std::cerr << "internal error in handle_value, unhandled statement ";
          PrintStatement(stmt, 1);
          std::cerr << std::endl;
          exit(-1);
      }  // switch stmt
      break;
    }
    case ValAction:
      std::cerr << "internal error, ValAction in handle_value" << std::endl;
      exit(-1);
  }  // switch act
}

/***** state transition *****/

void Step() {
  Frame* frame = state->stack->curr;
  if (!frame->todo) {
    std::cerr << "runtime error: fell off end of function " << frame->name
         << " without `return`" << std::endl;
    exit(-1);
  }

  Action* act = frame->todo->curr;
  switch (act->tag) {
    case DeleteTmpAction:
      std::cerr << "internal error in step, did not expect DeleteTmpAction" << std::endl;
      break;
    case ExpToLValAction:
      std::cerr << "internal error in step, did not expect ExpToLValAction" << std::endl;
      break;
    case ValAction:
      HandleValue();
      break;
    case LValAction:
      StepLvalue();
      break;
    case ExpressionAction:
      StepExp();
      break;
    case StatementAction:
      StepStmt();
      break;
  }  // switch
}

/***** interpret the whole program *****/

auto InterpProgram(std::list<Declaration*>* fs) -> int {
  state = new State();  // runtime state
  std::cout << "********** initializing globals **********" << std::endl;
  InitGlobals(fs);

  Expression* arg =
      MakeTuple(0, new std::vector<std::pair<std::string, Expression*>>());
  Expression* call_main = MakeCall(0, MakeVar(0, "main"), arg);
  Cons<Action*>* todo =
      MakeCons(MakeExpAct(call_main), (Cons<Action*>*)nullptr);
  auto* scope = new Scope(globals, std::list<std::string>());
  auto* frame = new Frame("top", MakeCons(scope, (Cons<Scope*>*)nullptr), todo);
  state->stack = MakeCons(frame, (Cons<Frame*>*)nullptr);

  std::cout << "********** calling main function **********" << std::endl;
  PrintState(cout);

  while (Length(state->stack) > 1 || Length(state->stack->curr->todo) > 1 ||
         state->stack->curr->todo->curr->tag != ValAction) {
    Step();
    PrintState(cout);
  }
  Value* v = state->stack->curr->todo->curr->u.val;
  return ValToInt(v, 0);
}

/***** interpret an expression (at compile-time) *****/

auto InterpExp(Env* env, Expression* e) -> Value* {
  Cons<Action*>* todo = MakeCons(MakeExpAct(e), (Cons<Action*>*)nullptr);
  auto* scope = new Scope(env, std::list<std::string>());
  auto* frame =
      new Frame("InterpExp", MakeCons(scope, (Cons<Scope*>*)nullptr), todo);
  state->stack = MakeCons(frame, (Cons<Frame*>*)nullptr);

  while (Length(state->stack) > 1 || Length(state->stack->curr->todo) > 1 ||
         state->stack->curr->todo->curr->tag != ValAction) {
    Step();
  }
  Value* v = state->stack->curr->todo->curr->u.val;
  return v;
}
