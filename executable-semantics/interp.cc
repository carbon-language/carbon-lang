#include <vector>
#include <map>
#include <iostream>
#include <iterator>
#include "interp.h"
#include "typecheck.h"

using std::vector;
using std::map;
using std::cout;
using std::cerr;
using std::endl;

State* state;

Env* pattern_match(Value* pat, Value* val, Env*, list<string>&, int);
void handle_value();

/***** Value Operations *****/

int to_integer(Value* v) {
  switch (v->tag) {
  case IntV:
    return v->u.integer;
  default:
    cerr << "expected an integer, not ";
    print_value(v, cerr);
    exit(-1);
  }
}

void check_alive(Value* v, int lineno) {
  if (! v->alive) {
    cerr << lineno << ": undefined behavior: access to dead value ";
    print_value(v, cerr);
    cerr << endl;
    exit(-1);
  }
}

Value* make_int_val(int i) {
  Value* v = new Value();
  v->alive = true;
  v->tag = IntV;
  v->u.integer = i;
  return v;
}

Value* make_bool_val(bool b) {
  Value* v = new Value();
  v->alive = true;
  v->tag = BoolV;
  v->u.boolean = b;
  return v;
}

Value* make_fun_val(string name, Value* param, Statement* body) {
  Value* v = new Value();
  v->alive = true;
  v->tag = FunV;
  v->u.fun.name = new string(name);
  v->u.fun.param = param;
  v->u.fun.body = body;
  return v;
}

Value* make_ptr_val(address addr) {
  Value* v = new Value();
  v->alive = true;
  v->tag = PtrV;
  v->u.ptr = addr;
  return v;
}

Value* make_struct_val(Value* type, Value* inits) {
  Value* v = new Value();
  v->alive = true;
  v->tag = StructV;
  v->u.struct_val.type = type;
  v->u.struct_val.inits = inits;
  return v;
}

Value* make_tuple_val(vector<pair<string,address> >* elts) {
  Value* v = new Value();
  v->alive = true;
  v->tag = TupleV;
  v->u.tuple.elts = elts;
  return v;
}

Value* make_alt_val(string alt_name, string choice_name, Value* arg) {
  Value* v = new Value();
  v->alive = true;
  v->tag = AltV;
  v->u.alt.alt_name = new string(alt_name);
  v->u.alt.choice_name = new string(choice_name);
  v->u.alt.arg = arg;
  return v;
}

Value* make_alt_cons(string alt_name, string choice_name) {
  Value* v = new Value();
  v->alive = true;
  v->tag = AltConsV;
  v->u.alt.alt_name = new string(alt_name);
  v->u.alt.choice_name = new string(choice_name);
  return v;
}

Value* make_var_pat_val(string name, Value* type) {
  Value* v = new Value();
  v->alive = true;
  v->tag = VarPatV;
  v->u.var_pat.name = new string(name);
  v->u.var_pat.type = type;
  return v;
}

Value* make_var_type_val(string name) {
  Value* v = new Value();
  v->alive = true;
  v->tag = VarTV;
  v->u.var_type = new string(name);
  return v;
}

Value* make_int_type_val() {
  Value* v = new Value();
  v->alive = true;
  v->tag = IntTV;
  return v;
}

Value* make_bool_type_val() {
  Value* v = new Value();
  v->alive = true;
  v->tag = BoolTV;
  return v;
}

Value* make_type_type_val() {
  Value* v = new Value();
  v->alive = true;
  v->tag = TypeTV;
  return v;
}

Value* make_auto_type_val() {
  Value* v = new Value();
  v->alive = true;
  v->tag = AutoTV;
  return v;
}

Value* make_fun_type_val(Value* param, Value* ret) {
  Value* v = new Value();
  v->alive = true;
  v->tag = FunctionTV;
  v->u.fun_type.param = param;
  v->u.fun_type.ret = ret;
  return v;
}

Value* make_ptr_type_val(Value* type) {
  Value* v = new Value();
  v->alive = true;
  v->tag = PointerTV;
  v->u.ptr_type.type = type;
  return v;
}

Value* make_struct_type_val(string name, VarValues* fields, VarValues* methods){
  Value* v = new Value();
  v->alive = true;
  v->tag = StructTV;
  v->u.struct_type.name = new string(name);
  v->u.struct_type.fields = fields;
  v->u.struct_type.methods = methods;
  return v;
}

Value* make_tuple_type_val(VarValues* fields) {
  Value* v = new Value();
  v->alive = true;
  v->tag = TupleTV;
  v->u.tuple_type.fields = fields;
  return v;
}

Value* make_void_type_val() {
  Value* v = new Value();
  v->alive = true;
  v->tag = TupleTV;
  v->u.tuple_type.fields = new VarValues();
  return v;
}


Value* make_choice_type_val(string* name, list<pair<string, Value* > >* alts){
  Value* v = new Value();
  v->alive = true;
  v->tag = ChoiceTV;
  v->u.choice_type.name = name;
  v->u.choice_type.alternatives = alts;
  return v;
}

/**** Auxiliary Functions ****/


address allocate_value(Value* v) {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  address a = state->heap.size();
  state->heap.push_back(v);
  return a;
}

Value* copy_val(Value* val, int lineno) {
  check_alive(val, lineno);
  switch (val->tag) {
  case TupleV: {
    auto elts = new vector<pair<string,address> >();
    for (auto i = val->u.tuple.elts->begin();
         i != val->u.tuple.elts->end(); ++i) {
      Value* elt = copy_val(state->heap[i->second], lineno);
      elts->push_back(make_pair(i->first, allocate_value(elt)));
    }
    return make_tuple_val(elts);
  }
  case AltV: {
    Value* arg = copy_val(val->u.alt.arg, lineno);
    return make_alt_val(*val->u.alt.alt_name, *val->u.alt.choice_name, arg);
  }
  case StructV: {
    Value* inits = copy_val(val->u.struct_val.inits, lineno);
    return make_struct_val(val->u.struct_val.type, inits);
  }
  case IntV:
    return make_int_val(val->u.integer);
  case BoolV:
    return make_bool_val(val->u.boolean);
  case FunV:
    return make_fun_val(*val->u.fun.name, val->u.fun.param, val->u.fun.body);
  case PtrV:
    return make_ptr_val(val->u.ptr);
  case FunctionTV:
    return make_fun_type_val(copy_val(val->u.fun_type.param, lineno),
                             copy_val(val->u.fun_type.ret, lineno));
    
  case PointerTV:
    return make_ptr_type_val(copy_val(val->u.ptr_type.type, lineno));
  case IntTV:
    return make_int_type_val();
  case BoolTV:
    return make_bool_type_val();
  case TypeTV:
    return make_type_type_val();
  case VarTV:
    return make_var_type_val(* val->u.var_type);
  case AutoTV:
    return make_auto_type_val();
  case TupleTV: {
    auto new_fields = new VarValues();
    for (auto i = val->u.tuple_type.fields->begin();
         i != val->u.tuple_type.fields->end(); ++i) {
      auto v = copy_val(i->second, lineno);
      new_fields->push_back(make_pair(i->first, v));
    }
    return make_tuple_type_val(new_fields);
  }
  case StructTV: case ChoiceTV: 
  case VarPatV: case AltConsV: 
    return val; // no need to copy these because they are immutable?
    // No, they need to be copied so they don't get killed. -Jeremy
  }
}

void kill_value(Value* val) {
  val->alive = false;
  switch (val->tag) {
  case AltV:
    kill_value(val->u.alt.arg);
    break;
  case StructV:
    kill_value(val->u.struct_val.inits);
    break;
  case TupleV:
    for (auto i = val->u.tuple.elts->begin();
         i != val->u.tuple.elts->end(); ++i) {
      if (state->heap[i->second]->alive)
        kill_value(state->heap[i->second]);
      else {
        cerr << "runtime error, killing an already dead value" << endl;
        exit(-1);
      }
    }
    break;
  default:
    break;
  }
}

void print_env(Env* env, std::ostream& out) {
  if (env) {
    cout << env->key << ": ";
    print_value(state->heap[env->value], out);
    cout << ", ";
    print_env(env->next, out);
  }
}

void print_value(Value* val, std::ostream& out) {
  if (! val->alive) {
    out << "!!";
  }
  switch (val->tag) {
  case AltConsV: {
    out << * val->u.alt_cons.choice_name << "." << * val->u.alt_cons.alt_name;
    break;
  }
  case VarPatV: {
    print_value(val->u.var_pat.type, out);
    out << ": " << *val->u.var_pat.name;
    break;
  }
  case AltV: {
    out << "alt "
         << * val->u.alt.choice_name
         << "."
         << * val->u.alt.alt_name
         << " ";
    print_value(val->u.alt.arg, out);
    break;
  }
  case StructV: {
    out << * val->u.struct_val.type->u.struct_type.name;
    print_value(val->u.struct_val.inits, out);
    break;
  }
  case TupleV: {
    out << "(";
    int i = 0;
    for (auto elt = val->u.tuple.elts->begin();
         elt != val->u.tuple.elts->end(); ++elt, ++i) {
      if (i != 0)
        out << ", ";
      out << elt->first << " = ";
      print_value(state->heap[elt->second], out);
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
    out << "fun<" << * val->u.fun.name << ">";
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
    print_value(val->u.ptr_type.type, out);
    out << ")";
    break;
  case FunctionTV:
    out << "fn ";
    print_value(val->u.fun_type.param, out);
    out << " -> ";
    print_value(val->u.fun_type.ret, out);
    break;
  case VarTV:
    out << * val->u.var_type;
    break;
  case TupleTV: {
    out << "Tuple(";
    int i = 0;
    for (auto elt = val->u.tuple_type.fields->begin();
         elt != val->u.tuple_type.fields->end(); ++elt, ++i) {
      if (i != 0)
        out << ", ";
      out << elt->first << " = ";
      print_value(elt->second, out);
    }
    out << ")";
    break;
  }
  case StructTV:
    out << "struct " << * val->u.struct_type.name;
    break;
  case ChoiceTV:
    out << "choice " << * val->u.choice_type.name;
    break;
  }
}

/***** Action Operations *****/

void print_act(Action* act, std::ostream& out) {
  switch (act->tag) {
  case DeleteTmpAction:
    cout << "delete_tmp(" << act->u.delete_ << ")";
    break;
  case ExpToLValAction:
    out << "exp=>lval";
    break;
  case LValAction:
  case ExpressionAction:
    print_exp(act->u.exp);
    break;
  case StatementAction:
    print_stmt(act->u.stmt, 1);
    break;
  case ValAction:
    print_value(act->u.val, out);
    break;
  }
  out << "<" << act->pos << ">";
  if (act->results.size() > 0) {
    out << "(";
    for (auto iter = act->results.begin(); iter != act->results.end(); ++iter) {
      if (*iter)
        print_value(*iter, out);
      out << ",";
    }
    out << ")";
  }
}

void print_act_list(Cons<Action*>* ls, std::ostream& out) {
  if (ls) {
    print_act(ls->curr, out);
    if (ls->next) {
      out << " :: ";
      print_act_list(ls->next, out);
    }
  }
}

Action* make_exp_act(Expression* e) {
  Action* act = new Action();
  act->tag = ExpressionAction;
  act->u.exp = e;
  act->pos = -1;
  return act;
}

Action* make_lval_act(Expression* e) {
  Action* act = new Action();
  act->tag = LValAction;
  act->u.exp = e;
  act->pos = -1;
  return act;
}

Action* make_stmt_act(Statement* s) {
  Action* act = new Action();
  act->tag = StatementAction;
  act->u.stmt = s;
  act->pos = -1;
  return act;
}

Action* make_val_act(Value* v) {
  Action* act = new Action();
  act->tag = ValAction;
  act->u.val = v;
  act->pos = -1;
  return act;
}

Action* make_exp_to_lval_act() {
  Action* act = new Action();
  act->tag = ExpToLValAction;
  act->pos = -1;
  return act;
}

Action* make_delete_act(address a) {
  Action* act = new Action();
  act->tag = DeleteTmpAction;
  act->pos = -1;
  act->u.delete_ = a;
  return act;
}

/***** Frame and State Operations *****/

void print_frame(Frame* frame, std::ostream& out) {
  out << frame->name;
  out << "{";
  print_act_list(frame->todo, out);
  out << "}"; 
}

void print_stack(Cons<Frame*>* ls, std::ostream& out) {
  if (ls) {
    print_frame(ls->curr, out);
    if (ls->next) {
      out << " :: ";
      print_stack(ls->next, out);
    }
  }
}

void print_heap(vector<Value*>& heap, std::ostream& out) {
  for (auto iter = heap.begin(); iter != heap.end(); ++iter) {
    if (*iter) {
      print_value(*iter, out);
    } else {
      out << "_";
    }
    out << ", ";
  }
}

Env* current_env(State* state) {
  Frame* frame = state->stack->curr;
  return frame->scopes->curr->env;
}

void print_state(std::ostream& out) {
  out << "{" << endl;
  out << "stack: ";
  print_stack(state->stack, out);
  out << endl << "heap: ";
  print_heap(state->heap, out);
  out << endl << "env: ";
  print_env(current_env(state), out);
  out << endl << "}" << endl;
}

/***** Auxilliary Functions *****/


int val_to_int(Value* v, int lineno) {
  check_alive(v, lineno);
  switch (v->tag) {
  case IntV:
    return v->u.integer;
  default:
    cerr << lineno << ": runtime error: expected an integer" << endl;
    exit(-1);
  }
}

int val_to_bool(Value* v, int lineno) {
  check_alive(v, lineno);
  switch (v->tag) {
  case BoolV:
    return v->u.boolean;
  default:
    cerr << "runtime type error: expected a Boolean" << endl;
    exit(-1);
  }
}

address val_to_ptr(Value* v, int lineno) {
  check_alive(v, lineno);
  switch (v->tag) {
  case PtrV:
    return v->u.ptr;
  default:
    cerr << "runtime type error: expected a pointer, not ";
    print_value(v, cerr);
    cerr << endl;
    exit(-1);
  }
}

bool fields_value_equal(VarValues* ts1, VarValues* ts2, int lineno) {
  if (ts1->size() == ts2->size()) {
    for (auto iter1 = ts1->begin(); iter1 != ts1->end(); ++iter1) {
      try {
        auto t2 = find_alist(iter1->first, ts2);
        if (! value_equal(iter1->second, t2, lineno))
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

bool value_equal(Value* v1, Value* v2, int lineno) {
  check_alive(v1, lineno);
  check_alive(v2, lineno);
  return (v1->tag == IntV && v2->tag == IntV && v1->u.integer == v2->u.integer)
    || (v1->tag == BoolV && v2->tag == BoolV && v1->u.boolean == v2->u.boolean)
    || (v1->tag == PtrV && v2->tag == PtrV && v1->u.ptr == v2->u.ptr)
    || (v1->tag == FunV && v2->tag == FunV && v1->u.fun.body == v2->u.fun.body)
    || (v1->tag == TupleV && v2->tag == TupleV
        && fields_value_equal(v1->u.tuple_type.fields,
                              v2->u.tuple_type.fields, lineno))
    // TODO: struct and alternative values
    || type_equal(v1, v2);
}

Value* eval_prim(Operator op, const vector<Value*>& args, int lineno) {
  switch (op) {
  case Neg:
    return make_int_val(- val_to_int(args[0], lineno));
  case Add:
    return make_int_val(val_to_int(args[0], lineno)
                        + val_to_int(args[1], lineno));
  case Sub:
    return make_int_val(val_to_int(args[0], lineno)
                        - val_to_int(args[1], lineno));
  case Not:
    return make_bool_val(! val_to_bool(args[0], lineno));
  case And:
    return make_bool_val(val_to_bool(args[0], lineno)
                         && val_to_bool(args[1], lineno));
  case Or:
    return make_bool_val(val_to_bool(args[0], lineno)
                         || val_to_bool(args[1], lineno));
  case Eq:
    return make_bool_val(value_equal(args[0], args[1], lineno));
  }
}

Env* globals;
  
void init_globals(list<Declaration*>* fs) {
  globals = 0;
  for (auto iter = fs->begin(); iter != fs->end(); ++iter) {
    switch ((*iter)->tag) {
    case ChoiceDeclaration: {
      auto d = *iter;
      auto alts = new VarValues();
      for (auto i = d->u.choice_def.alternatives->begin();
           i != d->u.choice_def.alternatives->end(); ++i) {
        auto t = to_type(d->u.choice_def.lineno, interp_exp(0, i->second));
        alts->push_back(make_pair(i->first, t));
      }
      auto ct = make_choice_type_val(d->u.choice_def.name, alts);
      auto a = allocate_value(ct);
      globals = new Env(* d->u.choice_def.name, a, globals);
      break;
    }
    case StructDeclaration: {
      auto d = *iter;
      auto fields = new VarValues();
      auto methods = new VarValues();
      for (auto i = d->u.struct_def->members->begin();
           i != d->u.struct_def->members->end(); ++i) {
        switch ((*i)->tag) {
        case FieldMember: {
          auto t = to_type(d->u.struct_def->lineno,
                           interp_exp(0, (*i)->u.field.type));
          fields->push_back(make_pair(* (*i)->u.field.name, t));
          break;
        }
        }
      }
      auto st = make_struct_type_val(*d->u.struct_def->name, fields, methods);
      auto a = allocate_value(st);
      globals = new Env(* d->u.struct_def->name, a, globals);
      break;
    }
    case FunctionDeclaration: {
      struct FunctionDefinition* fun = (*iter)->u.fun_def;
      Env* env = 0;
      VarValues* implicit_params = 0;
      auto pt = interp_exp(env, fun->param_pattern);
      auto f = make_fun_val(fun->name, pt, fun->body);
      address a = allocate_value(f);
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
void call_function(int lineno, vector<Value*> operas, State* state) {
  check_alive(operas[0], lineno);
  switch (operas[0]->tag) {
  case FunV: {
    Env* env = globals;
    // Bind arguments to parameters
    list<string> params;
    env = pattern_match(operas[0]->u.fun.param, operas[1], env, params, lineno);
    if (!env) {
      cerr << "internal error in call_function, pattern match failed" << endl;
      exit(-1);
    }
    // Create the new frame and push it on the stack
    Scope* scope = new Scope(env, params);
    Frame* frame = new Frame(* operas[0]->u.fun.name,
                             cons(scope, (Cons<Scope*>*)0),
                             cons(make_stmt_act(operas[0]->u.fun.body),
                                  (Cons<Action*>*)0));
    state->stack = cons(frame, state->stack);
    break;
  }
  case StructTV: {
    Value* arg = copy_val(operas[1], lineno);
    Value* sv = make_struct_val(operas[0], arg);
    Frame* frame = state->stack->curr;
    frame->todo = cons(make_val_act(sv), frame->todo);
    break;
  }
  case AltConsV: {
    Value* arg = copy_val(operas[1], lineno);
    Value* av = make_alt_val(* operas[0]->u.alt_cons.alt_name,
                             * operas[0]->u.alt_cons.choice_name,
                             arg);
    Frame* frame = state->stack->curr;
    frame->todo = cons(make_val_act(av), frame->todo);
    break;
  }
  default:
    cerr << lineno << ": in call, expected a function, not ";
    print_value(operas[0], cerr);
    cerr << endl;
    exit(-1);
  }
}

void kill_scope(int lineno, Scope* scope) {
  for (auto l = scope->locals.begin(); l != scope->locals.end(); ++l) {
    address a = lookup(lineno, scope->env, *l, print_error_string);
    kill_value(state->heap[a]);
  }
}

void kill_locals(int lineno, Frame* frame) {
  Cons<Scope*>* scopes = frame->scopes;
  for (Scope* scope = scopes->curr; scopes; scopes = scopes->next) {
    kill_scope(lineno, scope);
  }
}

void create_tuple(Frame* frame, Action* act, Expression* exp) {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  auto elts = new vector<pair<string,address>>();
  auto f = act->u.exp->u.tuple.fields->begin();
  for (auto i = act->results.begin(); i != act->results.end(); ++i, ++f) {
    address a = allocate_value(*i); // copy?
    elts->push_back(make_pair(f->first, a));
  }
  Value* tv = make_tuple_val(elts);
  frame->todo = cons(make_val_act(tv), frame->todo->next);
}

Value* to_value(Expression* value) {
  switch (value->tag) {
  case Integer:
    return make_int_val(value->u.integer);
  case Boolean:
    return make_bool_val(value->u.boolean);
  case IntT:
    return make_int_type_val();
  case BoolT:
    return make_bool_type_val();
  case TypeT:
    return make_type_type_val();
  case FunctionT:
    // instead add to patterns?
  default:
    cerr << "internal error in to_value, didn't expect ";
    print_exp(value);
    cerr << endl;
    exit(-1);
  }
}


//
// Returns 0 if the value doesn't match the pattern
//
Env* pattern_match(Value* p, Value* v, Env* env, 
                   list<string>& vars, int lineno) {
  cout << "pattern_match(";
  print_value(p, cout);
  cout << ", ";
  print_value(v, cout);
  cout << ")" << endl;
  switch (p->tag) {
  case VarPatV: {
    address a = allocate_value(copy_val(v, lineno));
    vars.push_back(*p->u.var_pat.name);
    return new Env(*p->u.var_pat.name, a, env);
  }
  case TupleV: 
    switch (v->tag) {
    case TupleV: {
      if (p->u.tuple.elts->size() != v->u.tuple.elts->size()) {
        cerr << "runtime error: arity mismatch in tuple pattern match" << endl;
        exit(-1);
      }
      for (auto i = p->u.tuple.elts->begin(); i != p->u.tuple.elts->end();
           ++i) {
        address a = find_field(i->first, v->u.tuple.elts);
        env = pattern_match(state->heap[i->second], state->heap[a],
                            env, vars, lineno);
      }
      return env;
    }
    default:
      cerr << "internal error, expected a tuple value in pattern, not ";
      print_value(v, cerr);
      cerr << endl;
      exit(-1);
    }
  case AltV:
    switch (v->tag) {
    case AltV: {
      if (*p->u.alt.choice_name != *v->u.alt.choice_name
          || *p->u.alt.alt_name != *v->u.alt.alt_name)
        return 0;
      env = pattern_match(p->u.alt.arg, v->u.alt.arg, env, vars, lineno);
      return env;
    }
    default:
      cerr << "internal error, expected a choice alternative in pattern, not ";
      print_value(v, cerr);
      cerr << endl;
      exit(-1);
    }
  case FunctionTV:
    switch (v->tag) {
    case FunctionTV:
      env = pattern_match(p->u.fun_type.param, v->u.fun_type.param, env, vars,
                          lineno);
      env = pattern_match(p->u.fun_type.ret, v->u.fun_type.ret, env, vars,
                          lineno);
      return env;
    default:
      return 0;
    }
  default:
    if (value_equal(p, v, lineno))
      return env;
    else
      return 0;
  }
}

void pattern_assignment(Value* pat, Value* val, int lineno) {
  switch (pat->tag) {
  case PtrV:
    state->heap[val_to_ptr(pat, lineno)] = val;
    break;
  case TupleV: {
    switch (val->tag) {
    case TupleV: {
      if (pat->u.tuple.elts->size() != val->u.tuple.elts->size()) {
        cerr << "runtime error: arity mismatch in tuple pattern match" << endl;
        exit(-1);
      }
      for (auto i = pat->u.tuple.elts->begin(); i != pat->u.tuple.elts->end();
           ++i) {
        address a = find_field(i->first, val->u.tuple.elts);
        pattern_assignment(state->heap[i->second], state->heap[a], lineno);
      }
      break;
    }
    default:
      cerr << "internal error, expected a tuple value on right-hand-side, not ";
      print_value(val, cerr);
      cerr << endl;
      exit(-1);
    }
    break;
  }
  case AltV: {
    switch (val->tag) {
    case AltV: {
      if (*pat->u.alt.choice_name != *val->u.alt.choice_name
          || *pat->u.alt.alt_name != *val->u.alt.alt_name) {
        cerr << "internal error in pattern assignment" << endl;
        exit(-1);
      }
      pattern_assignment(pat->u.alt.arg, val->u.alt.arg, lineno);
      break;
    }
    default:
      cerr << "internal error, expected an alternative in left-hand-side, not ";
      print_value(val, cerr);
      cerr << endl;
      exit(-1);
    }
    break;
  }
  default:
    if (! value_equal(pat, val, lineno)) {
      cerr << "internal error in pattern assignment" << endl;
      exit(-1);
    }
  }
}


/***** state transitions for lvalues *****/

void step_lvalue() {
  Frame* frame = state->stack->curr;
  Action* act = frame->todo->curr;
  Expression* exp = act->u.exp;
  cout << "--- step lvalue "; print_exp(exp); cout << " --->" << endl;
  switch (exp->tag) {
  case Variable: {
    //    { {x :: C, E, F} :: S, H}
    // -> { {E(x) :: C, E, F} :: S, H}
    address a = lookup(exp->lineno, current_env(state), *(exp->u.variable.name),
                       print_error_string);
    Value* v = make_ptr_val(a);
    check_alive(v, exp->lineno);
    frame->todo = cons(make_val_act(v), frame->todo->next);
    break;
  }
  case GetField: {
    //    { {e.f :: C, E, F} :: S, H}
    // -> { e :: [].f :: C, E, F} :: S, H}
    frame->todo = cons(make_lval_act(exp->u.get_field.aggregate), frame->todo);
    act->pos++;
    break;
  }
  case Index: {
    //    { {e[i] :: C, E, F} :: S, H}
    // -> { e :: [][i] :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(exp->u.index.aggregate), frame->todo);
    act->pos++;
    break;
  }
  case Tuple: {
    //    { {(f1=e1,...) :: C, E, F} :: S, H}
    // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
    Expression* e1 = (*exp->u.tuple.fields)[0].second;
    frame->todo = cons(make_lval_act(e1), frame->todo);
    act->pos++;
    break;
  }
  case Integer: case Boolean: case Call: case PrimitiveOp: 
  case IntT: case BoolT: case TypeT: case FunctionT: case AutoT:
  case PatternVariable: {
    frame->todo = cons(make_exp_act(exp),
                       cons(make_exp_to_lval_act(),
                            frame->todo->next));
  }
  }
}

/***** state transitions for expressions *****/

void step_exp() {
  Frame* frame = state->stack->curr;
  Action* act = frame->todo->curr;
  Expression* exp = act->u.exp;
  cout << "--- step exp "; print_exp(exp); cout << " --->" << endl;
  switch (exp->tag) {
  case PatternVariable: {
    frame->todo = cons(make_exp_act(exp->u.pattern_variable.type), frame->todo);
    act->pos++;
    break;
  }
  case Index: {
    //    { { e[i] :: C, E, F} :: S, H}
    // -> { { e :: [][i] :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(exp->u.index.aggregate), frame->todo);
    act->pos++;
    break;
  }
  case Tuple: {
    if (exp->u.tuple.fields->size() > 0) {
      //    { {(f1=e1,...) :: C, E, F} :: S, H}
      // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
      Expression* e1 = (*exp->u.tuple.fields)[0].second;
      frame->todo = cons(make_exp_act(e1), frame->todo);
      act->pos++;
    } else {
      create_tuple(frame, act, exp);
    }
    break;
  }
  case GetField: {
    //    { { e.f :: C, E, F} :: S, H}
    // -> { { e :: [].f :: C, E, F} :: S, H}
    frame->todo = cons(make_lval_act(exp->u.get_field.aggregate), frame->todo);
    act->pos++;
    break;
  }
  case Variable: {
    // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
    address a = lookup(exp->lineno, current_env(state), *(exp->u.variable.name),
                       print_error_string);
    Value* v = state->heap[a];
    frame->todo = cons(make_val_act(v), frame->todo->next);
    break;
  }
  case Integer:
    // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
    frame->todo = cons(make_val_act(make_int_val(exp->u.integer)),
                       frame->todo->next);
    break;
  case Boolean:
    // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
    frame->todo = cons(make_val_act(make_bool_val(exp->u.boolean)),
                       frame->todo->next);
    break;
  case PrimitiveOp:
    if (exp->u.primitive_op.arguments->size() > 0) {
      //    { {op(e :: es) :: C, E, F} :: S, H}
      // -> { e :: op([] :: es) :: C, E, F} :: S, H}
      frame->todo = cons(make_exp_act(exp->u.primitive_op.arguments->front()),
                         frame->todo);
      act->pos++;
    } else {
      //    { {v :: op(]) :: C, E, F} :: S, H}
      // -> { {eval_prim(op, ()) :: C, E, F} :: S, H}
      Value* v = eval_prim(exp->u.primitive_op.operator_, act->results,
                           exp->lineno);
      frame->todo = cons(make_val_act(v), frame->todo->next->next);
    }
    break;
  case Call:
    //    { {e1(e2) :: C, E, F} :: S, H}
    // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(exp->u.call.function), frame->todo);
    act->pos++;
    break;
  case IntT: {
    Value* v = make_int_type_val();
    frame->todo = cons(make_val_act(v), frame->todo->next);
    break;
  }
  case BoolT: {
    Value* v = make_bool_type_val();
    frame->todo = cons(make_val_act(v), frame->todo->next);
    break;
  }
  case AutoT: {
    Value* v = make_auto_type_val();
    frame->todo = cons(make_val_act(v), frame->todo->next);
    break;
  }
  case TypeT: {
    Value* v = make_type_type_val();
    frame->todo = cons(make_val_act(v), frame->todo->next);
    break;
  }
  case FunctionT: {
    frame->todo = cons(make_exp_act(exp->u.function_type.parameter),
                       frame->todo);
    act->pos++;
    break;
  }
  } // switch (exp->tag)
}

/***** state transitions for statements *****/

bool is_while_act(Action* act) {
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

bool is_block_act(Action* act) {
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

void step_stmt() {
  Frame* frame = state->stack->curr;
  Action* act = frame->todo->curr;
  Statement* stmt = act->u.stmt;
  cout << "--- step stmt "; print_stmt(stmt, 1); cout << " --->" << endl;
  switch (stmt->tag) {
  case Match:
    //    { { (match (e) ...) :: C, E, F} :: S, H}
    // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(stmt->u.match_stmt.exp), frame->todo);
    act->pos++;
    break;
  case While:
    //    { { (while (e) s) :: C, E, F} :: S, H}
    // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(stmt->u.while_stmt.cond), frame->todo);
    act->pos++;
    break;
  case Break:
    //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
    // -> { { C, E', F} :: S, H}
    frame->todo = frame->todo->next;
    while (frame->todo && ! is_while_act(frame->todo->curr)) {
      if (is_block_act(frame->todo->curr)) {
        kill_scope(stmt->lineno, frame->scopes->curr);
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
    while (frame->todo && ! is_while_act(frame->todo->curr)) {
      if (is_block_act(frame->todo->curr)) {
        kill_scope(stmt->lineno, frame->scopes->curr);
        frame->scopes = frame->scopes->next;
      }
      frame->todo = frame->todo->next;
    }
    break;
  case Block: {
    if (act->pos == -1) {
      Scope* scope = new Scope(current_env(state), list<string>());
      frame->scopes = cons(scope, frame->scopes);
      frame->todo = cons(make_stmt_act(stmt->u.block.stmt),
                         frame->todo);
      act->pos++;
    } else {
      Scope* scope = frame->scopes->curr;
      kill_scope(stmt->lineno, scope);
      frame->scopes = frame->scopes->next;
      frame->todo = frame->todo->next;
    }
    break;
  }
  case VariableDefinition:
    //    { {(var x = e) :: C, E, F} :: S, H}
    // -> { {e :: (var x = []) :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(stmt->u.variable_definition.init),
                       frame->todo);
    act->pos++;
    break;
  case ExpressionStatement:
    //    { {e :: C, E, F} :: S, H}
    // -> { {e :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(stmt->u.exp), frame->todo);
    break;
  case Assign:
    //    { {(lv = e) :: C, E, F} :: S, H}
    // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
    frame->todo = cons(make_lval_act(stmt->u.assign.lhs),
                          frame->todo);
    act->pos++;
    break;
  case If:
    //    { {(if (e) thn else els) :: C, E, F} :: S, H}
    // -> { { e :: (if ([]) thn else els) :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(stmt->u.if_stmt.cond), frame->todo);
    act->pos++;
    break;
  case Return:
    //    { {return e :: C, E, F} :: S, H}
    // -> { {e :: return [] :: C, E, F} :: S, H}
    frame->todo = cons(make_exp_act(stmt->u.return_stmt), frame->todo);
    act->pos++;
    break;
  case Sequence:
    //    { { (s1,s2) :: C, E, F} :: S, H}
    // -> { { s1 :: s2 :: C, E, F} :: S, H}
    Cons<Action*>* todo = frame->todo->next;
    if (stmt->u.sequence.next) {
      todo = cons(make_stmt_act(stmt->u.sequence.next), todo);
    }
    frame->todo = cons(make_stmt_act(stmt->u.sequence.stmt), todo);
    break;
  }
}

address get_member(address a, string f) {
  vector<pair<string,address> >*  fields;
  Value* v = state->heap[a];
  switch (v->tag) {
  case StructV:
    fields = v->u.struct_val.inits->u.tuple.elts;
    try {
      return find_field(f, fields);
    } catch (std::domain_error de) {
      cerr << "runtime error, member " << f << " not in ";
      print_value(v, cerr); cerr << endl;
      exit(-1);
    }
    break;
  case TupleV:
    fields = v->u.tuple.elts;
    try {
      return find_field(f, fields);
    } catch (std::domain_error de) {
      cerr << "field " << f << " not in ";
      print_value(v, cerr); cerr << endl;
      exit(-1);
    }
    break;
  case ChoiceTV: {
    try {
      find_alist(f, v->u.choice_type.alternatives);
      auto ac = make_alt_cons(f, * v->u.choice_type.name);
      return allocate_value(ac);
    } catch (std::domain_error de) {
      cerr << "alternative " << f << " not in ";
      print_value(v, cerr); cerr << endl;
      exit(-1);
    }
    break;
  }
  default:
    cerr << "field access not allowed for value ";
    print_value(v, cerr);
    cerr << endl;
    exit(-1);
  }
}

Cons<Action*>* insert_delete(Action* del, Cons<Action*>* todo) {
  if (todo) {
    switch (todo->curr->tag) {
    case StatementAction: {
      // This places the delete before the enclosing statement.
      // Not sure if that is OK. Conceptually it should go after
      // but that is tricky for some statements, like 'return'. -Jeremy
      return cons(del, todo);
    }
    case LValAction: case ExpressionAction: case ValAction:
    case ExpToLValAction: case DeleteTmpAction:
      return cons(todo->curr, insert_delete(del, todo->next));
    }
  } else {
    return cons(del, todo);
  }
}

/***** State transition for handling a value *****/

void handle_value() {
  Frame* frame = state->stack->curr;
  Action* val_act = frame->todo->curr;
  Action* act = frame->todo->next->curr;
  act->results.push_back(val_act->u.val);
  act->pos++;
  
  cout << "--- handle value "; print_value(val_act->u.val, cout);
  cout << " with "; print_act(act, cout); cout << " --->" << endl;
             
  switch (act->tag) {
  case DeleteTmpAction: {
    kill_value(state->heap[act->u.delete_]);
    frame->todo = cons(val_act, frame->todo->next->next);
    break;
  }
  case ExpToLValAction: {
    address a = allocate_value(act->results[0]);
    auto del = make_delete_act(a);
    frame->todo = cons(make_val_act(make_ptr_val(a)),
                       insert_delete(del, frame->todo->next->next));
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
        address a = get_member(val_to_ptr(str, exp->lineno),
                               * exp->u.get_field.field);
        frame->todo = cons(make_val_act(make_ptr_val(a)),
                           frame->todo->next->next);
      } catch (std::domain_error de) {
        cerr << "field " << * exp->u.get_field.field << " not in ";
        print_value(str, cerr);
        cerr << endl;
        exit(-1);
      }
      break;
    }      
    case Index: {
      if (act->pos == 1) {
        frame->todo = cons(make_exp_act(exp->u.index.offset),
                           frame->todo->next);
      } else if (act->pos == 2) {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Value* tuple = act->results[0];
        string f = std::to_string(to_integer(act->results[1]));
        try {
          address a = find_field(f, tuple->u.tuple.elts);
          frame->todo = cons(make_val_act(make_ptr_val(a)),
                             frame->todo->next->next);
        } catch (std::domain_error de) {
          cerr << "runtime error: field " << f << "not in ";
          print_value(tuple, cerr);
          cerr << endl;
          exit(-1);
        }
      }
      break;
    }
    case Tuple: {
      if (act->pos != exp->u.tuple.fields->size()) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S, H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S, H}
        Expression* elt = (*exp->u.tuple.fields)[act->pos].second;
        frame->todo = cons(make_lval_act(elt), frame->todo->next);
      } else {
        frame->todo = frame->todo->next;
        create_tuple(frame, act, exp);
      }      
      break;
    }
    default:
      cerr << "internal error in handle_value, LValAction" << endl;
      exit(-1);
    }
    break;
  }
  case ExpressionAction: {
    Expression* exp = act->u.exp;
    switch (exp->tag) {
    case PatternVariable: {
      auto v = make_var_pat_val(* exp->u.pattern_variable.name,
                                act->results[0]);
      frame->todo = cons(make_val_act(v), frame->todo->next->next);
      break;
    }      
    case Tuple: {
      if (act->pos != exp->u.tuple.fields->size()) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S, H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S, H}
        Expression* elt = (*exp->u.tuple.fields)[act->pos].second;
        frame->todo = cons(make_exp_act(elt), frame->todo->next);
      } else {
        frame->todo = frame->todo->next;
        create_tuple(frame, act, exp);
      }      
      break;
    }
    case Index: {
      if (act->pos == 1) {
        frame->todo = cons(make_exp_act(exp->u.index.offset),
                           frame->todo->next);
      } else if (act->pos == 2) {
        auto tuple = act->results[0];;
        switch (tuple->tag) {
        case TupleV: {
          //    { { v :: [][i] :: C, E, F} :: S, H}
          // -> { { v_i :: C, E, F} : S, H}
          string f = std::to_string(to_integer(act->results[1]));
          try {
            auto a = find_field(f, tuple->u.tuple.elts);
            frame->todo = cons(make_val_act(state->heap[a]),
                               frame->todo->next->next);
          } catch (std::domain_error de) {
            cerr << "runtime error, field " << f
                 << " not in ";
            print_value(tuple, cerr);
            cerr << endl;
            exit(-1);
          }
          break;
        }
        default:
          cerr << "runtime type error, expected a tuple in field access, not ";
          print_value(tuple, cerr);
          exit(-1);
        }
      }
      break;
    }
    case GetField: {
      //    { { v :: [].f :: C, E, F} :: S, H}
      // -> { { v_f :: C, E, F} : S, H}
      auto a = get_member(val_to_ptr(act->results[0], exp->lineno),
                         * exp->u.get_field.field);
      frame->todo = cons(make_val_act(state->heap[a]),
                         frame->todo->next->next);
      break;
    }
    case PrimitiveOp: {
      if (act->pos != exp->u.primitive_op.arguments->size()) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Expression* arg = (*exp->u.primitive_op.arguments)[act->pos];
        frame->todo = cons(make_exp_act(arg), frame->todo->next);
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        Value* v = eval_prim(exp->u.primitive_op.operator_, act->results,
                             exp->lineno);
        frame->todo = cons(make_val_act(v), frame->todo->next->next);
      }
      break;
    }
    case Call: {
      if (act->pos == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        frame->todo = cons(make_exp_act(exp->u.call.argument),
                           frame->todo->next);
      } else if (act->pos == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        frame->todo = frame->todo->next->next;
        call_function(exp->lineno, act->results, state);
      } else {
        cerr << "internal error in handle_value with Call" << endl;
        exit(-1);
      }
      break;
    }
    case FunctionT: {
      if (act->pos == 2) {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        Value* v = make_fun_type_val(act->results[0], act->results[1]);
        frame->todo = cons(make_val_act(v), frame->todo->next->next);
      } else {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        frame->todo = cons(make_exp_act(exp->u.function_type.return_type),
                           frame->todo->next);
      }
      break;
    }
    case Variable: case Integer: case Boolean: 
    case IntT: case BoolT: case TypeT: case AutoT:
      cerr << "internal error, bad expression context in handle_value" << endl;
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
        frame->todo = cons(make_exp_act(stmt->u.variable_definition.pat),
                           frame->todo->next);
      } else if (act->pos == 2) {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Value* v = act->results[0];
        Value* p = act->results[1];
        //address a = allocate_value(copy_val(v));
        frame->scopes->curr->env =
          pattern_match(p, v, frame->scopes->curr->env, 
                        frame->scopes->curr->locals, stmt->lineno);
        if (!frame->scopes->curr->env) {
          cerr << stmt->lineno
               << ": internal error in variable definition, match failed"
               << endl;
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
        frame->todo = cons(make_exp_act(stmt->u.assign.rhs),
                           frame->todo->next);
      } else if (act->pos == 2) {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->results[0];
        auto val = act->results[1];
        pattern_assignment(pat, val, stmt->lineno);
        frame->todo = frame->todo->next->next;
      }
      break;
    case If:
      if (val_to_bool(act->results[0], stmt->lineno)) {
        //    { {true :: if ([]) thn else els :: C, E, F} :: S, H}
        // -> { { thn :: C, E, F } :: S, H}
        frame->todo = cons(make_stmt_act(stmt->u.if_stmt.thn),
                           frame->todo->next->next);
      } else {
        //    { {false :: if ([]) thn else els :: C, E, F} :: S, H}
        // -> { { els :: C, E, F } :: S, H}
        frame->todo = cons(make_stmt_act(stmt->u.if_stmt.els),
                           frame->todo->next->next);
      }
      break;
    case While:
      if (val_to_bool(act->results[0], stmt->lineno)) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        frame->todo->next->curr->pos = -1;
        frame->todo->next->curr->results.clear();
        frame->todo = cons(make_stmt_act(stmt->u.while_stmt.body),
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
        * odd: start interpretting the pattern of a clause
        * even: finished interpretting the pattern, now try to match

        Regarding act->results:
        * 0: the value that we're matching
        * 1: the pattern for clause 0
        * 2: the pattern for clause 1
        * ...
      */
      auto clause_num = (act->pos - 1) / 2;
      if (clause_num >= stmt->u.match_stmt.clauses->size()) {
        frame->todo = frame->todo->next->next;
        break;
      }
      auto c = stmt->u.match_stmt.clauses->begin();
      std::advance(c, clause_num);
      
      if (act->pos % 2 == 1) {
        // start interpreting the pattern of the clause
        //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
        // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
        frame->todo = cons(make_exp_act(c->first), frame->todo->next);
      } else { // try to match
        auto v = act->results[0];
        auto pat = act->results[clause_num + 1];
        auto env = current_env(state);
        list<string> vars;
        Env* new_env = pattern_match(pat, v, env, vars, stmt->lineno);
        if (new_env) { // we have a match, start the body
          Scope* new_scope = new Scope(new_env, vars);
          frame->scopes = cons(new_scope, frame->scopes);
          Statement* body_block = make_block(stmt->lineno, c->second);
          Action* body_act = make_stmt_act(body_block);
          body_act->pos = 0;
          frame->todo = cons(make_stmt_act(c->second),
                             cons(body_act, frame->todo->next->next));
        } else {
          act->pos++;
          clause_num = (act->pos - 1) / 2;
          if (clause_num < stmt->u.match_stmt.clauses->size()) {
            // move on to the next clause
            c = stmt->u.match_stmt.clauses->begin();
            std::advance(c, clause_num);
            frame->todo = cons(make_exp_act(c->first), frame->todo->next);
          } else { // No more clauses in match
            frame->todo = frame->todo->next->next;
          }
        }
      }
      break;
    }
    case Return: {
      //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
      // -> { {v :: C', E', F'} :: S, H}
      Value* ret_val = copy_val(val_act->u.val, stmt->lineno);
      kill_locals(stmt->lineno, frame);
      state->stack = state->stack->next;
      frame = state->stack->curr;
      frame->todo = cons(make_val_act(ret_val), frame->todo);
      break;
    }
    case Block:
    case Sequence:
    case Break:
    case Continue:
      cerr << "internal error in handle_value, unhandled statement ";
      print_stmt(stmt, 1);
      cerr << endl;
      exit(-1);
    } // switch stmt
    break;
  }
  case ValAction:
    cerr << "internal error, ValAction in handle_value" << endl;
    exit(-1);
  } // switch act
}

/***** state transition *****/

void step() {
  Frame* frame = state->stack->curr;
  if (! frame->todo) {
    cerr << "runtime error: fell off end of function " << frame->name
         << " without `return`" << endl;
    exit(-1);
  }
  
  Action* act = frame->todo->curr;
  switch (act->tag) {
  case DeleteTmpAction:
    cerr << "internal error in step, did not expect DeleteTmpAction" << endl;
    break;
  case ExpToLValAction:
    cerr << "internal error in step, did not expect ExpToLValAction" << endl;
    break;
  case ValAction:
    handle_value();
    break;
  case LValAction:
    step_lvalue();
    break;
  case ExpressionAction:
    step_exp();
    break;
  case StatementAction:
    step_stmt();
    break;
  } // switch
}

/***** interpret the whole program *****/

int interp_program(list<Declaration*>* fs) {
  state = new State(); // runtime state
  cout << "********** initializing globals **********" << endl;
  init_globals(fs);

  Expression* arg = make_tuple(0, new vector<pair<string,Expression*> >());
  Expression* call_main = make_call(0, make_var(0, "main"), arg);
  Cons<Action*>* todo = cons(make_exp_act(call_main), (Cons<Action*>*)0);
  Scope* scope = new Scope(globals, list<string>());
  Frame* frame = new Frame("top", cons(scope, (Cons<Scope*>*)0), todo);
  state->stack = cons(frame, (Cons<Frame*>*)0);

  cout << "********** calling main function **********" << endl;  
  print_state(cout);

  while (length(state->stack) > 1
         || length(state->stack->curr->todo) > 1
         || state->stack->curr->todo->curr->tag != ValAction) {
    step();
    print_state(cout);
  }
  Value* v = state->stack->curr->todo->curr->u.val;
  return val_to_int(v, 0);
}

/***** interpret an expression (at compile-time) *****/

Value* interp_exp(Env* env, Expression* e) {
  Cons<Action*>* todo = cons(make_exp_act(e), (Cons<Action*>*)0);
  Scope* scope = new Scope(env, list<string>());
  Frame* frame = new Frame("interp_exp", cons(scope, (Cons<Scope*>*)0), todo);
  state->stack = cons(frame, (Cons<Frame*>*)0);

  while (length(state->stack) > 1
         || length(state->stack->curr->todo) > 1
         || state->stack->curr->todo->curr->tag != ValAction) {
    step();
  }
  Value* v = state->stack->curr->todo->curr->u.val;
  return v;
}
