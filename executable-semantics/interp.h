#ifndef INTERP_H
#define INTERP_H

#include "ast.h"
#include "assoc_list.h"
#include "cons_list.h"
#include <vector>
#include <list>
using std::vector;
using std::list;

struct Value;

typedef unsigned int address;
typedef list< pair<string, Value*> > VarValues;
typedef AList<string, address> Env;

/***** Values *****/

enum ValKind { IntV, FunV, PtrV, BoolV, StructV, AltV, TupleV,
               VarTV, IntTV, BoolTV, TypeTV, FunctionTV, PointerTV, AutoTV,
               TupleTV, StructTV, ChoiceTV, VarPatV,
               AltConsV };

struct Value {
  ValKind tag;
  bool alive;
  union {
    int integer;
    bool boolean;
    struct { string* name;
      Value* param;
      Statement* body;
    } fun;
    struct { Value* type; Value* inits; } struct_val;
    struct { string* alt_name; string* choice_name; } alt_cons;
    struct { string* alt_name; string* choice_name; Value* arg; } alt;
    struct { vector<pair<string,address> >* elts; } tuple;
    address ptr;
    string* var_type;
    struct { string* name; Value* type; } var_pat;
    struct { Value* param; Value* ret; } fun_type;
    struct { Value* type; } ptr_type;
    struct { string* name; VarValues* fields; VarValues* methods; } struct_type;
    struct { string* name; VarValues* fields; } tuple_type;
    struct { string* name; VarValues* alternatives; } choice_type;
    struct { list<string*>* params; Value* type; } implicit;
  } u;
};

Value* make_int_val(int i);
Value* make_bool_val(bool b);
Value* make_fun_val(string name, VarValues* implicit_params, Value* param,
                    Statement* body, vector<Value*>* implicit_args);
Value* make_ptr_val(address addr);
Value* make_struct_val(Value* type, vector<pair<string,address> >* inits);
Value* make_tuple_val(vector<pair<string,address> >* elts);
Value* make_alt_val(string alt_name, string choice_name, Value* arg);
Value* make_alt_cons(string alt_name, string choice_name);

Value* make_var_pat_val(string name, Value* type);

Value* make_var_type_val(string name);
Value* make_int_type_val();
Value* make_auto_type_val();
Value* make_bool_type_val();
Value* make_type_type_val();
Value* make_fun_type_val(Value* param, Value* ret);
Value* make_ptr_type_val(Value* type);
Value* make_struct_type_val(string name, VarValues* fields, VarValues* methods);
Value* make_tuple_type_val(VarValues* fields);
Value* make_void_type_val();
Value* make_choice_type_val(string* name, VarValues* alts);

bool value_equal(Value* v1, Value* v2, int lineno);

/***** Actions *****/

enum ActionKind { LValAction, ExpressionAction, StatementAction, ValAction,
                  ExpToLValAction, DeleteTmpAction };

struct Action {
  ActionKind tag;
  union {
    Expression* exp;      // for LValAction and ExpressionAction
    Statement* stmt;
    Value* val;           // for finished actions with a value (ValAction)
    address delete_;
  } u;
  int pos;                // position or state of the action
  vector<Value*> results; // results from subexpression
};
typedef Cons<Action*>* ActionList;

/***** Scopes *****/

struct Scope {
  Scope(Env* e, const list<string>& l) : env(e), locals(l) { }
  Env* env;
  list<string> locals;
};

/***** Frames and State *****/

struct Frame {
  Cons<Action*>* todo;
  Cons<Scope*>* scopes; 
  string name;
  Frame(string n, Cons<Scope*>* s, Cons<Action*>* c)
    : name(n), scopes(s), todo(c) { }
};

struct State {
  Cons<Frame*>* stack;
  vector<Value*> heap;
};

extern State* state;

void print_value(Value* val, std::ostream& out);
void print_env(Env* env);
address allocate_value(Value* v);
Value* copy_val(Value* val, int lineno);
int to_integer(Value* v);

/***** Interpreters *****/

int interp_program(list<Declaration*>* fs);
Value* interp_exp(Env* env, Expression* e);

#endif
