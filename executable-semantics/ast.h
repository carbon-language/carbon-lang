#ifndef AST_H
#define AST_H

#include <stdlib.h>
#include <string>
#include <list>
#include <vector>
#include <exception>

using std::string;
using std::list;
using std::vector;
using std::pair;

/***** Utilities *****/

template<class T>
void print_list(list<T*>* ts, void(*printer)(T*), const char* sep) {
  int i = 0;
  for (auto iter = ts->begin(); iter != ts->end(); ++iter, ++i) {
    if (i != 0)
      printf("%s", sep);
    printer(*iter);
  }
}

template<class T>
void print_vector(vector<T*>* ts, void(*printer)(T*), const char* sep) {
  int i = 0;
  for (auto iter = ts->begin(); iter != ts->end(); ++iter, ++i) {
    if (i != 0)
      printf("%s", sep);
    printer(*iter);
  }
}

char *read_file(FILE* fp);

extern char* input;

/***** Forward Declarations *****/

struct LValue;
struct Expression;
struct Statement;
struct FunctionDefinition;

typedef list< pair<string, Expression*> > VarTypes;

/***** Expressions *****/

enum ExpressionKind { Variable, PatternVariable, Integer, Boolean,
                      PrimitiveOp, Call, Tuple, Index, GetField,
                      IntT, BoolT, TypeT, FunctionT, AutoT };
enum Operator { Neg, Add, Sub, Not, And, Or, Eq };

struct Expression {
  int lineno;
  ExpressionKind tag;
  union {
    struct { string* name; } variable;
    struct { Expression* aggregate; string* field; } get_field;
    struct { Expression* aggregate; Expression* offset; } index;
    struct { string* name; Expression* type; } pattern_variable;
    int integer;
    bool boolean;
    struct { vector<pair<string,Expression*> >* fields; } tuple;
    struct { Operator operator_; vector<Expression*>* arguments; } primitive_op;
    struct { Expression* function; Expression* argument; } call;
    struct { Expression* parameter; Expression* return_type; } function_type;
  } u;
};

Expression* make_var(int lineno, string var);
Expression* make_var_pat(int lineno, string var, Expression* type);
Expression* make_int(int lineno, int i);
Expression* make_bool(int lineno, bool b);
Expression* make_op(int lineno, Operator op, vector<Expression*>* args);
Expression* make_unop(int lineno, enum Operator op, Expression* arg);
Expression* make_binop(int lineno, enum Operator op,
                       Expression* arg1, Expression* arg2);
Expression* make_call(int lineno, Expression* fun, Expression* arg);
Expression* make_get_field(int lineno, Expression* exp, string field);
Expression* make_tuple(int lineno, vector<pair<string,Expression*> >* args);
Expression* make_index(int lineno, Expression* exp, Expression* i);

Expression* make_type_type(int lineno);
Expression* make_int_type(int lineno);
Expression* make_bool_type(int lineno);
Expression* make_fun_type(int lineno, Expression* param, Expression* ret);
Expression* make_auto_type(int lineno);

void print_exp(Expression*);

/***** Expression or Field List *****/
/*
  This is used in the parsing of tuples and parenthesized expressions.
 */

enum ExpOrFieldListKind { Exp, FieldList };

struct ExpOrFieldList {
  ExpOrFieldListKind tag;
  union {
    Expression* exp;
    list<pair<string,Expression*> >* fields;
  } u;
};

ExpOrFieldList* make_exp(Expression* exp);
ExpOrFieldList* make_field_list(list<pair<string,Expression*> >* fields);
ExpOrFieldList* cons_field(ExpOrFieldList* e1, ExpOrFieldList* e2);

/***** Statements *****/

enum StatementKind { ExpressionStatement, Assign, VariableDefinition,
                     If, Return, Sequence, Block, While, Break, Continue,
                     Match };

struct Statement {
  int lineno;
  StatementKind tag;
  union {
    Expression* exp;
    struct { Expression* lhs; Expression* rhs; } assign;
    struct { Expression* pat; Expression* init; } variable_definition;
    struct { Expression* cond; Statement* then_; Statement* else_; } if_stmt;
    Expression* return_stmt;
    struct { Statement* stmt; Statement* next; } sequence;
    struct { Statement* stmt; } block;
    struct { Expression* cond; Statement* body; } while_stmt;
    struct {
      Expression* exp;
      list< pair<Expression*,Statement*> >* clauses;
    } match_stmt;
  } u;
};

Statement* make_exp_stmt(int lineno, Expression* exp);
Statement* make_assign(int lineno, Expression* lhs, Expression* rhs);
Statement* make_var_def(int lineno, Expression* pat, Expression* init);
Statement* make_if(int lineno, Expression* cond, Statement* then_,
                   Statement* else_);
Statement* make_return(int lineno, Expression* e);
Statement* make_seq(int lineno, Statement* s1, Statement* s2);
Statement* make_block(int lineno, Statement* s);
Statement* make_while(int lineno, Expression* cond, Statement* body);
Statement* make_break(int lineno);
Statement* make_continue(int lineno);
Statement* make_match(int lineno, Expression* exp,
                      list< pair<Expression*,Statement*> >* clauses);

void print_stmt(Statement*, int);

/***** Function Definitions *****/

struct FunctionDefinition {
  int lineno;
  string name;
  Expression* param_pattern;
  Expression* return_type;
  Statement* body;
};

/***** Struct Members *****/

enum MemberKind { FieldMember };

struct Member {
  int lineno;
  MemberKind tag;
  union {
    struct { string* name; Expression* type; } field;
  } u;
};

Member* make_field(int lineno, string name, Expression* type);

/***** Declarations *****/

struct StructDefinition {
  int lineno;
  string* name;
  list<Member*>* members;
};

enum DeclarationKind { FunctionDeclaration, StructDeclaration,
                       ChoiceDeclaration };

struct Declaration {
  DeclarationKind tag;
  union {
    struct FunctionDefinition* fun_def;
    struct StructDefinition* struct_def;
    struct {
      int lineno;
      string* name;
      list<pair<string, Expression*> >* alternatives;
    } choice_def;
  } u;
};


struct FunctionDefinition*
make_fun_def(int lineno, string name, Expression* ret_type,
             Expression* param, Statement* body);
void print_fun_def(struct FunctionDefinition*);
void print_fun_def_depth(struct FunctionDefinition*, int);

Declaration* make_fun_decl(struct FunctionDefinition* f);
Declaration* make_struct_decl(int lineno, string name, list<Member*>* members);
Declaration* make_choice_decl(int lineno, string name,
                              list<pair<string, Expression*> >* alts);

void print_decl(Declaration*);

void print_string(string* s);

template<class T>
T find_field(string field, vector<pair<string,T> >* inits) {
  for (auto i = inits->begin(); i != inits->end(); ++i) {
    if (i->first == field)
      return i->second;
  }
  throw std::domain_error(field);
}

template<class T>
T find_alist(string field, list<pair<string,T> >* inits) {
  for (auto i = inits->begin(); i != inits->end(); ++i) {
    if (i->first == field)
      return i->second;
  }
  throw std::domain_error(field);
}


#endif
