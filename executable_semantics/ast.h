// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AST_H
#define AST_H

#include <stdlib.h>

#include <exception>
#include <list>
#include <string>
#include <vector>

using std::list;
using std::pair;
using std::string;
using std::vector;

/***** Utilities *****/

template <class T>
void PrintList(list<T*>* ts, void (*printer)(T*), const char* sep) {
  int i = 0;
  for (auto iter = ts->begin(); iter != ts->end(); ++iter, ++i) {
    if (i != 0)
      printf("%s", sep);
    printer(*iter);
  }
}

template <class T>
void PrintVector(vector<T*>* ts, void (*printer)(T*), const char* sep) {
  int i = 0;
  for (auto iter = ts->begin(); iter != ts->end(); ++iter, ++i) {
    if (i != 0)
      printf("%s", sep);
    printer(*iter);
  }
}

char* ReadFile(FILE* fp);

extern char* input;

/***** Forward Declarations *****/

struct LValue;
struct Expression;
struct Statement;
struct FunctionDefinition;

typedef list<pair<string, Expression*> > VarTypes;

/***** Expressions *****/

enum ExpressionKind {
  Variable,
  PatternVariable,
  Integer,
  Boolean,
  PrimitiveOp,
  Call,
  Tuple,
  Index,
  GetField,
  IntT,
  BoolT,
  TypeT,
  FunctionT,
  AutoT
};
enum Operator { Neg, Add, Sub, Not, And, Or, Eq };

struct Expression {
  int lineno;
  ExpressionKind tag;
  union {
    struct {
      string* name;
    } variable;
    struct {
      Expression* aggregate;
      string* field;
    } get_field;
    struct {
      Expression* aggregate;
      Expression* offset;
    } index;
    struct {
      string* name;
      Expression* type;
    } pattern_variable;
    int integer;
    bool boolean;
    struct {
      vector<pair<string, Expression*> >* fields;
    } tuple;
    struct {
      Operator operator_;
      vector<Expression*>* arguments;
    } primitive_op;
    struct {
      Expression* function;
      Expression* argument;
    } call;
    struct {
      Expression* parameter;
      Expression* return_type;
    } function_type;
  } u;
};

Expression* MakeVar(int lineno, string var);
Expression* MakeVarPat(int lineno, string var, Expression* type);
Expression* MakeInt(int lineno, int i);
Expression* MakeBool(int lineno, bool b);
Expression* MakeOp(int lineno, Operator op, vector<Expression*>* args);
Expression* MakeUnOp(int lineno, enum Operator op, Expression* arg);
Expression* MakeBinOp(int lineno, enum Operator op, Expression* arg1,
                       Expression* arg2);
Expression* MakeCall(int lineno, Expression* fun, Expression* arg);
Expression* MakeGetField(int lineno, Expression* exp, string field);
Expression* MakeTuple(int lineno, vector<pair<string, Expression*> >* args);
Expression* MakeIndex(int lineno, Expression* exp, Expression* i);

Expression* MakeTypeType(int lineno);
Expression* MakeIntType(int lineno);
Expression* MakeBoolType(int lineno);
Expression* MakeFunType(int lineno, Expression* param, Expression* ret);
Expression* MakeAutoType(int lineno);

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
    list<pair<string, Expression*> >* fields;
  } u;
};

ExpOrFieldList* MakeExp(Expression* exp);
ExpOrFieldList* MakeFieldList(list<pair<string, Expression*> >* fields);
ExpOrFieldList* MakeConstructorField(ExpOrFieldList* e1, ExpOrFieldList* e2);

/***** Statements *****/

enum StatementKind {
  ExpressionStatement,
  Assign,
  VariableDefinition,
  If,
  Return,
  Sequence,
  Block,
  While,
  Break,
  Continue,
  Match
};

struct Statement {
  int lineno;
  StatementKind tag;
  union {
    Expression* exp;
    struct {
      Expression* lhs;
      Expression* rhs;
    } assign;
    struct {
      Expression* pat;
      Expression* init;
    } variable_definition;
    struct {
      Expression* cond;
      Statement* then_;
      Statement* else_;
    } if_stmt;
    Expression* return_stmt;
    struct {
      Statement* stmt;
      Statement* next;
    } sequence;
    struct {
      Statement* stmt;
    } block;
    struct {
      Expression* cond;
      Statement* body;
    } while_stmt;
    struct {
      Expression* exp;
      list<pair<Expression*, Statement*> >* clauses;
    } match_stmt;
  } u;
};

Statement* MakeExp_stmt(int lineno, Expression* exp);
Statement* MakeAssign(int lineno, Expression* lhs, Expression* rhs);
Statement* MakeVar_def(int lineno, Expression* pat, Expression* init);
Statement* MakeIf(int lineno, Expression* cond, Statement* then_,
                   Statement* else_);
Statement* MakeReturn(int lineno, Expression* e);
Statement* MakeSeq(int lineno, Statement* s1, Statement* s2);
Statement* MakeBlock(int lineno, Statement* s);
Statement* MakeWhile(int lineno, Expression* cond, Statement* body);
Statement* MakeBreak(int lineno);
Statement* MakeContinue(int lineno);
Statement* MakeMatch(int lineno, Expression* exp,
                      list<pair<Expression*, Statement*> >* clauses);

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
    struct {
      string* name;
      Expression* type;
    } field;
  } u;
};

Member* MakeField(int lineno, string name, Expression* type);

/***** Declarations *****/

struct StructDefinition {
  int lineno;
  string* name;
  list<Member*>* members;
};

enum DeclarationKind {
  FunctionDeclaration,
  StructDeclaration,
  ChoiceDeclaration
};

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

struct FunctionDefinition* MakeFun_def(int lineno, string name,
                                        Expression* ret_type, Expression* param,
                                        Statement* body);
void print_fun_def(struct FunctionDefinition*);
void print_fun_def_depth(struct FunctionDefinition*, int);

Declaration* MakeFun_decl(struct FunctionDefinition* f);
Declaration* MakeStruct_decl(int lineno, string name, list<Member*>* members);
Declaration* MakeChoice_decl(int lineno, string name,
                              list<pair<string, Expression*> >* alts);

void print_decl(Declaration*);

void print_string(string* s);

template <class T>
T find_field(string field, vector<pair<string, T> >* inits) {
  for (auto i = inits->begin(); i != inits->end(); ++i) {
    if (i->first == field)
      return i->second;
  }
  throw std::domain_error(field);
}

template <class T>
T find_alist(string field, list<pair<string, T> >* inits) {
  for (auto i = inits->begin(); i != inits->end(); ++i) {
    if (i->first == field)
      return i->second;
  }
  throw std::domain_error(field);
}

#endif
