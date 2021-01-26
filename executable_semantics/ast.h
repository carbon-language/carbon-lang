// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AST_H
#define AST_H

#include <cstdlib>
#include <exception>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>

/***** Utilities *****/

template <class T>
void PrintList(std::list<T*>* ts, void (*printer)(T*), const char* sep) {
  int i = 0;
  for (auto iter = ts->begin(); iter != ts->end(); ++iter, ++i) {
    if (i != 0) {
      printf("%s", sep);
    }
    printer(*iter);
  }
}

template <class T>
void PrintVector(std::vector<T*>* ts, void (*printer)(T*), const char* sep) {
  int i = 0;
  for (auto iter = ts->begin(); iter != ts->end(); ++iter, ++i) {
    if (i != 0) {
      printf("%s", sep);
    }
    printer(*iter);
  }
}

auto ReadFile(FILE* fp) -> char*;

extern char* input;

/***** Forward Declarations *****/

struct LValue;
struct Expression;
struct Statement;
struct FunctionDefinition;

using VarTypes = std::list<std::pair<std::string, Expression*>>;

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
      std::string* name;
    } variable;
    struct {
      Expression* aggregate;
      std::string* field;
    } get_field;
    struct {
      Expression* aggregate;
      Expression* offset;
    } index;
    struct {
      std::string* name;
      Expression* type;
    } pattern_variable;
    int integer;
    bool boolean;
    struct {
      std::vector<std::pair<std::string, Expression*>>* fields;
    } tuple;
    struct {
      Operator operator_;
      std::vector<Expression*>* arguments;
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

auto MakeVar(int lineno, std::string var) -> Expression*;
auto MakeVarPat(int lineno, std::string var, Expression* type) -> Expression*;
auto MakeInt(int lineno, int i) -> Expression*;
auto MakeBool(int lineno, bool b) -> Expression*;
auto MakeOp(int lineno, Operator op, std::vector<Expression*>* args)
    -> Expression*;
auto MakeUnOp(int lineno, enum Operator op, Expression* arg) -> Expression*;
auto MakeBinOp(int lineno, enum Operator op, Expression* arg1, Expression* arg2)
    -> Expression*;
auto MakeCall(int lineno, Expression* fun, Expression* arg) -> Expression*;
auto MakeGetField(int lineno, Expression* exp, std::string field)
    -> Expression*;
auto MakeTuple(int lineno,
               std::vector<std::pair<std::string, Expression*>>* args)
    -> Expression*;
auto MakeIndex(int lineno, Expression* exp, Expression* i) -> Expression*;

auto MakeTypeType(int lineno) -> Expression*;
auto MakeIntType(int lineno) -> Expression*;
auto MakeBoolType(int lineno) -> Expression*;
auto MakeFunType(int lineno, Expression* param, Expression* ret) -> Expression*;
auto MakeAutoType(int lineno) -> Expression*;

void PrintExp(Expression*);

/***** Expression or Field List *****/
/*
  This is used in the parsing of tuples and parenthesized expressions.
 */

enum ExpOrFieldListKind { Exp, FieldList };

struct ExpOrFieldList {
  ExpOrFieldListKind tag;
  union {
    Expression* exp;
    std::list<std::pair<std::string, Expression*>>* fields;
  } u;
};

auto MakeExp(Expression* exp) -> ExpOrFieldList*;
auto MakeFieldList(std::list<std::pair<std::string, Expression*>>* fields)
    -> ExpOrFieldList*;
auto MakeConstructorField(ExpOrFieldList* e1, ExpOrFieldList* e2)
    -> ExpOrFieldList*;

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
      Statement* then_stmt;
      Statement* else_stmt;
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
      std::list<std::pair<Expression*, Statement*>>* clauses;
    } match_stmt;
  } u;
};

auto MakeExpStmt(int lineno, Expression* exp) -> Statement*;
auto MakeAssign(int lineno, Expression* lhs, Expression* rhs) -> Statement*;
auto MakeVarDef(int lineno, Expression* pat, Expression* init) -> Statement*;
auto MakeIf(int lineno, Expression* cond, Statement* then_stmt,
            Statement* else_stmt) -> Statement*;
auto MakeReturn(int lineno, Expression* e) -> Statement*;
auto MakeSeq(int lineno, Statement* s1, Statement* s2) -> Statement*;
auto MakeBlock(int lineno, Statement* s) -> Statement*;
auto MakeWhile(int lineno, Expression* cond, Statement* body) -> Statement*;
auto MakeBreak(int lineno) -> Statement*;
auto MakeContinue(int lineno) -> Statement*;
auto MakeMatch(int lineno, Expression* exp,
               std::list<std::pair<Expression*, Statement*>>* clauses)
    -> Statement*;

void PrintStatement(Statement*, int);

/***** Function Definitions *****/

struct FunctionDefinition {
  int lineno;
  std::string name;
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
      std::string* name;
      Expression* type;
    } field;
  } u;
};

auto MakeField(int lineno, std::string name, Expression* type) -> Member*;

/***** Declarations *****/

struct StructDefinition {
  int lineno;
  std::string* name;
  std::list<Member*>* members;
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
      std::string* name;
      std::list<std::pair<std::string, Expression*>>* alternatives;
    } choice_def;
  } u;
};

auto MakeFunDef(int lineno, std::string name, Expression* ret_type,
                Expression* param, Statement* body)
    -> struct FunctionDefinition*;
void PrintFunDef(struct FunctionDefinition*);
void PrintFunDefDepth(struct FunctionDefinition*, int);

auto MakeFunDecl(struct FunctionDefinition* f) -> Declaration*;
auto MakeStructDecl(int lineno, std::string name, std::list<Member*>* members)
    -> Declaration*;
auto MakeChoiceDecl(int lineno, std::string name,
                    std::list<std::pair<std::string, Expression*>>* alts)
    -> Declaration*;

void PrintDecl(Declaration*);

void PrintString(std::string* s);

template <class T>
auto FindField(const std::string& field,
               std::vector<std::pair<std::string, T>>* inits) -> T {
  for (auto i = inits->begin(); i != inits->end(); ++i) {
    if (i->first == field) {
      return i->second;
    }
  }
  throw std::domain_error(field);
}

template <class T>
auto FindAlist(const std::string& field,
               std::list<std::pair<std::string, T>>* inits) -> T {
  for (auto i = inits->begin(); i != inits->end(); ++i) {
    if (i->first == field) {
      return i->second;
    }
  }
  throw std::domain_error(field);
}

#endif
