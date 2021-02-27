// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <string>
#include <vector>

namespace Carbon {

enum class ExpressionKind {
  AutoT,
  BoolT,
  Boolean,
  Call,
  FunctionT,
  GetField,
  Index,
  IntT,
  Integer,
  PatternVariable,
  PrimitiveOp,
  Tuple,
  TypeT,
  Variable,
};

enum class Operator {
  Add,
  And,
  Eq,
  Neg,
  Not,
  Or,
  Sub,
};

struct Expression {
  int line_num;
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
      Operator op;
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

auto MakeVar(int line_num, std::string var) -> Expression*;
auto MakeVarPat(int line_num, std::string var, Expression* type) -> Expression*;
auto MakeInt(int line_num, int i) -> Expression*;
auto MakeBool(int line_num, bool b) -> Expression*;
auto MakeOp(int line_num, Operator op, std::vector<Expression*>* args)
    -> Expression*;
auto MakeUnOp(int line_num, enum Operator op, Expression* arg) -> Expression*;
auto MakeBinOp(int line_num, enum Operator op, Expression* arg1,
               Expression* arg2) -> Expression*;
auto MakeCall(int line_num, Expression* fun, Expression* arg) -> Expression*;
auto MakeGetField(int line_num, Expression* exp, std::string field)
    -> Expression*;
auto MakeTuple(int line_num,
               std::vector<std::pair<std::string, Expression*>>* args)
    -> Expression*;
auto MakeIndex(int line_num, Expression* exp, Expression* i) -> Expression*;

auto MakeTypeType(int line_num) -> Expression*;
auto MakeIntType(int line_num) -> Expression*;
auto MakeBoolType(int line_num) -> Expression*;
auto MakeFunType(int line_num, Expression* param, Expression* ret)
    -> Expression*;
auto MakeAutoType(int line_num) -> Expression*;

void PrintExp(Expression* exp);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
