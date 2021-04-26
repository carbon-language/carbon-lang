// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <string>
#include <vector>

namespace Carbon {

struct Expression;

// A FieldInitializer represents the initialization of a single tuple field.
struct FieldInitializer {
  // The field name. For a positional field, this may be empty.
  std::string name;

  // The expression that initializes the field.
  const Expression* expression;
};

enum class ExpressionKind {
  AutoT,
  BoolT,
  Boolean,
  Call,
  FunctionT,
  GetField,
  Index,
  IntT,
  ContinuationT,  // The type of a continuation value.
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
      const Expression* aggregate;
      std::string* field;
    } get_field;

    struct {
      const Expression* aggregate;
      const Expression* offset;
    } index;

    struct {
      std::string* name;
      const Expression* type;
    } pattern_variable;

    int integer;
    bool boolean;

    struct {
      std::vector<FieldInitializer>* fields;
    } tuple;

    struct {
      Operator op;
      std::vector<const Expression*>* arguments;
    } primitive_op;

    struct {
      const Expression* function;
      const Expression* argument;
    } call;

    struct {
      const Expression* parameter;
      const Expression* return_type;
    } function_type;

  } u;
};

auto MakeVar(int line_num, std::string var) -> const Expression*;
auto MakeVarPat(int line_num, std::string var, const Expression* type)
    -> const Expression*;
auto MakeInt(int line_num, int i) -> const Expression*;
auto MakeBool(int line_num, bool b) -> const Expression*;
auto MakeOp(int line_num, Operator op, std::vector<const Expression*>* args)
    -> const Expression*;
auto MakeUnOp(int line_num, enum Operator op, const Expression* arg)
    -> const Expression*;
auto MakeBinOp(int line_num, enum Operator op, const Expression* arg1,
               const Expression* arg2) -> const Expression*;
auto MakeCall(int line_num, const Expression* fun, const Expression* arg)
    -> const Expression*;
auto MakeGetField(int line_num, const Expression* exp, std::string field)
    -> const Expression*;
auto MakeTuple(int line_num, std::vector<FieldInitializer>* args)
    -> const Expression*;
// Create an AST node for an empty tuple.
auto MakeUnit(int line_num) -> const Expression*;
auto MakeIndex(int line_num, const Expression* exp, const Expression* i)
    -> const Expression*;

auto MakeTypeType(int line_num) -> const Expression*;
auto MakeIntType(int line_num) -> const Expression*;
auto MakeBoolType(int line_num) -> const Expression*;
auto MakeFunType(int line_num, const Expression* param, const Expression* ret)
    -> const Expression*;
auto MakeAutoType(int line_num) -> const Expression*;
// Returns a Continuation type AST node at the given source location,
// which is the type of a continuation value.
auto MakeContinuationType(int line_num) -> const Expression*;

void PrintExp(const Expression* exp);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
