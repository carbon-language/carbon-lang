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

struct Expression;

struct Variable {
  std::string* name;
};

struct FieldAccess {
  const Expression* aggregate;
  std::string* field;
};

struct Index {
  const Expression* aggregate;
  const Expression* offset;
};

struct PatternVariable {
  std::string* name;
  const Expression* type;
};

struct Tuple {
  std::vector<std::pair<std::string, const Expression*>>* fields;
};

struct PrimitiveOperator {
  Operator op;
  std::vector<const Expression*>* arguments;
};

struct Call {
  const Expression* function;
  const Expression* argument;
};

struct FunctionType {
  const Expression* parameter;
  const Expression* return_type;
};

struct Expression {
  int line_num;
  ExpressionKind tag;

  Variable GetVariable() const;
  FieldAccess GetFieldAccess() const;
  Index GetIndex() const;
  PatternVariable GetPatternVariable() const;
  int GetInteger() const;
  bool GetBoolean() const;
  Tuple GetTuple() const;
  PrimitiveOperator GetPrimitiveOperator() const;
  Call GetCall() const;
  FunctionType GetFunctionType() const;

 private:
  union {
    Variable variable;
    FieldAccess get_field;
    Index index;
    PatternVariable pattern_variable;
    int integer;
    bool boolean;
    Tuple tuple;
    PrimitiveOperator primitive_op;
    Call call;
    FunctionType function_type;
  } u;

  // TODO: replace these constructors functions with real constructors
  //
  // RANT: The following long list of friend declarations is an
  // example of a problem in the design of C++. It is so focused on
  // classes and objects that it fails for modular procedural
  // programming. There are better ways to control access, for
  // example, going back to the module system of in CLU programming
  // language in the 1970's. -Jeremy

  friend auto MakeVar(int line_num, std::string var) -> const Expression*;
  friend auto MakeVarPat(int line_num, std::string var, const Expression* type)
      -> const Expression*;
  friend auto MakeInt(int line_num, int i) -> const Expression*;
  friend auto MakeBool(int line_num, bool b) -> const Expression*;
  friend auto MakeOp(int line_num, Operator op,
                     std::vector<const Expression*>* args) -> const Expression*;
  friend auto MakeUnOp(int line_num, enum Operator op, const Expression* arg)
      -> const Expression*;
  friend auto MakeBinOp(int line_num, enum Operator op, const Expression* arg1,
                        const Expression* arg2) -> const Expression*;
  friend auto MakeCall(int line_num, const Expression* fun,
                       const Expression* arg) -> const Expression*;
  friend auto MakeGetField(int line_num, const Expression* exp,
                           std::string field) -> const Expression*;
  friend auto MakeTuple(
      int line_num,
      std::vector<std::pair<std::string, const Expression*>>* args)
      -> const Expression*;
  friend auto MakeUnit(int line_num) -> const Expression*;
  friend auto MakeIndex(int line_num, const Expression* exp,
                        const Expression* i) -> const Expression*;
  friend auto MakeTypeType(int line_num) -> const Expression*;
  friend auto MakeIntType(int line_num) -> const Expression*;
  friend auto MakeBoolType(int line_num) -> const Expression*;
  friend auto MakeFunType(int line_num, const Expression* param,
                          const Expression* ret) -> const Expression*;
  friend auto MakeAutoType(int line_num) -> const Expression*;
  friend auto MakeContinuationType(int line_num) -> const Expression*;
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
auto MakeTuple(int line_num,
               std::vector<std::pair<std::string, const Expression*>>* args)
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
