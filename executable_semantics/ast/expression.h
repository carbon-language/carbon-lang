// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <string>
#include <variant>
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

struct IntLiteral {
  int value;
};

struct BoolLiteral {
  bool value;
};

struct Tuple {
  std::vector<FieldInitializer>* fields;
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

struct AutoT {
};

struct BoolT {
};

struct IntT {
};

struct ContinuationT {
};

struct TypeT {
};

struct Expression {
  int line_num;
  ExpressionKind tag;

  static auto MakeVar(int line_num, std::string var) -> const Expression*;
  static auto MakeVarPat(int line_num, std::string var, const Expression* type)
      -> const Expression*;
  static auto MakeInt(int line_num, int i) -> const Expression*;
  static auto MakeBool(int line_num, bool b) -> const Expression*;
  static auto MakeOp(int line_num, Operator op,
                     std::vector<const Expression*>* args) -> const Expression*;
  static auto MakeUnOp(int line_num, enum Operator op, const Expression* arg)
      -> const Expression*;
  static auto MakeBinOp(int line_num, enum Operator op, const Expression* arg1,
                        const Expression* arg2) -> const Expression*;
  static auto MakeCall(int line_num, const Expression* fun,
                       const Expression* arg) -> const Expression*;
  static auto MakeGetField(int line_num, const Expression* exp,
                           std::string field) -> const Expression*;
  static auto MakeTuple(int line_num, std::vector<FieldInitializer>* args)
      -> const Expression*;
  static auto MakeUnit(int line_num) -> const Expression*;
  static auto MakeIndex(int line_num, const Expression* exp,
                        const Expression* i) -> const Expression*;
  static auto MakeTypeType(int line_num) -> const Expression*;
  static auto MakeIntType(int line_num) -> const Expression*;
  static auto MakeBoolType(int line_num) -> const Expression*;
  static auto MakeFunType(int line_num, const Expression* param,
                          const Expression* ret) -> const Expression*;
  static auto MakeAutoType(int line_num) -> const Expression*;
  static auto MakeContinuationType(int line_num) -> const Expression*;

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
  std::variant<
    Variable,
    FieldAccess,
    Index,
    PatternVariable,
    IntLiteral,
    BoolLiteral,
    Tuple,
    PrimitiveOperator,
    Call,
    FunctionType,
    AutoT,
    BoolT,
    IntT,
    ContinuationT,
    TypeT
  > value;
};

void PrintExp(const Expression* exp);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
