// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <string>
#include <variant>
#include <vector>

#include "common/indirect_value.h"

namespace Carbon {

struct Expression;

// A FieldInitializer represents the initialization of a single tuple field.
struct FieldInitializer {
  // The field name. For a positional field, this may be empty.
  std::string name;

  // The expression that initializes the field.
  IndirectValue<Expression> expression;
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
  Deref,
  Eq,
  Mul,
  Neg,
  Not,
  Or,
  Sub,
  Ptr,
};

struct Expression;

struct Variable {
  static constexpr ExpressionKind Kind = ExpressionKind::Variable;
  std::string name;
};

struct FieldAccess {
  static constexpr ExpressionKind Kind = ExpressionKind::GetField;
  IndirectValue<Expression> aggregate;
  std::string field;
};

struct Index {
  static constexpr ExpressionKind Kind = ExpressionKind::Index;
  IndirectValue<Expression> aggregate;
  IndirectValue<Expression> offset;
};

struct PatternVariable {
  static constexpr ExpressionKind Kind = ExpressionKind::PatternVariable;
  std::string name;
  IndirectValue<Expression> type;
};

struct IntLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::Integer;
  int value;
};

struct BoolLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::Boolean;
  bool value;
};

struct Tuple {
  static constexpr ExpressionKind Kind = ExpressionKind::Tuple;
  std::vector<FieldInitializer> fields;
};

struct PrimitiveOperator {
  static constexpr ExpressionKind Kind = ExpressionKind::PrimitiveOp;
  Operator op;
  std::vector<Expression> arguments;
};

struct Call {
  static constexpr ExpressionKind Kind = ExpressionKind::Call;
  IndirectValue<Expression> function;
  IndirectValue<Expression> argument;
};

struct FunctionType {
  static constexpr ExpressionKind Kind = ExpressionKind::FunctionT;
  IndirectValue<Expression> parameter;
  IndirectValue<Expression> return_type;
};

struct AutoT {
  static constexpr ExpressionKind Kind = ExpressionKind::AutoT;
};

struct BoolT {
  static constexpr ExpressionKind Kind = ExpressionKind::BoolT;
};

struct IntT {
  static constexpr ExpressionKind Kind = ExpressionKind::IntT;
};

struct ContinuationT {
  static constexpr ExpressionKind Kind = ExpressionKind::ContinuationT;
};

struct TypeT {
  static constexpr ExpressionKind Kind = ExpressionKind::TypeT;
};

struct Expression {
  int line_num;
  inline auto tag() const -> ExpressionKind;

  static auto MakeVar(int line_num, std::string var) -> const Expression*;
  static auto MakeVarPat(int line_num, std::string var, Expression type)
      -> const Expression*;
  static auto MakeInt(int line_num, int i) -> const Expression*;
  static auto MakeBool(int line_num, bool b) -> const Expression*;
  static auto MakeOp(int line_num, Operator op, std::vector<Expression> args)
      -> const Expression*;
  static auto MakeUnOp(int line_num, enum Operator op, Expression arg)
      -> const Expression*;
  static auto MakeBinOp(int line_num, enum Operator op, Expression arg1,
                        Expression arg2) -> const Expression*;
  static auto MakeCall(int line_num, Expression fun, Expression arg)
      -> const Expression*;
  static auto MakeGetField(int line_num, const Expression* exp,
                           std::string field) -> const Expression*;
  static auto MakeTuple(int line_num, std::vector<FieldInitializer> args)
      -> const Expression*;
  static auto MakeIndex(int line_num, Expression exp, Expression i)
      -> const Expression*;
  static auto MakeTypeType(int line_num) -> const Expression*;
  static auto MakeIntType(int line_num) -> const Expression*;
  static auto MakeBoolType(int line_num) -> const Expression*;
  static auto MakeFunType(int line_num, Expression param, Expression ret)
      -> const Expression*;
  static auto MakeAutoType(int line_num) -> const Expression*;
  static auto MakeContinuationType(int line_num) -> const Expression*;

  auto GetVariable() const -> const Variable&;
  auto GetFieldAccess() const -> const FieldAccess&;
  auto GetIndex() const -> const Index&;
  auto GetPatternVariable() const -> const PatternVariable&;
  auto GetInteger() const -> int;
  auto GetBoolean() const -> bool;
  auto GetTuple() const -> const Tuple&;
  auto GetPrimitiveOperator() const -> const PrimitiveOperator&;
  auto GetCall() const -> const Call&;
  auto GetFunctionType() const -> const FunctionType&;

 private:
  std::variant<Variable, FieldAccess, Index, PatternVariable, IntLiteral,
               BoolLiteral, Tuple, PrimitiveOperator, Call, FunctionType, AutoT,
               BoolT, IntT, ContinuationT, TypeT>
      value;
};

void PrintExp(const Expression* exp);

// Implementation details only beyond this point

struct TagVisitor {
  template <typename Alternative>
  auto operator()(const Alternative&) -> ExpressionKind {
    return Alternative::Kind;
  }
};

auto Expression::tag() const -> ExpressionKind {
  return std::visit(TagVisitor(), value);
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
