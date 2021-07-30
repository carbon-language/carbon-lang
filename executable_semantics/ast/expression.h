// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/syntax/paren_contents.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

struct Expression;
class Pattern;

// Converts paren_contents to an Expression, interpreting the parentheses as
// grouping if their contents permit that interpretation, or as forming a
// tuple otherwise.
auto ExpressionFromParenContents(
    int line_num, const ParenContents<Expression>& paren_contents)
    -> const Expression*;

// Converts paren_contents to an Expression, interpreting the parentheses as
// forming a tuple.
auto TupleExpressionFromParenContents(
    int line_num, const ParenContents<Expression>& paren_contents)
    -> const Expression*;

// A FieldInitializer represents the initialization of a single tuple field.
struct FieldInitializer {
  FieldInitializer(std::string name, const Expression* expression)
      : name(std::move(name)), expression(expression) {}

  // The field name. Cannot be empty.
  std::string name;

  // The expression that initializes the field.
  const Expression* expression;
};

enum class ExpressionKind {
  BoolTypeLiteral,
  BoolLiteral,
  CallExpression,
  FunctionTypeLiteral,
  FieldAccessExpression,
  IndexExpression,
  IntTypeLiteral,
  ContinuationTypeLiteral,  // The type of a continuation value.
  IntLiteral,
  PrimitiveOperatorExpression,
  TupleLiteral,
  TypeTypeLiteral,
  IdentifierExpression,
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

// Provides the detail of a return's expression, where it may be either
// implicitly `()` or an explicit expression.
struct ReturnInfo {
  // Default constructor for FunctionDefinition.
  ReturnInfo() : ReturnInfo(-1) {}
  // An implicit return.
  explicit ReturnInfo(int line_num);
  // An explicit return.
  explicit ReturnInfo(const Expression* exp);

  // True if the expression was implicitly constructed.
  bool is_implicit;

  // The expression to use for the return.
  const Expression* exp;
};

struct IdentifierExpression {
  static constexpr ExpressionKind Kind = ExpressionKind::IdentifierExpression;
  std::string name;
};

struct FieldAccessExpression {
  static constexpr ExpressionKind Kind = ExpressionKind::FieldAccessExpression;
  const Expression* aggregate;
  std::string field;
};

struct IndexExpression {
  static constexpr ExpressionKind Kind = ExpressionKind::IndexExpression;
  const Expression* aggregate;
  const Expression* offset;
};

struct IntLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::IntLiteral;
  int value;
};

struct BoolLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::BoolLiteral;
  bool value;
};

struct TupleLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::TupleLiteral;
  std::vector<FieldInitializer> fields;
};

struct PrimitiveOperatorExpression {
  static constexpr ExpressionKind Kind =
      ExpressionKind::PrimitiveOperatorExpression;
  Operator op;
  std::vector<const Expression*> arguments;
};

struct CallExpression {
  static constexpr ExpressionKind Kind = ExpressionKind::CallExpression;
  const Expression* function;
  const Expression* argument;
};

struct FunctionTypeLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::FunctionTypeLiteral;
  const Expression* parameter;
  ReturnInfo return_type;
};

struct BoolTypeLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::BoolTypeLiteral;
};

struct IntTypeLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::IntTypeLiteral;
};

struct ContinuationTypeLiteral {
  static constexpr ExpressionKind Kind =
      ExpressionKind::ContinuationTypeLiteral;
};

struct TypeTypeLiteral {
  static constexpr ExpressionKind Kind = ExpressionKind::TypeTypeLiteral;
};

struct Expression {
  static auto MakeIdentifierExpression(int line_num, std::string var)
      -> const Expression*;
  static auto MakeIntLiteral(int line_num, int i) -> const Expression*;
  static auto MakeBoolLiteral(int line_num, bool b) -> const Expression*;
  static auto MakePrimitiveOperatorExpression(
      int line_num, Operator op, std::vector<const Expression*> args)
      -> const Expression*;
  static auto MakeCallExpression(int line_num, const Expression* fun,
                                 const Expression* arg) -> const Expression*;
  static auto MakeFieldAccessExpression(int line_num, const Expression* exp,
                                        std::string field) -> const Expression*;
  static auto MakeTupleLiteral(int line_num, std::vector<FieldInitializer> args)
      -> const Expression*;
  static auto MakeIndexExpression(int line_num, const Expression* exp,
                                  const Expression* i) -> const Expression*;
  static auto MakeTypeTypeLiteral(int line_num) -> const Expression*;
  static auto MakeIntTypeLiteral(int line_num) -> const Expression*;
  static auto MakeBoolTypeLiteral(int line_num) -> const Expression*;
  static auto MakeFunctionTypeLiteral(int line_num, const Expression* param,
                                      ReturnInfo ret) -> const Expression*;
  static auto MakeContinuationTypeLiteral(int line_num) -> const Expression*;

  auto GetIdentifierExpression() const -> const IdentifierExpression&;
  auto GetFieldAccessExpression() const -> const FieldAccessExpression&;
  auto GetIndexExpression() const -> const IndexExpression&;
  auto GetIntLiteral() const -> int;
  auto GetBoolLiteral() const -> bool;
  auto GetTupleLiteral() const -> const TupleLiteral&;
  auto GetPrimitiveOperatorExpression() const
      -> const PrimitiveOperatorExpression&;
  auto GetCallExpression() const -> const CallExpression&;
  auto GetFunctionTypeLiteral() const -> const FunctionTypeLiteral&;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  inline auto tag() const -> ExpressionKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

  int line_num;

 private:
  std::variant<IdentifierExpression, FieldAccessExpression, IndexExpression,
               IntLiteral, BoolLiteral, TupleLiteral,
               PrimitiveOperatorExpression, CallExpression, FunctionTypeLiteral,
               BoolTypeLiteral, IntTypeLiteral, ContinuationTypeLiteral,
               TypeTypeLiteral>
      value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
