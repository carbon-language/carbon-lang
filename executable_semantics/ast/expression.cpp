// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <optional>

#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

auto ExpressionFromParenContents(
    int line_num, const ParenContents<Expression>& paren_contents)
    -> const Expression* {
  std::optional<const Expression*> single_term = paren_contents.SingleTerm();
  if (single_term.has_value()) {
    return *single_term;
  } else {
    return TupleExpressionFromParenContents(line_num, paren_contents);
  }
}

auto TupleExpressionFromParenContents(
    int line_num, const ParenContents<Expression>& paren_contents)
    -> const Expression* {
  return Expression::MakeTupleLiteral(
      line_num, paren_contents.TupleElements<FieldInitializer>(line_num));
}

auto Expression::GetIdentifierExpression() const
    -> const IdentifierExpression& {
  return std::get<IdentifierExpression>(value);
}

auto Expression::GetFieldAccessExpression() const
    -> const FieldAccessExpression& {
  return std::get<FieldAccessExpression>(value);
}

auto Expression::GetIndexExpression() const -> const IndexExpression& {
  return std::get<IndexExpression>(value);
}

auto Expression::GetIntLiteral() const -> int {
  return std::get<IntLiteral>(value).value;
}

auto Expression::GetBoolLiteral() const -> bool {
  return std::get<BoolLiteral>(value).value;
}

auto Expression::GetTupleLiteral() const -> const TupleLiteral& {
  return std::get<TupleLiteral>(value);
}

auto Expression::GetPrimitiveOperatorExpression() const
    -> const PrimitiveOperatorExpression& {
  return std::get<PrimitiveOperatorExpression>(value);
}

auto Expression::GetCallExpression() const -> const CallExpression& {
  return std::get<CallExpression>(value);
}

auto Expression::GetFunctionTypeLiteral() const -> const FunctionTypeLiteral& {
  return std::get<FunctionTypeLiteral>(value);
}

auto Expression::MakeTypeTypeLiteral(int line_num) -> const Expression* {
  auto* t = global_arena->New<Expression>();
  t->line_num = line_num;
  t->value = TypeTypeLiteral();
  return t;
}

auto Expression::MakeIntTypeLiteral(int line_num) -> const Expression* {
  auto* t = global_arena->New<Expression>();
  t->line_num = line_num;
  t->value = IntTypeLiteral();
  return t;
}

auto Expression::MakeBoolTypeLiteral(int line_num) -> const Expression* {
  auto* t = global_arena->New<Expression>();
  t->line_num = line_num;
  t->value = BoolTypeLiteral();
  return t;
}

// Returns a Continuation type AST node at the given source location.
auto Expression::MakeContinuationTypeLiteral(int line_num)
    -> const Expression* {
  auto* type = global_arena->New<Expression>();
  type->line_num = line_num;
  type->value = ContinuationTypeLiteral();
  return type;
}

auto Expression::MakeFunctionTypeLiteral(int line_num,
                                         const Expression* parameter,
                                         const Expression* return_type,
                                         bool is_omitted_return_type)
    -> const Expression* {
  auto* t = global_arena->New<Expression>();
  t->line_num = line_num;
  t->value =
      FunctionTypeLiteral({.parameter = parameter,
                           .return_type = return_type,
                           .is_omitted_return_type = is_omitted_return_type});
  return t;
}

auto Expression::MakeIdentifierExpression(int line_num, std::string var)
    -> const Expression* {
  auto* v = global_arena->New<Expression>();
  v->line_num = line_num;
  v->value = IdentifierExpression({.name = std::move(var)});
  return v;
}

auto Expression::MakeIntLiteral(int line_num, int i) -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value = IntLiteral({.value = i});
  return e;
}

auto Expression::MakeBoolLiteral(int line_num, bool b) -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value = BoolLiteral({.value = b});
  return e;
}

auto Expression::MakePrimitiveOperatorExpression(
    int line_num, enum Operator op, std::vector<const Expression*> args)
    -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value =
      PrimitiveOperatorExpression({.op = op, .arguments = std::move(args)});
  return e;
}

auto Expression::MakeCallExpression(int line_num, const Expression* fun,
                                    const Expression* arg)
    -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value = CallExpression({.function = fun, .argument = arg});
  return e;
}

auto Expression::MakeFieldAccessExpression(int line_num, const Expression* exp,
                                           std::string field)
    -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value =
      FieldAccessExpression({.aggregate = exp, .field = std::move(field)});
  return e;
}

auto Expression::MakeTupleLiteral(int line_num,
                                  std::vector<FieldInitializer> args)
    -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value = TupleLiteral({.fields = std::move(args)});
  return e;
}

auto Expression::MakeIndexExpression(int line_num, const Expression* exp,
                                     const Expression* i) -> const Expression* {
  auto* e = global_arena->New<Expression>();
  e->line_num = line_num;
  e->value = IndexExpression({.aggregate = exp, .offset = i});
  return e;
}

static void PrintOp(llvm::raw_ostream& out, Operator op) {
  switch (op) {
    case Operator::Add:
      out << "+";
      break;
    case Operator::Neg:
    case Operator::Sub:
      out << "-";
      break;
    case Operator::Mul:
    case Operator::Deref:
    case Operator::Ptr:
      out << "*";
      break;
    case Operator::Not:
      out << "not";
      break;
    case Operator::And:
      out << "and";
      break;
    case Operator::Or:
      out << "or";
      break;
    case Operator::Eq:
      out << "==";
      break;
  }
}

static void PrintFields(llvm::raw_ostream& out,
                        const std::vector<FieldInitializer>& fields) {
  llvm::ListSeparator sep;
  for (const auto& field : fields) {
    out << sep << field.name << " = " << field.expression;
  }
}

void Expression::Print(llvm::raw_ostream& out) const {
  switch (tag()) {
    case ExpressionKind::IndexExpression:
      out << *GetIndexExpression().aggregate << "["
          << *GetIndexExpression().offset << "]";
      break;
    case ExpressionKind::FieldAccessExpression:
      out << *GetFieldAccessExpression().aggregate << "."
          << GetFieldAccessExpression().field;
      break;
    case ExpressionKind::TupleLiteral:
      out << "(";
      PrintFields(out, GetTupleLiteral().fields);
      out << ")";
      break;
    case ExpressionKind::IntLiteral:
      out << GetIntLiteral();
      break;
    case ExpressionKind::BoolLiteral:
      out << (GetBoolLiteral() ? "true" : "false");
      break;
    case ExpressionKind::PrimitiveOperatorExpression: {
      out << "(";
      PrimitiveOperatorExpression op = GetPrimitiveOperatorExpression();
      if (op.arguments.size() == 0) {
        PrintOp(out, op.op);
      } else if (op.arguments.size() == 1) {
        PrintOp(out, op.op);
        out << " " << *op.arguments[0];
      } else if (op.arguments.size() == 2) {
        out << *op.arguments[0] << " ";
        PrintOp(out, op.op);
        out << " " << *op.arguments[1];
      }
      out << ")";
      break;
    }
    case ExpressionKind::IdentifierExpression:
      out << GetIdentifierExpression().name;
      break;
    case ExpressionKind::CallExpression:
      out << *GetCallExpression().function;
      if (GetCallExpression().argument->tag() == ExpressionKind::TupleLiteral) {
        out << *GetCallExpression().argument;
      } else {
        out << "(" << *GetCallExpression().argument << ")";
      }
      break;
    case ExpressionKind::BoolTypeLiteral:
      out << "Bool";
      break;
    case ExpressionKind::IntTypeLiteral:
      out << "Int";
      break;
    case ExpressionKind::TypeTypeLiteral:
      out << "Type";
      break;
    case ExpressionKind::ContinuationTypeLiteral:
      out << "Continuation";
      break;
    case ExpressionKind::FunctionTypeLiteral:
      out << "fn " << *GetFunctionTypeLiteral().parameter << " -> "
          << *GetFunctionTypeLiteral().return_type;
      break;
  }
}

}  // namespace Carbon
