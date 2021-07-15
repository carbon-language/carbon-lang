// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <iostream>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

namespace Carbon {

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

auto Expression::GetBindingExpression() const -> const BindingExpression& {
  return std::get<BindingExpression>(value);
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
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = TypeTypeLiteral();
  return t;
}

auto Expression::MakeIntTypeLiteral(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = IntTypeLiteral();
  return t;
}

auto Expression::MakeBoolTypeLiteral(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = BoolTypeLiteral();
  return t;
}

auto Expression::MakeAutoTypeLiteral(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = AutoTypeLiteral();
  return t;
}

// Returns a Continuation type AST node at the given source location.
auto Expression::MakeContinuationTypeLiteral(int line_num)
    -> const Expression* {
  auto* type = new Expression();
  type->line_num = line_num;
  type->value = ContinuationTypeLiteral();
  return type;
}

auto Expression::MakeFunctionTypeLiteral(int line_num, const Expression* param,
                                         const Expression* ret)
    -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = FunctionTypeLiteral({.parameter = param, .return_type = ret});
  return t;
}

auto Expression::MakeIdentifierExpression(int line_num, std::string var)
    -> const Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->value = IdentifierExpression({.name = std::move(var)});
  return v;
}

auto Expression::MakeBindingExpression(int line_num, std::string var,
                                       const Expression* type)
    -> const Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->value = BindingExpression({.name = std::move(var), .type = type});
  return v;
}

auto Expression::MakeIntLiteral(int line_num, int i) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = IntLiteral({.value = i});
  return e;
}

auto Expression::MakeBoolLiteral(int line_num, bool b) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = BoolLiteral({.value = b});
  return e;
}

auto Expression::MakePrimitiveOperatorExpression(
    int line_num, enum Operator op, std::vector<const Expression*> args)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value =
      PrimitiveOperatorExpression({.op = op, .arguments = std::move(args)});
  return e;
}

auto Expression::MakeCallExpression(int line_num, const Expression* fun,
                                    const Expression* arg)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = CallExpression({.function = fun, .argument = arg});
  return e;
}

auto Expression::MakeFieldAccessExpression(int line_num, const Expression* exp,
                                           std::string field)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value =
      FieldAccessExpression({.aggregate = exp, .field = std::move(field)});
  return e;
}

auto Expression::MakeTupleLiteral(int line_num,
                                  std::vector<FieldInitializer> args)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  int i = 0;
  bool seen_named_member = false;
  for (auto& arg : args) {
    if (arg.name == "") {
      if (seen_named_member) {
        llvm::report_fatal_error(
            llvm::Twine(line_num) +
            ": positional members must come before named members");
      }
      arg.name = std::to_string(i);
      ++i;
    } else {
      seen_named_member = true;
    }
  }
  e->value = TupleLiteral({.fields = args});
  return e;
}

auto Expression::MakeIndexExpression(int line_num, const Expression* exp,
                                     const Expression* i) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = IndexExpression({.aggregate = exp, .offset = i});
  return e;
}

static void PrintOp(Operator op) {
  switch (op) {
    case Operator::Add:
      std::cout << "+";
      break;
    case Operator::Neg:
    case Operator::Sub:
      std::cout << "-";
      break;
    case Operator::Mul:
    case Operator::Deref:
    case Operator::Ptr:
      std::cout << "*";
      break;
    case Operator::Not:
      std::cout << "not";
      break;
    case Operator::And:
      std::cout << "and";
      break;
    case Operator::Or:
      std::cout << "or";
      break;
    case Operator::Eq:
      std::cout << "==";
      break;
  }
}

static void PrintFields(const std::vector<FieldInitializer>& fields) {
  int i = 0;
  for (auto iter = fields.begin(); iter != fields.end(); ++iter, ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << iter->name << " = ";
    PrintExp(iter->expression);
  }
}

void PrintExp(const Expression* e) {
  switch (e->tag()) {
    case ExpressionKind::IndexExpression:
      PrintExp(e->GetIndexExpression().aggregate);
      std::cout << "[";
      PrintExp(e->GetIndexExpression().offset);
      std::cout << "]";
      break;
    case ExpressionKind::FieldAccessExpression:
      PrintExp(e->GetFieldAccessExpression().aggregate);
      std::cout << ".";
      std::cout << e->GetFieldAccessExpression().field;
      break;
    case ExpressionKind::TupleLiteral:
      std::cout << "(";
      PrintFields(e->GetTupleLiteral().fields);
      std::cout << ")";
      break;
    case ExpressionKind::IntLiteral:
      std::cout << e->GetIntLiteral();
      break;
    case ExpressionKind::BoolLiteral:
      std::cout << std::boolalpha;
      std::cout << e->GetBoolLiteral();
      break;
    case ExpressionKind::PrimitiveOperatorExpression: {
      std::cout << "(";
      PrimitiveOperatorExpression op = e->GetPrimitiveOperatorExpression();
      if (op.arguments.size() == 0) {
        PrintOp(op.op);
      } else if (op.arguments.size() == 1) {
        PrintOp(op.op);
        std::cout << " ";
        auto iter = op.arguments.begin();
        PrintExp(*iter);
      } else if (op.arguments.size() == 2) {
        auto iter = op.arguments.begin();
        PrintExp(*iter);
        std::cout << " ";
        PrintOp(op.op);
        std::cout << " ";
        ++iter;
        PrintExp(*iter);
      }
      std::cout << ")";
      break;
    }
    case ExpressionKind::IdentifierExpression:
      std::cout << e->GetIdentifierExpression().name;
      break;
    case ExpressionKind::BindingExpression:
      PrintExp(e->GetBindingExpression().type);
      std::cout << ": ";
      std::cout << e->GetBindingExpression().name;
      break;
    case ExpressionKind::CallExpression:
      PrintExp(e->GetCallExpression().function);
      if (e->GetCallExpression().argument->tag() ==
          ExpressionKind::TupleLiteral) {
        PrintExp(e->GetCallExpression().argument);
      } else {
        std::cout << "(";
        PrintExp(e->GetCallExpression().argument);
        std::cout << ")";
      }
      break;
    case ExpressionKind::BoolTypeLiteral:
      std::cout << "Bool";
      break;
    case ExpressionKind::IntTypeLiteral:
      std::cout << "Int";
      break;
    case ExpressionKind::TypeTypeLiteral:
      std::cout << "Type";
      break;
    case ExpressionKind::AutoTypeLiteral:
      std::cout << "auto";
      break;
    case ExpressionKind::ContinuationTypeLiteral:
      std::cout << "Continuation";
      break;
    case ExpressionKind::FunctionTypeLiteral:
      std::cout << "fn ";
      PrintExp(e->GetFunctionTypeLiteral().parameter);
      std::cout << " -> ";
      PrintExp(e->GetFunctionTypeLiteral().return_type);
      break;
  }
}

}  // namespace Carbon
