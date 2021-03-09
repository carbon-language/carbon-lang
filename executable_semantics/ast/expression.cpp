// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <iostream>

namespace Carbon {

static void PrintOp(Operator op) {
  switch (op) {
    case Operator::Neg:
      std::cout << "-";
      break;
    case Operator::Add:
      std::cout << "+";
      break;
    case Operator::Sub:
      std::cout << "-";
      break;
    case Operator::Not:
      std::cout << "!";
      break;
    case Operator::And:
      std::cout << "&&";
      break;
    case Operator::Or:
      std::cout << "||";
      break;
    case Operator::Eq:
      std::cout << "==";
      break;
  }
}

auto IndexExpression::Print() const -> void {
  aggregate.Print();
  std::cout << "[";
  offset.Print();
  std::cout << "]";
}

auto GetFieldExpression::Print() const -> void {
  aggregate.Print();
  std::cout << ".";
  std::cout << fieldName;
}

auto TupleExpression::Print() const -> void {
  std::cout << "(";
  const char* separator = "";
  for (auto iter = fields->begin(); iter != fields->end(); ++iter) {
    std::cout << separator << iter->first << " = ";
    PrintExp(iter->second);
    separator = ", ";
  }
  std::cout << ")";
}

auto IntegerExpression::Print() const -> void { std::cout << value; }

auto BooleanExpression::Print() const -> void {
  std::cout << std::boolalpha;
  std::cout << e->u.boolean;
}

auto PrimitiveOperatorExpression::Print() const -> void {
  std::cout << "(";
  auto p = arguments.begin();
  if (arguments.size() == 2) {
    (p++)->Print();
    std::cout << " ";
  }
  PrintOp(operation);
  while (p != arguments.end()) {
    std::cout << " ";
    (p++)->Print();
  }
  std::cout << ")";
}

auto VariableExpression::Print() const -> void { std::cout << name; }

auto PatternVariableExpression::Print() const -> void {
  type.Print();
  std::cout << ": " << name;
}

auto CallExpression::Print() const -> void {
  function.Print();
  if (auto t = argumentTuple.As<TupleExpression>()) {
    t.Print();
  } else {
    std::cout << "(";
    argumentTuple.Print();
    std::cout << ")";
  }
}

auto BoolTypeExpression::Print() const -> void { std::cout << "Bool"; }

auto IntTypeExpression::Print() const -> void { std::cout << "Int"; }

auto AutoTypeExpression::Print() const -> void { std::cout << "auto"; }

case ExpressionKind::AutoT:
  std::cout << "auto";
  break;
case ExpressionKind::FunctionT:
  std::cout << "fn ";
  PrintExp(e->u.function_type.parameter);
  std::cout << " -> ";
  PrintExp(e->u.function_type.return_type);
  break;
}  // namespace Carbon
}

}  // namespace Carbon
