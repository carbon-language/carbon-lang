// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <iostream>

#include "executable_semantics/utility/fatal.h"

namespace Carbon {

static void PrintOp(PrimitiveOperatorExpression::Operation op) {
  switch (op) {
    case PrimitiveOperatorExpression::Operation::Neg:
      std::cout << "-";
      break;
    case PrimitiveOperatorExpression::Operation::Add:
      std::cout << "+";
      break;
    case PrimitiveOperatorExpression::Operation::Sub:
      std::cout << "-";
      break;
    case PrimitiveOperatorExpression::Operation::Not:
      std::cout << "!";
      break;
    case PrimitiveOperatorExpression::Operation::And:
      std::cout << "&&";
      break;
    case PrimitiveOperatorExpression::Operation::Or:
      std::cout << "||";
      break;
    case PrimitiveOperatorExpression::Operation::Eq:
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
  for (auto const& e : elements) {
    std::cout << separator << e.first << " = ";
    e.second.Print();
    separator = ", ";
  }
  std::cout << ")";
}

auto IntegerExpression::Print() const -> void { std::cout << value; }

auto BooleanExpression::Print() const -> void {
  std::cout << std::boolalpha;
  std::cout << value;
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
    t->Print();
  } else {
    std::cout << "(";
    argumentTuple.Print();
    std::cout << ")";
  }
}

auto BoolTypeExpression::Print() const -> void { std::cout << "Bool"; }

auto IntTypeExpression::Print() const -> void { std::cout << "Int"; }

auto TypeTypeExpression::Print() const -> void { std::cout << "Type"; }

auto AutoTypeExpression::Print() const -> void { std::cout << "auto"; }

auto FunctionTypeExpression::Print() const -> void {
  std::cout << "fn ";
  parameterTupleType.Print();
  std::cout << " -> ";
  returnType.Print();
}

auto ExpressionSource::fatalLValAction() const -> void {
  fatal("internal error in handle_value, LValAction");
}

auto ExpressionSource::fatalBadExpressionContext() const -> void {
  fatal("internal error, bad expression context in handle_value");
}

}  // namespace Carbon
