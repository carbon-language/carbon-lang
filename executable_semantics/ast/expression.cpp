// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <iostream>

namespace Carbon {

auto Expression::GetVariable() const -> const Variable& {
  return std::get<Variable>(value);
}

auto Expression::GetFieldAccess() const -> const FieldAccess& {
  return std::get<FieldAccess>(value);
}

auto Expression::GetIndex() const -> const Index& {
  return std::get<Index>(value);
}

auto Expression::GetPatternVariable() const -> const PatternVariable& {
  return std::get<PatternVariable>(value);
}

auto Expression::GetInteger() const -> int {
  return std::get<IntLiteral>(value).value;
}

auto Expression::GetBoolean() const -> bool {
  return std::get<BoolLiteral>(value).value;
}

auto Expression::GetTuple() const -> const Tuple& {
  return std::get<Tuple>(value);
}

auto Expression::GetPrimitiveOperator() const -> const PrimitiveOperator& {
  return std::get<PrimitiveOperator>(value);
}

auto Expression::GetCall() const -> const Call& {
  return std::get<Call>(value);
}

auto Expression::GetFunctionType() const -> const FunctionType& {
  return std::get<FunctionType>(value);
}

auto Expression::MakeTypeType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = TypeT();
  return t;
}

auto Expression::MakeIntType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = IntT();
  return t;
}

auto Expression::MakeBoolType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = BoolT();
  return t;
}

auto Expression::MakeAutoType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = AutoT();
  return t;
}

// Returns a Continuation type AST node at the given source location.
auto Expression::MakeContinuationType(int line_num) -> const Expression* {
  auto* type = new Expression();
  type->line_num = line_num;
  type->value = ContinuationT();
  return type;
}

auto Expression::MakeFunType(int line_num, const Expression* param,
                             const Expression* ret) -> const Expression* {
  auto* t = new Expression();
  t->line_num = line_num;
  t->value = FunctionType({.parameter = param, .return_type = ret});
  return t;
}

auto Expression::MakeVar(int line_num, std::string var) -> const Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->value = Variable({.name = std::move(var)});
  return v;
}

auto Expression::MakeVarPat(int line_num, std::string var,
                            const Expression* type) -> const Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->value = PatternVariable({.name = std::move(var), .type = type});
  return v;
}

auto Expression::MakeInt(int line_num, int i) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = IntLiteral({.value = i});
  return e;
}

auto Expression::MakeBool(int line_num, bool b) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = BoolLiteral({.value = b});
  return e;
}

auto Expression::MakeOp(int line_num, enum Operator op,
                        std::vector<const Expression*> args)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = PrimitiveOperator({.op = op, .arguments = std::move(args)});
  return e;
}

auto Expression::MakeUnOp(int line_num, enum Operator op, const Expression* arg)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = PrimitiveOperator({.op = op, .arguments = {arg}});
  return e;
}

auto Expression::MakeBinOp(int line_num, enum Operator op,
                           const Expression* arg1, const Expression* arg2)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = PrimitiveOperator({.op = op, .arguments = {arg1, arg2}});
  return e;
}

auto Expression::MakeCall(int line_num, const Expression* fun,
                          const Expression* arg) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = Call({.function = fun, .argument = arg});
  return e;
}

auto Expression::MakeGetField(int line_num, const Expression* exp,
                              std::string field) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = FieldAccess({.aggregate = exp, .field = std::move(field)});
  return e;
}

auto Expression::MakeTuple(int line_num, std::vector<FieldInitializer> args)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  int i = 0;
  bool seen_named_member = false;
  for (auto& arg : args) {
    if (arg.name == "") {
      if (seen_named_member) {
        std::cerr << line_num
                  << ": positional members must come before named members"
                  << std::endl;
        exit(-1);
      }
      arg.name = std::to_string(i);
      ++i;
    } else {
      seen_named_member = true;
    }
  }
  e->value = Tuple({.fields = args});
  return e;
}

auto Expression::MakeIndex(int line_num, const Expression* exp,
                           const Expression* i) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->value = Index({.aggregate = exp, .offset = i});
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
    case ExpressionKind::Index:
      PrintExp(e->GetIndex().aggregate);
      std::cout << "[";
      PrintExp(e->GetIndex().offset);
      std::cout << "]";
      break;
    case ExpressionKind::GetField:
      PrintExp(e->GetFieldAccess().aggregate);
      std::cout << ".";
      std::cout << e->GetFieldAccess().field;
      break;
    case ExpressionKind::Tuple:
      std::cout << "(";
      PrintFields(e->GetTuple().fields);
      std::cout << ")";
      break;
    case ExpressionKind::Integer:
      std::cout << e->GetInteger();
      break;
    case ExpressionKind::Boolean:
      std::cout << std::boolalpha;
      std::cout << e->GetBoolean();
      break;
    case ExpressionKind::PrimitiveOp: {
      std::cout << "(";
      PrimitiveOperator op = e->GetPrimitiveOperator();
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
    case ExpressionKind::Variable:
      std::cout << e->GetVariable().name;
      break;
    case ExpressionKind::PatternVariable:
      PrintExp(e->GetPatternVariable().type);
      std::cout << ": ";
      std::cout << e->GetPatternVariable().name;
      break;
    case ExpressionKind::Call:
      PrintExp(e->GetCall().function);
      if (e->GetCall().argument->tag() == ExpressionKind::Tuple) {
        PrintExp(e->GetCall().argument);
      } else {
        std::cout << "(";
        PrintExp(e->GetCall().argument);
        std::cout << ")";
      }
      break;
    case ExpressionKind::BoolT:
      std::cout << "Bool";
      break;
    case ExpressionKind::IntT:
      std::cout << "Int";
      break;
    case ExpressionKind::TypeT:
      std::cout << "Type";
      break;
    case ExpressionKind::AutoT:
      std::cout << "auto";
      break;
    case ExpressionKind::ContinuationT:
      std::cout << "Continuation";
      break;
    case ExpressionKind::FunctionT:
      std::cout << "fn ";
      PrintExp(e->GetFunctionType().parameter);
      std::cout << " -> ";
      PrintExp(e->GetFunctionType().return_type);
      break;
  }
}

}  // namespace Carbon
