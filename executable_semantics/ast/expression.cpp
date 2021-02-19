// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <iostream>

namespace Carbon {

auto MakeTypeType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::TypeT;
  t->line_num = line_num;
  return t;
}

auto MakeIntType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::IntT;
  t->line_num = line_num;
  return t;
}

auto MakeBoolType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::BoolT;
  t->line_num = line_num;
  return t;
}

auto MakeAutoType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::AutoT;
  t->line_num = line_num;
  return t;
}

auto MakeFunType(int line_num, Expression* param, Expression* ret)
    -> Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::FunctionT;
  t->line_num = line_num;
  t->u.function_type.parameter = param;
  t->u.function_type.return_type = ret;
  return t;
}

auto MakeVar(int line_num, std::string var) -> Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->tag = ExpressionKind::Variable;
  v->u.variable.name = new std::string(std::move(var));
  return v;
}

auto MakeVarPat(int line_num, std::string var, Expression* type)
    -> Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->tag = ExpressionKind::PatternVariable;
  v->u.pattern_variable.name = new std::string(std::move(var));
  v->u.pattern_variable.type = type;
  return v;
}

auto MakeInt(int line_num, int i) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Integer;
  e->u.integer = i;
  return e;
}

auto MakeBool(int line_num, bool b) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Boolean;
  e->u.boolean = b;
  return e;
}

auto MakeOp(int line_num, enum Operator op, std::vector<Expression*>* args)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::PrimitiveOp;
  e->u.primitive_op.op = op;
  e->u.primitive_op.arguments = args;
  return e;
}

auto MakeUnOp(int line_num, enum Operator op, Expression* arg) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::PrimitiveOp;
  e->u.primitive_op.op = op;
  auto* args = new std::vector<Expression*>();
  args->push_back(arg);
  e->u.primitive_op.arguments = args;
  return e;
}

auto MakeBinOp(int line_num, enum Operator op, Expression* arg1,
               Expression* arg2) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::PrimitiveOp;
  e->u.primitive_op.op = op;
  auto* args = new std::vector<Expression*>();
  args->push_back(arg1);
  args->push_back(arg2);
  e->u.primitive_op.arguments = args;
  return e;
}

auto MakeCall(int line_num, Expression* fun, Expression* arg) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Call;
  e->u.call.function = fun;
  e->u.call.argument = arg;
  return e;
}

auto MakeGetField(int line_num, Expression* exp, std::string field)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::GetField;
  e->u.get_field.aggregate = exp;
  e->u.get_field.field = new std::string(std::move(field));
  return e;
}

auto MakeTuple(int line_num,
               std::vector<std::pair<std::string, Expression*>>* args)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Tuple;
  int i = 0;
  for (auto& arg : *args) {
    if (arg.first == "") {
      arg.first = std::to_string(i);
      ++i;
    }
  }
  e->u.tuple.fields = args;
  return e;
}

auto MakeIndex(int line_num, Expression* exp, Expression* i) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Index;
  e->u.index.aggregate = exp;
  e->u.index.offset = i;
  return e;
}

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

static void PrintFields(
    std::vector<std::pair<std::string, Expression*>>* fields) {
  int i = 0;
  for (auto iter = fields->begin(); iter != fields->end(); ++iter, ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << iter->first << " = ";
    PrintExp(iter->second);
  }
}

void PrintExp(Expression* e) {
  switch (e->tag) {
    case ExpressionKind::Index:
      PrintExp(e->u.index.aggregate);
      std::cout << "[";
      PrintExp(e->u.index.offset);
      std::cout << "]";
      break;
    case ExpressionKind::GetField:
      PrintExp(e->u.get_field.aggregate);
      std::cout << ".";
      std::cout << *e->u.get_field.field;
      break;
    case ExpressionKind::Tuple:
      std::cout << "(";
      PrintFields(e->u.tuple.fields);
      std::cout << ")";
      break;
    case ExpressionKind::Integer:
      std::cout << e->u.integer;
      break;
    case ExpressionKind::Boolean:
      std::cout << std::boolalpha;
      std::cout << e->u.boolean;
      break;
    case ExpressionKind::PrimitiveOp:
      std::cout << "(";
      if (e->u.primitive_op.arguments->size() == 0) {
        PrintOp(e->u.primitive_op.op);
      } else if (e->u.primitive_op.arguments->size() == 1) {
        PrintOp(e->u.primitive_op.op);
        std::cout << " ";
        auto iter = e->u.primitive_op.arguments->begin();
        PrintExp(*iter);
      } else if (e->u.primitive_op.arguments->size() == 2) {
        auto iter = e->u.primitive_op.arguments->begin();
        PrintExp(*iter);
        std::cout << " ";
        PrintOp(e->u.primitive_op.op);
        std::cout << " ";
        ++iter;
        PrintExp(*iter);
      }
      std::cout << ")";
      break;
    case ExpressionKind::Variable:
      std::cout << *e->u.variable.name;
      break;
    case ExpressionKind::PatternVariable:
      PrintExp(e->u.pattern_variable.type);
      std::cout << ": ";
      std::cout << *e->u.pattern_variable.name;
      break;
    case ExpressionKind::Call:
      PrintExp(e->u.call.function);
      if (e->u.call.argument->tag == ExpressionKind::Tuple) {
        PrintExp(e->u.call.argument);
      } else {
        std::cout << "(";
        PrintExp(e->u.call.argument);
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
    case ExpressionKind::FunctionT:
      std::cout << "fn ";
      PrintExp(e->u.function_type.parameter);
      std::cout << " -> ";
      PrintExp(e->u.function_type.return_type);
      break;
  }
}

}  // namespace Carbon
