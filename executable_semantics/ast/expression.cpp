// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <cassert>
#include <iostream>

namespace Carbon {

Variable Expression::GetVariable() const {
  assert(tag == ExpressionKind::Variable);
  return u.variable;
}

FieldAccess Expression::GetFieldAccess() const {
  assert(tag == ExpressionKind::GetField);
  return u.get_field;
}

Index Expression::GetIndex() const {
  assert(tag == ExpressionKind::Index);
  return u.index;
}

PatternVariable Expression::GetPatternVariable() const {
  assert(tag == ExpressionKind::PatternVariable);
  return u.pattern_variable;
}

int Expression::GetInteger() const {
  assert(tag == ExpressionKind::Integer);
  return u.integer;
}

bool Expression::GetBoolean() const {
  assert(tag == ExpressionKind::Boolean);
  return u.boolean;
}

Tuple Expression::GetTuple() const {
  assert(tag == ExpressionKind::Tuple);
  return u.tuple;
}

PrimitiveOperator Expression::GetPrimitiveOperator() const {
  assert(tag == ExpressionKind::PrimitiveOp);
  return u.primitive_op;
}

Call Expression::GetCall() const {
  assert(tag == ExpressionKind::Call);
  return u.call;
}

FunctionType Expression::GetFunctionType() const {
  assert(tag == ExpressionKind::FunctionT);
  return u.function_type;
}

auto Expression::MakeTypeType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::TypeT;
  t->line_num = line_num;
  return t;
}

auto Expression::MakeIntType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::IntT;
  t->line_num = line_num;
  return t;
}

auto Expression::MakeBoolType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::BoolT;
  t->line_num = line_num;
  return t;
}

auto Expression::MakeAutoType(int line_num) -> const Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::AutoT;
  t->line_num = line_num;
  return t;
}

// Returns a Continuation type AST node at the given source location.
auto Expression::MakeContinuationType(int line_num) -> const Expression* {
  auto* type = new Expression();
  type->tag = ExpressionKind::ContinuationT;
  type->line_num = line_num;
  return type;
}

auto Expression::MakeFunType(int line_num, const Expression* param,
                             const Expression* ret) -> const Expression* {
  auto* t = new Expression();
  t->tag = ExpressionKind::FunctionT;
  t->line_num = line_num;
  t->u.function_type.parameter = param;
  t->u.function_type.return_type = ret;
  return t;
}

auto Expression::MakeVar(int line_num, std::string var) -> const Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->tag = ExpressionKind::Variable;
  v->u.variable.name = new std::string(std::move(var));
  return v;
}

auto Expression::MakeVarPat(int line_num, std::string var,
                            const Expression* type) -> const Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->tag = ExpressionKind::PatternVariable;
  v->u.pattern_variable.name = new std::string(std::move(var));
  v->u.pattern_variable.type = type;
  return v;
}

auto Expression::MakeInt(int line_num, int i) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Integer;
  e->u.integer = i;
  return e;
}

auto Expression::MakeBool(int line_num, bool b) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Boolean;
  e->u.boolean = b;
  return e;
}

auto Expression::MakeOp(int line_num, enum Operator op,
                        std::vector<const Expression*>* args)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::PrimitiveOp;
  e->u.primitive_op.op = op;
  e->u.primitive_op.arguments = args;
  return e;
}

auto Expression::MakeUnOp(int line_num, enum Operator op, const Expression* arg)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::PrimitiveOp;
  e->u.primitive_op.op = op;
  auto* args = new std::vector<const Expression*>();
  args->push_back(arg);
  e->u.primitive_op.arguments = args;
  return e;
}

auto Expression::MakeBinOp(int line_num, enum Operator op,
                           const Expression* arg1, const Expression* arg2)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::PrimitiveOp;
  e->u.primitive_op.op = op;
  auto* args = new std::vector<const Expression*>();
  args->push_back(arg1);
  args->push_back(arg2);
  e->u.primitive_op.arguments = args;
  return e;
}

auto Expression::MakeCall(int line_num, const Expression* fun,
                          const Expression* arg) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Call;
  e->u.call.function = fun;
  e->u.call.argument = arg;
  return e;
}

auto Expression::MakeGetField(int line_num, const Expression* exp,
                              std::string field) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::GetField;
  e->u.get_field.aggregate = exp;
  e->u.get_field.field = new std::string(std::move(field));
  return e;
}

auto Expression::MakeTuple(int line_num, std::vector<FieldInitializer>* args)
    -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Tuple;
  int i = 0;
  bool seen_named_member = false;
  for (auto& arg : *args) {
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
  e->u.tuple.fields = args;
  return e;
}

// Create an AST node for an empty tuple.
// TODO(geoffromer): remove this and rewrite its callers to use
// `MakeTuple(line_num, {})`, once that works.
auto Expression::MakeUnit(int line_num) -> const Expression* {
  auto* unit = new Expression();
  unit->line_num = line_num;
  unit->tag = ExpressionKind::Tuple;
  auto* args = new std::vector<FieldInitializer>();
  unit->u.tuple.fields = args;
  return unit;
}

auto Expression::MakeIndex(int line_num, const Expression* exp,
                           const Expression* i) -> const Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = ExpressionKind::Index;
  e->u.index.aggregate = exp;
  e->u.index.offset = i;
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

static void PrintFields(std::vector<FieldInitializer>* fields) {
  int i = 0;
  for (auto iter = fields->begin(); iter != fields->end(); ++iter, ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << iter->name << " = ";
    PrintExp(iter->expression);
  }
}

void PrintExp(const Expression* e) {
  switch (e->tag) {
    case ExpressionKind::Index:
      PrintExp(e->GetIndex().aggregate);
      std::cout << "[";
      PrintExp(e->GetIndex().offset);
      std::cout << "]";
      break;
    case ExpressionKind::GetField:
      PrintExp(e->GetFieldAccess().aggregate);
      std::cout << ".";
      std::cout << *e->GetFieldAccess().field;
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
      if (op.arguments->size() == 0) {
        PrintOp(op.op);
      } else if (op.arguments->size() == 1) {
        PrintOp(op.op);
        std::cout << " ";
        auto iter = op.arguments->begin();
        PrintExp(*iter);
      } else if (op.arguments->size() == 2) {
        auto iter = op.arguments->begin();
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
      std::cout << *e->GetVariable().name;
      break;
    case ExpressionKind::PatternVariable:
      PrintExp(e->GetPatternVariable().type);
      std::cout << ": ";
      std::cout << *e->GetPatternVariable().name;
      break;
    case ExpressionKind::Call:
      PrintExp(e->GetCall().function);
      if (e->GetCall().argument->tag == ExpressionKind::Tuple) {
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
