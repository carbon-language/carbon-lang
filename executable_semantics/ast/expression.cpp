// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <optional>

#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

using llvm::cast;

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
  return global_arena->New<TupleLiteral>(
      line_num, paren_contents.TupleElements<FieldInitializer>(line_num));
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
  switch (Tag()) {
    case Expression::Kind::IndexExpression: {
      const auto& index = cast<IndexExpression>(*this);
      out << *index.Aggregate() << "[" << *index.Offset() << "]";
      break;
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*this);
      out << *access.Aggregate() << "." << access.Field();
      break;
    }
    case Expression::Kind::TupleLiteral:
      out << "(";
      PrintFields(out, cast<TupleLiteral>(*this).Fields());
      out << ")";
      break;
    case Expression::Kind::IntLiteral:
      out << cast<IntLiteral>(*this).Val();
      break;
    case Expression::Kind::BoolLiteral:
      out << (cast<BoolLiteral>(*this).Val() ? "true" : "false");
      break;
    case Expression::Kind::PrimitiveOperatorExpression: {
      out << "(";
      PrimitiveOperatorExpression op = cast<PrimitiveOperatorExpression>(*this);
      if (op.Arguments().size() == 0) {
        PrintOp(out, op.Op());
      } else if (op.Arguments().size() == 1) {
        PrintOp(out, op.Op());
        out << " " << *op.Arguments()[0];
      } else if (op.Arguments().size() == 2) {
        out << *op.Arguments()[0] << " ";
        PrintOp(out, op.Op());
        out << " " << *op.Arguments()[1];
      }
      out << ")";
      break;
    }
    case Expression::Kind::IdentifierExpression:
      out << cast<IdentifierExpression>(*this).Name();
      break;
    case Expression::Kind::CallExpression: {
      const auto& call = cast<CallExpression>(*this);
      out << *call.Function();
      if (call.Argument()->Tag() == Expression::Kind::TupleLiteral) {
        out << *call.Argument();
      } else {
        out << "(" << *call.Argument() << ")";
      }
      break;
    }
    case Expression::Kind::BoolTypeLiteral:
      out << "Bool";
      break;
    case Expression::Kind::IntTypeLiteral:
      out << "i32";
      break;
    case Expression::Kind::TypeTypeLiteral:
      out << "Type";
      break;
    case Expression::Kind::ContinuationTypeLiteral:
      out << "Continuation";
      break;
    case Expression::Kind::FunctionTypeLiteral: {
      const auto& fn = cast<FunctionTypeLiteral>(*this);
      out << "fn " << *fn.Parameter() << " -> " << *fn.ReturnType();
      break;
    }
  }
}

}  // namespace Carbon
