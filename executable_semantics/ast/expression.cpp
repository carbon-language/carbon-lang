// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <optional>

#include "executable_semantics/ast/unimplemented.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

using llvm::cast;
using llvm::isa;

auto ExpressionFromParenContents(
    Nonnull<Arena*> arena, SourceLocation source_loc,
    const ParenContents<Expression>& paren_contents) -> Nonnull<Expression*> {
  std::optional<Nonnull<Expression*>> single_term = paren_contents.SingleTerm();
  if (single_term.has_value()) {
    return *single_term;
  } else {
    return TupleExpressionFromParenContents(arena, source_loc, paren_contents);
  }
}

auto TupleExpressionFromParenContents(
    Nonnull<Arena*> arena, SourceLocation source_loc,
    const ParenContents<Expression>& paren_contents) -> Nonnull<Expression*> {
  return arena->New<TupleLiteral>(source_loc, paren_contents.elements);
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
                        const std::vector<FieldInitializer>& fields,
                        std::string_view separator) {
  llvm::ListSeparator sep;
  for (const auto& field : fields) {
    out << sep << "." << field.name() << separator << field.expression();
  }
}

void Expression::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Expression::Kind::IndexExpression: {
      const auto& index = cast<IndexExpression>(*this);
      out << index.aggregate() << "[" << index.offset() << "]";
      break;
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*this);
      out << access.aggregate() << "." << access.field();
      break;
    }
    case Expression::Kind::TupleLiteral: {
      out << "(";
      llvm::ListSeparator sep;
      for (Nonnull<const Expression*> field :
           cast<TupleLiteral>(*this).fields()) {
        out << sep << *field;
      }
      out << ")";
      break;
    }
    case Expression::Kind::StructLiteral:
      out << "{";
      PrintFields(out, cast<StructLiteral>(*this).fields(), " = ");
      out << "}";
      break;
    case Expression::Kind::StructTypeLiteral:
      out << "{";
      PrintFields(out, cast<StructTypeLiteral>(*this).fields(), ": ");
      out << "}";
      break;
    case Expression::Kind::IntLiteral:
      out << cast<IntLiteral>(*this).value();
      break;
    case Expression::Kind::BoolLiteral:
      out << (cast<BoolLiteral>(*this).value() ? "true" : "false");
      break;
    case Expression::Kind::PrimitiveOperatorExpression: {
      out << "(";
      PrimitiveOperatorExpression op = cast<PrimitiveOperatorExpression>(*this);
      switch (op.arguments().size()) {
        case 0:
          PrintOp(out, op.op());
          break;
        case 1:
          PrintOp(out, op.op());
          out << " " << *op.arguments()[0];
          break;
        case 2:
          out << *op.arguments()[0] << " ";
          PrintOp(out, op.op());
          out << " " << *op.arguments()[1];
          break;
        default:
          FATAL() << "Unexpected argument count: " << op.arguments().size();
      }
      out << ")";
      break;
    }
    case Expression::Kind::IdentifierExpression:
      out << cast<IdentifierExpression>(*this).name();
      break;
    case Expression::Kind::CallExpression: {
      const auto& call = cast<CallExpression>(*this);
      out << call.function();
      if (isa<TupleLiteral>(call.argument())) {
        out << call.argument();
      } else {
        out << "(" << call.argument() << ")";
      }
      break;
    }
    case Expression::Kind::BoolTypeLiteral:
      out << "Bool";
      break;
    case Expression::Kind::IntTypeLiteral:
      out << "i32";
      break;
    case Expression::Kind::StringLiteral:
      out << "\"";
      out.write_escaped(cast<StringLiteral>(*this).value());
      out << "\"";
      break;
    case Expression::Kind::StringTypeLiteral:
      out << "String";
      break;
    case Expression::Kind::TypeTypeLiteral:
      out << "Type";
      break;
    case Expression::Kind::ContinuationTypeLiteral:
      out << "Continuation";
      break;
    case Expression::Kind::FunctionTypeLiteral: {
      const auto& fn = cast<FunctionTypeLiteral>(*this);
      out << "fn " << fn.parameter() << " -> " << fn.return_type();
      break;
    }
    case Expression::Kind::IntrinsicExpression:
      out << "intrinsic_expression(";
      switch (cast<IntrinsicExpression>(*this).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          out << "print";
      }
      out << ")";
      break;
    case Expression::Kind::Unimplemented: {
      cast<Unimplemented<Expression>>(*this).PrintImpl(out);
      break;
    }
  }
}

}  // namespace Carbon
