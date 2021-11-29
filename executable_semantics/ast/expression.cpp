// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <map>
#include <optional>

#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

using llvm::cast;
using llvm::isa;

auto IntrinsicExpression::FindIntrinsic(std::string_view name,
                                        SourceLocation source_loc)
    -> Intrinsic {
  static const auto& intrinsic_map =
      *new std::map<std::string_view, Intrinsic>({{"print", Intrinsic::Print}});
  name.remove_prefix(std::strlen("__intrinsic_"));
  auto it = intrinsic_map.find(name);
  if (it == intrinsic_map.end()) {
    FATAL_COMPILATION_ERROR(source_loc) << "Unknown intrinsic '" << name << "'";
  }
  return it->second;
}

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
    const ParenContents<Expression>& paren_contents) -> Nonnull<TupleLiteral*> {
  return arena->New<TupleLiteral>(source_loc, paren_contents.elements);
}

Expression::~Expression() = default;

auto ToString(Operator op) -> std::string_view {
  switch (op) {
    case Operator::Add:
      return "+";
    case Operator::Neg:
    case Operator::Sub:
      return "-";
    case Operator::Mul:
    case Operator::Deref:
    case Operator::Ptr:
      return "*";
    case Operator::Not:
      return "not";
    case Operator::And:
      return "and";
    case Operator::Or:
      return "or";
    case Operator::Eq:
      return "==";
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
    case ExpressionKind::IndexExpression: {
      const auto& index = cast<IndexExpression>(*this);
      out << index.aggregate() << "[" << index.offset() << "]";
      break;
    }
    case ExpressionKind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*this);
      out << access.aggregate() << "." << access.field();
      break;
    }
    case ExpressionKind::TupleLiteral: {
      out << "(";
      llvm::ListSeparator sep;
      for (Nonnull<const Expression*> field :
           cast<TupleLiteral>(*this).fields()) {
        out << sep << *field;
      }
      out << ")";
      break;
    }
    case ExpressionKind::StructLiteral:
      out << "{";
      PrintFields(out, cast<StructLiteral>(*this).fields(), " = ");
      out << "}";
      break;
    case ExpressionKind::StructTypeLiteral:
      out << "{";
      PrintFields(out, cast<StructTypeLiteral>(*this).fields(), ": ");
      out << "}";
      break;
    case ExpressionKind::IntLiteral:
      out << cast<IntLiteral>(*this).value();
      break;
    case ExpressionKind::BoolLiteral:
      out << (cast<BoolLiteral>(*this).value() ? "true" : "false");
      break;
    case ExpressionKind::PrimitiveOperatorExpression: {
      out << "(";
      const auto& op = cast<PrimitiveOperatorExpression>(*this);
      switch (op.arguments().size()) {
        case 0:
          out << ToString(op.op());
          break;
        case 1:
          out << ToString(op.op()) << " " << *op.arguments()[0];
          break;
        case 2:
          out << *op.arguments()[0] << " " << ToString(op.op()) << " "
              << *op.arguments()[1];
          break;
        default:
          FATAL() << "Unexpected argument count: " << op.arguments().size();
      }
      out << ")";
      break;
    }
    case ExpressionKind::IdentifierExpression:
      out << cast<IdentifierExpression>(*this).name();
      break;
    case ExpressionKind::CallExpression: {
      const auto& call = cast<CallExpression>(*this);
      out << call.function();
      if (isa<TupleLiteral>(call.argument())) {
        out << call.argument();
      } else {
        out << "(" << call.argument() << ")";
      }
      break;
    }
    case ExpressionKind::BoolTypeLiteral:
      out << "Bool";
      break;
    case ExpressionKind::IntTypeLiteral:
      out << "i32";
      break;
    case ExpressionKind::StringLiteral:
      out << "\"";
      out.write_escaped(cast<StringLiteral>(*this).value());
      out << "\"";
      break;
    case ExpressionKind::StringTypeLiteral:
      out << "String";
      break;
    case ExpressionKind::TypeTypeLiteral:
      out << "Type";
      break;
    case ExpressionKind::ContinuationTypeLiteral:
      out << "Continuation";
      break;
    case ExpressionKind::FunctionTypeLiteral: {
      const auto& fn = cast<FunctionTypeLiteral>(*this);
      out << "fn " << fn.parameter() << " -> " << fn.return_type();
      break;
    }
    case ExpressionKind::IntrinsicExpression:
      out << "intrinsic_expression(";
      switch (cast<IntrinsicExpression>(*this).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          out << "print";
      }
      out << ")";
  }
}

}  // namespace Carbon
