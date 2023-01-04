// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/expression.h"

#include <map>
#include <optional>

#include "explorer/ast/pattern.h"
#include "explorer/common/arena.h"
#include "explorer/common/error_builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

using llvm::cast;
using llvm::isa;

auto IntrinsicExpression::FindIntrinsic(std::string_view name,
                                        SourceLocation source_loc)
    -> ErrorOr<Intrinsic> {
  // TODO: Remove Print special casing once we have variadics or overloads.
  if (name == "Print") {
    return Intrinsic::Print;
  }
  static const auto& intrinsic_map = *new std::map<std::string_view, Intrinsic>(
      {{"print", Intrinsic::Print},
       {"new", Intrinsic::Alloc},
       {"delete", Intrinsic::Dealloc},
       {"rand", Intrinsic::Rand},
       {"int_eq", Intrinsic::IntEq},
       {"int_compare", Intrinsic::IntCompare},
       {"int_bit_complement", Intrinsic::IntBitComplement},
       {"int_bit_and", Intrinsic::IntBitAnd},
       {"int_bit_or", Intrinsic::IntBitOr},
       {"int_bit_xor", Intrinsic::IntBitXor},
       {"int_left_shift", Intrinsic::IntLeftShift},
       {"int_right_shift", Intrinsic::IntRightShift},
       {"str_eq", Intrinsic::StrEq},
       {"str_compare", Intrinsic::StrCompare},
       {"assert", Intrinsic::Assert}});
  name.remove_prefix(std::strlen("__intrinsic_"));
  auto it = intrinsic_map.find(name);
  if (it == intrinsic_map.end()) {
    return ProgramError(source_loc) << "Unknown intrinsic '" << name << "'";
  }
  return it->second;
}

auto IntrinsicExpression::name() const -> std::string_view {
  switch (intrinsic()) {
    case IntrinsicExpression::Intrinsic::Print:
      // TODO: Remove Print special casing once we have variadics or overloads.
      return "Print";
    case IntrinsicExpression::Intrinsic::Alloc:
      return "__intrinsic_new";
    case IntrinsicExpression::Intrinsic::Dealloc:
      return "__intrinsic_delete";
    case IntrinsicExpression::Intrinsic::Rand:
      return "__intrinsic_rand";
    case IntrinsicExpression::Intrinsic::IntEq:
      return "__intrinsic_int_eq";
    case IntrinsicExpression::Intrinsic::IntCompare:
      return "__intrinsic_int_compare";
    case IntrinsicExpression::Intrinsic::IntBitComplement:
      return "__intrinsic_int_bit_complement";
    case IntrinsicExpression::Intrinsic::IntBitAnd:
      return "__intrinsic_int_bit_and";
    case IntrinsicExpression::Intrinsic::IntBitOr:
      return "__intrinsic_int_bit_or";
    case IntrinsicExpression::Intrinsic::IntBitXor:
      return "__intrinsic_int_bit_xor";
    case IntrinsicExpression::Intrinsic::IntLeftShift:
      return "__intrinsic_int_left_shift";
    case IntrinsicExpression::Intrinsic::IntRightShift:
      return "__intrinsic_int_right_shift";
    case IntrinsicExpression::Intrinsic::StrEq:
      return "__intrinsic_str_eq";
    case IntrinsicExpression::Intrinsic::StrCompare:
      return "__intrinsic_str_compare";
    case IntrinsicExpression::Intrinsic::Assert:
      return "__intrinsic_assert";
  }
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
    case Operator::As:
      return "as";
    case Operator::AddressOf:
    case Operator::BitwiseAnd:
      return "&";
    case Operator::BitwiseOr:
      return "|";
    case Operator::BitwiseXor:
    case Operator::Complement:
      return "^";
    case Operator::BitShiftLeft:
      return "<<";
    case Operator::BitShiftRight:
      return ">>";
    case Operator::Div:
      return "/";
    case Operator::Neg:
    case Operator::Sub:
      return "-";
    case Operator::Mul:
    case Operator::Deref:
    case Operator::Ptr:
      return "*";
    case Operator::Not:
      return "not";
    case Operator::NotEq:
      return "!=";
    case Operator::And:
      return "and";
    case Operator::Or:
      return "or";
    case Operator::Eq:
      return "==";
    case Operator::Mod:
      return "%";
    case Operator::Less:
      return "<";
    case Operator::LessEq:
      return "<=";
    case Operator::Greater:
      return ">";
    case Operator::GreaterEq:
      return ">=";
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
      out << index.object() << "[" << index.offset() << "]";
      break;
    }
    case ExpressionKind::SimpleMemberAccessExpression: {
      const auto& access = cast<SimpleMemberAccessExpression>(*this);
      out << access.object() << "." << access.member_name();
      break;
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      const auto& access = cast<CompoundMemberAccessExpression>(*this);
      out << access.object() << ".(" << access.path() << ")";
      break;
    }
    case ExpressionKind::BaseAccessExpression: {
      const auto& access = cast<BaseAccessExpression>(*this);
      out << access.object() << ".base";
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
    case ExpressionKind::OperatorExpression: {
      out << "(";
      const auto& op = cast<OperatorExpression>(*this);
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
          CARBON_FATAL() << "Unexpected argument count: "
                         << op.arguments().size();
      }
      out << ")";
      break;
    }
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
    case ExpressionKind::FunctionTypeLiteral: {
      const auto& fn = cast<FunctionTypeLiteral>(*this);
      out << "fn " << fn.parameter() << " -> " << fn.return_type();
      break;
    }
    case ExpressionKind::IntrinsicExpression: {
      const auto& iexp = cast<IntrinsicExpression>(*this);
      out << iexp.name() << iexp.args();
      break;
    }
    case ExpressionKind::IfExpression: {
      const auto& if_expr = cast<IfExpression>(*this);
      out << "if " << if_expr.condition() << " then "
          << if_expr.then_expression() << " else " << if_expr.else_expression();
      break;
    }
    case ExpressionKind::WhereExpression: {
      const auto& where = cast<WhereExpression>(*this);
      out << where.self_binding().type() << " where ";
      llvm::ListSeparator sep(" and ");
      for (const WhereClause* clause : where.clauses()) {
        out << sep << *clause;
      }
      break;
    }
    case ExpressionKind::BuiltinConvertExpression: {
      // These don't represent source syntax, so just print the original
      // expression.
      out << *cast<BuiltinConvertExpression>(this)->source_expression();
      break;
    }
    case ExpressionKind::UnimplementedExpression: {
      const auto& unimplemented = cast<UnimplementedExpression>(*this);
      out << "UnimplementedExpression<" << unimplemented.label() << ">(";
      llvm::ListSeparator sep;
      for (Nonnull<const AstNode*> child : unimplemented.children()) {
        out << sep << *child;
      }
      out << ")";
      break;
    }
    case ExpressionKind::ArrayTypeLiteral: {
      const auto& array_literal = cast<ArrayTypeLiteral>(*this);
      out << "[" << array_literal.element_type_expression() << "; "
          << array_literal.size_expression() << "]";
      break;
    }
    case ExpressionKind::IdentifierExpression:
    case ExpressionKind::DotSelfExpression:
    case ExpressionKind::IntLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
    case ExpressionKind::ValueLiteral:
      PrintID(out);
      break;
  }
}

void Expression::PrintID(llvm::raw_ostream& out) const {
  switch (kind()) {
    case ExpressionKind::IdentifierExpression:
      out << cast<IdentifierExpression>(*this).name();
      break;
    case ExpressionKind::DotSelfExpression:
      out << ".Self";
      break;
    case ExpressionKind::IntLiteral:
      out << cast<IntLiteral>(*this).value();
      break;
    case ExpressionKind::BoolLiteral:
      out << (cast<BoolLiteral>(*this).value() ? "true" : "false");
      break;
    case ExpressionKind::BoolTypeLiteral:
      out << "bool";
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
      out << "type";
      break;
    case ExpressionKind::ContinuationTypeLiteral:
      out << "Continuation";
      break;
    case ExpressionKind::ValueLiteral:
      // TODO: For layering reasons, we can't print out the value from here.
      out << "ValueLiteral";
      break;
    case ExpressionKind::IndexExpression:
    case ExpressionKind::SimpleMemberAccessExpression:
    case ExpressionKind::CompoundMemberAccessExpression:
    case ExpressionKind::BaseAccessExpression:
    case ExpressionKind::IfExpression:
    case ExpressionKind::WhereExpression:
    case ExpressionKind::BuiltinConvertExpression:
    case ExpressionKind::TupleLiteral:
    case ExpressionKind::StructLiteral:
    case ExpressionKind::StructTypeLiteral:
    case ExpressionKind::CallExpression:
    case ExpressionKind::OperatorExpression:
    case ExpressionKind::IntrinsicExpression:
    case ExpressionKind::UnimplementedExpression:
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::ArrayTypeLiteral:
      out << "...";
      break;
  }
}

WhereClause::~WhereClause() = default;

void WhereClause::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case WhereClauseKind::IsWhereClause: {
      const auto& clause = cast<IsWhereClause>(*this);
      out << clause.type() << " is " << clause.constraint();
      break;
    }
    case WhereClauseKind::EqualsWhereClause: {
      const auto& clause = cast<EqualsWhereClause>(*this);
      out << clause.lhs() << " == " << clause.rhs();
      break;
    }
    case WhereClauseKind::RewriteWhereClause: {
      const auto& clause = cast<RewriteWhereClause>(*this);
      out << "." << clause.member_name() << " = " << clause.replacement();
      break;
    }
  }
}

void WhereClause::PrintID(llvm::raw_ostream& out) const { out << "..."; }

}  // namespace Carbon
