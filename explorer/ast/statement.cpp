// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/statement.h"

#include "common/check.h"
#include "explorer/ast/declaration.h"
#include "explorer/base/arena.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Statement::~Statement() = default;

void Statement::PrintDepth(int depth, int indent_num_spaces,
                           llvm::raw_ostream& out) const {
  if (depth == 0) {
    out.indent(indent_num_spaces) << " ... ";
    return;
  }
  switch (kind()) {
    case StatementKind::Match: {
      const auto& match = cast<Match>(*this);
      out.indent(indent_num_spaces) << "match (" << match.expression() << ") {";
      if (depth < 0 || depth > 1) {
        out << "\n";
        for (const auto& clause : match.clauses()) {
          out.indent(indent_num_spaces + 2)
              << "case " << clause.pattern() << " => ";
          clause.statement().PrintDepth(depth - 1, indent_num_spaces + 2, out);
          out << "\n";
        }
      } else {
        out.indent(indent_num_spaces) << "...";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case StatementKind::While: {
      const auto& while_stmt = cast<While>(*this);
      out.indent(indent_num_spaces)
          << "while (" << while_stmt.condition() << ")\n";
      while_stmt.body().PrintDepth(depth - 1, indent_num_spaces + 2, out);
      break;
    }
    case StatementKind::For: {
      const auto& for_stmt = cast<For>(*this);
      out.indent(indent_num_spaces)
          << "for (" << for_stmt.variable_declaration() << " in "
          << for_stmt.loop_target() << ")\n";
      for_stmt.body().PrintDepth(depth - 1, indent_num_spaces + 2, out);
      break;
    }
    case StatementKind::Break:
      out.indent(indent_num_spaces) << "break;";
      break;
    case StatementKind::Continue:
      out.indent(indent_num_spaces) << "continue;";
      break;
    case StatementKind::VariableDefinition: {
      const auto& var = cast<VariableDefinition>(*this);
      if (var.is_returned()) {
        out.indent(indent_num_spaces) << "returned ";
      }
      out.indent(var.is_returned() ? 0 : indent_num_spaces)
          << "var " << var.pattern();
      if (var.has_init()) {
        out << " = " << var.init();
      }
      out << ";";
      break;
    }
    case StatementKind::ExpressionStatement:
      out.indent(indent_num_spaces)
          << cast<ExpressionStatement>(*this).expression() << ";";
      break;
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(*this);
      out.indent(indent_num_spaces)
          << assign.lhs() << " " << AssignOperatorToString(assign.op()) << " "
          << assign.rhs() << ";";
      break;
    }
    case StatementKind::IncrementDecrement: {
      const auto& inc_dec = cast<IncrementDecrement>(*this);
      out.indent(indent_num_spaces)
          << (inc_dec.is_increment() ? "++" : "--") << inc_dec.argument();
      break;
    }
    case StatementKind::If: {
      const auto& if_stmt = cast<If>(*this);
      out.indent(indent_num_spaces) << "if (" << if_stmt.condition() << ") ";
      if_stmt.then_block().PrintDepth(depth - 1, indent_num_spaces, out);
      if (if_stmt.else_block()) {
        out << "\n";
        out.indent(indent_num_spaces) << "else\n";
        (*if_stmt.else_block())
            ->PrintDepth(depth - 1, indent_num_spaces + 2, out);
      }
      break;
    }
    case StatementKind::ReturnVar: {
      out.indent(indent_num_spaces) << "return var;";
      break;
    }
    case StatementKind::ReturnExpression: {
      const auto& ret = cast<ReturnExpression>(*this);
      if (ret.is_omitted_expression()) {
        out.indent(indent_num_spaces) << "return;";
      } else {
        out.indent(indent_num_spaces) << "return " << ret.expression() << ";";
      }
      break;
    }
    case StatementKind::Block: {
      const auto& block = cast<Block>(*this);
      const auto statements = block.statements();
      out << "{\n";
      for (const auto* statement : statements) {
        statement->PrintDepth(depth, indent_num_spaces + 2, out);
        out << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
  }
}

auto AssignOperatorToString(AssignOperator op) -> std::string_view {
  switch (op) {
    case AssignOperator::Plain:
      return "=";
    case AssignOperator::Add:
      return "+=";
    case AssignOperator::Div:
      return "/=";
    case AssignOperator::Mul:
      return "*=";
    case AssignOperator::Mod:
      return "%=";
    case AssignOperator::Sub:
      return "-=";
    case AssignOperator::And:
      return "&=";
    case AssignOperator::Or:
      return "|=";
    case AssignOperator::Xor:
      return "^=";
    case AssignOperator::ShiftLeft:
      return "<<=";
    case AssignOperator::ShiftRight:
      return ">>=";
  }
}

Return::Return(CloneContext& context, const Return& other)
    : Statement(context, other), function_(context.Remap(other.function_)) {}

}  // namespace Carbon
