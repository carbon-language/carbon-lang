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

void Statement::PrintID(llvm::raw_ostream& out) const {
  switch (kind()) {
    case StatementKind::Match:
      out << "match (...) { ... }";
      break;
    case StatementKind::While:
      out << "while (...) { ... }";
      break;
    case StatementKind::For:
      out << "for (...) { ... }";
      break;
    case StatementKind::Break:
      out << "break;";
      break;
    case StatementKind::Continue:
      out << "continue;";
      break;
    case StatementKind::VariableDefinition: {
      const auto& var = cast<VariableDefinition>(*this);
      if (var.is_returned()) {
        out << "returned ";
      }
      out << "var ...";
      if (var.has_init()) {
        out << " = ...";
      }
      out << ";";
      break;
    }
    case StatementKind::ExpressionStatement:
      out << "<expression>;";
      break;
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(*this);
      out << "... " << AssignOperatorToString(assign.op()) << " ...;";
      break;
    }
    case StatementKind::IncrementDecrement: {
      const auto& inc_dec = cast<IncrementDecrement>(*this);
      out << (inc_dec.is_increment() ? "++" : "--") << "...;";
      break;
    }
    case StatementKind::If: {
      const auto& if_stmt = cast<If>(*this);
      out << "if (...) { ... }";
      if (if_stmt.else_block()) {
        out << " else { ... }";
      }
      break;
    }
    case StatementKind::ReturnVar:
      out << "return var;";
      break;
    case StatementKind::ReturnExpression: {
      const auto& ret = cast<ReturnExpression>(*this);
      if (ret.is_omitted_expression()) {
        out << "return;";
      } else {
        out << "return ...;";
      }
      break;
    }
    case StatementKind::Block:
      out << "{ ... }";
      break;
  }
}

void Statement::PrintIndent(int indent_num_spaces,
                            llvm::raw_ostream& out) const {
  out.indent(indent_num_spaces);

  switch (kind()) {
    case StatementKind::Match: {
      const auto& match = cast<Match>(*this);
      out << "match (" << match.expression() << ") {\n";
      for (const auto& clause : match.clauses()) {
        out.indent(indent_num_spaces + 2)
            << "case " << clause.pattern() << " =>\n";
        clause.statement().PrintIndent(indent_num_spaces + 2, out);
        out << "\n";
      }
      out.indent(indent_num_spaces) << "}";
      break;
    }
    case StatementKind::While: {
      const auto& while_stmt = cast<While>(*this);
      out << "while (" << while_stmt.condition() << ")\n";
      while_stmt.body().PrintIndent(indent_num_spaces, out);
      break;
    }
    case StatementKind::For: {
      const auto& for_stmt = cast<For>(*this);
      out << "for (" << for_stmt.variable_declaration() << " in "
          << for_stmt.loop_target() << ")\n";
      for_stmt.body().PrintIndent(indent_num_spaces, out);
      break;
    }
    case StatementKind::Break:
      out << "break;";
      break;
    case StatementKind::Continue:
      out << "continue;";
      break;
    case StatementKind::VariableDefinition: {
      const auto& var = cast<VariableDefinition>(*this);
      if (var.is_returned()) {
        out << "returned ";
      }
      out << "var " << var.pattern();
      if (var.has_init()) {
        out << " = " << var.init();
      }
      out << ";";
      break;
    }
    case StatementKind::ExpressionStatement:
      out << cast<ExpressionStatement>(*this).expression() << ";";
      break;
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(*this);
      out << assign.lhs() << " " << AssignOperatorToString(assign.op()) << " "
          << assign.rhs() << ";";
      break;
    }
    case StatementKind::IncrementDecrement: {
      const auto& inc_dec = cast<IncrementDecrement>(*this);
      out << (inc_dec.is_increment() ? "++" : "--") << inc_dec.argument()
          << ";";
      break;
    }
    case StatementKind::If: {
      const auto& if_stmt = cast<If>(*this);
      out << "if (" << if_stmt.condition() << ")\n";
      if_stmt.then_block().PrintIndent(indent_num_spaces, out);
      if (if_stmt.else_block()) {
        out << "\n";
        out.indent(indent_num_spaces) << "else\n";
        (*if_stmt.else_block())->PrintIndent(indent_num_spaces, out);
      }
      break;
    }
    case StatementKind::ReturnVar: {
      out << "return var;";
      break;
    }
    case StatementKind::ReturnExpression: {
      const auto& ret = cast<ReturnExpression>(*this);
      if (ret.is_omitted_expression()) {
        out << "return;";
      } else {
        out << "return " << ret.expression() << ";";
      }
      break;
    }
    case StatementKind::Block: {
      const auto& block = cast<Block>(*this);
      const auto statements = block.statements();
      out << "{\n";
      for (const auto* statement : statements) {
        statement->PrintIndent(indent_num_spaces + 2, out);
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
