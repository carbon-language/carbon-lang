// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/resolve_unformed.h"

#include <unordered_map>

#include "common/check.h"
#include "explorer/ast/ast.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/common/error_builders.h"
#include "explorer/common/nonnull.h"

using llvm::cast;

namespace Carbon {

// Aggregate information about a AstNode being analyzed.
struct FlowFact {
  bool may_be_formed;
};

// Traverses the sub-AST rooted at the given node, resolving the formed/unformed
// states of local variables within it and updating the flow facts.
static auto ResolveUnformed(
    Nonnull<const Expression*> expression,
    std::unordered_map<Nonnull<const AstNode*>, FlowFact>& flow_facts,
    bool set_formed) -> ErrorOr<Success>;
static auto ResolveUnformed(
    Nonnull<const Pattern*> pattern,
    std::unordered_map<Nonnull<const AstNode*>, FlowFact>& flow_facts,
    bool has_init) -> ErrorOr<Success>;
static auto ResolveUnformed(
    Nonnull<const Statement*> statement,
    std::unordered_map<Nonnull<const AstNode*>, FlowFact>& flow_facts)
    -> ErrorOr<Success>;
static auto ResolveUnformed(Nonnull<const Declaration*> declaration)
    -> ErrorOr<Success>;

static auto ResolveUnformed(
    Nonnull<const Expression*> expression,
    std::unordered_map<Nonnull<const AstNode*>, FlowFact>& flow_facts,
    const bool set_formed) -> ErrorOr<Success> {
  switch (expression->kind()) {
    case ExpressionKind::IdentifierExpression: {
      auto& identifier = cast<IdentifierExpression>(*expression);
      auto fact = flow_facts.find(&identifier.value_node().base());
      // TODO: @slaterlatiao add all available value nodes to flow facts and use
      // CARBON_CHECK on the following line.
      if (fact == flow_facts.end()) {
        break;
      }
      if (set_formed) {
        fact->second.may_be_formed = true;
      } else if (!fact->second.may_be_formed) {
        return CompilationError(identifier.source_loc())
               << "use of uninitialized variable " << identifier.name();
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*expression);
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&call.argument(), flow_facts, /*set_formed=*/false));
      break;
    }
    case ExpressionKind::TupleLiteral:
      for (Nonnull<const Expression*> field :
           cast<TupleLiteral>(*expression).fields()) {
        CARBON_RETURN_IF_ERROR(
            ResolveUnformed(field, flow_facts, /*set_formed=*/false));
      }
      break;
    case ExpressionKind::OperatorExpression: {
      auto& opt_exp = cast<OperatorExpression>(*expression);
      if (opt_exp.op() == Operator::AddressOf) {
        CARBON_CHECK(opt_exp.arguments().size() == 1)
            << "OperatorExpression with op & can only have 1 argument";
        CARBON_RETURN_IF_ERROR(
            // When a variable is taken address of, defer the unformed check to
            // runtime. A more sound analysis can be implemented when a
            // points-to analysis is available.
            ResolveUnformed(opt_exp.arguments().front(), flow_facts,
                            /*set_formed=*/true));
      } else {
        for (Nonnull<const Expression*> operand : opt_exp.arguments()) {
          CARBON_RETURN_IF_ERROR(
              ResolveUnformed(operand, flow_facts, /*set_formed=*/false));
        }
      }
      break;
    }
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
    case ExpressionKind::IndexExpression:
    case ExpressionKind::SimpleMemberAccessExpression:
    case ExpressionKind::CompoundMemberAccessExpression:
    case ExpressionKind::IfExpression:
    case ExpressionKind::WhereExpression:
    case ExpressionKind::StructLiteral:
    case ExpressionKind::StructTypeLiteral:
    case ExpressionKind::IntrinsicExpression:
    case ExpressionKind::UnimplementedExpression:
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::ArrayTypeLiteral:
    case ExpressionKind::InstantiateImpl:
      break;
  }
  return Success();
}

static auto ResolveUnformed(
    Nonnull<const Pattern*> pattern,
    std::unordered_map<Nonnull<const AstNode*>, FlowFact>& flow_facts,
    const bool has_init) -> ErrorOr<Success> {
  switch (pattern->kind()) {
    case PatternKind::BindingPattern:
      flow_facts.insert(
          {Nonnull<const AstNode*>(&cast<BindingPattern>(*pattern)),
           {has_init}});
      break;
    case PatternKind::TuplePattern:
      for (Nonnull<const Pattern*> field :
           cast<TuplePattern>(*pattern).fields()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(field, flow_facts, has_init));
      }
      break;
    case PatternKind::GenericBinding:
    case PatternKind::AlternativePattern:
    case PatternKind::ExpressionPattern:
    case PatternKind::AutoPattern:
    case PatternKind::VarPattern:
    case PatternKind::AddrPattern:
      // do nothing
      break;
  }
  return Success();
}

static auto ResolveUnformed(
    Nonnull<const Statement*> statement,
    std::unordered_map<Nonnull<const AstNode*>, FlowFact>& flow_facts)
    -> ErrorOr<Success> {
  switch (statement->kind()) {
    case StatementKind::Block: {
      auto& block = cast<Block>(*statement);
      for (auto* block_statement : block.statements()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(block_statement, flow_facts));
      }
      break;
    }
    case StatementKind::VariableDefinition: {
      auto& def = cast<VariableDefinition>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&def.pattern(), flow_facts,
                                             /*has_init=*/def.has_init()));
      break;
    }
    case StatementKind::ReturnVar:
      // TODO: @slaterlatiao: Implement this flow.
      break;
    case StatementKind::ReturnExpression: {
      auto& ret_exp_stmt = cast<ReturnExpression>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&ret_exp_stmt.expression(),
                                             flow_facts, /*set_formed=*/false));
      break;
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*statement);
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&assign.lhs(), flow_facts, /*set_formed=*/true));
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&assign.rhs(), flow_facts, /*set_formed=*/false));
      break;
    }
    case StatementKind::ExpressionStatement: {
      auto& exp_stmt = cast<ExpressionStatement>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&exp_stmt.expression(), flow_facts,
                                             /*set_formed=*/false));
      break;
    }
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::If:
    case StatementKind::While:
    case StatementKind::Match:
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
    case StatementKind::For:
      // do nothing
      break;
  }
  return Success();
}

static auto ResolveUnformed(Nonnull<const Declaration*> declaration)
    -> ErrorOr<Success> {
  switch (declaration->kind()) {
    // Checks formed/unformed state intraprocedurally.
    // Can be extended to an interprocedural analysis when a call graph is
    // available.
    case DeclarationKind::FunctionDeclaration: {
      auto& function = cast<FunctionDeclaration>(*declaration);
      if (function.body().has_value()) {
        std::unordered_map<Nonnull<const AstNode*>, FlowFact> flow_facts;
        CARBON_RETURN_IF_ERROR(ResolveUnformed(*function.body(), flow_facts));
      }
      break;
    }
    case DeclarationKind::ClassDeclaration:
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::ChoiceDeclaration:
    case DeclarationKind::VariableDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration:
    case DeclarationKind::SelfDeclaration:
    case DeclarationKind::AliasDeclaration:
      // do nothing
      break;
  }
  return Success();
}

auto ResolveUnformed(const AST& ast) -> ErrorOr<Success> {
  for (auto declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(ResolveUnformed(declaration));
  }
  return Success();
}

}  // namespace Carbon
