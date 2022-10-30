// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/resolve_unformed.h"

#include <unordered_map>

#include "common/check.h"
#include "explorer/ast/ast.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/common/nonnull.h"

using llvm::cast;

namespace Carbon {

auto FlowFacts::TakeAction(Nonnull<const AstNode*> node, ActionType action,
                           SourceLocation source_loc, const std::string& name)
    -> ErrorOr<Success> {
  switch (action) {
    case ActionType::AddInit: {
      AddFact(node, FormedState::MustBeFormed);
      break;
    }
    case ActionType::AddUninit: {
      AddFact(node, FormedState::Unformed);
      break;
    }
    case ActionType::Form: {
      // TODO: Use CARBON_CHECK when we are able to handle global variables.
      auto entry = facts_.find(node);
      if (entry != facts_.end() &&
          entry->second.formed_state == FormedState::Unformed) {
        entry->second.formed_state = FormedState::MayBeFormed;
      }
      break;
    }
    case ActionType::Check: {
      // TODO: @slaterlatiao add all available value nodes to flow facts and use
      // CARBON_CHECK on the following line.
      auto entry = facts_.find(node);
      if (entry != facts_.end() &&
          entry->second.formed_state == FormedState::Unformed) {
        return ProgramError(source_loc)
               << "use of uninitialized variable " << name;
      }
      break;
    }
    case ActionType::None:
      break;
  }
  return Success();
}

// Traverses the sub-AST rooted at the given node, resolving the formed/unformed
// states of local variables within it and updating the flow facts.
static auto ResolveUnformed(Nonnull<const Expression*> expression,
                            FlowFacts& flow_facts, FlowFacts::ActionType action)
    -> ErrorOr<Success>;
static auto ResolveUnformed(Nonnull<const Pattern*> pattern,
                            FlowFacts& flow_facts, FlowFacts::ActionType action)
    -> ErrorOr<Success>;
static auto ResolveUnformed(Nonnull<const Statement*> statement,
                            FlowFacts& flow_facts, FlowFacts::ActionType action)
    -> ErrorOr<Success>;
static auto ResolveUnformed(Nonnull<const Declaration*> declaration)
    -> ErrorOr<Success>;

static auto ResolveUnformed(Nonnull<const Expression*> expression,
                            FlowFacts& flow_facts, FlowFacts::ActionType action)
    -> ErrorOr<Success> {
  switch (expression->kind()) {
    case ExpressionKind::IdentifierExpression: {
      const auto& identifier = cast<IdentifierExpression>(*expression);
      CARBON_RETURN_IF_ERROR(
          flow_facts.TakeAction(&identifier.value_node().base(), action,
                                identifier.source_loc(), identifier.name()));
      break;
    }
    case ExpressionKind::CallExpression: {
      const auto& call = cast<CallExpression>(*expression);
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&call.argument(), flow_facts, action));
      break;
    }
    case ExpressionKind::TupleLiteral:
      for (Nonnull<const Expression*> field :
           cast<TupleLiteral>(*expression).fields()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(field, flow_facts, action));
      }
      break;
    case ExpressionKind::OperatorExpression: {
      const auto& opt_exp = cast<OperatorExpression>(*expression);
      if (opt_exp.op() == Operator::AddressOf) {
        CARBON_CHECK(opt_exp.arguments().size() == 1)
            << "OperatorExpression with op & can only have 1 argument";
        CARBON_RETURN_IF_ERROR(
            // When a variable is taken address of, defer the unformed check to
            // runtime. A more sound analysis can be implemented when a
            // points-to analysis is available.
            ResolveUnformed(opt_exp.arguments().front(), flow_facts,
                            FlowFacts::ActionType::Form));
      } else {
        for (Nonnull<const Expression*> operand : opt_exp.arguments()) {
          CARBON_RETURN_IF_ERROR(ResolveUnformed(operand, flow_facts, action));
        }
      }
      break;
    }
    case ExpressionKind::StructLiteral:
      for (const FieldInitializer& init :
           cast<StructLiteral>(*expression).fields()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(&init.expression(), flow_facts,
                                               FlowFacts::ActionType::Check));
      }
      break;
    case ExpressionKind::SimpleMemberAccessExpression:
      CARBON_RETURN_IF_ERROR(ResolveUnformed(
          &cast<SimpleMemberAccessExpression>(*expression).object(), flow_facts,
          FlowFacts::ActionType::Check));
      break;
    case ExpressionKind::BuiltinConvertExpression:
      CARBON_RETURN_IF_ERROR(ResolveUnformed(
          cast<BuiltinConvertExpression>(*expression).source_expression(),
          flow_facts, FlowFacts::ActionType::Check));
      break;
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
    case ExpressionKind::CompoundMemberAccessExpression:
    case ExpressionKind::IfExpression:
    case ExpressionKind::WhereExpression:
    case ExpressionKind::StructTypeLiteral:
    case ExpressionKind::IntrinsicExpression:
    case ExpressionKind::UnimplementedExpression:
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::ArrayTypeLiteral:
      break;
  }
  return Success();
}

static auto ResolveUnformed(Nonnull<const Pattern*> pattern,
                            FlowFacts& flow_facts, FlowFacts::ActionType action)
    -> ErrorOr<Success> {
  switch (pattern->kind()) {
    case PatternKind::BindingPattern: {
      const auto& binding_pattern = cast<BindingPattern>(*pattern);
      CARBON_RETURN_IF_ERROR(flow_facts.TakeAction(&binding_pattern, action,
                                                   binding_pattern.source_loc(),
                                                   binding_pattern.name()));
    } break;
    case PatternKind::TuplePattern:
      for (Nonnull<const Pattern*> field :
           cast<TuplePattern>(*pattern).fields()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(field, flow_facts, action));
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

static auto ResolveUnformed(Nonnull<const Statement*> statement,
                            FlowFacts& flow_facts, FlowFacts::ActionType action)
    -> ErrorOr<Success> {
  switch (statement->kind()) {
    case StatementKind::Block: {
      const auto& block = cast<Block>(*statement);
      for (const auto* block_statement : block.statements()) {
        CARBON_RETURN_IF_ERROR(
            ResolveUnformed(block_statement, flow_facts, action));
      }
      break;
    }
    case StatementKind::VariableDefinition: {
      const auto& def = cast<VariableDefinition>(*statement);
      if (def.has_init()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(&def.pattern(), flow_facts,
                                               FlowFacts::ActionType::AddInit));
        CARBON_RETURN_IF_ERROR(ResolveUnformed(&def.init(), flow_facts,
                                               FlowFacts::ActionType::Check));
      } else {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(
            &def.pattern(), flow_facts, FlowFacts::ActionType::AddUninit));
      }
      break;
    }
    case StatementKind::ReturnVar: {
      const auto& ret_var = cast<ReturnVar>(*statement);
      const auto& binding_pattern =
          cast<BindingPattern>(ret_var.value_node().base());
      CARBON_RETURN_IF_ERROR(
          flow_facts.TakeAction(&binding_pattern, FlowFacts::ActionType::Check,
                                ret_var.source_loc(), binding_pattern.name()));
      break;
    }
    case StatementKind::ReturnExpression: {
      const auto& ret_exp_stmt = cast<ReturnExpression>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&ret_exp_stmt.expression(),
                                             flow_facts,
                                             FlowFacts::ActionType::Check));
      break;
    }
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(*statement);
      if (assign.lhs().kind() == ExpressionKind::IdentifierExpression) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(&assign.lhs(), flow_facts,
                                               FlowFacts::ActionType::Form));
      } else {
        // TODO: Support checking non-identifier lhs expression.
        CARBON_RETURN_IF_ERROR(ResolveUnformed(&assign.lhs(), flow_facts,
                                               FlowFacts::ActionType::None));
      }
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&assign.rhs(), flow_facts,
                                             FlowFacts::ActionType::Check));
      break;
    }
    case StatementKind::ExpressionStatement: {
      const auto& exp_stmt = cast<ExpressionStatement>(*statement);
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&exp_stmt.expression(), flow_facts, action));
      break;
    }
    case StatementKind::If: {
      const auto& if_stmt = cast<If>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&if_stmt.condition(), flow_facts,
                                             FlowFacts::ActionType::Check));
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&if_stmt.then_block(), flow_facts, action));
      if (if_stmt.else_block().has_value()) {
        CARBON_RETURN_IF_ERROR(
            ResolveUnformed(*if_stmt.else_block(), flow_facts, action));
      }
      break;
    }
    case StatementKind::While: {
      const auto& while_stmt = cast<While>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(
          &while_stmt.condition(), flow_facts, FlowFacts::ActionType::Check));
      CARBON_RETURN_IF_ERROR(
          ResolveUnformed(&while_stmt.body(), flow_facts, action));
      break;
    }
    case StatementKind::Match: {
      const auto& match = cast<Match>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveUnformed(&match.expression(), flow_facts,
                                             FlowFacts::ActionType::Check));
      for (const auto& clause : match.clauses()) {
        CARBON_RETURN_IF_ERROR(ResolveUnformed(&clause.pattern(), flow_facts,
                                               FlowFacts::ActionType::Check));
        CARBON_RETURN_IF_ERROR(
            ResolveUnformed(&clause.statement(), flow_facts, action));
      }
      break;
    }
    case StatementKind::Break:
    case StatementKind::Continue:
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
    case DeclarationKind::FunctionDeclaration:
    case DeclarationKind::DestructorDeclaration: {
      const auto& callable = cast<CallableDeclaration>(*declaration);
      if (callable.body().has_value()) {
        FlowFacts flow_facts;
        CARBON_RETURN_IF_ERROR(ResolveUnformed(*callable.body(), flow_facts,
                                               FlowFacts::ActionType::None));
      }
      break;
    }
    case DeclarationKind::ClassDeclaration:
    case DeclarationKind::MixDeclaration:
    case DeclarationKind::MixinDeclaration:
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::ChoiceDeclaration:
    case DeclarationKind::VariableDeclaration:
    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration:
    case DeclarationKind::SelfDeclaration:
    case DeclarationKind::AliasDeclaration:
      // do nothing
      break;
  }
  return Success();
}

auto ResolveUnformed(const AST& ast) -> ErrorOr<Success> {
  for (auto* declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(ResolveUnformed(declaration));
  }
  return Success();
}

}  // namespace Carbon
