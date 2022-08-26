// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/ast_to_proto.h"

#include <optional>

#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using ::llvm::cast;
using ::llvm::isa;

static auto ExpressionToProto(const Expression& expression)
    -> Fuzzing::Expression;
static auto PatternToProto(const Pattern& pattern) -> Fuzzing::Pattern;
static auto StatementToProto(const Statement& statement) -> Fuzzing::Statement;
static auto DeclarationToProto(const Declaration& declaration)
    -> Fuzzing::Declaration;

static auto LibraryNameToProto(const LibraryName& library_name)
    -> Fuzzing::LibraryName {
  Fuzzing::LibraryName library_name_proto;
  library_name_proto.set_package_name(library_name.package);
  if (!library_name.path.empty()) {
    library_name_proto.set_path(library_name.path);
  }
  return library_name_proto;
}

static auto OperatorToProtoEnum(const Operator op)
    -> Fuzzing::OperatorExpression::Operator {
  switch (op) {
    case Operator::AddressOf:
      return Fuzzing::OperatorExpression::AddressOf;
    case Operator::As:
      return Fuzzing::OperatorExpression::As;
    case Operator::Deref:
      return Fuzzing::OperatorExpression::Deref;
    case Operator::Neg:
      return Fuzzing::OperatorExpression::Neg;
    case Operator::Not:
      return Fuzzing::OperatorExpression::Not;
    case Operator::Ptr:
      return Fuzzing::OperatorExpression::Ptr;
    case Operator::Add:
      return Fuzzing::OperatorExpression::Add;
    case Operator::And:
      return Fuzzing::OperatorExpression::And;
    case Operator::Eq:
      return Fuzzing::OperatorExpression::Eq;
    case Operator::Less:
      return Fuzzing::OperatorExpression::Less;
    case Operator::LessEq:
      return Fuzzing::OperatorExpression::LessEq;
    case Operator::Greater:
      return Fuzzing::OperatorExpression::Greater;
    case Operator::GreaterEq:
      return Fuzzing::OperatorExpression::GreaterEq;
    case Operator::Mul:
      return Fuzzing::OperatorExpression::Mul;
    case Operator::Mod:
      return Fuzzing::OperatorExpression::Mod;
    case Operator::Or:
      return Fuzzing::OperatorExpression::Or;
    case Operator::Sub:
      return Fuzzing::OperatorExpression::Sub;
    case Operator::BitwiseAnd:
      return Fuzzing::OperatorExpression::BitwiseAnd;
    case Operator::BitwiseOr:
      return Fuzzing::OperatorExpression::BitwiseOr;
    case Operator::BitwiseXor:
      return Fuzzing::OperatorExpression::BitwiseXor;
    case Operator::BitShiftLeft:
      return Fuzzing::OperatorExpression::BitShiftLeft;
    case Operator::BitShiftRight:
      return Fuzzing::OperatorExpression::BitShiftRight;
    case Operator::Complement:
      return Fuzzing::OperatorExpression::Complement;
  }
}

static auto FieldInitializerToProto(const FieldInitializer& field)
    -> Fuzzing::FieldInitializer {
  Fuzzing::FieldInitializer field_proto;
  field_proto.set_name(field.name());
  *field_proto.mutable_expression() = ExpressionToProto(field.expression());
  return field_proto;
}

static auto TupleLiteralExpressionToProto(const TupleLiteral& tuple_literal)
    -> Fuzzing::TupleLiteralExpression {
  Fuzzing::TupleLiteralExpression tuple_literal_proto;
  for (Nonnull<const Expression*> field : tuple_literal.fields()) {
    *tuple_literal_proto.add_fields() = ExpressionToProto(*field);
  }
  return tuple_literal_proto;
}

static auto ExpressionToProto(const Expression& expression)
    -> Fuzzing::Expression {
  Fuzzing::Expression expression_proto;
  switch (expression.kind()) {
    case ExpressionKind::InstantiateImpl:
    case ExpressionKind::ValueLiteral: {
      // These do not correspond to source syntax.
      break;
    }
    case ExpressionKind::CallExpression: {
      const auto& call = cast<CallExpression>(expression);
      auto* call_proto = expression_proto.mutable_call();
      *call_proto->mutable_function() = ExpressionToProto(call.function());
      *call_proto->mutable_argument() = ExpressionToProto(call.argument());
      break;
    }

    case ExpressionKind::FunctionTypeLiteral: {
      const auto& fun_type = cast<FunctionTypeLiteral>(expression);
      auto* fun_type_proto = expression_proto.mutable_function_type();
      *fun_type_proto->mutable_parameter() =
          TupleLiteralExpressionToProto(fun_type.parameter());
      *fun_type_proto->mutable_return_type() =
          ExpressionToProto(fun_type.return_type());
      break;
    }

    case ExpressionKind::SimpleMemberAccessExpression: {
      const auto& simple_member_access =
          cast<SimpleMemberAccessExpression>(expression);
      if (isa<DotSelfExpression>(simple_member_access.object())) {
        // The parser rewrites `.Foo` into `.Self.Foo`. Undo this
        // transformation.
        auto* designator_proto = expression_proto.mutable_designator();
        designator_proto->set_name(simple_member_access.member_name());
        break;
      }
      auto* simple_member_access_proto =
          expression_proto.mutable_simple_member_access();
      simple_member_access_proto->set_field(simple_member_access.member_name());
      *simple_member_access_proto->mutable_object() =
          ExpressionToProto(simple_member_access.object());
      break;
    }

    case ExpressionKind::CompoundMemberAccessExpression: {
      const auto& simple_member_access =
          cast<CompoundMemberAccessExpression>(expression);
      auto* simple_member_access_proto =
          expression_proto.mutable_compound_member_access();
      *simple_member_access_proto->mutable_object() =
          ExpressionToProto(simple_member_access.object());
      *simple_member_access_proto->mutable_path() =
          ExpressionToProto(simple_member_access.path());
      break;
    }

    case ExpressionKind::IndexExpression: {
      const auto& index = cast<IndexExpression>(expression);
      auto* index_proto = expression_proto.mutable_index();
      *index_proto->mutable_object() = ExpressionToProto(index.object());
      *index_proto->mutable_offset() = ExpressionToProto(index.offset());
      break;
    }

    case ExpressionKind::OperatorExpression: {
      const auto& operator_expr = cast<OperatorExpression>(expression);
      auto* operator_proto = expression_proto.mutable_operator_();
      operator_proto->set_op(OperatorToProtoEnum(operator_expr.op()));
      for (Nonnull<const Expression*> arg : operator_expr.arguments()) {
        *operator_proto->add_arguments() = ExpressionToProto(*arg);
      }
      break;
    }

    case ExpressionKind::TupleLiteral:
      *expression_proto.mutable_tuple_literal() =
          TupleLiteralExpressionToProto(cast<TupleLiteral>(expression));
      break;

    case ExpressionKind::StructLiteral: {
      const auto& struct_literal = cast<StructLiteral>(expression);
      auto* struct_literal_proto = expression_proto.mutable_struct_literal();
      for (const FieldInitializer& field : struct_literal.fields()) {
        *struct_literal_proto->add_fields() = FieldInitializerToProto(field);
      }
      break;
    }

    case ExpressionKind::StructTypeLiteral: {
      const auto& struct_type_literal = cast<StructTypeLiteral>(expression);
      auto* struct_type_literal_proto =
          expression_proto.mutable_struct_type_literal();
      for (const FieldInitializer& field : struct_type_literal.fields()) {
        *struct_type_literal_proto->add_fields() =
            FieldInitializerToProto(field);
      }
      break;
    }

    case ExpressionKind::IdentifierExpression: {
      const auto& identifier = cast<IdentifierExpression>(expression);
      auto* identifier_proto = expression_proto.mutable_identifier();
      identifier_proto->set_name(identifier.name());
      break;
    }

    case ExpressionKind::WhereExpression: {
      const auto& where = cast<WhereExpression>(expression);
      auto* where_proto = expression_proto.mutable_where();
      *where_proto->mutable_base() =
          ExpressionToProto(where.self_binding().type());
      for (const WhereClause* where : where.clauses()) {
        Fuzzing::WhereClause clause_proto;
        switch (where->kind()) {
          case WhereClauseKind::IsWhereClause: {
            auto* is_proto = clause_proto.mutable_is();
            *is_proto->mutable_type() =
                ExpressionToProto(cast<IsWhereClause>(where)->type());
            *is_proto->mutable_constraint() =
                ExpressionToProto(cast<IsWhereClause>(where)->constraint());
            break;
          }
          case WhereClauseKind::EqualsWhereClause: {
            auto* equals_proto = clause_proto.mutable_equals();
            *equals_proto->mutable_lhs() =
                ExpressionToProto(cast<EqualsWhereClause>(where)->lhs());
            *equals_proto->mutable_rhs() =
                ExpressionToProto(cast<EqualsWhereClause>(where)->rhs());
            break;
          }
        }
        *where_proto->add_clauses() = clause_proto;
      }
      break;
    }

    case ExpressionKind::DotSelfExpression: {
      auto* designator_proto = expression_proto.mutable_designator();
      designator_proto->set_name("Self");
      break;
    }

    case ExpressionKind::IntrinsicExpression: {
      const auto& intrinsic = cast<IntrinsicExpression>(expression);
      auto* call_proto = expression_proto.mutable_call();
      call_proto->mutable_function()->mutable_identifier()->set_name(
          std::string(intrinsic.name()));
      *call_proto->mutable_argument() = ExpressionToProto(intrinsic.args());
      break;
    }

    case ExpressionKind::IfExpression: {
      const auto& if_expression = cast<IfExpression>(expression);
      auto* if_proto = expression_proto.mutable_if_expression();
      *if_proto->mutable_condition() =
          ExpressionToProto(if_expression.condition());
      *if_proto->mutable_then_expression() =
          ExpressionToProto(if_expression.then_expression());
      *if_proto->mutable_else_expression() =
          ExpressionToProto(if_expression.else_expression());
      break;
    }

    case ExpressionKind::BoolTypeLiteral:
      expression_proto.mutable_bool_type_literal();
      break;

    case ExpressionKind::BoolLiteral:
      expression_proto.mutable_bool_literal()->set_value(
          cast<BoolLiteral>(expression).value());
      break;

    case ExpressionKind::IntTypeLiteral:
      expression_proto.mutable_int_type_literal();
      break;

    case ExpressionKind::IntLiteral:
      expression_proto.mutable_int_literal()->set_value(
          cast<IntLiteral>(expression).value());
      break;

    case ExpressionKind::StringLiteral:
      expression_proto.mutable_string_literal()->set_value(
          cast<StringLiteral>(expression).value());
      break;

    case ExpressionKind::StringTypeLiteral:
      expression_proto.mutable_string_type_literal();
      break;

    case ExpressionKind::ContinuationTypeLiteral:
      expression_proto.mutable_continuation_type_literal();
      break;

    case ExpressionKind::TypeTypeLiteral:
      expression_proto.mutable_type_type_literal();
      break;

    case ExpressionKind::UnimplementedExpression:
      expression_proto.mutable_unimplemented_expression();
      break;

    case ExpressionKind::ArrayTypeLiteral: {
      const auto& array_literal = cast<ArrayTypeLiteral>(expression);
      Fuzzing::ArrayTypeLiteral* array_literal_proto =
          expression_proto.mutable_array_type_literal();
      *array_literal_proto->mutable_element_type() =
          ExpressionToProto(array_literal.element_type_expression());
      *array_literal_proto->mutable_size() =
          ExpressionToProto(array_literal.size_expression());
      break;
    }
  }
  return expression_proto;
}

static auto BindingPatternToProto(const BindingPattern& pattern)
    -> Fuzzing::BindingPattern {
  Fuzzing::BindingPattern pattern_proto;
  pattern_proto.set_name(pattern.name());
  *pattern_proto.mutable_type() = PatternToProto(pattern.type());
  return pattern_proto;
}

static auto GenericBindingToProto(const GenericBinding& binding)
    -> Fuzzing::GenericBinding {
  Fuzzing::GenericBinding binding_proto;
  binding_proto.set_name(binding.name());
  *binding_proto.mutable_type() = ExpressionToProto(binding.type());
  return binding_proto;
}

static auto TuplePatternToProto(const TuplePattern& tuple_pattern)
    -> Fuzzing::TuplePattern {
  Fuzzing::TuplePattern tuple_pattern_proto;
  for (Nonnull<const Pattern*> field : tuple_pattern.fields()) {
    *tuple_pattern_proto.add_fields() = PatternToProto(*field);
  }
  return tuple_pattern_proto;
}

static auto PatternToProto(const Pattern& pattern) -> Fuzzing::Pattern {
  Fuzzing::Pattern pattern_proto;
  switch (pattern.kind()) {
    case PatternKind::GenericBinding: {
      const auto& binding = cast<GenericBinding>(pattern);
      *pattern_proto.mutable_generic_binding() = GenericBindingToProto(binding);
      break;
    }
    case PatternKind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      *pattern_proto.mutable_binding_pattern() = BindingPatternToProto(binding);
      break;
    }
    case PatternKind::TuplePattern:
      *pattern_proto.mutable_tuple_pattern() =
          TuplePatternToProto(cast<TuplePattern>(pattern));
      break;

    case PatternKind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(pattern);
      auto* alternative_proto = pattern_proto.mutable_alternative_pattern();
      alternative_proto->set_alternative_name(alternative.alternative_name());
      *alternative_proto->mutable_choice_type() =
          ExpressionToProto(alternative.choice_type());
      *alternative_proto->mutable_arguments() =
          TuplePatternToProto(alternative.arguments());
      break;
    }

    case PatternKind::ExpressionPattern:
      *pattern_proto.mutable_expression_pattern()->mutable_expression() =
          ExpressionToProto(cast<ExpressionPattern>(pattern).expression());
      break;

    case PatternKind::AutoPattern:
      pattern_proto.mutable_auto_pattern();
      break;

    case PatternKind::VarPattern:
      *pattern_proto.mutable_var_pattern()->mutable_pattern() =
          PatternToProto(cast<VarPattern>(pattern).pattern());
      break;
    case PatternKind::AddrPattern:
      *pattern_proto.mutable_addr_pattern()->mutable_binding_pattern() =
          BindingPatternToProto(cast<AddrPattern>(pattern).binding());
      break;
  }
  return pattern_proto;
}

static auto BlockStatementToProto(const Block& block)
    -> Fuzzing::BlockStatement {
  Fuzzing::BlockStatement block_proto;
  for (Nonnull<const Statement*> statement : block.statements()) {
    *block_proto.add_statements() = StatementToProto(*statement);
  }
  return block_proto;
}

static auto StatementToProto(const Statement& statement) -> Fuzzing::Statement {
  Fuzzing::Statement statement_proto;
  switch (statement.kind()) {
    case StatementKind::ExpressionStatement:
      *statement_proto.mutable_expression_statement()->mutable_expression() =
          ExpressionToProto(cast<ExpressionStatement>(statement).expression());
      break;

    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(statement);
      auto* assign_proto = statement_proto.mutable_assign();
      *assign_proto->mutable_lhs() = ExpressionToProto(assign.lhs());
      *assign_proto->mutable_rhs() = ExpressionToProto(assign.rhs());
      break;
    }

    case StatementKind::VariableDefinition: {
      const auto& def = cast<VariableDefinition>(statement);
      auto* def_proto = statement_proto.mutable_variable_definition();
      *def_proto->mutable_pattern() = PatternToProto(def.pattern());
      if (def.has_init()) {
        *def_proto->mutable_init() = ExpressionToProto(def.init());
      }
      def_proto->set_is_returned(def.is_returned());
      break;
    }

    case StatementKind::If: {
      const auto& if_stmt = cast<If>(statement);
      auto* if_proto = statement_proto.mutable_if_statement();
      *if_proto->mutable_condition() = ExpressionToProto(if_stmt.condition());
      *if_proto->mutable_then_block() =
          BlockStatementToProto(if_stmt.then_block());
      if (if_stmt.else_block().has_value()) {
        *if_proto->mutable_else_block() =
            BlockStatementToProto(**if_stmt.else_block());
      }
      break;
    }

    case StatementKind::ReturnVar: {
      statement_proto.mutable_return_var_statement();
      break;
    }

    case StatementKind::ReturnExpression: {
      const auto& ret = cast<ReturnExpression>(statement);
      auto* ret_proto = statement_proto.mutable_return_expression_statement();
      if (!ret.is_omitted_expression()) {
        *ret_proto->mutable_expression() = ExpressionToProto(ret.expression());
      } else {
        ret_proto->set_is_omitted_expression(true);
      }
      break;
    }

    case StatementKind::Block:
      *statement_proto.mutable_block() =
          BlockStatementToProto(cast<Block>(statement));
      break;

    case StatementKind::While: {
      const auto& while_stmt = cast<While>(statement);
      auto* while_proto = statement_proto.mutable_while_statement();
      *while_proto->mutable_condition() =
          ExpressionToProto(while_stmt.condition());
      *while_proto->mutable_body() = BlockStatementToProto(while_stmt.body());
      break;
    }

    case StatementKind::Match: {
      const auto& match = cast<Match>(statement);
      auto* match_proto = statement_proto.mutable_match();
      *match_proto->mutable_expression() =
          ExpressionToProto(match.expression());
      for (const Match::Clause& clause : match.clauses()) {
        auto* clause_proto = match_proto->add_clauses();
        const bool is_default_clause =
            clause.pattern().kind() == PatternKind::BindingPattern &&
            cast<BindingPattern>(clause.pattern()).name() == AnonymousName;
        if (is_default_clause) {
          clause_proto->set_is_default(true);
        } else {
          *clause_proto->mutable_pattern() = PatternToProto(clause.pattern());
        }
        *clause_proto->mutable_statement() =
            StatementToProto(clause.statement());
      }
      break;
    }

    case StatementKind::Continuation: {
      const auto& continuation = cast<Continuation>(statement);
      auto* continuation_proto = statement_proto.mutable_continuation();
      continuation_proto->set_name(continuation.name());
      *continuation_proto->mutable_body() =
          BlockStatementToProto(continuation.body());
      break;
    }

    case StatementKind::Run:
      *statement_proto.mutable_run()->mutable_argument() =
          ExpressionToProto(cast<Run>(statement).argument());
      break;

    case StatementKind::Await:
      // Initializes with the default value; there's nothing to set.
      statement_proto.mutable_await_statement();
      break;

    case StatementKind::Break:
      // Initializes with the default value; there's nothing to set.
      statement_proto.mutable_break_statement();
      break;

    case StatementKind::Continue:
      // Initializes with the default value; there's nothing to set.
      statement_proto.mutable_continue_statement();
      break;

    case StatementKind::For: {
      const auto& for_stmt = cast<For>(statement);
      auto* for_proto = statement_proto.mutable_for_statement();
      *for_proto->mutable_var_decl() =
          BindingPatternToProto(for_stmt.variable_declaration());
      *for_proto->mutable_target() = ExpressionToProto(for_stmt.loop_target());
      *for_proto->mutable_body() = BlockStatementToProto(for_stmt.body());
      break;
    }
  }
  return statement_proto;
}

static auto ReturnTermToProto(const ReturnTerm& return_term)
    -> Fuzzing::ReturnTerm {
  Fuzzing::ReturnTerm return_term_proto;
  if (return_term.is_omitted()) {
    return_term_proto.set_kind(Fuzzing::ReturnTerm::Omitted);
  } else if (return_term.is_auto()) {
    return_term_proto.set_kind(Fuzzing::ReturnTerm::Auto);
  } else {
    return_term_proto.set_kind(Fuzzing::ReturnTerm::Expression);
    *return_term_proto.mutable_type() =
        ExpressionToProto(**return_term.type_expression());
  }
  return return_term_proto;
}

static auto DeclarationToProto(const Declaration& declaration)
    -> Fuzzing::Declaration {
  Fuzzing::Declaration declaration_proto;
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration: {
      const auto& function = cast<FunctionDeclaration>(declaration);
      auto* function_proto = declaration_proto.mutable_function();
      function_proto->set_name(function.name());
      for (Nonnull<const GenericBinding*> binding :
           function.deduced_parameters()) {
        *function_proto->add_deduced_parameters() =
            GenericBindingToProto(*binding);
      }
      if (function.is_method()) {
        switch (function.me_pattern().kind()) {
          case PatternKind::AddrPattern:
            *function_proto->mutable_me_pattern() =
                PatternToProto(cast<AddrPattern>(function.me_pattern()));
            break;
          case PatternKind::BindingPattern:
            *function_proto->mutable_me_pattern() =
                PatternToProto(cast<BindingPattern>(function.me_pattern()));
            break;
          default:
            // Parser shouldn't allow me_pattern to be anything other than
            // AddrPattern or BindingPattern
            CARBON_FATAL() << "me_pattern in method declaration can be either "
                              "AddrPattern or BindingPattern. Actual pattern: "
                           << function.me_pattern();
            break;
        }
      }
      *function_proto->mutable_param_pattern() =
          TuplePatternToProto(function.param_pattern());
      *function_proto->mutable_return_term() =
          ReturnTermToProto(function.return_term());
      if (function.body().has_value()) {
        *function_proto->mutable_body() =
            BlockStatementToProto(**function.body());
      }
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(declaration);
      auto* class_proto = declaration_proto.mutable_class_declaration();
      class_proto->set_name(class_decl.name());
      if (class_decl.type_params().has_value()) {
        *class_proto->mutable_type_params() =
            TuplePatternToProto(**class_decl.type_params());
      }
      for (Nonnull<const Declaration*> member : class_decl.members()) {
        *class_proto->add_members() = DeclarationToProto(*member);
      }
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(declaration);
      auto* choice_proto = declaration_proto.mutable_choice();
      choice_proto->set_name(choice.name());
      for (Nonnull<const AlternativeSignature*> alternative :
           choice.alternatives()) {
        auto* alternative_proto = choice_proto->add_alternatives();
        alternative_proto->set_name(alternative->name());
        *alternative_proto->mutable_signature() =
            TupleLiteralExpressionToProto(alternative->signature());
      }
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(declaration);
      auto* var_proto = declaration_proto.mutable_variable();
      *var_proto->mutable_binding() = BindingPatternToProto(var.binding());
      if (var.has_initializer()) {
        *var_proto->mutable_initializer() =
            ExpressionToProto(var.initializer());
      }
      break;
    }

    case DeclarationKind::AssociatedConstantDeclaration: {
      const auto& assoc = cast<AssociatedConstantDeclaration>(declaration);
      auto* let_proto = declaration_proto.mutable_let();
      *let_proto->mutable_pattern() = PatternToProto(assoc.binding());
      break;
    }

    case DeclarationKind::InterfaceDeclaration: {
      const auto& interface = cast<InterfaceDeclaration>(declaration);
      auto* interface_proto = declaration_proto.mutable_interface();
      interface_proto->set_name(interface.name());
      for (const auto& member : interface.members()) {
        *interface_proto->add_members() = DeclarationToProto(*member);
      }
      *interface_proto->mutable_self() =
          GenericBindingToProto(*interface.self());
      break;
    }

    case DeclarationKind::ImplDeclaration: {
      const auto& impl = cast<ImplDeclaration>(declaration);
      auto* impl_proto = declaration_proto.mutable_impl();
      switch (impl.kind()) {
        case ImplKind::InternalImpl:
          impl_proto->set_kind(Fuzzing::ImplDeclaration::InternalImpl);
          break;
        case ImplKind::ExternalImpl:
          impl_proto->set_kind(Fuzzing::ImplDeclaration::ExternalImpl);
          break;
      }
      *impl_proto->mutable_impl_type() = ExpressionToProto(*impl.impl_type());
      *impl_proto->mutable_interface() = ExpressionToProto(impl.interface());
      for (const auto& member : impl.members()) {
        *impl_proto->add_members() = DeclarationToProto(*member);
      }
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      CARBON_FATAL() << "Unreachable SelfDeclaration in DeclarationToProto().";
    }

    case DeclarationKind::AliasDeclaration: {
      const auto& alias = cast<AliasDeclaration>(declaration);
      auto* alias_proto = declaration_proto.mutable_alias();
      alias_proto->set_name(alias.name());
      *alias_proto->mutable_target() = ExpressionToProto(alias.target());
      break;
    }
  }
  return declaration_proto;
}

Fuzzing::CompilationUnit AstToProto(const AST& ast) {
  Fuzzing::CompilationUnit compilation_unit;
  *compilation_unit.mutable_package_statement() =
      LibraryNameToProto(ast.package);
  compilation_unit.set_is_api(ast.is_api);
  for (const Declaration* declaration : ast.declarations) {
    *compilation_unit.add_declarations() = DeclarationToProto(*declaration);
  }
  return compilation_unit;
}

}  // namespace Carbon
