// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/ast_to_proto.h"

#include <optional>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/generic_binding.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

namespace {

using ::llvm::cast;

auto ExpressionToProto(const Expression& expression) -> Fuzzing::Expression;
auto PatternToProto(const Pattern& pattern) -> Fuzzing::Pattern;
auto StatementToProto(const Statement& statement) -> Fuzzing::Statement;
auto DeclarationToProto(const Declaration& declaration) -> Fuzzing::Declaration;

auto LibraryNameToProto(const LibraryName& library_name)
    -> Fuzzing::LibraryName {
  Fuzzing::LibraryName library_name_proto;
  library_name_proto.set_package_name(library_name.package);
  if (!library_name.path.empty()) {
    library_name_proto.set_path(library_name.path);
  }
  return library_name_proto;
}

auto OperatorToUnaryProtoEnum(const Operator op)
    -> std::optional<Fuzzing::UnaryOperatorExpression::UnaryOperator> {
  switch (op) {
    case Operator::AddressOf:
      return Fuzzing::UnaryOperatorExpression::OP_ADDRESS_OF;
    case Operator::Deref:
      return Fuzzing::UnaryOperatorExpression::OP_DEREF;
    case Operator::Neg:
      return Fuzzing::UnaryOperatorExpression::OP_NEG;
    case Operator::Not:
      return Fuzzing::UnaryOperatorExpression::OP_NOT;
    case Operator::Ptr:
      return Fuzzing::UnaryOperatorExpression::OP_PTR;

    case Operator::Add:
    case Operator::And:
    case Operator::Eq:
    case Operator::Mul:
    case Operator::Or:
    case Operator::Sub:
      return std::nullopt;
  }
}

auto OperatorToBinaryProtoEnum(const Operator op)
    -> std::optional<Fuzzing::BinaryOperatorExpression::BinaryOperator> {
  switch (op) {
    case Operator::Add:
      return Fuzzing::BinaryOperatorExpression::OP_ADD;
    case Operator::And:
      return Fuzzing::BinaryOperatorExpression::OP_AND;
    case Operator::Eq:
      return Fuzzing::BinaryOperatorExpression::OP_EQ;
    case Operator::Mul:
      return Fuzzing::BinaryOperatorExpression::OP_MUL;
    case Operator::Or:
      return Fuzzing::BinaryOperatorExpression::OP_OR;
    case Operator::Sub:
      return Fuzzing::BinaryOperatorExpression::OP_SUB;

    case Operator::AddressOf:
    case Operator::Deref:
    case Operator::Neg:
    case Operator::Not:
    case Operator::Ptr:
      return std::nullopt;
  }
}

auto FieldInitializerToProto(const FieldInitializer& field)
    -> Fuzzing::FieldInitializer {
  Fuzzing::FieldInitializer field_proto;
  field_proto.set_name(field.name());
  *field_proto.mutable_expression() = ExpressionToProto(field.expression());
  return field_proto;
}

auto TupleLiteralExpressionToProto(const TupleLiteral& tuple_literal)
    -> Fuzzing::TupleLiteralExpression {
  Fuzzing::TupleLiteralExpression tuple_literal_proto;
  for (Nonnull<const Expression*> field : tuple_literal.fields()) {
    *tuple_literal_proto.add_field() = ExpressionToProto(*field);
  }
  return tuple_literal_proto;
}

auto ExpressionToProto(const Expression& expression) -> Fuzzing::Expression {
  Fuzzing::Expression expression_proto;
  switch (expression.kind()) {
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
          ExpressionToProto(fun_type.parameter());
      *fun_type_proto->mutable_return_type() =
          ExpressionToProto(fun_type.return_type());
      break;
    }

    case ExpressionKind::FieldAccessExpression: {
      const auto& field_access = cast<FieldAccessExpression>(expression);
      auto* field_access_proto = expression_proto.mutable_field_access();
      field_access_proto->set_field(field_access.field());
      *field_access_proto->mutable_aggregate() =
          ExpressionToProto(field_access.aggregate());
      break;
    }

    case ExpressionKind::IndexExpression: {
      const auto& index = cast<IndexExpression>(expression);
      auto* index_proto = expression_proto.mutable_index();
      *index_proto->mutable_aggregate() = ExpressionToProto(index.aggregate());
      *index_proto->mutable_offset() = ExpressionToProto(index.offset());
      break;
    }

    case ExpressionKind::PrimitiveOperatorExpression: {
      const auto& primitive_operator =
          cast<PrimitiveOperatorExpression>(expression);
      if (const auto unary_op_enum =
              OperatorToUnaryProtoEnum(primitive_operator.op());
          unary_op_enum.has_value()) {
        CHECK(primitive_operator.arguments().size() == 1);
        auto* unary_operator_proto = expression_proto.mutable_unary_operator();
        unary_operator_proto->set_op(*unary_op_enum);
        *unary_operator_proto->mutable_arg() =
            ExpressionToProto(*primitive_operator.arguments()[0]);
      } else if (const auto binary_op_enum =
                     OperatorToBinaryProtoEnum(primitive_operator.op());
                 binary_op_enum.has_value()) {
        CHECK(primitive_operator.arguments().size() == 2);
        auto* binary_operator_proto =
            expression_proto.mutable_binary_operator();
        binary_operator_proto->set_op(*binary_op_enum);
        *binary_operator_proto->mutable_lhs() =
            ExpressionToProto(*primitive_operator.arguments()[0]);
        *binary_operator_proto->mutable_rhs() =
            ExpressionToProto(*primitive_operator.arguments()[1]);
      } else {
        FATAL() << "Unknown operator "
                << static_cast<int>(primitive_operator.op());
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
        *struct_literal_proto->add_field() = FieldInitializerToProto(field);
      }
      break;
    }

    case ExpressionKind::StructTypeLiteral: {
      const auto& struct_type_literal = cast<StructTypeLiteral>(expression);
      auto* struct_type_literal_proto =
          expression_proto.mutable_struct_type_literal();
      for (const FieldInitializer& field : struct_type_literal.fields()) {
        *struct_type_literal_proto->add_field() =
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

    case ExpressionKind::IntrinsicExpression: {
      const auto& intrinsic = cast<IntrinsicExpression>(expression);
      auto* intrinsic_proto = expression_proto.mutable_intrinsic();
      switch (intrinsic.intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          intrinsic_proto->set_intrinsic(
              Fuzzing::IntrinsicExpression::INTRINSIC_PRINT);
          break;
      }
      *intrinsic_proto->mutable_argument() =
          TupleLiteralExpressionToProto(intrinsic.args());
      break;
    }

    case ExpressionKind::IfExpression: {
      const auto& if_expression = cast<IfExpression>(expression);
      auto* if_proto = expression_proto.mutable_if_expression();
      if (if_expression.condition()) {
        *if_proto->mutable_condition() =
            ExpressionToProto(*if_expression.condition());
      }
      if (if_expression.then_expression()) {
        *if_proto->mutable_then_expression() =
            ExpressionToProto(*if_expression.then_expression());
      }
      if (if_expression.else_expression()) {
        *if_proto->mutable_else_expression() =
            ExpressionToProto(*if_expression.else_expression());
      }
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
  }
  return expression_proto;
}

auto BindingPatternToProto(const BindingPattern& pattern)
    -> Fuzzing::BindingPattern {
  Fuzzing::BindingPattern pattern_proto;
  pattern_proto.set_name(pattern.name());
  *pattern_proto.mutable_type() = PatternToProto(pattern.type());
  return pattern_proto;
}

auto TuplePatternToProto(const TuplePattern& tuple_pattern)
    -> Fuzzing::TuplePattern {
  Fuzzing::TuplePattern tuple_pattern_proto;
  for (Nonnull<const Pattern*> field : tuple_pattern.fields()) {
    *tuple_pattern_proto.add_field() = PatternToProto(*field);
  }
  return tuple_pattern_proto;
}

auto PatternToProto(const Pattern& pattern) -> Fuzzing::Pattern {
  Fuzzing::Pattern pattern_proto;
  switch (pattern.kind()) {
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
  }
  return pattern_proto;
}

auto BlockStatementToProto(const Block& block) -> Fuzzing::BlockStatement {
  Fuzzing::BlockStatement block_proto;
  for (Nonnull<const Statement*> statement : block.statements()) {
    *block_proto.add_statement() = StatementToProto(*statement);
  }
  return block_proto;
}

auto StatementToProto(const Statement& statement) -> Fuzzing::Statement {
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
      *def_proto->mutable_init() = ExpressionToProto(def.init());
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

    case StatementKind::Return: {
      const auto& ret = cast<Return>(statement);
      auto* ret_proto = statement_proto.mutable_return_statement();
      if (!ret.is_omitted_expression()) {
        *ret_proto->mutable_expression() = ExpressionToProto(ret.expression());
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
        auto* clause_proto = match_proto->add_clause();
        *clause_proto->mutable_pattern() = PatternToProto(clause.pattern());
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
      statement_proto.mutable_await();
      break;

    case StatementKind::Break:
      statement_proto.mutable_break_statement();
      break;

    case StatementKind::Continue:
      // Initializes with the default value; there's nothing to set.
      statement_proto.mutable_continue_statement();
      break;
  }
  return statement_proto;
}

auto ReturnTermToProto(const ReturnTerm& return_term) -> Fuzzing::ReturnTerm {
  Fuzzing::ReturnTerm return_term_proto;
  if (return_term.is_omitted()) {
    return_term_proto.set_kind(Fuzzing::ReturnTerm::RK_OMITTED);
  } else if (return_term.is_auto()) {
    return_term_proto.set_kind(Fuzzing::ReturnTerm::RK_AUTO);
  } else {
    return_term_proto.set_kind(Fuzzing::ReturnTerm::RK_EXPRESSION);
    *return_term_proto.mutable_type() =
        ExpressionToProto(**return_term.type_expression());
  }
  return return_term_proto;
}

auto GenericBindingToProto(const GenericBinding& binding)
    -> Fuzzing::GenericBinding {
  Fuzzing::GenericBinding binding_proto;
  binding_proto.set_name(binding.name());
  *binding_proto.mutable_type() = ExpressionToProto(binding.type());
  return binding_proto;
}

auto DeclarationToProto(const Declaration& declaration)
    -> Fuzzing::Declaration {
  Fuzzing::Declaration declaration_proto;
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration: {
      const auto& function = cast<FunctionDeclaration>(declaration);
      auto* function_proto = declaration_proto.mutable_function();
      function_proto->set_name(function.name());
      for (Nonnull<const GenericBinding*> binding :
           function.deduced_parameters()) {
        *function_proto->add_deduced_parameter() =
            GenericBindingToProto(*binding);
      }
      if (function.is_method()) {
        *function_proto->mutable_me_pattern() =
            BindingPatternToProto(function.me_pattern());
      }
      *function_proto->mutable_param_pattern() =
          TuplePatternToProto(function.param_pattern());
      if (function.return_term().type_expression().has_value()) {
        *function_proto->mutable_return_term() =
            ReturnTermToProto(function.return_term());
      }
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
      for (Nonnull<const Declaration*> member : class_decl.members()) {
        *class_proto->add_member() = DeclarationToProto(*member);
      }
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(declaration);
      auto* choice_proto = declaration_proto.mutable_choice();
      choice_proto->set_name(choice.name());
      for (Nonnull<const AlternativeSignature*> alternative :
           choice.alternatives()) {
        auto* alternative_proto = choice_proto->add_alternative();
        alternative_proto->set_name(alternative->name());
        *alternative_proto->mutable_signature() =
            ExpressionToProto(alternative->signature());
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

    case DeclarationKind::InterfaceDeclaration: {
      const auto& interface = cast<InterfaceDeclaration>(declaration);
      auto* interface_proto = declaration_proto.mutable_interface();
      interface_proto->set_name(interface.name());
      for (const auto& member : interface.members()) {
        *interface_proto->add_member() = DeclarationToProto(*member);
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
          impl_proto->set_kind(Fuzzing::ImplDeclaration::INTERNAL_IMPL);
          break;
        case ImplKind::ExternalImpl:
          impl_proto->set_kind(Fuzzing::ImplDeclaration::EXTERNAL_IMPL);
          break;
      }
      *impl_proto->mutable_impl_type() = ExpressionToProto(*impl.impl_type());
      *impl_proto->mutable_interface() = ExpressionToProto(impl.interface());
      for (const auto& member : impl.members()) {
        *impl_proto->add_member() = DeclarationToProto(*member);
      }
      break;
    }
  }
  return declaration_proto;
}

}  // namespace

Fuzzing::CompilationUnit CarbonToProto(const AST& ast) {
  Fuzzing::CompilationUnit compilation_unit;
  *compilation_unit.mutable_package_statement() =
      LibraryNameToProto(ast.package);
  compilation_unit.set_is_api(ast.is_api);
  for (const Declaration* declaration : ast.declarations) {
    *compilation_unit.add_declaration() = DeclarationToProto(*declaration);
  }
  return compilation_unit;
}

}  // namespace Carbon
