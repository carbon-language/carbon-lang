// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/fuzzing/proto_to_carbon.h"

#include <string_view>

#include "common/fuzzing/carbon.pb.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

static auto ExpressionToCarbon(const Fuzzing::Expression& expression,
                               llvm::raw_ostream& out) -> void;
static auto PatternToCarbon(const Fuzzing::Pattern& pattern,
                            llvm::raw_ostream& out) -> void;
static auto StatementToCarbon(const Fuzzing::Statement& statement,
                              llvm::raw_ostream& out) -> void;
static auto DeclarationToCarbon(const Fuzzing::Declaration& declaration,
                                llvm::raw_ostream& out) -> void;

// Produces a valid Carbon identifier, which must match the regex
// `[A-Za-z_][A-Za-z0-9_]*`. In the case when `s` is generated by the
// fuzzing framework, it might contain invalid/non-printable characters.
static auto IdentifierToCarbon(std::string_view s, llvm::raw_ostream& out)
    -> void {
  if (s.empty()) {
    out << "EmptyIdentifier";
  } else {
    if (!llvm::isAlpha(s[0]) && s[0] != '_') {
      // Ensures that identifier starts with a valid character.
      out << 'x';
    }
    for (const char c : s) {
      if (llvm::isAlnum(c) || c == '_') {
        out << c;
      } else {
        out << llvm::toHex(c);
      }
    }
  }
}

static auto StringLiteralToCarbon(std::string_view s, llvm::raw_ostream& out) {
  out << '"';
  out.write_escaped(s, /*UseHexEscapes=*/true);
  out << '"';
}

static auto LibraryNameToCarbon(const Fuzzing::LibraryName& library,
                                llvm::raw_ostream& out) -> void {
  IdentifierToCarbon(library.package_name(), out);

  // Path is optional.
  if (library.has_path()) {
    out << " library ";
    // library.path() is a string literal.
    StringLiteralToCarbon(library.path(), out);
  }
}

static auto PrefixUnaryOperatorToCarbon(std::string_view op,
                                        const Fuzzing::Expression& arg,
                                        llvm::raw_ostream& out) -> void {
  out << op;
  ExpressionToCarbon(arg, out);
}

static auto PostfixUnaryOperatorToCarbon(const Fuzzing::Expression& arg,
                                         std::string_view op,
                                         llvm::raw_ostream& out) -> void {
  ExpressionToCarbon(arg, out);
  out << op;
}

static auto BinaryOperatorToCarbon(const Fuzzing::Expression& lhs,
                                   std::string_view op,
                                   const Fuzzing::Expression& rhs,
                                   llvm::raw_ostream& out) -> void {
  ExpressionToCarbon(lhs, out);
  out << op;
  ExpressionToCarbon(rhs, out);
}

static auto OperatorToCarbon(const Fuzzing::OperatorExpression& operator_expr,
                             llvm::raw_ostream& out) -> void {
  const Fuzzing::Expression& arg0 =
      !operator_expr.arguments().empty()
          ? operator_expr.arguments(0)
          : Fuzzing::Expression::default_instance();
  const Fuzzing::Expression& arg1 =
      operator_expr.arguments().size() > 1
          ? operator_expr.arguments(1)
          : Fuzzing::Expression::default_instance();
  out << "(";
  switch (operator_expr.op()) {
    case Fuzzing::OperatorExpression::UnknownOperator:
      // `-` is an arbitrary default to avoid getting invalid syntax.
      PrefixUnaryOperatorToCarbon("-", arg0, out);
      break;

    case Fuzzing::OperatorExpression::AddressOf:
      PrefixUnaryOperatorToCarbon("&", arg0, out);
      break;

    case Fuzzing::OperatorExpression::As:
      BinaryOperatorToCarbon(arg0, " as ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Deref:
      PrefixUnaryOperatorToCarbon("*", arg0, out);
      break;

    case Fuzzing::OperatorExpression::Mul:
      BinaryOperatorToCarbon(arg0, " * ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Div:
      BinaryOperatorToCarbon(arg0, " / ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Mod:
      BinaryOperatorToCarbon(arg0, " % ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Ptr:
      PostfixUnaryOperatorToCarbon(arg0, "*", out);
      break;

    case Fuzzing::OperatorExpression::Neg:
      PrefixUnaryOperatorToCarbon("-", arg0, out);
      break;

    case Fuzzing::OperatorExpression::Sub:
      BinaryOperatorToCarbon(arg0, " - ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Not:
      // Needs a space to 'unglue' from the operand.
      PrefixUnaryOperatorToCarbon("not ", arg0, out);
      break;

    case Fuzzing::OperatorExpression::Add:
      BinaryOperatorToCarbon(arg0, " + ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::And:
      BinaryOperatorToCarbon(arg0, " and ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Eq:
      BinaryOperatorToCarbon(arg0, " == ", arg1, out);
      break;
    case Fuzzing::OperatorExpression::Less:
      BinaryOperatorToCarbon(arg0, " < ", arg1, out);
      break;
    case Fuzzing::OperatorExpression::LessEq:
      BinaryOperatorToCarbon(arg0, " <= ", arg1, out);
      break;
    case Fuzzing::OperatorExpression::GreaterEq:
      BinaryOperatorToCarbon(arg0, " >= ", arg1, out);
      break;
    case Fuzzing::OperatorExpression::Greater:
      BinaryOperatorToCarbon(arg0, " > ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Or:
      BinaryOperatorToCarbon(arg0, " or ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::Complement:
      PrefixUnaryOperatorToCarbon("^", arg0, out);
      break;

    case Fuzzing::OperatorExpression::BitwiseAnd:
      BinaryOperatorToCarbon(arg0, " & ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::BitwiseOr:
      BinaryOperatorToCarbon(arg0, " | ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::BitwiseXor:
      BinaryOperatorToCarbon(arg0, " ^ ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::BitShiftLeft:
      BinaryOperatorToCarbon(arg0, " << ", arg1, out);
      break;

    case Fuzzing::OperatorExpression::BitShiftRight:
      BinaryOperatorToCarbon(arg0, " >> ", arg1, out);
      break;
    case Fuzzing::OperatorExpression::NotEq:
      BinaryOperatorToCarbon(arg0, " != ", arg1, out);
      break;
  }
  out << ")";
}

static auto FieldInitializerToCarbon(const Fuzzing::FieldInitializer& field,
                                     std::string_view separator,
                                     llvm::raw_ostream& out) -> void {
  out << ".";
  IdentifierToCarbon(field.name(), out);
  out << " " << separator << " ";
  ExpressionToCarbon(field.expression(), out);
}

static auto TupleLiteralExpressionToCarbon(
    const Fuzzing::TupleLiteralExpression& tuple_literal,
    llvm::raw_ostream& out) -> void {
  out << "(";
  llvm::ListSeparator sep;
  for (const auto& field : tuple_literal.fields()) {
    out << sep;
    ExpressionToCarbon(field, out);
  }
  if (tuple_literal.fields_size() == 1) {
    // Adding a trailing comma so that generated source will be parsed as a
    // tuple expression. See docs/design/tuples.md.
    out << ", ";
  }
  out << ")";
}

static auto ExpressionToCarbon(const Fuzzing::Expression& expression,
                               llvm::raw_ostream& out) -> void {
  switch (expression.kind_case()) {
    case Fuzzing::Expression::KIND_NOT_SET:
      // Arbitrary default for missing expressions to avoid invalid syntax.
      out << "true";
      break;

    case Fuzzing::Expression::kCall: {
      const auto& call = expression.call();
      ExpressionToCarbon(call.function(), out);
      if (call.argument().kind_case() == Fuzzing::Expression::kTupleLiteral) {
        TupleLiteralExpressionToCarbon(call.argument().tuple_literal(), out);
      } else {
        out << "(";
        ExpressionToCarbon(call.argument(), out);
        out << ")";
      }
      break;
    }

    case Fuzzing::Expression::kFunctionType: {
      const auto& fun_type = expression.function_type();
      out << "__Fn";
      TupleLiteralExpressionToCarbon(fun_type.parameter(), out);
      out << " -> ";
      ExpressionToCarbon(fun_type.return_type(), out);
      break;
    }

    case Fuzzing::Expression::kSimpleMemberAccess: {
      const auto& simple_member_access = expression.simple_member_access();
      ExpressionToCarbon(simple_member_access.object(), out);
      out << ".";
      IdentifierToCarbon(simple_member_access.field(), out);
      break;
    }

    case Fuzzing::Expression::kCompoundMemberAccess: {
      const auto& simple_member_access = expression.compound_member_access();
      ExpressionToCarbon(simple_member_access.object(), out);
      out << ".(";
      ExpressionToCarbon(simple_member_access.path(), out);
      out << ")";
      break;
    }

    case Fuzzing::Expression::kIndex: {
      const auto& index = expression.index();
      ExpressionToCarbon(index.object(), out);
      out << "[";
      ExpressionToCarbon(index.offset(), out);
      out << "]";
      break;
    }

    case Fuzzing::Expression::kOperator:
      OperatorToCarbon(expression.operator_(), out);
      break;

    case Fuzzing::Expression::kTupleLiteral: {
      TupleLiteralExpressionToCarbon(expression.tuple_literal(), out);
      break;
    }

    case Fuzzing::Expression::kStructLiteral: {
      const auto& struct_literal = expression.struct_literal();
      out << "{";
      llvm::ListSeparator sep;
      for (const auto& field : struct_literal.fields()) {
        out << sep;
        FieldInitializerToCarbon(field, "=", out);
      }
      out << "}";
      break;
    }

    case Fuzzing::Expression::kStructTypeLiteral: {
      const auto& struct_type_literal = expression.struct_type_literal();
      out << "{";
      llvm::ListSeparator sep;
      for (const auto& field : struct_type_literal.fields()) {
        out << sep;
        FieldInitializerToCarbon(field, ":", out);
      }
      out << "}";
      break;
    }

    case Fuzzing::Expression::kIdentifier: {
      const auto& identifier = expression.identifier();
      IdentifierToCarbon(identifier.name(), out);
      break;
    }

    case Fuzzing::Expression::kDesignator: {
      const auto& designator = expression.designator();
      out << ".";
      IdentifierToCarbon(designator.name(), out);
      break;
    }

    case Fuzzing::Expression::kIfExpression: {
      const auto& if_expression = expression.if_expression();
      out << "if ";
      ExpressionToCarbon(if_expression.condition(), out);
      out << " then ";
      ExpressionToCarbon(if_expression.then_expression(), out);
      out << " else ";
      ExpressionToCarbon(if_expression.else_expression(), out);
      break;
    }

    case Fuzzing::Expression::kBoolTypeLiteral:
      out << "bool";
      break;

    case Fuzzing::Expression::kBoolLiteral: {
      const auto& bool_literal = expression.bool_literal();
      out << (bool_literal.value() ? "true" : "false");
      break;
    }

    case Fuzzing::Expression::kIntTypeLiteral:
      out << "i32";
      break;

    case Fuzzing::Expression::kIntLiteral: {
      out << expression.int_literal().value();
      break;
    }

    case Fuzzing::Expression::kStringLiteral:
      StringLiteralToCarbon(expression.string_literal().value(), out);
      break;

    case Fuzzing::Expression::kStringTypeLiteral:
      out << "String";
      break;

    case Fuzzing::Expression::kContinuationTypeLiteral:
      out << "__Continuation";
      break;

    case Fuzzing::Expression::kTypeTypeLiteral:
      out << "type";
      break;

    case Fuzzing::Expression::kUnimplementedExpression:
      // Not really supported.
      // This is an arbitrary default to avoid getting invalid syntax.
      out << "1 __unimplemented_example_infix 2";
      break;

    case Fuzzing::Expression::kArrayTypeLiteral: {
      const Fuzzing::ArrayTypeLiteral& array_literal =
          expression.array_type_literal();
      out << "[";
      ExpressionToCarbon(array_literal.element_type(), out);
      out << "; ";
      ExpressionToCarbon(array_literal.size(), out);
      out << "]";
      break;
    }

    case Fuzzing::Expression::kWhere: {
      const Fuzzing::WhereExpression& where = expression.where();
      ExpressionToCarbon(where.base(), out);
      out << " where ";
      llvm::ListSeparator sep(" and ");
      for (const auto& clause : where.clauses()) {
        out << sep;
        switch (clause.kind_case()) {
          case Fuzzing::WhereClause::kImpls:
            ExpressionToCarbon(clause.impls().type(), out);
            out << " impls ";
            ExpressionToCarbon(clause.impls().constraint(), out);
            break;
          case Fuzzing::WhereClause::kEquals:
            ExpressionToCarbon(clause.equals().lhs(), out);
            out << " == ";
            ExpressionToCarbon(clause.equals().rhs(), out);
            break;
          case Fuzzing::WhereClause::kRewrite:
            out << "." << clause.rewrite().member_name() << " = ";
            ExpressionToCarbon(clause.rewrite().replacement(), out);
            break;
          case Fuzzing::WhereClause::KIND_NOT_SET:
            // Arbitrary default to avoid invalid syntax.
            out << ".Self == .Self";
            break;
        }
      }
      break;
    }
  }
}

static auto BindingPatternToCarbon(const Fuzzing::BindingPattern& pattern,
                                   llvm::raw_ostream& out) -> void {
  IdentifierToCarbon(pattern.name(), out);
  out << ": ";
  PatternToCarbon(pattern.type(), out);
}

static auto GenericBindingToCarbon(
    const Fuzzing::GenericBinding& generic_binding, llvm::raw_ostream& out) {
  IdentifierToCarbon(generic_binding.name(), out);
  out << ":! ";
  ExpressionToCarbon(generic_binding.type(), out);
}

static auto TuplePatternToCarbon(const Fuzzing::TuplePattern& tuple_pattern,
                                 llvm::raw_ostream& out) -> void {
  out << "(";
  llvm::ListSeparator sep;
  for (const auto& field : tuple_pattern.fields()) {
    out << sep;
    PatternToCarbon(field, out);
  }
  if (tuple_pattern.fields_size() == 1) {
    // Adding a trailing comma so that generated source will be parsed as a
    // tuple pattern expression. See docs/design/tuples.md.
    out << ", ";
  }
  out << ")";
}

static auto PatternToCarbon(const Fuzzing::Pattern& pattern,
                            llvm::raw_ostream& out) -> void {
  switch (pattern.kind_case()) {
    case Fuzzing::Pattern::KIND_NOT_SET:
      // Arbitrary default to avoid getting invalid syntax.
      out << "auto";
      break;

    case Fuzzing::Pattern::kBindingPattern:
      BindingPatternToCarbon(pattern.binding_pattern(), out);
      break;

    case Fuzzing::Pattern::kTuplePattern:
      TuplePatternToCarbon(pattern.tuple_pattern(), out);
      break;

    case Fuzzing::Pattern::kAlternativePattern: {
      const auto& alternative_pattern = pattern.alternative_pattern();
      ExpressionToCarbon(alternative_pattern.choice_type(), out);
      out << ".";
      IdentifierToCarbon(alternative_pattern.alternative_name(), out);
      TuplePatternToCarbon(alternative_pattern.arguments(), out);
      break;
    }

    // Arbitrary expression.
    case Fuzzing::Pattern::kExpressionPattern: {
      const auto& expression_pattern = pattern.expression_pattern();
      ExpressionToCarbon(expression_pattern.expression(), out);
      break;
    }

    case Fuzzing::Pattern::kAutoPattern:
      out << "auto";
      break;

    case Fuzzing::Pattern::kVarPattern:
      out << "var ";
      PatternToCarbon(pattern.var_pattern().pattern(), out);
      break;

    case Fuzzing::Pattern::kGenericBinding:
      GenericBindingToCarbon(pattern.generic_binding(), out);
      break;

    case Fuzzing::Pattern::kAddrPattern:
      out << "addr ";
      BindingPatternToCarbon(pattern.addr_pattern().binding_pattern(), out);
      break;
  }
}

static auto BlockStatementToCarbon(const Fuzzing::BlockStatement& block,
                                   llvm::raw_ostream& out) -> void {
  out << "{\n";
  for (const auto& statement : block.statements()) {
    StatementToCarbon(statement, out);
    out << "\n";
  }
  out << "}\n";
}

static auto StatementToCarbon(const Fuzzing::Statement& statement,
                              llvm::raw_ostream& out) -> void {
  switch (statement.kind_case()) {
    case Fuzzing::Statement::KIND_NOT_SET:
      // Arbitrary default to avoid getting invalid syntax.
      out << "true;\n";
      break;

    case Fuzzing::Statement::kExpressionStatement: {
      const auto& expression_statement = statement.expression_statement();
      ExpressionToCarbon(expression_statement.expression(), out);
      out << ";";
      break;
    }

    case Fuzzing::Statement::kAssign: {
      const auto& assign_statement = statement.assign();
      ExpressionToCarbon(assign_statement.lhs(), out);
      switch (assign_statement.op()) {
        case Fuzzing::AssignStatement::Plain:
          out << " = ";
          break;
        case Fuzzing::AssignStatement::Add:
          out << " += ";
          break;
        case Fuzzing::AssignStatement::And:
          out << " &= ";
          break;
        case Fuzzing::AssignStatement::Div:
          out << " /= ";
          break;
        case Fuzzing::AssignStatement::Mod:
          out << " %= ";
          break;
        case Fuzzing::AssignStatement::Mul:
          out << " *= ";
          break;
        case Fuzzing::AssignStatement::Or:
          out << " |= ";
          break;
        case Fuzzing::AssignStatement::ShiftLeft:
          out << " <<= ";
          break;
        case Fuzzing::AssignStatement::ShiftRight:
          out << " >>= ";
          break;
        case Fuzzing::AssignStatement::Sub:
          out << " -= ";
          break;
        case Fuzzing::AssignStatement::Xor:
          out << " ^= ";
          break;
      }
      ExpressionToCarbon(assign_statement.rhs(), out);
      out << ";";
      break;
    }

    case Fuzzing::Statement::kIncDec: {
      const auto& inc_dec_statement = statement.inc_dec();
      out << (inc_dec_statement.is_increment() ? "++" : "--");
      ExpressionToCarbon(inc_dec_statement.operand(), out);
      out << ";";
      break;
    }

    case Fuzzing::Statement::kVariableDefinition: {
      const auto& def = statement.variable_definition();
      if (def.is_returned()) {
        out << "returned ";
      }
      out << "var ";
      PatternToCarbon(def.pattern(), out);
      if (def.has_init()) {
        out << " = ";
        ExpressionToCarbon(def.init(), out);
      }
      out << ";";
      break;
    }

    case Fuzzing::Statement::kIfStatement: {
      const auto& if_statement = statement.if_statement();
      out << "if (";
      ExpressionToCarbon(if_statement.condition(), out);
      out << ") ";
      BlockStatementToCarbon(if_statement.then_block(), out);
      // `else` is optional.
      if (if_statement.has_else_block()) {
        out << " else ";
        BlockStatementToCarbon(if_statement.else_block(), out);
      }
      break;
    }

    case Fuzzing::Statement::kReturnVarStatement: {
      out << "return var;";
      break;
    }

    case Fuzzing::Statement::kReturnExpressionStatement: {
      const auto& ret = statement.return_expression_statement();
      out << "return";
      if (!ret.is_omitted_expression()) {
        out << " ";
        ExpressionToCarbon(ret.expression(), out);
      }
      out << ";";
      break;
    }

    case Fuzzing::Statement::kBlock:
      BlockStatementToCarbon(statement.block(), out);
      break;

    case Fuzzing::Statement::kWhileStatement: {
      const auto& while_statement = statement.while_statement();
      out << "while (";
      ExpressionToCarbon(while_statement.condition(), out);
      out << ") ";
      BlockStatementToCarbon(while_statement.body(), out);
      break;
    }
    case Fuzzing::Statement::kForStatement: {
      const auto& for_statement = statement.for_statement();
      out << "for (";
      BindingPatternToCarbon(for_statement.var_decl(), out);
      out << " in ";
      ExpressionToCarbon(for_statement.target(), out);
      out << ") ";
      BlockStatementToCarbon(for_statement.body(), out);
      break;
    }

    case Fuzzing::Statement::kMatch: {
      const auto& match = statement.match();
      out << "match (";
      ExpressionToCarbon(match.expression(), out);
      out << ") {";
      for (const auto& clause : match.clauses()) {
        if (clause.is_default()) {
          out << "default";
        } else {
          out << "case ";
          PatternToCarbon(clause.pattern(), out);
        }
        out << " => ";
        StatementToCarbon(clause.statement(), out);
      }
      out << "}";
      break;
    }

    case Fuzzing::Statement::kContinuation: {
      const auto& continuation = statement.continuation();
      out << "__continuation ";
      IdentifierToCarbon(continuation.name(), out);
      BlockStatementToCarbon(continuation.body(), out);
      break;
    }

    case Fuzzing::Statement::kRun: {
      const auto& run = statement.run();
      out << "__run ";
      ExpressionToCarbon(run.argument(), out);
      out << ";";
      break;
    }

    case Fuzzing::Statement::kAwaitStatement:
      out << "__await;";
      break;

    case Fuzzing::Statement::kBreakStatement:
      out << "break;";
      break;

    case Fuzzing::Statement::kContinueStatement:
      out << "continue;";
      break;
  }
}

static auto ReturnTermToCarbon(const Fuzzing::ReturnTerm& return_term,
                               llvm::raw_ostream& out) -> void {
  switch (return_term.kind()) {
    case Fuzzing::ReturnTerm::UnknownReturnKind:
    case Fuzzing::ReturnTerm::Omitted:
      break;
    case Fuzzing::ReturnTerm::Auto:
      out << " -> auto";
      break;
    case Fuzzing::ReturnTerm::Expression:
      out << " -> ";
      ExpressionToCarbon(return_term.type(), out);
      break;
  }
}

static auto DeclaredNameToCarbon(const Fuzzing::DeclaredName& name,
                                 llvm::raw_ostream& out) -> void {
  for (const std::string& qual : name.qualifiers()) {
    IdentifierToCarbon(qual, out);
    out << ".";
  }
  IdentifierToCarbon(name.name(), out);
}

static auto DeclarationToCarbon(const Fuzzing::Declaration& declaration,
                                llvm::raw_ostream& out) -> void {
  switch (declaration.kind_case()) {
    case Fuzzing::Declaration::KIND_NOT_SET: {
      // Arbitrary default to avoid getting invalid syntax.
      out << "var x: i32;";
      break;
    }

    case Fuzzing::Declaration::kNamespace: {
      out << "namespace ";
      DeclaredNameToCarbon(declaration.namespace_().name(), out);
      out << ";";
      break;
    }

    case Fuzzing::Declaration::kDestructor: {
      const auto& function = declaration.destructor();
      out << "destructor";
      llvm::ListSeparator sep;
      out << "[";
      if (function.has_self_pattern()) {
        // This is a class method.
        out << sep;
        PatternToCarbon(function.self_pattern(), out);
      }
      out << "]";

      // Body is optional.
      if (function.has_body()) {
        out << "\n";
        BlockStatementToCarbon(function.body(), out);
      } else {
        out << ";";
      }
      break;
    }
    case Fuzzing::Declaration::kFunction: {
      const auto& function = declaration.function();
      out << "fn ";
      DeclaredNameToCarbon(function.name(), out);

      if (!function.deduced_parameters().empty() ||
          function.has_self_pattern()) {
        out << "[";
        llvm::ListSeparator sep;
        for (const Fuzzing::GenericBinding& p : function.deduced_parameters()) {
          out << sep;
          GenericBindingToCarbon(p, out);
        }
        if (function.has_self_pattern()) {
          // This is a class method.
          out << sep;
          PatternToCarbon(function.self_pattern(), out);
        }
        out << "]";
      }
      TuplePatternToCarbon(function.param_pattern(), out);
      ReturnTermToCarbon(function.return_term(), out);

      // Body is optional.
      if (function.has_body()) {
        out << "\n";
        BlockStatementToCarbon(function.body(), out);
      } else {
        out << ";";
      }
      break;
    }

    case Fuzzing::Declaration::kClassDeclaration: {
      const auto& class_declaration = declaration.class_declaration();
      out << "class ";
      DeclaredNameToCarbon(class_declaration.name(), out);

      // type_params is optional.
      if (class_declaration.has_type_params()) {
        TuplePatternToCarbon(class_declaration.type_params(), out);
      }

      out << "{\n";
      for (const auto& member : class_declaration.members()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      break;
    }

    // EXPERIMENTAL MIXIN FEATURE
    case Fuzzing::Declaration::kMixin: {
      const auto& mixin_declaration = declaration.mixin();
      out << "__mixin ";
      DeclaredNameToCarbon(mixin_declaration.name(), out);

      // type params are not implemented yet
      // if (mixin_declaration.has_params()) {
      //  TuplePatternToCarbon(mixin_declaration.params(), out);
      //}

      out << "{\n";
      for (const auto& member : mixin_declaration.members()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      // TODO: need to handle interface.self()?
      break;
    }

    // EXPERIMENTAL MIXIN FEATURE
    case Fuzzing::Declaration::kMix: {
      const auto& mix_declaration = declaration.mix();
      out << "__mix ";
      ExpressionToCarbon(mix_declaration.mixin(), out);
      out << ";";
      break;
    }

    case Fuzzing::Declaration::kChoice: {
      const auto& choice = declaration.choice();
      out << "choice ";
      DeclaredNameToCarbon(choice.name(), out);

      out << "{";
      llvm::ListSeparator sep;
      for (const auto& alternative : choice.alternatives()) {
        out << sep;
        IdentifierToCarbon(alternative.name(), out);
        if (alternative.has_signature()) {
          TupleLiteralExpressionToCarbon(alternative.signature(), out);
        }
      }
      out << "}";
      break;
    }

    case Fuzzing::Declaration::kVariable: {
      const auto& var = declaration.variable();
      out << "var ";
      BindingPatternToCarbon(var.binding(), out);

      // Initializer is optional.
      if (var.has_initializer()) {
        out << " = ";
        ExpressionToCarbon(var.initializer(), out);
      }
      out << ";";
      break;
    }

    case Fuzzing::Declaration::kLet: {
      const auto& let = declaration.let();
      out << "let ";
      PatternToCarbon(let.pattern(), out);

      // TODO: Print out the initializer once it's supported.
      // if (let.has_initializer()) {
      //   out << " = ";
      //   ExpressionToCarbon(let.initializer(), out);
      // }
      out << ";";
      break;
    }

    case Fuzzing::Declaration::kInterfaceExtends: {
      const auto& extends = declaration.interface_extends();
      out << "extends ";
      ExpressionToCarbon(extends.base(), out);
      out << ";";
      break;
    }

    case Fuzzing::Declaration::kInterfaceImpl: {
      const auto& impl = declaration.interface_impl();
      out << "impl ";
      ExpressionToCarbon(impl.impl_type(), out);
      out << " as ";
      ExpressionToCarbon(impl.constraint(), out);
      out << ";";
      break;
    }

    case Fuzzing::Declaration::kInterface: {
      const auto& interface = declaration.interface();
      out << "interface ";
      DeclaredNameToCarbon(interface.name(), out);
      out << " {\n";
      for (const auto& member : interface.members()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      break;
    }

    case Fuzzing::Declaration::kConstraint: {
      const auto& constraint = declaration.constraint();
      out << "constraint ";
      DeclaredNameToCarbon(constraint.name(), out);
      out << " {\n";
      for (const auto& member : constraint.members()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      break;
    }

    case Fuzzing::Declaration::kImpl: {
      const auto& impl = declaration.impl();
      if (impl.kind() == Fuzzing::ImplDeclaration::ExternalImpl) {
        out << "external ";
      }
      out << "impl ";
      ExpressionToCarbon(impl.impl_type(), out);
      out << " as ";
      ExpressionToCarbon(impl.interface(), out);
      out << " {\n";
      for (const auto& member : impl.members()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      break;
    }

    case Fuzzing::Declaration::kMatchFirst: {
      const auto& match_first = declaration.match_first();
      out << "__match_first {\n";
      for (const auto& impl : match_first.impl_declarations()) {
        DeclarationToCarbon(impl, out);
        out << "\n";
      }
      out << "}";
      break;
    }

    case Fuzzing::Declaration::kAlias: {
      const auto& alias = declaration.alias();
      out << "alias ";
      DeclaredNameToCarbon(alias.name(), out);
      out << " = ";
      ExpressionToCarbon(alias.target(), out);
      out << ";";
      break;
    }
  }
}

static auto ProtoToCarbon(const Fuzzing::CompilationUnit& compilation_unit,
                          llvm::raw_ostream& out) -> void {
  out << "// Generated by proto_to_carbon.\n\n";
  out << "package ";
  LibraryNameToCarbon(compilation_unit.package_statement(), out);
  out << (compilation_unit.is_api() ? " api" : " impl") << ";\n";

  if (!compilation_unit.declarations().empty()) {
    out << "\n";
    for (const auto& declaration : compilation_unit.declarations()) {
      DeclarationToCarbon(declaration, out);
      out << "\n";
    }
  }
}

auto ProtoToCarbon(const Fuzzing::CompilationUnit& compilation_unit)
    -> std::string {
  std::string source;
  llvm::raw_string_ostream out(source);
  ProtoToCarbon(compilation_unit, out);
  return source;
}

}  // namespace Carbon
