// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/fuzzing/proto_to_carbon.h"

#include <string_view>

#include "common/fuzzing/carbon.pb.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

// TODO: use llvm::ListSeparator sep;

namespace Carbon {

static constexpr std::string_view AnonymousName = "_";

static auto ExpressionToCarbon(const Fuzzing::Expression& expression,
                               llvm::raw_ostream& out) -> void;
static auto PatternToCarbon(const Fuzzing::Pattern& pattern,
                            llvm::raw_ostream& out) -> void;
static auto StatementToCarbon(const Fuzzing::Statement& statement,
                              llvm::raw_ostream& out) -> void;
static auto DeclarationToCarbon(const Fuzzing::Declaration& declaration,
                                llvm::raw_ostream& out) -> void;

static auto FirstIdentifierCharToCarbon(char c) -> char {
  return llvm::isAlpha(c) || c == '_' ? c : 'F';
}

static auto NextIdentifierCharToCarbon(char c) -> char {
  return llvm::isAlnum(c) || c == '_' ? c : 'N';
}

// [A-Za-z_][A-Za-z0-9_]*
static auto IdentifierToCarbon(std::string_view s, llvm::raw_ostream& out)
    -> void {
  if (s.empty()) {
    out << "EMPTY";
  } else {
    for (size_t i = 0; i < s.size(); ++i) {
      if (i == 0) {
        out << FirstIdentifierCharToCarbon(s[i]);
      } else {
        out << NextIdentifierCharToCarbon(s[i]);
      }
    }
  }
}

static auto LibraryNameToCarbon(const Fuzzing::LibraryName& library,
                                llvm::raw_ostream& out) -> void {
  // TODO if (library.has_package_name())
  IdentifierToCarbon(library.package_name(), out);
  if (library.has_path()) {
    out << " library \"";
    IdentifierToCarbon(library.path(), out);
    out << "\"";
  }
}

static auto PrimitiveOperatorToString(
    const Fuzzing::PrimitiveOperatorExpression::Operator op)
    -> std::string_view {
  switch (op) {
    case Fuzzing::PrimitiveOperatorExpression::UnknownOperator:
      return "-";  // Arbitrary default to avoid getting invalid syntax.
    case Fuzzing::PrimitiveOperatorExpression::AddressOf:
      return "&";
    case Fuzzing::PrimitiveOperatorExpression::Deref:
    case Fuzzing::PrimitiveOperatorExpression::Mul:
    case Fuzzing::PrimitiveOperatorExpression::Ptr:
      return "*";
    case Fuzzing::PrimitiveOperatorExpression::Neg:
    case Fuzzing::PrimitiveOperatorExpression::Sub:
      return "-";
    case Fuzzing::PrimitiveOperatorExpression::Not:
      return "not ";  // Needs a space to 'unglue' from the operand.
    case Fuzzing::PrimitiveOperatorExpression::Add:
      return "+";
    case Fuzzing::PrimitiveOperatorExpression::And:
      return "and";
    case Fuzzing::PrimitiveOperatorExpression::Eq:
      return "==";
    case Fuzzing::PrimitiveOperatorExpression::Or:
      return "or";
  }
}

// .x = 1
static auto FieldInitializerToCarbon(const Fuzzing::FieldInitializer& field,
                                     std::string_view separator,
                                     llvm::raw_ostream& out) -> void {
  out << ".";
  // TODO if (field.has_name())
  IdentifierToCarbon(field.name(), out);
  out << " " << separator << " ";
  if (field.has_expression()) {
    ExpressionToCarbon(field.expression(), out);
  }
}

// ("a", 1)
static auto TupleLiteralExpressionToCarbon(
    const Fuzzing::TupleLiteralExpression& tuple_literal,
    llvm::raw_ostream& out) -> void {
  out << "(";
  for (int i = 0; i < tuple_literal.field_size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    ExpressionToCarbon(tuple_literal.field(i), out);
    if (tuple_literal.field_size() == 1) {
      out << ", ";  // Ensures interpretation as a tuple expression.
    }
  }
  out << ")";
}

static auto ExpressionToCarbon(const Fuzzing::Expression& expression,
                               llvm::raw_ostream& out) -> void {
  switch (expression.kind_case()) {
    case Fuzzing::Expression::KIND_NOT_SET:
      out << "true";  // Default for missing expressions to avoid invalid
                      // syntax.
      break;

    // func(1, 2)
    case Fuzzing::Expression::kCall: {
      const auto& call = expression.call();
      if (call.has_function()) {
        ExpressionToCarbon(call.function(), out);
      }
      if (call.argument().kind_case() == Fuzzing::Expression::kTupleLiteral) {
        TupleLiteralExpressionToCarbon(call.argument().tuple_literal(), out);
      } else {
        out << "(";
        ExpressionToCarbon(call.argument(), out);
        out << ")";
      }
      break;
    }

    // __Fn(i32, bool) -> String
    case Fuzzing::Expression::kFunctionType: {
      const auto& fun_type = expression.function_type();
      out << "__Fn";
      if (fun_type.has_parameter()) {
        ExpressionToCarbon(fun_type.parameter(), out);
      }
      if (fun_type.has_return_type()) {
        out << " -> ";
        ExpressionToCarbon(fun_type.return_type(), out);
      }
      break;
    }

    // s.f
    case Fuzzing::Expression::kFieldAccess: {
      const auto& field_access = expression.field_access();
      if (field_access.has_aggregate()) {
        ExpressionToCarbon(field_access.aggregate(), out);
      }
      out << ".";
      // TODO if (field_access.has_field())
      IdentifierToCarbon(field_access.field(), out);
      break;
    }

    // a[0]
    case Fuzzing::Expression::kIndex: {
      const auto& index = expression.index();
      if (index.has_aggregate()) {
        ExpressionToCarbon(index.aggregate(), out);
      }
      out << "[";
      if (index.has_offset()) {
        ExpressionToCarbon(index.offset(), out);
      }
      out << "]";
      break;
    }

    // -a, a + b
    case Fuzzing::Expression::kPrimitiveOperator: {
      const auto& primitive_operator = expression.primitive_operator();
      const std::string_view op =
          PrimitiveOperatorToString(primitive_operator.op());
      out << "(";
      switch (primitive_operator.argument().size()) {
        case 0:
          out << op;
          break;

        case 1: {
          const bool postix = primitive_operator.op() ==
                              Fuzzing::PrimitiveOperatorExpression::Ptr;
          if (!postix) {
            out << op;
          }
          ExpressionToCarbon(primitive_operator.argument(0), out);
          if (postix) {
            out << op;
          }
          break;
        }

        default:
          ExpressionToCarbon(primitive_operator.argument(0), out);
          out << " " << op << " ";
          ExpressionToCarbon(primitive_operator.argument(1), out);
          break;
      }
      out << ")";
      break;
    }

    // ("a", 1)
    case Fuzzing::Expression::kTupleLiteral: {
      TupleLiteralExpressionToCarbon(expression.tuple_literal(), out);
      break;
    }

    // {.x = 5, .y = 2}
    case Fuzzing::Expression::kStructLiteral: {
      const auto& struct_literal = expression.struct_literal();
      out << "{";
      for (int i = 0; i < struct_literal.field_size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        FieldInitializerToCarbon(struct_literal.field(i), "=", out);
      }
      out << "}";
      break;
    }

    // {.x: i32, .y: i32}
    case Fuzzing::Expression::kStructTypeLiteral: {
      const auto& struct_type_literal = expression.struct_type_literal();
      out << "{";
      for (int i = 0; i < struct_type_literal.field_size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        FieldInitializerToCarbon(struct_type_literal.field(i), ":", out);
      }
      out << "}";
      break;
    }

    // x
    case Fuzzing::Expression::kIdentifier: {
      const auto& identifier = expression.identifier();
      // TODO if (identifier.has_name())
      IdentifierToCarbon(identifier.name(), out);
      break;
    }

    // Print('a')
    case Fuzzing::Expression::kIntrinsic: {
      const auto& intrinsic = expression.intrinsic();
      if (intrinsic.has_intrinsic()) {
        switch (intrinsic.intrinsic()) {
          case Fuzzing::IntrinsicExpression::UnknownIntrinsic:
          case Fuzzing::IntrinsicExpression::Print:
            out << "__intrinsic_print";
            break;
        }
      }
      if (intrinsic.has_argument()) {
        TupleLiteralExpressionToCarbon(intrinsic.argument(), out);
      }
    } break;

    // if cond then true else false
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

    // Bool
    case Fuzzing::Expression::kBoolTypeLiteral:
      out << "Bool";
      break;

    // false
    case Fuzzing::Expression::kBoolLiteral: {
      const auto& bool_literal = expression.bool_literal();
      if (bool_literal.has_value()) {
        out << (bool_literal.value() ? "true" : "false");
      }
      break;
    }

    // i32
    case Fuzzing::Expression::kIntTypeLiteral:
      out << "i32";
      break;

    // 42
    case Fuzzing::Expression::kIntLiteral: {
      const auto& int_literal = expression.int_literal();
      // TODO if (int_literal.has_value())
      out << int_literal.value();
      break;
    }

    // "a\n\"b"
    case Fuzzing::Expression::kStringLiteral:
      out << '"';
      out.write_escaped(expression.string_literal().value());
      out << '"';
      break;

    // String
    case Fuzzing::Expression::kStringTypeLiteral:
      out << "String";
      break;

    // __Continuation
    case Fuzzing::Expression::kContinuationTypeLiteral:
      out << "__Continuation";
      break;

    // Type
    case Fuzzing::Expression::kTypeTypeLiteral:
      out << "Type";
      break;

    case Fuzzing::Expression::kUnimplementedExpression:
      // TODO
      break;
  }
}

// a: i32
static auto BindingPatternToCarbon(const Fuzzing::BindingPattern& pattern,
                                   llvm::raw_ostream& out) -> void {
  // TODO if (pattern.has_name()) {
  IdentifierToCarbon(pattern.name(), out);

  out << ": ";
  if (pattern.has_type()) {
    PatternToCarbon(pattern.type(), out);
  }
}

// (a: i32, b: auto)
static auto TuplePatternToCarbon(const Fuzzing::TuplePattern& tuple_pattern,
                                 llvm::raw_ostream& out) -> void {
  out << "(";
  for (int i = 0; i < tuple_pattern.field_size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    PatternToCarbon(tuple_pattern.field(i), out);
    // TODO xx
    //  if (tuple_pattern.field_size() == 1) {
    //    out << ", ";  // Ensures interpretation as a tuple expression.
    //  }
  }
  out << ")";
}

static auto PatternToCarbon(const Fuzzing::Pattern& pattern,
                            llvm::raw_ostream& out) -> void {
  switch (pattern.kind_case()) {
    case Fuzzing::Pattern::KIND_NOT_SET:
      // TODO
      break;

    // a: i32
    case Fuzzing::Pattern::kBindingPattern:
      BindingPatternToCarbon(pattern.binding_pattern(), out);
      break;

    // (a: i32, b: auto)
    case Fuzzing::Pattern::kTuplePattern:
      TuplePatternToCarbon(pattern.tuple_pattern(), out);
      break;

    // Ints.Two(a: auto, b: auto)
    case Fuzzing::Pattern::kAlternativePattern: {
      const auto& alternative_pattern = pattern.alternative_pattern();
      // TODO if (alternative_pattern.has_choice_type()) {
      ExpressionToCarbon(alternative_pattern.choice_type(), out);
      out << ".";
      // TODO if (alternative_pattern.has_alternative_name()) {
      IdentifierToCarbon(alternative_pattern.alternative_name(), out);

      TuplePatternToCarbon(alternative_pattern.arguments(), out);
      break;
    }

    // "A pattern that matches a value if it is equal to the value of a given
    // expression."
    case Fuzzing::Pattern::kExpressionPattern: {
      const auto& expression_pattern = pattern.expression_pattern();
      if (expression_pattern.has_expression()) {
        ExpressionToCarbon(expression_pattern.expression(), out);
      }
      break;
    }

    // auto
    case Fuzzing::Pattern::kAutoPattern:
      out << "auto";
      break;

    // var
    case Fuzzing::Pattern::kVarPattern:
      out << "var ";
      PatternToCarbon(pattern.var_pattern().pattern(), out);
      break;
  }
}

// { var x: i32 = 1; var y: i32 = 2; }
static auto BlockStatementToCarbon(const Fuzzing::BlockStatement& block,
                                   llvm::raw_ostream& out) -> void {
  out << "{\n";
  for (const auto& statement : block.statement()) {
    StatementToCarbon(statement, out);
    out << "\n";
  }
  out << "}\n";
}

static auto StatementToCarbon(const Fuzzing::Statement& statement,
                              llvm::raw_ostream& out) -> void {
  switch (statement.kind_case()) {
    case Fuzzing::Statement::KIND_NOT_SET:
      break;

    // f(1);
    case Fuzzing::Statement::kExpressionStatement: {
      const auto& expression_statement = statement.expression_statement();
      if (expression_statement.has_expression()) {
        ExpressionToCarbon(expression_statement.expression(), out);
      }
      out << ";";
      break;
    }

    // a = 1;
    case Fuzzing::Statement::kAssign: {
      const auto& assign_statement = statement.assign();
      if (assign_statement.has_lhs()) {
        ExpressionToCarbon(assign_statement.lhs(), out);
      }
      out << " = ";
      if (assign_statement.has_rhs()) {
        ExpressionToCarbon(assign_statement.rhs(), out);
      }
      out << ";";
      break;
    }

    // var a: i32 = 1;
    case Fuzzing::Statement::kVariableDefinition: {
      const auto& def = statement.variable_definition();
      out << "var ";
      if (def.has_pattern()) {
        PatternToCarbon(def.pattern(), out);
      }
      out << " = ";
      if (def.has_init()) {
        ExpressionToCarbon(def.init(), out);
      }
      out << ";";
      break;
    }

    // if (a == 1) { b = 2; } else { c = 3; }
    case Fuzzing::Statement::kIfStatement: {
      const auto& if_statement = statement.if_statement();
      out << "if (";
      if (if_statement.has_condition()) {
        ExpressionToCarbon(if_statement.condition(), out);
      }
      out << ") ";
      if (if_statement.has_then_block()) {
        BlockStatementToCarbon(if_statement.then_block(), out);
      } else {
        out << ";";
      }
      if (if_statement.has_else_block()) {
        out << " else ";
        BlockStatementToCarbon(if_statement.else_block(), out);
      }
      break;
    }

    // return 1;
    case Fuzzing::Statement::kReturnStatement: {
      const auto& ret = statement.return_statement();
      out << "return";
      if (ret.has_expression()) {
        out << " ";
        ExpressionToCarbon(ret.expression(), out);
      }
      out << ";";
      break;
    }

    case Fuzzing::Statement::kBlock:
      BlockStatementToCarbon(statement.block(), out);
      break;

    // while (x > 0) { x = x - 1; }
    case Fuzzing::Statement::kWhileStatement: {
      const auto& while_statement = statement.while_statement();
      out << "while (";
      if (while_statement.has_condition()) {
        ExpressionToCarbon(while_statement.condition(), out);
      }
      out << ") ";
      if (while_statement.has_body()) {
        BlockStatementToCarbon(while_statement.body(), out);
      } else {
        out << ";";
      }
      break;
    }

    // match (t) {
    //   case (a: auto, b: auto) =>
    //     return 0;
    //   default =>
    //     return 1;
    // }
    case Fuzzing::Statement::kMatch: {
      const auto& match = statement.match();
      out << "match (";
      if (match.has_expression()) {
        ExpressionToCarbon(match.expression(), out);
      }
      out << ") {";
      for (const auto& clause : match.clause()) {
        const bool is_default_clause =
            clause.pattern().kind_case() == Fuzzing::Pattern::kBindingPattern &&
            clause.pattern().binding_pattern().name() == AnonymousName;
        if (is_default_clause) {
          out << "default";
        } else {
          out << "case ";
          if (clause.has_pattern()) {
            PatternToCarbon(clause.pattern(), out);
          }
        }
        out << " => ";
        if (clause.has_statement()) {
          StatementToCarbon(clause.statement(), out);
        }
      }
      out << "}";
      break;
    }

    // __continuation k { x = x + 1; }
    case Fuzzing::Statement::kContinuation: {
      const auto& continuation = statement.continuation();
      out << "__continuation ";
      // TODO if (continuation.has_name())
      IdentifierToCarbon(continuation.name(), out);

      if (continuation.has_body()) {
        BlockStatementToCarbon(continuation.body(), out);
      } else {
        out << "{}\n";
      }
      break;
    }

    // __run k;
    case Fuzzing::Statement::kRun: {
      const auto& run = statement.run();
      out << "__run ";
      if (run.has_argument()) {
        ExpressionToCarbon(run.argument(), out);
      }
      out << ";";
      break;
    }

    // __await;
    case Fuzzing::Statement::kAwait:
      out << "__await;";
      break;

    // break;
    case Fuzzing::Statement::kBreakStatement:
      out << "break;";
      break;

    // continue;
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

static auto DeclarationToCarbon(const Fuzzing::Declaration& declaration,
                                llvm::raw_ostream& out) -> void {
  switch (declaration.kind_case()) {
    case Fuzzing::Declaration::KIND_NOT_SET:
      break;

    // fn f(x: i32) -> auto {  return x; }
    // fn g[T:! Type](x: i32, y: T) -> T {  return y; }
    case Fuzzing::Declaration::kFunction: {
      // FN identifier deduced_params receiver maybe_empty_tuple_pattern
      // return_term block
      const auto& function = declaration.function();
      out << "fn ";
      // TODO if (function.has_name()) {
      IdentifierToCarbon(function.name(), out);

      if (!function.deduced_parameter().empty()) {
        out << "[";
        for (int i = 0; i < function.deduced_parameter().size(); ++i) {
          const Fuzzing::GenericBinding& p = function.deduced_parameter(i);
          if (i > 0) {
            out << ", ";
          }
          IdentifierToCarbon(p.name(), out);
          out << ":! ";
          ExpressionToCarbon(p.type(), out);
        }
        out << "]";
      }
      if (function.has_me_pattern()) {  // TODO
        // This is a class method.
        out << "[";
        BindingPatternToCarbon(function.me_pattern(), out);
        out << "]";
      }
      if (function.has_param_pattern()) {
        TuplePatternToCarbon(function.param_pattern(), out);
      }
      ReturnTermToCarbon(function.return_term(), out);
      if (function.has_body()) {
        out << "\n";
        BlockStatementToCarbon(function.body(), out);
      } else {
        out << ";";
      }
      break;
    }

    // class Point { var x: i32; }
    case Fuzzing::Declaration::kClassDeclaration: {
      const auto& class_declaration = declaration.class_declaration();
      out << "class ";
      // TODO if (class_declaration.has_name()) {
      IdentifierToCarbon(class_declaration.name(), out);

      out << "{\n";
      for (const auto& member : class_declaration.member()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}\n";
      break;
    }

    // choice Ints { None, One(i32) }
    case Fuzzing::Declaration::kChoice: {
      const auto& choice = declaration.choice();
      out << "choice ";
      // TODO if (choice.has_name()) {
      IdentifierToCarbon(choice.name(), out);

      out << "{";
      for (int i = 0; i < choice.alternative().size(); ++i) {
        const auto& alternative = choice.alternative(i);
        if (i > 0) {
          out << ",\n";
        }
        // TODO if (alternative.has_name()) {
        IdentifierToCarbon(alternative.name(), out);

        if (alternative.has_signature()) {
          ExpressionToCarbon(alternative.signature(), out);
        }
      }
      out << "}";
      break;
    }

    // var x: i32;
    // var y: i32 = 1;
    case Fuzzing::Declaration::kVariable: {
      const auto& var = declaration.variable();
      out << "var ";
      if (var.has_binding()) {
        BindingPatternToCarbon(var.binding(), out);
      }
      if (var.has_initializer()) {
        out << " = ";
        ExpressionToCarbon(var.initializer(), out);
      }
      out << ";";
      break;
    }

    // interface Vector {
    //   fn Add[me: Self](b: Self) -> Self;
    // }
    case Fuzzing::Declaration::kInterface: {
      const auto& interface = declaration.interface();
      out << "interface ";
      IdentifierToCarbon(interface.name(), out);
      out << " {\n";
      for (const auto& member : interface.member()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      // TODO: need to handle interface.self()?
      break;
    }

    // impl Point as Vector {
    //   fn Add[me: Point](b: Point) -> Point { ... }
    // }
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
      for (const auto& member : impl.member()) {
        DeclarationToCarbon(member, out);
        out << "\n";
      }
      out << "}";
      break;
    }
  }
}

static auto ProtoToCarbon(const Fuzzing::CompilationUnit& compilation_unit,
                          llvm::raw_ostream& out) -> void {
  out << "// Generated by proto_to_carbon.\n\n";
  // TODO if (compilation_unit.has_package_statement())
  out << "package ";
  LibraryNameToCarbon(compilation_unit.package_statement(), out);
  out << (compilation_unit.is_api() ? " api" : " impl") << ";\n";

  if (!compilation_unit.declaration().empty()) {
    out << "\n";
    for (const auto& declaration : compilation_unit.declaration()) {
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
