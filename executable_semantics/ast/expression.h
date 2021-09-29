// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/paren_contents.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/arena.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Expression {
 public:
  enum class Kind {
    BoolTypeLiteral,
    BoolLiteral,
    CallExpression,
    FunctionTypeLiteral,
    FieldAccessExpression,
    IndexExpression,
    IntTypeLiteral,
    ContinuationTypeLiteral,  // The type of a continuation value.
    IntLiteral,
    PrimitiveOperatorExpression,
    StringLiteral,
    StringTypeLiteral,
    TupleLiteral,
    TypeTypeLiteral,
    IdentifierExpression,
    IntrinsicExpression,
  };

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs an Expression representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Expression(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  const Kind kind_;
  SourceLocation source_loc_;
};

// Converts paren_contents to an Expression, interpreting the parentheses as
// grouping if their contents permit that interpretation, or as forming a
// tuple otherwise.
auto ExpressionFromParenContents(
    Nonnull<Arena*> arena, SourceLocation source_loc,
    const ParenContents<Expression>& paren_contents) -> Nonnull<Expression*>;

// Converts paren_contents to an Expression, interpreting the parentheses as
// forming a tuple.
auto TupleExpressionFromParenContents(
    Nonnull<Arena*> arena, SourceLocation source_loc,
    const ParenContents<Expression>& paren_contents) -> Nonnull<Expression*>;

// A FieldInitializer represents the initialization of a single tuple field.
struct FieldInitializer {
  FieldInitializer(std::string name, Nonnull<Expression*> expression)
      : name(std::move(name)), expression(expression) {}

  // The field name. Cannot be empty.
  std::string name;

  // The expression that initializes the field.
  Nonnull<Expression*> expression;
};

enum class Operator {
  Add,
  And,
  Deref,
  Eq,
  Mul,
  Neg,
  Not,
  Or,
  Sub,
  Ptr,
};

class IdentifierExpression : public Expression {
 public:
  explicit IdentifierExpression(SourceLocation source_loc, std::string name)
      : Expression(Kind::IdentifierExpression, source_loc),
        name(std::move(name)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::IdentifierExpression;
  }

  auto Name() const -> const std::string& { return name; }

 private:
  std::string name;
};

class FieldAccessExpression : public Expression {
 public:
  explicit FieldAccessExpression(SourceLocation source_loc,
                                 Nonnull<Expression*> aggregate,
                                 std::string field)
      : Expression(Kind::FieldAccessExpression, source_loc),
        aggregate(aggregate),
        field(std::move(field)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::FieldAccessExpression;
  }

  auto Aggregate() const -> Nonnull<const Expression*> { return aggregate; }
  auto Aggregate() -> Nonnull<Expression*> { return aggregate; }
  auto Field() const -> const std::string& { return field; }

 private:
  Nonnull<Expression*> aggregate;
  std::string field;
};

class IndexExpression : public Expression {
 public:
  explicit IndexExpression(SourceLocation source_loc,
                           Nonnull<Expression*> aggregate,
                           Nonnull<Expression*> offset)
      : Expression(Kind::IndexExpression, source_loc),
        aggregate(aggregate),
        offset(offset) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::IndexExpression;
  }

  auto Aggregate() const -> Nonnull<const Expression*> { return aggregate; }
  auto Aggregate() -> Nonnull<Expression*> { return aggregate; }
  auto Offset() const -> Nonnull<const Expression*> { return offset; }
  auto Offset() -> Nonnull<Expression*> { return offset; }

 private:
  Nonnull<Expression*> aggregate;
  Nonnull<Expression*> offset;
};

class IntLiteral : public Expression {
 public:
  explicit IntLiteral(SourceLocation source_loc, int val)
      : Expression(Kind::IntLiteral, source_loc), val(val) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::IntLiteral;
  }

  auto Val() const -> int { return val; }

 private:
  int val;
};

class BoolLiteral : public Expression {
 public:
  explicit BoolLiteral(SourceLocation source_loc, bool val)
      : Expression(Kind::BoolLiteral, source_loc), val(val) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::BoolLiteral;
  }

  auto Val() const -> bool { return val; }

 private:
  bool val;
};

class StringLiteral : public Expression {
 public:
  explicit StringLiteral(SourceLocation source_loc, std::string val)
      : Expression(Kind::StringLiteral, source_loc), val(std::move(val)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::StringLiteral;
  }

  auto Val() const -> const std::string& { return val; }

 private:
  std::string val;
};

class StringTypeLiteral : public Expression {
 public:
  explicit StringTypeLiteral(SourceLocation source_loc)
      : Expression(Kind::StringTypeLiteral, source_loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::StringTypeLiteral;
  }
};

class TupleLiteral : public Expression {
 public:
  explicit TupleLiteral(SourceLocation source_loc)
      : TupleLiteral(source_loc, {}) {}

  explicit TupleLiteral(SourceLocation source_loc,
                        std::vector<FieldInitializer> fields)
      : Expression(Kind::TupleLiteral, source_loc), fields(std::move(fields)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::TupleLiteral;
  }

  auto Fields() const -> const std::vector<FieldInitializer>& { return fields; }

 private:
  std::vector<FieldInitializer> fields;
};

class PrimitiveOperatorExpression : public Expression {
 public:
  explicit PrimitiveOperatorExpression(
      SourceLocation source_loc, Operator op,
      std::vector<Nonnull<Expression*>> arguments)
      : Expression(Kind::PrimitiveOperatorExpression, source_loc),
        op(op),
        arguments(std::move(arguments)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::PrimitiveOperatorExpression;
  }

  auto Op() const -> Operator { return op; }
  auto Arguments() const -> llvm::ArrayRef<Nonnull<Expression*>> {
    return arguments;
  }
  auto Arguments() -> llvm::MutableArrayRef<Nonnull<Expression*>> {
    return arguments;
  }

 private:
  Operator op;
  std::vector<Nonnull<Expression*>> arguments;
};

class CallExpression : public Expression {
 public:
  explicit CallExpression(SourceLocation source_loc,
                          Nonnull<Expression*> function,
                          Nonnull<Expression*> argument)
      : Expression(Kind::CallExpression, source_loc),
        function(function),
        argument(argument) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::CallExpression;
  }

  auto Function() const -> Nonnull<const Expression*> { return function; }
  auto Function() -> Nonnull<Expression*> { return function; }
  auto Argument() const -> Nonnull<const Expression*> { return argument; }
  auto Argument() -> Nonnull<Expression*> { return argument; }

 private:
  Nonnull<Expression*> function;
  Nonnull<Expression*> argument;
};

class FunctionTypeLiteral : public Expression {
 public:
  explicit FunctionTypeLiteral(SourceLocation source_loc,
                               Nonnull<Expression*> parameter,
                               Nonnull<Expression*> return_type,
                               bool is_omitted_return_type)
      : Expression(Kind::FunctionTypeLiteral, source_loc),
        parameter(parameter),
        return_type(return_type),
        is_omitted_return_type(is_omitted_return_type) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::FunctionTypeLiteral;
  }

  auto Parameter() const -> Nonnull<const Expression*> { return parameter; }
  auto Parameter() -> Nonnull<Expression*> { return parameter; }
  auto ReturnType() const -> Nonnull<const Expression*> { return return_type; }
  auto ReturnType() -> Nonnull<Expression*> { return return_type; }
  auto IsOmittedReturnType() const -> bool { return is_omitted_return_type; }

 private:
  Nonnull<Expression*> parameter;
  Nonnull<Expression*> return_type;
  bool is_omitted_return_type;
};

class BoolTypeLiteral : public Expression {
 public:
  explicit BoolTypeLiteral(SourceLocation source_loc)
      : Expression(Kind::BoolTypeLiteral, source_loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::BoolTypeLiteral;
  }
};

class IntTypeLiteral : public Expression {
 public:
  explicit IntTypeLiteral(SourceLocation source_loc)
      : Expression(Kind::IntTypeLiteral, source_loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::IntTypeLiteral;
  }
};

class ContinuationTypeLiteral : public Expression {
 public:
  explicit ContinuationTypeLiteral(SourceLocation source_loc)
      : Expression(Kind::ContinuationTypeLiteral, source_loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::ContinuationTypeLiteral;
  }
};

class TypeTypeLiteral : public Expression {
 public:
  explicit TypeTypeLiteral(SourceLocation source_loc)
      : Expression(Kind::TypeTypeLiteral, source_loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::TypeTypeLiteral;
  }
};

class IntrinsicExpression : public Expression {
 public:
  enum class IntrinsicKind {
    Print,
  };

  explicit IntrinsicExpression(IntrinsicKind intrinsic)
      : Expression(Kind::IntrinsicExpression, SourceLocation("<intrinsic>", 0)),
        intrinsic(intrinsic) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->kind() == Kind::IntrinsicExpression;
  }

  auto Intrinsic() const -> IntrinsicKind { return intrinsic; }

 private:
  IntrinsicKind intrinsic;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
