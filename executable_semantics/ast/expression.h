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
    StructLiteral,
    StructTypeLiteral,
    TypeTypeLiteral,
    IdentifierExpression,
    IntrinsicExpression,
  };

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return tag; }

  auto SourceLoc() const -> SourceLocation { return loc; }

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs an Expression representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Expression(Kind tag, SourceLocation loc) : tag(tag), loc(loc) {}

 private:
  const Kind tag;
  SourceLocation loc;
};

// Converts paren_contents to an Expression, interpreting the parentheses as
// grouping if their contents permit that interpretation, or as forming a
// tuple otherwise.
auto ExpressionFromParenContents(
    Nonnull<Arena*> arena, SourceLocation loc,
    const ParenContents<Expression>& paren_contents) -> Nonnull<Expression*>;

// Converts paren_contents to an Expression, interpreting the parentheses as
// forming a tuple.
auto TupleExpressionFromParenContents(
    Nonnull<Arena*> arena, SourceLocation loc,
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
  explicit IdentifierExpression(SourceLocation loc, std::string name)
      : Expression(Kind::IdentifierExpression, loc), name(std::move(name)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IdentifierExpression;
  }

  auto Name() const -> const std::string& { return name; }

 private:
  std::string name;
};

class FieldAccessExpression : public Expression {
 public:
  explicit FieldAccessExpression(SourceLocation loc,
                                 Nonnull<Expression*> aggregate,
                                 std::string field)
      : Expression(Kind::FieldAccessExpression, loc),
        aggregate(aggregate),
        field(std::move(field)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::FieldAccessExpression;
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
  explicit IndexExpression(SourceLocation loc, Nonnull<Expression*> aggregate,
                           Nonnull<Expression*> offset)
      : Expression(Kind::IndexExpression, loc),
        aggregate(aggregate),
        offset(offset) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IndexExpression;
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
  explicit IntLiteral(SourceLocation loc, int val)
      : Expression(Kind::IntLiteral, loc), val(val) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IntLiteral;
  }

  auto Val() const -> int { return val; }

 private:
  int val;
};

class BoolLiteral : public Expression {
 public:
  explicit BoolLiteral(SourceLocation loc, bool val)
      : Expression(Kind::BoolLiteral, loc), val(val) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::BoolLiteral;
  }

  auto Val() const -> bool { return val; }

 private:
  bool val;
};

class StringLiteral : public Expression {
 public:
  explicit StringLiteral(SourceLocation loc, std::string val)
      : Expression(Kind::StringLiteral, loc), val(std::move(val)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::StringLiteral;
  }

  auto Val() const -> const std::string& { return val; }

 private:
  std::string val;
};

class StringTypeLiteral : public Expression {
 public:
  explicit StringTypeLiteral(SourceLocation loc)
      : Expression(Kind::StringTypeLiteral, loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::StringTypeLiteral;
  }
};

class TupleLiteral : public Expression {
 public:
  explicit TupleLiteral(SourceLocation loc) : TupleLiteral(loc, {}) {}

  explicit TupleLiteral(SourceLocation loc,
                        std::vector<FieldInitializer> fields)
      : Expression(Kind::TupleLiteral, loc), fields(std::move(fields)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::TupleLiteral;
  }

  auto Fields() const -> const std::vector<FieldInitializer>& { return fields; }

 private:
  std::vector<FieldInitializer> fields;
};

// A non-empty literal value of a struct type.
//
// It can't be empty because the syntax `{}` is a struct type literal as well
// as a literal value of that type, so for consistency we always represent it
// as a StructTypeLiteral rather than let it oscillate unpredictably between
// the two.
class StructLiteral : public Expression {
 public:
  explicit StructLiteral(SourceLocation loc,
                         std::vector<FieldInitializer> fields)
      : Expression(Kind::StructLiteral, loc), fields_(std::move(fields)) {
    CHECK(!fields_.empty())
        << "`{}` is represented as a StructTypeLiteral, not a StructLiteral.";
  }

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::StructLiteral;
  }

  auto fields() const -> const std::vector<FieldInitializer>& {
    return fields_;
  }

 private:
  std::vector<FieldInitializer> fields_;
};

// A literal representing a struct type.
//
// Code that handles this type may sometimes need to have special-case handling
// for `{}`, which is a struct value in addition to being a struct type.
class StructTypeLiteral : public Expression {
 public:
  explicit StructTypeLiteral(SourceLocation loc) : StructTypeLiteral(loc, {}) {}

  explicit StructTypeLiteral(SourceLocation loc,
                             std::vector<FieldInitializer> fields)
      : Expression(Kind::StructTypeLiteral, loc), fields_(std::move(fields)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::StructTypeLiteral;
  }

  auto fields() const -> const std::vector<FieldInitializer>& {
    return fields_;
  }

 private:
  std::vector<FieldInitializer> fields_;
};

class PrimitiveOperatorExpression : public Expression {
 public:
  explicit PrimitiveOperatorExpression(
      SourceLocation loc, Operator op,
      std::vector<Nonnull<Expression*>> arguments)
      : Expression(Kind::PrimitiveOperatorExpression, loc),
        op(op),
        arguments(std::move(arguments)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::PrimitiveOperatorExpression;
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
  explicit CallExpression(SourceLocation loc, Nonnull<Expression*> function,
                          Nonnull<Expression*> argument)
      : Expression(Kind::CallExpression, loc),
        function(function),
        argument(argument) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::CallExpression;
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
  explicit FunctionTypeLiteral(SourceLocation loc,
                               Nonnull<Expression*> parameter,
                               Nonnull<Expression*> return_type,
                               bool is_omitted_return_type)
      : Expression(Kind::FunctionTypeLiteral, loc),
        parameter(parameter),
        return_type(return_type),
        is_omitted_return_type(is_omitted_return_type) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::FunctionTypeLiteral;
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
  explicit BoolTypeLiteral(SourceLocation loc)
      : Expression(Kind::BoolTypeLiteral, loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::BoolTypeLiteral;
  }
};

class IntTypeLiteral : public Expression {
 public:
  explicit IntTypeLiteral(SourceLocation loc)
      : Expression(Kind::IntTypeLiteral, loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IntTypeLiteral;
  }
};

class ContinuationTypeLiteral : public Expression {
 public:
  explicit ContinuationTypeLiteral(SourceLocation loc)
      : Expression(Kind::ContinuationTypeLiteral, loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::ContinuationTypeLiteral;
  }
};

class TypeTypeLiteral : public Expression {
 public:
  explicit TypeTypeLiteral(SourceLocation loc)
      : Expression(Kind::TypeTypeLiteral, loc) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::TypeTypeLiteral;
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
    return exp->Tag() == Kind::IntrinsicExpression;
  }

  auto Intrinsic() const -> IntrinsicKind { return intrinsic; }

 private:
  IntrinsicKind intrinsic;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
