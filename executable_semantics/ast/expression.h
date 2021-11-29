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
#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/paren_contents.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/arena.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Value;

class Expression : public virtual AstNode {
 public:
  // The value category of a Carbon expression indicates whether it evaluates
  // to a variable or a value. A variable can be mutated, and can have its
  // address taken, whereas a value cannot.
  enum class ValueCategory {
    // A variable. This roughly corresponds to a C/C++ lvalue.
    Var,
    // A value. This roughly corresponds to a C/C++ rvalue.
    Let,
  };

  ~Expression() override = 0;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  static auto classof(const AstNode* node) {
    return InheritsFromExpression(node->kind());
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> ExpressionKind {
    return static_cast<ExpressionKind>(root_kind());
  }

  // The static type of this expression. Cannot be called before typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of this expression. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) { static_type_ = type; }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

  // The value category of this expression. Cannot be called before
  // typechecking.
  auto value_category() const -> ValueCategory { return *value_category_; }

  // Sets the value category of this expression. Can be called multiple times,
  // but the argument must have the same value each time.
  void set_value_category(ValueCategory value_category) {
    CHECK(!value_category_.has_value() || value_category == *value_category_);
    value_category_ = value_category;
  }

 protected:
  // Constructs an Expression representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Expression() = default;

 private:
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<ValueCategory> value_category_;
};

// A FieldInitializer represents the initialization of a single struct field.
class FieldInitializer {
 public:
  FieldInitializer(std::string name, Nonnull<Expression*> expression)
      : name_(std::move(name)), expression_(expression) {}

  auto name() const -> const std::string& { return name_; }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }

 private:
  // The field name. Cannot be empty.
  std::string name_;

  // The expression that initializes the field.
  Nonnull<Expression*> expression_;
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

// Returns the lexical representation of `op`, such as "+" for `Add`.
auto ToString(Operator op) -> std::string_view;

class IdentifierExpression : public Expression {
 public:
  explicit IdentifierExpression(SourceLocation source_loc, std::string name)
      : AstNode(AstNodeKind::IdentifierExpression, source_loc),
        name_(std::move(name)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIdentifierExpression(node->kind());
  }

  auto name() const -> const std::string& { return name_; }

 private:
  std::string name_;
};

class FieldAccessExpression : public Expression {
 public:
  explicit FieldAccessExpression(SourceLocation source_loc,
                                 Nonnull<Expression*> aggregate,
                                 std::string field)
      : AstNode(AstNodeKind::FieldAccessExpression, source_loc),
        aggregate_(aggregate),
        field_(std::move(field)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFieldAccessExpression(node->kind());
  }

  auto aggregate() const -> const Expression& { return *aggregate_; }
  auto aggregate() -> Expression& { return *aggregate_; }
  auto field() const -> const std::string& { return field_; }

 private:
  Nonnull<Expression*> aggregate_;
  std::string field_;
};

class IndexExpression : public Expression {
 public:
  explicit IndexExpression(SourceLocation source_loc,
                           Nonnull<Expression*> aggregate,
                           Nonnull<Expression*> offset)
      : AstNode(AstNodeKind::IndexExpression, source_loc),
        aggregate_(aggregate),
        offset_(offset) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIndexExpression(node->kind());
  }

  auto aggregate() const -> const Expression& { return *aggregate_; }
  auto aggregate() -> Expression& { return *aggregate_; }
  auto offset() const -> const Expression& { return *offset_; }
  auto offset() -> Expression& { return *offset_; }

 private:
  Nonnull<Expression*> aggregate_;
  Nonnull<Expression*> offset_;
};

class IntLiteral : public Expression {
 public:
  explicit IntLiteral(SourceLocation source_loc, int value)
      : AstNode(AstNodeKind::IntLiteral, source_loc), value_(value) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIntLiteral(node->kind());
  }

  auto value() const -> int { return value_; }

 private:
  int value_;
};

class BoolLiteral : public Expression {
 public:
  explicit BoolLiteral(SourceLocation source_loc, bool value)
      : AstNode(AstNodeKind::BoolLiteral, source_loc), value_(value) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromBoolLiteral(node->kind());
  }

  auto value() const -> bool { return value_; }

 private:
  bool value_;
};

class StringLiteral : public Expression {
 public:
  explicit StringLiteral(SourceLocation source_loc, std::string value)
      : AstNode(AstNodeKind::StringLiteral, source_loc),
        value_(std::move(value)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromStringLiteral(node->kind());
  }

  auto value() const -> const std::string& { return value_; }

 private:
  std::string value_;
};

class StringTypeLiteral : public Expression {
 public:
  explicit StringTypeLiteral(SourceLocation source_loc)
      : AstNode(AstNodeKind::StringTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromStringTypeLiteral(node->kind());
  }
};

class TupleLiteral : public Expression {
 public:
  explicit TupleLiteral(SourceLocation source_loc)
      : TupleLiteral(source_loc, {}) {}

  explicit TupleLiteral(SourceLocation source_loc,
                        std::vector<Nonnull<Expression*>> fields)
      : AstNode(AstNodeKind::TupleLiteral, source_loc),
        fields_(std::move(fields)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromTupleLiteral(node->kind());
  }

  auto fields() const -> llvm::ArrayRef<Nonnull<const Expression*>> {
    return fields_;
  }
  auto fields() -> llvm::ArrayRef<Nonnull<Expression*>> { return fields_; }

 private:
  std::vector<Nonnull<Expression*>> fields_;
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
      : AstNode(AstNodeKind::StructLiteral, loc), fields_(std::move(fields)) {
    CHECK(!fields_.empty())
        << "`{}` is represented as a StructTypeLiteral, not a StructLiteral.";
  }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromStructLiteral(node->kind());
  }

  auto fields() const -> llvm::ArrayRef<FieldInitializer> { return fields_; }
  auto fields() -> llvm::MutableArrayRef<FieldInitializer> { return fields_; }

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
      : AstNode(AstNodeKind::StructTypeLiteral, loc),
        fields_(std::move(fields)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromStructTypeLiteral(node->kind());
  }

  auto fields() const -> llvm::ArrayRef<FieldInitializer> { return fields_; }
  auto fields() -> llvm::MutableArrayRef<FieldInitializer> { return fields_; }

 private:
  std::vector<FieldInitializer> fields_;
};

class PrimitiveOperatorExpression : public Expression {
 public:
  explicit PrimitiveOperatorExpression(
      SourceLocation source_loc, Operator op,
      std::vector<Nonnull<Expression*>> arguments)
      : AstNode(AstNodeKind::PrimitiveOperatorExpression, source_loc),
        op_(op),
        arguments_(std::move(arguments)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromPrimitiveOperatorExpression(node->kind());
  }

  auto op() const -> Operator { return op_; }
  auto arguments() const -> llvm::ArrayRef<Nonnull<Expression*>> {
    return arguments_;
  }
  auto arguments() -> llvm::MutableArrayRef<Nonnull<Expression*>> {
    return arguments_;
  }

 private:
  Operator op_;
  std::vector<Nonnull<Expression*>> arguments_;
};

class CallExpression : public Expression {
 public:
  explicit CallExpression(SourceLocation source_loc,
                          Nonnull<Expression*> function,
                          Nonnull<Expression*> argument)
      : AstNode(AstNodeKind::CallExpression, source_loc),
        function_(function),
        argument_(argument) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromCallExpression(node->kind());
  }

  auto function() const -> const Expression& { return *function_; }
  auto function() -> Expression& { return *function_; }
  auto argument() const -> const Expression& { return *argument_; }
  auto argument() -> Expression& { return *argument_; }

 private:
  Nonnull<Expression*> function_;
  Nonnull<Expression*> argument_;
};

class FunctionTypeLiteral : public Expression {
 public:
  explicit FunctionTypeLiteral(SourceLocation source_loc,
                               Nonnull<Expression*> parameter,
                               Nonnull<Expression*> return_type)
      : AstNode(AstNodeKind::FunctionTypeLiteral, source_loc),
        parameter_(parameter),
        return_type_(return_type) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFunctionTypeLiteral(node->kind());
  }

  auto parameter() const -> const Expression& { return *parameter_; }
  auto parameter() -> Expression& { return *parameter_; }
  auto return_type() const -> const Expression& { return *return_type_; }
  auto return_type() -> Expression& { return *return_type_; }

 private:
  Nonnull<Expression*> parameter_;
  Nonnull<Expression*> return_type_;
};

class BoolTypeLiteral : public Expression {
 public:
  explicit BoolTypeLiteral(SourceLocation source_loc)
      : AstNode(AstNodeKind::BoolTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromBoolTypeLiteral(node->kind());
  }
};

class IntTypeLiteral : public Expression {
 public:
  explicit IntTypeLiteral(SourceLocation source_loc)
      : AstNode(AstNodeKind::IntTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIntTypeLiteral(node->kind());
  }
};

class ContinuationTypeLiteral : public Expression {
 public:
  explicit ContinuationTypeLiteral(SourceLocation source_loc)
      : AstNode(AstNodeKind::ContinuationTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromContinuationTypeLiteral(node->kind());
  }
};

class TypeTypeLiteral : public Expression {
 public:
  explicit TypeTypeLiteral(SourceLocation source_loc)
      : AstNode(AstNodeKind::TypeTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromTypeTypeLiteral(node->kind());
  }
};

class IntrinsicExpression : public Expression {
 public:
  enum class Intrinsic {
    Print,
  };

  explicit IntrinsicExpression(Intrinsic intrinsic)
      : AstNode(AstNodeKind::IntrinsicExpression,
                SourceLocation("<intrinsic>", 0)),
        intrinsic_(intrinsic) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIntrinsicExpression(node->kind());
  }

  auto intrinsic() const -> Intrinsic { return intrinsic_; }

 private:
  Intrinsic intrinsic_;
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
    const ParenContents<Expression>& paren_contents) -> Nonnull<TupleLiteral*>;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
