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
#include "executable_semantics/ast/static_scope.h"
#include "executable_semantics/ast/value_category.h"
#include "executable_semantics/common/arena.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Value;

class Expression : public AstNode {
 public:
  ~Expression() override = 0;

  void Print(llvm::raw_ostream& out) const override;

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
  Expression(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}

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
      : Expression(AstNodeKind::IdentifierExpression, source_loc),
        name_(std::move(name)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIdentifierExpression(node->kind());
  }

  auto name() const -> const std::string& { return name_; }

  // Returns the NamedEntityView this identifier refers to. Cannot be called
  // before name resolution.
  auto named_entity() const -> const NamedEntityView& { return *named_entity_; }

  // Sets the value returned by named_entity. Can be called only once,
  // during name resolution.
  void set_named_entity(NamedEntityView named_entity) {
    CHECK(!named_entity_.has_value());
    named_entity_ = std::move(named_entity);
  }

 private:
  std::string name_;
  std::optional<NamedEntityView> named_entity_;
};

class FieldAccessExpression : public Expression {
 public:
  explicit FieldAccessExpression(SourceLocation source_loc,
                                 Nonnull<Expression*> aggregate,
                                 std::string field)
      : Expression(AstNodeKind::FieldAccessExpression, source_loc),
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
      : Expression(AstNodeKind::IndexExpression, source_loc),
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
      : Expression(AstNodeKind::IntLiteral, source_loc), value_(value) {}

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
      : Expression(AstNodeKind::BoolLiteral, source_loc), value_(value) {}

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
      : Expression(AstNodeKind::StringLiteral, source_loc),
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
      : Expression(AstNodeKind::StringTypeLiteral, source_loc) {}

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
      : Expression(AstNodeKind::TupleLiteral, source_loc),
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
      : Expression(AstNodeKind::StructLiteral, loc),
        fields_(std::move(fields)) {
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
      : Expression(AstNodeKind::StructTypeLiteral, loc),
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
      : Expression(AstNodeKind::PrimitiveOperatorExpression, source_loc),
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
      : Expression(AstNodeKind::CallExpression, source_loc),
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
      : Expression(AstNodeKind::FunctionTypeLiteral, source_loc),
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
      : Expression(AstNodeKind::BoolTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromBoolTypeLiteral(node->kind());
  }
};

class IntTypeLiteral : public Expression {
 public:
  explicit IntTypeLiteral(SourceLocation source_loc)
      : Expression(AstNodeKind::IntTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIntTypeLiteral(node->kind());
  }
};

class ContinuationTypeLiteral : public Expression {
 public:
  explicit ContinuationTypeLiteral(SourceLocation source_loc)
      : Expression(AstNodeKind::ContinuationTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromContinuationTypeLiteral(node->kind());
  }
};

class TypeTypeLiteral : public Expression {
 public:
  explicit TypeTypeLiteral(SourceLocation source_loc)
      : Expression(AstNodeKind::TypeTypeLiteral, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromTypeTypeLiteral(node->kind());
  }
};

class IntrinsicExpression : public Expression {
 public:
  enum class Intrinsic {
    Print,
  };

  explicit IntrinsicExpression(std::string_view intrinsic_name,
                               Nonnull<TupleLiteral*> args,
                               SourceLocation source_loc)
      : Expression(AstNodeKind::IntrinsicExpression, source_loc),
        intrinsic_(FindIntrinsic(intrinsic_name, source_loc)),
        args_(args) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIntrinsicExpression(node->kind());
  }

  auto intrinsic() const -> Intrinsic { return intrinsic_; }
  auto args() const -> const TupleLiteral& { return *args_; }
  auto args() -> TupleLiteral& { return *args_; }

 private:
  // Returns the enumerator corresponding to the intrinsic named `name`,
  // or raises a fatal compile error if there is no such enumerator.
  static auto FindIntrinsic(std::string_view name, SourceLocation source_loc)
      -> Intrinsic;

  Intrinsic intrinsic_;
  Nonnull<TupleLiteral*> args_;
};

// An expression whose semantics have not been implemented. This can be used
// as a placeholder during development, in order to implement and test parsing
// of a new expression syntax without having to implement its semantics.
class UnimplementedExpression : public Expression {
 public:
  // Constructs an UnimplementedExpression with the given label and the given
  // children, which must all be convertible to Nonnull<AstNode*>. The label
  // should correspond roughly to the name of the class that will eventually
  // replace this usage of UnimplementedExpression.
  template <typename... Children>
  UnimplementedExpression(SourceLocation source_loc, std::string label,
                          Children... children)
      : Expression(AstNodeKind::UnimplementedExpression, source_loc),
        label_(std::move(label)) {
    AddChildren(children...);
  }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromUnimplementedExpression(node->kind());
  }

  auto label() const -> std::string_view { return label_; }
  auto children() const -> llvm::ArrayRef<Nonnull<const AstNode*>> {
    return children_;
  }

 private:
  void AddChildren() {}

  template <typename... Children>
  void AddChildren(Nonnull<AstNode*> child, Children... children) {
    children_.push_back(child);
    AddChildren(children...);
  }

  std::string label_;
  std::vector<Nonnull<AstNode*>> children_;
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
