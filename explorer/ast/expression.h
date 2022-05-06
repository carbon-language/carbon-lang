// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_AST_EXPRESSION_H_
#define EXPLORER_AST_EXPRESSION_H_

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/ast_node.h"
#include "explorer/ast/paren_contents.h"
#include "explorer/ast/static_scope.h"
#include "explorer/ast/value_category.h"
#include "explorer/common/arena.h"
#include "explorer/common/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Value;
class VariableType;
class ImplBinding;

class Expression : public AstNode {
 public:
  ~Expression() override = 0;

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) {
    return InheritsFromExpression(node->kind());
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> ExpressionKind {
    return static_cast<ExpressionKind>(root_kind());
  }

  // The static type of this expression. Cannot be called before typechecking.
  auto static_type() const -> const Value& {
    CARBON_CHECK(static_type_.has_value());
    return **static_type_;
  }

  // Sets the static type of this expression. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  // The value category of this expression. Cannot be called before
  // typechecking.
  auto value_category() const -> ValueCategory { return *value_category_; }

  // Sets the value category of this expression. Can be called multiple times,
  // but the argument must have the same value each time.
  void set_value_category(ValueCategory value_category) {
    CARBON_CHECK(!value_category_.has_value() ||
                 value_category == *value_category_);
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
  AddressOf,
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

  // Returns the ValueNodeView this identifier refers to. Cannot be called
  // before name resolution.
  auto value_node() const -> const ValueNodeView& { return *value_node_; }

  // Sets the value returned by value_node. Can be called only once,
  // during name resolution.
  void set_value_node(ValueNodeView value_node) {
    CARBON_CHECK(!value_node_.has_value());
    value_node_ = std::move(value_node);
  }

 private:
  std::string name_;
  std::optional<ValueNodeView> value_node_;
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

  // If `aggregate` has a generic type, returns the `ImplBinding` that
  // identifies its witness table. Otherwise, returns `std::nullopt`. Should not
  // be called before typechecking.
  auto impl() const -> std::optional<Nonnull<const ImplBinding*>> {
    return impl_;
  }

  // Can only be called once, during typechecking.
  void set_impl(Nonnull<const ImplBinding*> impl) {
    CARBON_CHECK(!impl_.has_value());
    impl_ = impl;
  }

 private:
  Nonnull<Expression*> aggregate_;
  std::string field_;
  std::optional<Nonnull<const ImplBinding*>> impl_;
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
    CARBON_CHECK(!fields_.empty())
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

class GenericBinding;

using BindingMap =
    std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>;

using ImplExpMap = std::map<Nonnull<const ImplBinding*>, Nonnull<Expression*>>;

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

  // Maps each of `function`'s impl bindings to an expression
  // that constructs a witness table.
  // Should not be called before typechecking, or if `function` is not
  // a generic function.
  auto impls() const -> const ImplExpMap& { return impls_; }

  // Can only be called once, during typechecking.
  void set_impls(const ImplExpMap& impls) {
    CARBON_CHECK(impls_.empty());
    impls_ = impls;
  }

  auto deduced_args() const -> const BindingMap& { return deduced_args_; }

  void set_deduced_args(const BindingMap& deduced_args) {
    deduced_args_ = deduced_args;
  }

 private:
  Nonnull<Expression*> function_;
  Nonnull<Expression*> argument_;
  ImplExpMap impls_;
  BindingMap deduced_args_;
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

  // Returns the enumerator corresponding to the intrinsic named `name`,
  // or raises a fatal compile error if there is no such enumerator.
  static auto FindIntrinsic(std::string_view name, SourceLocation source_loc)
      -> ErrorOr<Intrinsic>;

  explicit IntrinsicExpression(Intrinsic intrinsic, Nonnull<TupleLiteral*> args,
                               SourceLocation source_loc)
      : Expression(AstNodeKind::IntrinsicExpression, source_loc),
        intrinsic_(intrinsic),
        args_(args) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIntrinsicExpression(node->kind());
  }

  auto intrinsic() const -> Intrinsic { return intrinsic_; }
  auto args() const -> const TupleLiteral& { return *args_; }
  auto args() -> TupleLiteral& { return *args_; }

 private:
  Intrinsic intrinsic_;
  Nonnull<TupleLiteral*> args_;
};

class IfExpression : public Expression {
 public:
  explicit IfExpression(SourceLocation source_loc,
                        Nonnull<Expression*> condition,
                        Nonnull<Expression*> then_expression,
                        Nonnull<Expression*> else_expression)
      : Expression(AstNodeKind::IfExpression, source_loc),
        condition_(condition),
        then_expression_(then_expression),
        else_expression_(else_expression) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIfExpression(node->kind());
  }

  auto condition() const -> const Expression& { return *condition_; }
  auto condition() -> Expression& { return *condition_; }

  auto then_expression() const -> const Expression& {
    return *then_expression_;
  }
  auto then_expression() -> Expression& { return *then_expression_; }

  auto else_expression() const -> const Expression& {
    return *else_expression_;
  }
  auto else_expression() -> Expression& { return *else_expression_; }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Expression*> then_expression_;
  Nonnull<Expression*> else_expression_;
};

// Instantiate a generic impl.
class InstantiateImpl : public Expression {
 public:
  using ImplementsCarbonValueNode = void;

  explicit InstantiateImpl(SourceLocation source_loc,
                           Nonnull<Expression*> generic_impl,
                           const BindingMap& type_args, const ImplExpMap& impls)
      : Expression(AstNodeKind::InstantiateImpl, source_loc),
        generic_impl_(generic_impl),
        type_args_(type_args),
        impls_(impls) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInstantiateImpl(node->kind());
  }
  auto generic_impl() const -> Nonnull<Expression*> { return generic_impl_; }
  auto type_args() const -> const BindingMap& { return type_args_; }

  // Maps each of the impl bindings to an expression that constructs
  // the witness table for that impl.
  auto impls() const -> const ImplExpMap& { return impls_; }

 private:
  Nonnull<Expression*> generic_impl_;
  BindingMap type_args_;
  ImplExpMap impls_;
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

// A literal representing a statically-sized array type.
class ArrayTypeLiteral : public Expression {
 public:
  // Constructs an array type literal which uses the given expressions to
  // represent the element type and size.
  ArrayTypeLiteral(SourceLocation source_loc,
                   Nonnull<Expression*> element_type_expression,
                   Nonnull<Expression*> size_expression)
      : Expression(AstNodeKind::ArrayTypeLiteral, source_loc),
        element_type_expression_(element_type_expression),
        size_expression_(size_expression) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromArrayTypeLiteral(node->kind());
  }

  auto element_type_expression() const -> const Expression& {
    return *element_type_expression_;
  }
  auto element_type_expression() -> Expression& {
    return *element_type_expression_;
  }

  auto size_expression() const -> const Expression& {
    return *size_expression_;
  }
  auto size_expression() -> Expression& { return *size_expression_; }

 private:
  Nonnull<Expression*> element_type_expression_;
  Nonnull<Expression*> size_expression_;
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

#endif  // EXPLORER_AST_EXPRESSION_H_
