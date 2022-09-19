// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_EXPRESSION_H_
#define CARBON_EXPLORER_AST_EXPRESSION_H_

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/ast_node.h"
#include "explorer/ast/bindings.h"
#include "explorer/ast/member.h"
#include "explorer/ast/paren_contents.h"
#include "explorer/ast/static_scope.h"
#include "explorer/ast/value_category.h"
#include "explorer/common/arena.h"
#include "explorer/common/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Value;
class MemberName;
class VariableType;
class InterfaceType;
class ImplBinding;
class GenericBinding;

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

  // Determines whether the expression has already been type-checked. Should
  // only be used by type-checking.
  auto is_type_checked() -> bool {
    return static_type_.has_value() && value_category_.has_value();
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
  As,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  BitShiftLeft,
  BitShiftRight,
  Complement,
  Deref,
  Div,
  Eq,
  Less,
  LessEq,
  Greater,
  GreaterEq,
  Mul,
  Mod,
  Neg,
  Not,
  NotEq,
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

  // Sets the value returned by value_node. Can be called only during name
  // resolution.
  void set_value_node(ValueNodeView value_node) {
    CARBON_CHECK(!value_node_.has_value() || value_node_ == value_node);
    value_node_ = std::move(value_node);
  }

 private:
  std::string name_;
  std::optional<ValueNodeView> value_node_;
};

// A `.Self` expression within either a `:!` binding or a standalone `where`
// expression.
//
// In a `:!` binding, the type of `.Self` is always `Type`. For example, in
// `A:! AddableWith(.Self)`, the expression `.Self` refers to the same type as
// `A`, but with type `Type`.
//
// In a `where` binding, the type of `.Self` is the constraint preceding the
// `where` keyword. For example, in `Foo where .Result is Bar(.Self)`, the type
// of `.Self` is `Foo`.
class DotSelfExpression : public Expression {
 public:
  explicit DotSelfExpression(SourceLocation source_loc)
      : Expression(AstNodeKind::DotSelfExpression, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromDotSelfExpression(node->kind());
  }

  // The self binding. Cannot be called before name resolution.
  auto self_binding() const -> const GenericBinding& { return **self_binding_; }
  auto self_binding() -> GenericBinding& { return **self_binding_; }

  // Sets the self binding. Called only during name resolution.
  void set_self_binding(Nonnull<GenericBinding*> self_binding) {
    CARBON_CHECK(!self_binding_.has_value() || self_binding_ == self_binding);
    self_binding_ = self_binding;
  }

 private:
  std::string name_;
  std::optional<Nonnull<GenericBinding*>> self_binding_;
};

class SimpleMemberAccessExpression : public Expression {
 public:
  explicit SimpleMemberAccessExpression(SourceLocation source_loc,
                                        Nonnull<Expression*> object,
                                        std::string member_name)
      : Expression(AstNodeKind::SimpleMemberAccessExpression, source_loc),
        object_(object),
        member_name_(std::move(member_name)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromSimpleMemberAccessExpression(node->kind());
  }

  auto object() const -> const Expression& { return *object_; }
  auto object() -> Expression& { return *object_; }
  auto member_name() const -> const std::string& { return member_name_; }

  // Returns the `Member` that the member name resolved to.
  // Should not be called before typechecking.
  auto member() const -> const Member& {
    CARBON_CHECK(member_.has_value());
    return *member_;
  }

  // Can only be called once, during typechecking.
  void set_member(Member member) {
    CARBON_CHECK(!member_.has_value());
    member_ = member;
  }

  // Returns true if the field is a method that has a "me" declaration in an
  // AddrPattern.
  auto is_field_addr_me_method() const -> bool {
    return is_field_addr_me_method_;
  }

  // Can only be called once, during typechecking.
  void set_is_field_addr_me_method() { is_field_addr_me_method_ = true; }

  // If `object` has a generic type, returns the `ImplBinding` that
  // identifies its witness table. Otherwise, returns `std::nullopt`. Should not
  // be called before typechecking.
  auto impl() const -> std::optional<Nonnull<const Expression*>> {
    return impl_;
  }

  // Can only be called once, during typechecking.
  void set_impl(Nonnull<const Expression*> impl) {
    CARBON_CHECK(!impl_.has_value());
    impl_ = impl;
  }

  // If `object` is a constrained type parameter and `member` was found in an
  // interface, returns that interface. Should not be called before
  // typechecking.
  auto found_in_interface() const
      -> std::optional<Nonnull<const InterfaceType*>> {
    return found_in_interface_;
  }

  // Can only be called once, during typechecking.
  void set_found_in_interface(Nonnull<const InterfaceType*> interface) {
    CARBON_CHECK(!found_in_interface_.has_value());
    found_in_interface_ = interface;
  }

 private:
  Nonnull<Expression*> object_;
  std::string member_name_;
  std::optional<Member> member_;
  bool is_field_addr_me_method_ = false;
  std::optional<Nonnull<const Expression*>> impl_;
  std::optional<Nonnull<const InterfaceType*>> found_in_interface_;
};

// A compound member access expression of the form `object.(path)`.
//
// `path` is required to have `TypeOfMemberName` type, and describes the member
// being accessed, which is one of:
//
// -   An instance member of a type: `object.(Type.member)`.
// -   A non-instance member of an interface: `Type.(Interface.member)` or
//     `object.(Interface.member)`.
// -   An instance member of an interface: `object.(Interface.member)` or
//     `object.(Type.(Interface.member))`.
//
// Note that the `path` is evaluated during type-checking, not at runtime, so
// the corresponding `member` is determined statically.
class CompoundMemberAccessExpression : public Expression {
 public:
  explicit CompoundMemberAccessExpression(SourceLocation source_loc,
                                          Nonnull<Expression*> object,
                                          Nonnull<Expression*> path)
      : Expression(AstNodeKind::CompoundMemberAccessExpression, source_loc),
        object_(object),
        path_(path) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromCompoundMemberAccessExpression(node->kind());
  }

  auto object() const -> const Expression& { return *object_; }
  auto object() -> Expression& { return *object_; }
  auto path() const -> const Expression& { return *path_; }
  auto path() -> Expression& { return *path_; }

  // Returns the `MemberName` value that evaluation of the path produced.
  // Should not be called before typechecking.
  auto member() const -> const MemberName& {
    CARBON_CHECK(member_.has_value());
    return **member_;
  }

  // Can only be called once, during typechecking.
  void set_member(Nonnull<const MemberName*> member) {
    CARBON_CHECK(!member_.has_value());
    member_ = member;
  }

  // Returns the expression to use to compute the witness table, if this
  // expression names an interface member.
  auto impl() const -> std::optional<Nonnull<const Expression*>> {
    return impl_;
  }

  // Can only be called once, during typechecking.
  void set_impl(Nonnull<const Expression*> impl) {
    CARBON_CHECK(!impl_.has_value());
    impl_ = impl;
  }

  // Can only be called by type-checking, if a conversion was required.
  void set_object(Nonnull<Expression*> object) { object_ = object; }

 private:
  Nonnull<Expression*> object_;
  Nonnull<Expression*> path_;
  std::optional<Nonnull<const MemberName*>> member_;
  std::optional<Nonnull<const Expression*>> impl_;
};

class IndexExpression : public Expression {
 public:
  explicit IndexExpression(SourceLocation source_loc,
                           Nonnull<Expression*> object,
                           Nonnull<Expression*> offset)
      : Expression(AstNodeKind::IndexExpression, source_loc),
        object_(object),
        offset_(offset) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIndexExpression(node->kind());
  }

  auto object() const -> const Expression& { return *object_; }
  auto object() -> Expression& { return *object_; }
  auto offset() const -> const Expression& { return *offset_; }
  auto offset() -> Expression& { return *offset_; }

 private:
  Nonnull<Expression*> object_;
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

class OperatorExpression : public Expression {
 public:
  explicit OperatorExpression(SourceLocation source_loc, Operator op,
                              std::vector<Nonnull<Expression*>> arguments)
      : Expression(AstNodeKind::OperatorExpression, source_loc),
        op_(op),
        arguments_(std::move(arguments)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromOperatorExpression(node->kind());
  }

  auto op() const -> Operator { return op_; }
  auto arguments() const -> llvm::ArrayRef<Nonnull<Expression*>> {
    return arguments_;
  }
  auto arguments() -> llvm::MutableArrayRef<Nonnull<Expression*>> {
    return arguments_;
  }

  // Set the rewritten form of this expression. Can only be called during type
  // checking.
  auto set_rewritten_form(const Expression* rewritten_form) -> void {
    CARBON_CHECK(!rewritten_form_.has_value()) << "rewritten form set twice";
    rewritten_form_ = rewritten_form;
    set_static_type(&rewritten_form->static_type());
    set_value_category(rewritten_form->value_category());
  }
  // Get the rewritten form of this expression. A rewritten form is used when
  // the expression is rewritten as a function call on an interface. A
  // rewritten form is not used when providing built-in operator semantics.
  auto rewritten_form() const -> std::optional<Nonnull<const Expression*>> {
    return rewritten_form_;
  }

 private:
  Operator op_;
  std::vector<Nonnull<Expression*>> arguments_;
  std::optional<Nonnull<const Expression*>> rewritten_form_;
};

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

  // Can only be called by type-checking, if a conversion was required.
  void set_argument(Nonnull<Expression*> argument) { argument_ = argument; }

 private:
  Nonnull<Expression*> function_;
  Nonnull<Expression*> argument_;
  ImplExpMap impls_;
  BindingMap deduced_args_;
};

class FunctionTypeLiteral : public Expression {
 public:
  explicit FunctionTypeLiteral(SourceLocation source_loc,
                               Nonnull<TupleLiteral*> parameter,
                               Nonnull<Expression*> return_type)
      : Expression(AstNodeKind::FunctionTypeLiteral, source_loc),
        parameter_(parameter),
        return_type_(return_type) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFunctionTypeLiteral(node->kind());
  }

  auto parameter() const -> const TupleLiteral& { return *parameter_; }
  auto parameter() -> TupleLiteral& { return *parameter_; }
  auto return_type() const -> const Expression& { return *return_type_; }
  auto return_type() -> Expression& { return *return_type_; }

 private:
  Nonnull<TupleLiteral*> parameter_;
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

// A literal value. This is used in desugaring, and can't be expressed in
// source syntax.
class ValueLiteral : public Expression {
 public:
  // Value literals are created by type-checking, and so are created with their
  // type and value category already known.
  ValueLiteral(SourceLocation source_loc, Nonnull<const Value*> value,
               Nonnull<const Value*> type, ValueCategory value_category)
      : Expression(AstNodeKind::ValueLiteral, source_loc), value_(value) {
    set_static_type(type);
    set_value_category(value_category);
  }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromValueLiteral(node->kind());
  }

  auto value() const -> const Value& { return *value_; }

 private:
  Nonnull<const Value*> value_;
};

class IntrinsicExpression : public Expression {
 public:
  enum class Intrinsic {
    Print,
    Alloc,
    Dealloc,
    Rand,
    IntEq,
    StrEq,
    StrCompare,
    IntCompare,
    IntBitAnd,
    IntBitOr,
    IntBitXor,
    IntBitComplement,
    IntLeftShift,
    IntRightShift,
    Assert,
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
  auto name() const -> std::string_view;
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

  // Can only be called by type-checking, if a conversion was required.
  void set_condition(Nonnull<Expression*> condition) { condition_ = condition; }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Expression*> then_expression_;
  Nonnull<Expression*> else_expression_;
};

// A clause appearing on the right-hand side of a `where` operator that forms a
// more precise constraint from a more general one.
class WhereClause : public AstNode {
 public:
  ~WhereClause() override = 0;

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) {
    return InheritsFromWhereClause(node->kind());
  }

  auto kind() const -> WhereClauseKind {
    return static_cast<WhereClauseKind>(root_kind());
  }

 protected:
  WhereClause(WhereClauseKind kind, SourceLocation source_loc)
      : AstNode(static_cast<AstNodeKind>(kind), source_loc) {}
};

// An `is` where clause.
//
// For example, `ConstraintA where .Type is ConstraintB` requires that the
// associated type `.Type` implements the constraint `ConstraintB`.
class IsWhereClause : public WhereClause {
 public:
  explicit IsWhereClause(SourceLocation source_loc, Nonnull<Expression*> type,
                         Nonnull<Expression*> constraint)
      : WhereClause(WhereClauseKind::IsWhereClause, source_loc),
        type_(type),
        constraint_(constraint) {}

  static auto classof(const AstNode* node) {
    return InheritsFromIsWhereClause(node->kind());
  }

  auto type() const -> const Expression& { return *type_; }
  auto type() -> Expression& { return *type_; }

  auto constraint() const -> const Expression& { return *constraint_; }
  auto constraint() -> Expression& { return *constraint_; }

 private:
  Nonnull<Expression*> type_;
  Nonnull<Expression*> constraint_;
};

// An `==` where clause.
//
// For example, `Constraint where .Type == i32` requires that the associated
// type `.Type` is `i32`.
class EqualsWhereClause : public WhereClause {
 public:
  explicit EqualsWhereClause(SourceLocation source_loc,
                             Nonnull<Expression*> lhs, Nonnull<Expression*> rhs)
      : WhereClause(WhereClauseKind::EqualsWhereClause, source_loc),
        lhs_(lhs),
        rhs_(rhs) {}

  static auto classof(const AstNode* node) {
    return InheritsFromEqualsWhereClause(node->kind());
  }

  auto lhs() const -> const Expression& { return *lhs_; }
  auto lhs() -> Expression& { return *lhs_; }

  auto rhs() const -> const Expression& { return *rhs_; }
  auto rhs() -> Expression& { return *rhs_; }

 private:
  Nonnull<Expression*> lhs_;
  Nonnull<Expression*> rhs_;
};

// A `where` expression: `AddableWith(i32) where .Result == i32`.
//
// The first operand is rewritten to a generic binding, for example
// `.Self:! AddableWith(i32)`, which may be used in the clauses.
class WhereExpression : public Expression {
 public:
  explicit WhereExpression(SourceLocation source_loc,
                           Nonnull<GenericBinding*> self_binding,
                           std::vector<Nonnull<WhereClause*>> clauses)
      : Expression(AstNodeKind::WhereExpression, source_loc),
        self_binding_(self_binding),
        clauses_(std::move(clauses)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromWhereExpression(node->kind());
  }

  auto self_binding() const -> const GenericBinding& { return *self_binding_; }
  auto self_binding() -> GenericBinding& { return *self_binding_; }

  auto clauses() const -> llvm::ArrayRef<Nonnull<const WhereClause*>> {
    return clauses_;
  }
  auto clauses() -> llvm::ArrayRef<Nonnull<WhereClause*>> { return clauses_; }

 private:
  Nonnull<GenericBinding*> self_binding_;
  std::vector<Nonnull<WhereClause*>> clauses_;
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

#endif  // CARBON_EXPLORER_AST_EXPRESSION_H_
