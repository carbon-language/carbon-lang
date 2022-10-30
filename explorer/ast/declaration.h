// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_DECLARATION_H_
#define CARBON_EXPLORER_AST_DECLARATION_H_

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/ast/ast_node.h"
#include "explorer/ast/impl_binding.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/return_term.h"
#include "explorer/ast/statement.h"
#include "explorer/ast/static_scope.h"
#include "explorer/ast/value_category.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class MixinPseudoType;
class ConstraintType;

// Abstract base class of all AST nodes representing patterns.
//
// Declaration and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Declaration must provide a `classof` operation, and
// every concrete derived class must have a corresponding enumerator
// in `Kind`; see https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html for
// details.
class Declaration : public AstNode {
 public:
  ~Declaration() override = 0;

  Declaration(const Declaration&) = delete;
  auto operator=(const Declaration&) -> Declaration& = delete;

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromDeclaration(node->kind());
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> DeclarationKind {
    return static_cast<DeclarationKind>(root_kind());
  }

  // The static type of the declared entity. Cannot be called before
  // typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of the declared entity. Can only be called once,
  // during typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

  // Sets the value returned by constant_value(). Can only be called once,
  // during typechecking.
  void set_constant_value(Nonnull<const Value*> value) {
    CARBON_CHECK(!constant_value_.has_value());
    constant_value_ = value;
  }

  // See static_scope.h for API.
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return constant_value_;
  }

  // See static_scope.h for API.
  auto symbolic_identity() const -> std::optional<Nonnull<const Value*>> {
    return constant_value_;
  }

  // Returns whether this node has been declared.
  auto is_declared() const -> bool { return is_declared_; }

  // Set that this node is declared. Should only be called once, by the
  // type-checker, once the node is ready to be named and used.
  void set_is_declared() {
    CARBON_CHECK(!is_declared_) << "should not be declared twice";
    is_declared_ = true;
  }

  // Returns whether this node has been fully type-checked.
  auto is_type_checked() const -> bool { return is_type_checked_; }

  // Set that this node is type-checked. Should only be called once, by the
  // type-checker, once full type-checking is complete.
  void set_is_type_checked() {
    CARBON_CHECK(!is_type_checked_) << "should not be type-checked twice";
    is_type_checked_ = true;
  }

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}

 private:
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const Value*>> constant_value_;
  bool is_declared_ = false;
  bool is_type_checked_ = false;
};

class CallableDeclaration : public Declaration {
 public:
  CallableDeclaration(AstNodeKind kind, SourceLocation loc, std::string name,
                      std::vector<Nonnull<GenericBinding*>> deduced_params,
                      std::optional<Nonnull<Pattern*>> me_pattern,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body)
      : Declaration(kind, loc),
        name_(std::move(name)),
        deduced_parameters_(std::move(deduced_params)),
        me_pattern_(me_pattern),
        param_pattern_(param_pattern),
        return_term_(return_term),
        body_(body) {}

  void PrintDepth(int depth, llvm::raw_ostream& out) const;

  // TODO: Move name() and name_ to FunctionDeclaration
  auto name() const -> const std::string& { return name_; }
  auto deduced_parameters() const
      -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_parameters_;
  }
  auto deduced_parameters() -> llvm::ArrayRef<Nonnull<GenericBinding*>> {
    return deduced_parameters_;
  }
  auto me_pattern() const -> const Pattern& { return **me_pattern_; }
  auto me_pattern() -> Pattern& { return **me_pattern_; }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_term() const -> const ReturnTerm& { return return_term_; }
  auto return_term() -> ReturnTerm& { return return_term_; }
  auto body() const -> std::optional<Nonnull<const Block*>> { return body_; }
  auto body() -> std::optional<Nonnull<Block*>> { return body_; }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

  auto is_method() const -> bool { return me_pattern_.has_value(); }

 private:
  std::string name_;
  std::vector<Nonnull<GenericBinding*>> deduced_parameters_;
  std::optional<Nonnull<Pattern*>> me_pattern_;
  Nonnull<TuplePattern*> param_pattern_;
  ReturnTerm return_term_;
  std::optional<Nonnull<Block*>> body_;
};

class FunctionDeclaration : public CallableDeclaration {
 public:
  using ImplementsCarbonValueNode = void;

  static auto Create(Nonnull<Arena*> arena, SourceLocation source_loc,
                     std::string name,
                     std::vector<Nonnull<AstNode*>> deduced_params,
                     Nonnull<TuplePattern*> param_pattern,
                     ReturnTerm return_term,
                     std::optional<Nonnull<Block*>> body)
      -> ErrorOr<Nonnull<FunctionDeclaration*>>;

  // Use `Create()` instead. This is public only so Arena::New() can call it.
  FunctionDeclaration(SourceLocation source_loc, std::string name,
                      std::vector<Nonnull<GenericBinding*>> deduced_params,
                      std::optional<Nonnull<Pattern*>> me_pattern,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body)
      : CallableDeclaration(AstNodeKind::FunctionDeclaration, source_loc,
                            std::move(name), std::move(deduced_params),
                            me_pattern, param_pattern, return_term, body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFunctionDeclaration(node->kind());
  }
};

class DestructorDeclaration : public CallableDeclaration {
 public:
  using ImplementsCarbonValueNode = void;

  static auto CreateDestructor(Nonnull<Arena*> arena, SourceLocation source_loc,
                               std::vector<Nonnull<AstNode*>> deduced_params,
                               Nonnull<TuplePattern*> param_pattern,
                               ReturnTerm return_term,
                               std::optional<Nonnull<Block*>> body)
      -> ErrorOr<Nonnull<DestructorDeclaration*>>;

  // Use `Create()` instead. This is public only so Arena::New() can call it.
  DestructorDeclaration(SourceLocation source_loc,
                        std::vector<Nonnull<GenericBinding*>> deduced_params,
                        std::optional<Nonnull<Pattern*>> me_pattern,
                        Nonnull<TuplePattern*> param_pattern,
                        ReturnTerm return_term,
                        std::optional<Nonnull<Block*>> body)
      : CallableDeclaration(AstNodeKind::DestructorDeclaration, source_loc,
                            "destructor", std::move(deduced_params), me_pattern,
                            param_pattern, return_term, body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromDestructorDeclaration(node->kind());
  }
};

class SelfDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  explicit SelfDeclaration(SourceLocation source_loc)
      : Declaration(AstNodeKind::SelfDeclaration, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromSelfDeclaration(node->kind());
  }

  static auto name() -> std::string_view { return "Self"; }
  auto value_category() const -> ValueCategory { return ValueCategory::Let; }
};

enum class ClassExtensibility { None, Base, Abstract };

class ClassDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ClassDeclaration(SourceLocation source_loc, std::string name,
                   Nonnull<SelfDeclaration*> self_decl,
                   ClassExtensibility extensibility,
                   std::optional<Nonnull<TuplePattern*>> type_params,
                   std::optional<Nonnull<Expression*>> base,
                   std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::ClassDeclaration, source_loc),
        name_(std::move(name)),
        extensibility_(extensibility),
        self_decl_(self_decl),
        type_params_(type_params),
        base_expr_(base),
        members_(std::move(members)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromClassDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto extensibility() const -> ClassExtensibility { return extensibility_; }
  auto type_params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return type_params_;
  }
  auto type_params() -> std::optional<Nonnull<TuplePattern*>> {
    return type_params_;
  }
  auto base_expr() const -> std::optional<Nonnull<Expression*>> {
    return base_expr_;
  }
  auto self() const -> Nonnull<const SelfDeclaration*> { return self_decl_; }
  auto self() -> Nonnull<SelfDeclaration*> { return self_decl_; }

  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }
  auto destructor() const -> std::optional<Nonnull<DestructorDeclaration*>> {
    for (const auto& x : members_) {
      if (x->kind() == DeclarationKind::DestructorDeclaration) {
        return llvm::cast<DestructorDeclaration>(x);
      }
    }
    return std::nullopt;
  }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

  auto base() const -> std::optional<Nonnull<const ClassDeclaration*>> {
    return base_;
  }
  void set_base(Nonnull<const ClassDeclaration*> base_decl) {
    base_ = base_decl;
  }

 private:
  std::string name_;
  ClassExtensibility extensibility_;
  Nonnull<SelfDeclaration*> self_decl_;
  std::optional<Nonnull<TuplePattern*>> type_params_;
  std::optional<Nonnull<Expression*>> base_expr_;
  std::vector<Nonnull<Declaration*>> members_;
  std::optional<Nonnull<FunctionDeclaration*>> destructor_;
  std::optional<Nonnull<const ClassDeclaration*>> base_;
};

// EXPERIMENTAL MIXIN FEATURE
class MixinDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  MixinDeclaration(SourceLocation source_loc, std::string name,
                   std::optional<Nonnull<TuplePattern*>> params,
                   Nonnull<GenericBinding*> self,
                   std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::MixinDeclaration, source_loc),
        name_(std::move(name)),
        params_(params),
        self_(self),
        members_(std::move(members)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMixinDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return params_;
  }
  auto params() -> std::optional<Nonnull<TuplePattern*>> { return params_; }
  auto self() const -> Nonnull<const GenericBinding*> { return self_; }
  auto self() -> Nonnull<GenericBinding*> { return self_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  std::string name_;
  std::optional<Nonnull<TuplePattern*>> params_;
  Nonnull<GenericBinding*> self_;
  std::vector<Nonnull<Declaration*>> members_;
};

// EXPERIMENTAL MIXIN FEATURE
class MixDeclaration : public Declaration {
 public:
  MixDeclaration(SourceLocation source_loc,
                 std::optional<Nonnull<Expression*>> mixin_type)
      : Declaration(AstNodeKind::MixDeclaration, source_loc),
        mixin_(mixin_type) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMixDeclaration(node->kind());
  }

  auto mixin() const -> const Expression& { return **mixin_; }
  auto mixin() -> Expression& { return **mixin_; }

  auto mixin_value() const -> const MixinPseudoType& { return *mixin_value_; }
  void set_mixin_value(Nonnull<const MixinPseudoType*> mixin_value) {
    mixin_value_ = mixin_value;
  }

 private:
  std::optional<Nonnull<Expression*>> mixin_;
  Nonnull<const MixinPseudoType*> mixin_value_;
};

class AlternativeSignature : public AstNode {
 public:
  AlternativeSignature(SourceLocation source_loc, std::string name,
                       Nonnull<TupleLiteral*> signature)
      : AstNode(AstNodeKind::AlternativeSignature, source_loc),
        name_(std::move(name)),
        signature_(signature) {}

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAlternativeSignature(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto signature() const -> const TupleLiteral& { return *signature_; }
  auto signature() -> TupleLiteral& { return *signature_; }

 private:
  std::string name_;
  Nonnull<TupleLiteral*> signature_;
};

class ChoiceDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ChoiceDeclaration(SourceLocation source_loc, std::string name,
                    std::optional<Nonnull<TuplePattern*>> type_params,
                    std::vector<Nonnull<AlternativeSignature*>> alternatives)
      : Declaration(AstNodeKind::ChoiceDeclaration, source_loc),
        name_(std::move(name)),
        type_params_(type_params),
        alternatives_(std::move(alternatives)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromChoiceDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }

  auto type_params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return type_params_;
  }
  auto type_params() -> std::optional<Nonnull<TuplePattern*>> {
    return type_params_;
  }

  auto alternatives() const
      -> llvm::ArrayRef<Nonnull<const AlternativeSignature*>> {
    return alternatives_;
  }
  auto alternatives() -> llvm::ArrayRef<Nonnull<AlternativeSignature*>> {
    return alternatives_;
  }

  void set_members(const std::vector<NamedValue>& members) {
    members_ = members;
  }
  auto members() const -> std::vector<NamedValue> { return members_; }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  std::string name_;
  std::optional<Nonnull<TuplePattern*>> type_params_;
  std::vector<Nonnull<AlternativeSignature*>> alternatives_;
  std::vector<NamedValue> members_;
};

// Global variable definition implements the Declaration concept.
class VariableDeclaration : public Declaration {
 public:
  VariableDeclaration(SourceLocation source_loc,
                      Nonnull<BindingPattern*> binding,
                      std::optional<Nonnull<Expression*>> initializer,
                      ValueCategory value_category)
      : Declaration(AstNodeKind::VariableDeclaration, source_loc),
        binding_(binding),
        initializer_(initializer),
        value_category_(value_category) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromVariableDeclaration(node->kind());
  }

  auto binding() const -> const BindingPattern& { return *binding_; }
  auto binding() -> BindingPattern& { return *binding_; }
  auto initializer() const -> const Expression& { return **initializer_; }
  auto initializer() -> Expression& { return **initializer_; }
  auto value_category() const -> ValueCategory { return value_category_; }

  auto has_initializer() const -> bool { return initializer_.has_value(); }

  // Can only be called by type-checking, if a conversion was required.
  void set_initializer(Nonnull<Expression*> initializer) {
    CARBON_CHECK(has_initializer()) << "should not add a new initializer";
    initializer_ = initializer;
  }

 private:
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  Nonnull<BindingPattern*> binding_;
  std::optional<Nonnull<Expression*>> initializer_;
  ValueCategory value_category_;
};

class InterfaceDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  InterfaceDeclaration(Nonnull<Arena*> arena, SourceLocation source_loc,
                       std::string name,
                       std::optional<Nonnull<TuplePattern*>> params,
                       std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::InterfaceDeclaration, source_loc),
        name_(std::move(name)),
        params_(params),
        self_type_(arena->New<SelfDeclaration>(source_loc)),
        members_(std::move(members)) {
    // `interface X` has `Self:! X`.
    auto* self_type_ref = arena->New<IdentifierExpression>(source_loc, name);
    self_type_ref->set_value_node(self_type_);
    self_ = arena->New<GenericBinding>(source_loc, "Self", self_type_ref);
  }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return params_;
  }
  auto params() -> std::optional<Nonnull<TuplePattern*>> { return params_; }
  // Get the type of `Self`, which is a reference to the interface itself, with
  // parameters mapped to their values. For example, in `interface X(T:!
  // Type)`, the self type is `X(T)`.
  auto self_type() const -> Nonnull<const SelfDeclaration*> {
    return self_type_;
  }
  auto self_type() -> Nonnull<SelfDeclaration*> { return self_type_; }
  auto self() const -> Nonnull<const GenericBinding*> { return self_; }
  auto self() -> Nonnull<GenericBinding*> { return self_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

  // Get the constraint type corresponding to this interface, or nullopt if
  // this interface is incomplete.
  auto constraint_type() const
      -> std::optional<Nonnull<const ConstraintType*>> {
    return constraint_type_;
  }

  // Set the constraint type corresponding to this interface. Can only be set
  // once, by type-checking.
  void set_constraint_type(Nonnull<const ConstraintType*> constraint_type) {
    CARBON_CHECK(!constraint_type_);
    constraint_type_ = constraint_type;
  }

 private:
  std::string name_;
  std::optional<Nonnull<TuplePattern*>> params_;
  Nonnull<SelfDeclaration*> self_type_;
  Nonnull<GenericBinding*> self_;
  std::vector<Nonnull<Declaration*>> members_;
  std::optional<Nonnull<const ConstraintType*>> constraint_type_;
};

// An `extends` declaration in an interface.
class InterfaceExtendsDeclaration : public Declaration {
 public:
  InterfaceExtendsDeclaration(SourceLocation source_loc,
                              Nonnull<Expression*> base)
      : Declaration(AstNodeKind::InterfaceExtendsDeclaration, source_loc),
        base_(base) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceExtendsDeclaration(node->kind());
  }

  auto base() const -> const Expression* { return base_; }
  auto base() -> Expression* { return base_; }

 private:
  Nonnull<Expression*> base_;
};

// An `impl ... as` declaration in an interface.
class InterfaceImplDeclaration : public Declaration {
 public:
  InterfaceImplDeclaration(SourceLocation source_loc,
                           Nonnull<Expression*> impl_type,
                           Nonnull<Expression*> constraint)
      : Declaration(AstNodeKind::InterfaceImplDeclaration, source_loc),
        impl_type_(impl_type),
        constraint_(constraint) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceImplDeclaration(node->kind());
  }

  auto impl_type() const -> const Expression* { return impl_type_; }
  auto impl_type() -> Expression* { return impl_type_; }

  auto constraint() const -> const Expression* { return constraint_; }
  auto constraint() -> Expression* { return constraint_; }

 private:
  Nonnull<Expression*> impl_type_;
  Nonnull<Expression*> constraint_;
};

class AssociatedConstantDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  AssociatedConstantDeclaration(SourceLocation source_loc,
                                Nonnull<GenericBinding*> binding)
      : Declaration(AstNodeKind::AssociatedConstantDeclaration, source_loc),
        binding_(binding) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAssociatedConstantDeclaration(node->kind());
  }

  auto binding() const -> const GenericBinding& { return *binding_; }
  auto binding() -> GenericBinding& { return *binding_; }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  Nonnull<GenericBinding*> binding_;
};

enum class ImplKind { InternalImpl, ExternalImpl };

class ImplDeclaration : public Declaration {
 public:
  static auto Create(Nonnull<Arena*> arena, SourceLocation source_loc,
                     ImplKind kind, Nonnull<Expression*> impl_type,
                     Nonnull<Expression*> interface,
                     std::vector<Nonnull<AstNode*>> deduced_params,
                     std::vector<Nonnull<Declaration*>> members)
      -> ErrorOr<Nonnull<ImplDeclaration*>>;

  // Use `Create` instead.
  ImplDeclaration(SourceLocation source_loc, ImplKind kind,
                  Nonnull<Expression*> impl_type,
                  Nonnull<SelfDeclaration*> self_decl,
                  Nonnull<Expression*> interface,
                  std::vector<Nonnull<GenericBinding*>> deduced_params,
                  std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::ImplDeclaration, source_loc),
        kind_(kind),
        impl_type_(impl_type),
        self_decl_(self_decl),
        interface_(interface),
        deduced_parameters_(std::move(deduced_params)),
        members_(std::move(members)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromImplDeclaration(node->kind());
  }
  // Return whether this is an external or internal impl.
  auto kind() const -> ImplKind { return kind_; }
  // Return the type that is doing the implementing.
  auto impl_type() const -> Nonnull<Expression*> { return impl_type_; }
  // Return the interface that is being implemented.
  auto interface() const -> const Expression& { return *interface_; }
  auto interface() -> Expression& { return *interface_; }
  void set_constraint_type(Nonnull<const ConstraintType*> constraint_type) {
    constraint_type_ = constraint_type;
  }
  auto constraint_type() const -> Nonnull<const ConstraintType*> {
    return *constraint_type_;
  }
  // Returns the deduced parameters specified on the impl declaration. This
  // does not include any generic parameters from enclosing scopes.
  auto deduced_parameters() const
      -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_parameters_;
  }
  auto deduced_parameters() -> llvm::ArrayRef<Nonnull<GenericBinding*>> {
    return deduced_parameters_;
  }
  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }
  auto value_category() const -> ValueCategory { return ValueCategory::Let; }
  void set_impl_bindings(llvm::ArrayRef<Nonnull<const ImplBinding*>> imps) {
    impl_bindings_ = imps;
  }
  auto impl_bindings() const -> llvm::ArrayRef<Nonnull<const ImplBinding*>> {
    return impl_bindings_;
  }
  auto self() const -> Nonnull<const SelfDeclaration*> { return self_decl_; }
  auto self() -> Nonnull<SelfDeclaration*> { return self_decl_; }

 private:
  ImplKind kind_;
  Nonnull<Expression*> impl_type_;
  Nonnull<SelfDeclaration*> self_decl_;
  Nonnull<Expression*> interface_;
  std::optional<Nonnull<const ConstraintType*>> constraint_type_;
  std::vector<Nonnull<GenericBinding*>> deduced_parameters_;
  std::vector<Nonnull<Declaration*>> members_;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings_;
};

class AliasDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  explicit AliasDeclaration(SourceLocation source_loc, std::string name,
                            Nonnull<Expression*> target)
      : Declaration(AstNodeKind::AliasDeclaration, source_loc),
        name_(std::move(name)),
        target_(target) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAliasDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto target() const -> const Expression& { return *target_; }
  auto target() -> Expression& { return *target_; }
  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  std::string name_;
  Nonnull<Expression*> target_;
};

// Return the name of a declaration, if it has one.
auto GetName(const Declaration&) -> std::optional<std::string_view>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_DECLARATION_H_
