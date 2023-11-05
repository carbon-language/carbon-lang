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
#include "explorer/ast/clone_context.h"
#include "explorer/ast/expression_category.h"
#include "explorer/ast/impl_binding.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/return_term.h"
#include "explorer/ast/statement.h"
#include "explorer/ast/value_node.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class MixinPseudoType;
class ConstraintType;
class NominalClassType;
class MatchFirstDeclaration;

// Abstract base class of all AST nodes representing declarations.
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

  virtual void PrintIndent(int indent_num_spaces, llvm::raw_ostream& out) const;

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

  // See value_node.h for API.
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return constant_value_;
  }

  // See value_node.h for API.
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
  explicit Declaration(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}

  explicit Declaration(CloneContext& context, const Declaration& other)
      : AstNode(context, other),
        static_type_(context.Clone(other.static_type_)),
        constant_value_(context.Clone(other.constant_value_)),
        is_declared_(other.is_declared_),
        is_type_checked_(other.is_type_checked_) {}

 private:
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const Value*>> constant_value_;
  bool is_declared_ = false;
  bool is_type_checked_ = false;
};

// Determine whether two declarations declare the same entity.
inline auto DeclaresSameEntity(const Declaration& first,
                               const Declaration& second) -> bool {
  return &first == &second;
}

// A name being declared in a named declaration.
class DeclaredName : public Printable<DeclaredName> {
 public:
  struct NameComponent {
    SourceLocation source_loc;
    std::string name;
  };

  explicit DeclaredName(SourceLocation loc, std::string name)
      : components_{{loc, std::move(name)}} {}

  void Print(llvm::raw_ostream& out) const;

  void Append(SourceLocation loc, std::string name) {
    components_.push_back({loc, std::move(name)});
  }

  // Returns the location of the first name component.
  auto source_loc() const -> SourceLocation {
    return components_.front().source_loc;
  }

  // Returns whether this is a qualified name, as opposed to a simple
  // single-identifier name.
  auto is_qualified() const { return components_.size() > 1; }

  // Returns a range containing the components of the name other than the final
  // component.
  auto qualifiers() const -> llvm::ArrayRef<NameComponent> {
    return llvm::ArrayRef(components_).drop_back();
  }

  // Returns the innermost name, which is the unqualified name of the entity
  // being declared. For example in `fn Namespace.Func();`, returns `"Func"`.
  auto inner_name() const -> std::string_view {
    return components_.back().name;
  }

 private:
  std::vector<NameComponent> components_;
};

// A declaration of a namespace.
class NamespaceDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  explicit NamespaceDeclaration(SourceLocation source_loc, DeclaredName name)
      : Declaration(AstNodeKind::NamespaceDeclaration, source_loc),
        name_(std::move(name)) {}

  explicit NamespaceDeclaration(CloneContext& context,
                                const NamespaceDeclaration& other)
      : Declaration(context, other), name_(other.name_) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromNamespaceDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }
  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

 private:
  DeclaredName name_;
};

// A function's virtual override keyword.
enum class VirtualOverride { None, Abstract, Virtual, Impl };

class CallableDeclaration : public Declaration {
 public:
  CallableDeclaration(AstNodeKind kind, SourceLocation loc,
                      std::vector<Nonnull<GenericBinding*>> deduced_params,
                      std::optional<Nonnull<Pattern*>> self_pattern,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body,
                      VirtualOverride virt_override)
      : Declaration(kind, loc),
        deduced_parameters_(std::move(deduced_params)),
        self_pattern_(self_pattern),
        param_pattern_(param_pattern),
        return_term_(return_term),
        body_(body),
        virt_override_(virt_override) {}

  explicit CallableDeclaration(CloneContext& context,
                               const CallableDeclaration& other)
      : Declaration(context, other),
        deduced_parameters_(context.Clone(other.deduced_parameters_)),
        self_pattern_(context.Clone(other.self_pattern_)),
        param_pattern_(context.Clone(other.param_pattern_)),
        return_term_(context.Clone(other.return_term_)),
        body_(context.Clone(other.body_)),
        virt_override_(other.virt_override_) {}

  void PrintIndent(int indent_num_spaces,
                   llvm::raw_ostream& out) const override;
  auto deduced_parameters() const
      -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_parameters_;
  }
  auto deduced_parameters() -> llvm::ArrayRef<Nonnull<GenericBinding*>> {
    return deduced_parameters_;
  }
  auto self_pattern() const -> const Pattern& { return **self_pattern_; }
  auto self_pattern() -> Pattern& { return **self_pattern_; }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_term() const -> const ReturnTerm& { return return_term_; }
  auto return_term() -> ReturnTerm& { return return_term_; }
  auto body() const -> std::optional<Nonnull<const Block*>> { return body_; }
  auto body() -> std::optional<Nonnull<Block*>> { return body_; }
  auto virt_override() const -> VirtualOverride { return virt_override_; }

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

  auto is_method() const -> bool { return self_pattern_.has_value(); }

 private:
  std::vector<Nonnull<GenericBinding*>> deduced_parameters_;
  std::optional<Nonnull<Pattern*>> self_pattern_;
  Nonnull<TuplePattern*> param_pattern_;
  ReturnTerm return_term_;
  std::optional<Nonnull<Block*>> body_;
  VirtualOverride virt_override_;
};

class FunctionDeclaration : public CallableDeclaration {
 public:
  using ImplementsCarbonValueNode = void;

  static auto Create(Nonnull<Arena*> arena, SourceLocation source_loc,
                     DeclaredName name,
                     std::vector<Nonnull<AstNode*>> deduced_params,
                     Nonnull<TuplePattern*> param_pattern,
                     ReturnTerm return_term,
                     std::optional<Nonnull<Block*>> body,
                     VirtualOverride virt_override)
      -> ErrorOr<Nonnull<FunctionDeclaration*>>;

  // Use `Create()` instead. This is public only so Arena::New() can call it.
  FunctionDeclaration(SourceLocation source_loc, DeclaredName name,
                      std::vector<Nonnull<GenericBinding*>> deduced_params,
                      std::optional<Nonnull<Pattern*>> self_pattern,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body,
                      VirtualOverride virt_override)
      : CallableDeclaration(AstNodeKind::FunctionDeclaration, source_loc,
                            std::move(deduced_params), self_pattern,
                            param_pattern, return_term, body, virt_override),
        name_(std::move(name)) {}

  explicit FunctionDeclaration(CloneContext& context,
                               const FunctionDeclaration& other)
      : CallableDeclaration(context, other), name_(other.name_) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFunctionDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }

 private:
  DeclaredName name_;
};

class DestructorDeclaration : public CallableDeclaration {
 public:
  using ImplementsCarbonValueNode = void;

  static auto CreateDestructor(Nonnull<Arena*> arena, SourceLocation source_loc,
                               std::vector<Nonnull<AstNode*>> deduced_params,
                               Nonnull<TuplePattern*> param_pattern,
                               ReturnTerm return_term,
                               std::optional<Nonnull<Block*>> body,
                               VirtualOverride virt_override)
      -> ErrorOr<Nonnull<DestructorDeclaration*>>;

  // Use `Create()` instead. This is public only so Arena::New() can call it.
  DestructorDeclaration(SourceLocation source_loc,
                        std::vector<Nonnull<GenericBinding*>> deduced_params,
                        std::optional<Nonnull<Pattern*>> self_pattern,
                        Nonnull<TuplePattern*> param_pattern,
                        ReturnTerm return_term,
                        std::optional<Nonnull<Block*>> body,
                        VirtualOverride virt_override)
      : CallableDeclaration(AstNodeKind::DestructorDeclaration, source_loc,
                            std::move(deduced_params), self_pattern,
                            param_pattern, return_term, body, virt_override) {}

  explicit DestructorDeclaration(CloneContext& context,
                                 const DestructorDeclaration& other)
      : CallableDeclaration(context, other) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromDestructorDeclaration(node->kind());
  }
};

class SelfDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  explicit SelfDeclaration(SourceLocation source_loc)
      : Declaration(AstNodeKind::SelfDeclaration, source_loc) {}

  explicit SelfDeclaration(CloneContext& context, const SelfDeclaration& other)
      : Declaration(context, other) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromSelfDeclaration(node->kind());
  }

  static auto name() -> std::string_view { return "Self"; }
  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }
};

enum class ClassExtensibility { None, Base, Abstract };

class ClassDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ClassDeclaration(SourceLocation source_loc, DeclaredName name,
                   Nonnull<SelfDeclaration*> self_decl,
                   ClassExtensibility extensibility,
                   std::optional<Nonnull<TuplePattern*>> type_params,
                   std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::ClassDeclaration, source_loc),
        name_(std::move(name)),
        extensibility_(extensibility),
        self_decl_(self_decl),
        type_params_(type_params),
        members_(std::move(members)) {}

  explicit ClassDeclaration(CloneContext& context,
                            const ClassDeclaration& other);

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromClassDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }
  auto extensibility() const -> ClassExtensibility { return extensibility_; }
  auto type_params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return type_params_;
  }
  auto type_params() -> std::optional<Nonnull<TuplePattern*>> {
    return type_params_;
  }
  auto self() const -> Nonnull<const SelfDeclaration*> { return self_decl_; }
  auto self() -> Nonnull<SelfDeclaration*> { return self_decl_; }

  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }
  auto destructor() const
      -> std::optional<Nonnull<const DestructorDeclaration*>> {
    for (const auto& x : members_) {
      if (x->kind() == DeclarationKind::DestructorDeclaration) {
        return llvm::cast<DestructorDeclaration>(x);
      }
    }
    return std::nullopt;
  }

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

  // Returns the original base type, before instantiation & substitutions
  // Use `NominalClassType::base()` to get the instantiated type.
  auto base_type() const -> std::optional<Nonnull<const NominalClassType*>> {
    return base_type_;
  }
  void set_base_type(
      std::optional<Nonnull<const NominalClassType*>> base_type) {
    base_type_ = base_type;
  }

 private:
  DeclaredName name_;
  ClassExtensibility extensibility_;
  Nonnull<SelfDeclaration*> self_decl_;
  std::optional<Nonnull<TuplePattern*>> type_params_;
  std::vector<Nonnull<Declaration*>> members_;
  std::optional<Nonnull<const NominalClassType*>> base_type_;
};

// EXPERIMENTAL MIXIN FEATURE
class MixinDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  MixinDeclaration(SourceLocation source_loc, DeclaredName name,
                   std::optional<Nonnull<TuplePattern*>> params,
                   Nonnull<GenericBinding*> self,
                   std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::MixinDeclaration, source_loc),
        name_(std::move(name)),
        params_(params),
        self_(self),
        members_(std::move(members)) {}

  explicit MixinDeclaration(CloneContext& context,
                            const MixinDeclaration& other)
      : Declaration(context, other),
        name_(other.name_),
        params_(context.Clone(other.params_)),
        self_(context.Clone(other.self_)),
        members_(context.Clone(other.members_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMixinDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }
  auto params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return params_;
  }
  auto params() -> std::optional<Nonnull<TuplePattern*>> { return params_; }
  auto self() const -> Nonnull<const GenericBinding*> { return self_; }
  auto self() -> Nonnull<GenericBinding*> { return self_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

 private:
  DeclaredName name_;
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

  explicit MixDeclaration(CloneContext& context, const MixDeclaration& other);

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMixDeclaration(node->kind());
  }

  auto mixin() const -> const Expression& { return **mixin_; }
  auto mixin() -> Expression& { return **mixin_; }

  auto mixin_value() const -> const MixinPseudoType& { return **mixin_value_; }
  void set_mixin_value(Nonnull<const MixinPseudoType*> mixin_value) {
    mixin_value_ = mixin_value;
  }

 private:
  std::optional<Nonnull<Expression*>> mixin_;
  std::optional<Nonnull<const MixinPseudoType*>> mixin_value_;
};

class ExtendBaseDeclaration : public Declaration {
 public:
  ExtendBaseDeclaration(SourceLocation source_loc,
                        Nonnull<Expression*> base_class)
      : Declaration(AstNodeKind::ExtendBaseDeclaration, source_loc),
        base_class_(base_class) {}

  explicit ExtendBaseDeclaration(CloneContext& context,
                                 const ExtendBaseDeclaration& other);

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromExtendBaseDeclaration(node->kind());
  }

  auto base_class() const -> Nonnull<const Expression*> { return base_class_; }
  auto base_class() -> Nonnull<Expression*> { return base_class_; }

 private:
  Nonnull<Expression*> base_class_;
};

class AlternativeSignature : public AstNode {
 public:
  AlternativeSignature(SourceLocation source_loc, std::string name,
                       std::optional<Nonnull<TupleLiteral*>> parameters)
      : AstNode(AstNodeKind::AlternativeSignature, source_loc),
        name_(std::move(name)),
        parameters_(parameters) {}

  explicit AlternativeSignature(CloneContext& context,
                                const AlternativeSignature& other)
      : AstNode(context, other),
        name_(other.name_),
        parameters_(context.Clone(other.parameters_)),
        parameters_static_type_(context.Clone(other.parameters_static_type_)) {}

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAlternativeSignature(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto parameters() const -> std::optional<Nonnull<const TupleLiteral*>> {
    return parameters_;
  }
  auto parameters() -> std::optional<Nonnull<TupleLiteral*>> {
    return parameters_;
  }

  // The static type described by the parameters expression, if any. Cannot be
  // called before type checking. This will be nullopt after type checking if
  // this alternative does not have a parameter list.
  auto parameters_static_type() const -> std::optional<Nonnull<const Value*>> {
    return parameters_static_type_;
  }

  // Sets the static type of the declared entity. Can only be called once,
  // during typechecking.
  void set_parameters_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!parameters_static_type_.has_value());
    parameters_static_type_ = type;
  }

 private:
  std::string name_;
  std::optional<Nonnull<TupleLiteral*>> parameters_;
  std::optional<Nonnull<const Value*>> parameters_static_type_;
};

class ChoiceDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ChoiceDeclaration(SourceLocation source_loc, DeclaredName name,
                    std::optional<Nonnull<TuplePattern*>> type_params,
                    std::vector<Nonnull<AlternativeSignature*>> alternatives)
      : Declaration(AstNodeKind::ChoiceDeclaration, source_loc),
        name_(std::move(name)),
        type_params_(type_params),
        alternatives_(std::move(alternatives)) {}

  explicit ChoiceDeclaration(CloneContext& context,
                             const ChoiceDeclaration& other)
      : Declaration(context, other),
        name_(other.name_),
        type_params_(context.Clone(other.type_params_)),
        alternatives_(context.Clone(other.alternatives_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromChoiceDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }

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

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

  auto FindAlternative(std::string_view name) const
      -> std::optional<const AlternativeSignature*>;

 private:
  DeclaredName name_;
  std::optional<Nonnull<TuplePattern*>> type_params_;
  std::vector<Nonnull<AlternativeSignature*>> alternatives_;
};

// Global variable definition implements the Declaration concept.
class VariableDeclaration : public Declaration {
 public:
  VariableDeclaration(SourceLocation source_loc,
                      Nonnull<BindingPattern*> binding,
                      std::optional<Nonnull<Expression*>> initializer,
                      ExpressionCategory expression_category)
      : Declaration(AstNodeKind::VariableDeclaration, source_loc),
        binding_(binding),
        initializer_(initializer),
        expression_category_(expression_category) {}

  explicit VariableDeclaration(CloneContext& context,
                               const VariableDeclaration& other)
      : Declaration(context, other),
        binding_(context.Clone(other.binding_)),
        initializer_(context.Clone(other.initializer_)),
        expression_category_(other.expression_category_) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromVariableDeclaration(node->kind());
  }

  auto binding() const -> const BindingPattern& { return *binding_; }
  auto binding() -> BindingPattern& { return *binding_; }
  auto initializer() const -> const Expression& { return **initializer_; }
  auto initializer() -> Expression& { return **initializer_; }
  auto expression_category() const -> ExpressionCategory {
    return expression_category_;
  }

  auto has_initializer() const -> bool { return initializer_.has_value(); }

  // Can only be called by type-checking, if a conversion was required.
  void set_initializer(Nonnull<Expression*> initializer) {
    CARBON_CHECK(has_initializer()) << "should not add a new initializer";
    initializer_ = initializer;
  }

 private:
  Nonnull<BindingPattern*> binding_;
  std::optional<Nonnull<Expression*>> initializer_;
  ExpressionCategory expression_category_;
};

// Base class for constraint and interface declarations. Interfaces and named
// constraints behave the same in most respects, but only interfaces can
// introduce new associated functions and constants.
class ConstraintTypeDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ConstraintTypeDeclaration(AstNodeKind kind, Nonnull<Arena*> arena,
                            SourceLocation source_loc, DeclaredName name,
                            std::optional<Nonnull<TuplePattern*>> params,
                            std::vector<Nonnull<Declaration*>> members)
      : Declaration(kind, source_loc),
        name_(std::move(name)),
        params_(params),
        self_type_(arena->New<SelfDeclaration>(source_loc)),
        members_(std::move(members)) {
    // `interface X` has `Self:! X`.
    auto* self_type_ref = arena->New<IdentifierExpression>(
        source_loc, std::string(name_.inner_name()));
    self_type_ref->set_value_node(self_type_);
    self_ = arena->New<GenericBinding>(source_loc, "Self", self_type_ref,
                                       GenericBinding::BindingKind::Checked);
  }

  explicit ConstraintTypeDeclaration(CloneContext& context,
                                     const ConstraintTypeDeclaration& other);

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromConstraintTypeDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }
  auto params() const -> std::optional<Nonnull<const TuplePattern*>> {
    return params_;
  }
  auto params() -> std::optional<Nonnull<TuplePattern*>> { return params_; }
  // Get the type of `Self`, which is a reference to the interface itself, with
  // parameters mapped to their values. For example, in `interface X(T:!
  // type)`, the self type is `X(T)`.
  auto self_type() const -> Nonnull<const SelfDeclaration*> {
    return self_type_;
  }
  auto self_type() -> Nonnull<SelfDeclaration*> { return self_type_; }
  auto self() const -> Nonnull<const GenericBinding*> { return self_; }
  auto self() -> Nonnull<GenericBinding*> { return self_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

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
  DeclaredName name_;
  std::optional<Nonnull<TuplePattern*>> params_;
  Nonnull<SelfDeclaration*> self_type_;
  Nonnull<GenericBinding*> self_;
  std::vector<Nonnull<Declaration*>> members_;
  std::optional<Nonnull<const ConstraintType*>> constraint_type_;
};

// A `interface` declaration.
class InterfaceDeclaration : public ConstraintTypeDeclaration {
 public:
  using ImplementsCarbonValueNode = void;

  InterfaceDeclaration(Nonnull<Arena*> arena, SourceLocation source_loc,
                       DeclaredName name,
                       std::optional<Nonnull<TuplePattern*>> params,
                       std::vector<Nonnull<Declaration*>> members)
      : ConstraintTypeDeclaration(AstNodeKind::InterfaceDeclaration, arena,
                                  source_loc, std::move(name), params,
                                  std::move(members)) {}

  explicit InterfaceDeclaration(CloneContext& context,
                                const InterfaceDeclaration& other)
      : ConstraintTypeDeclaration(context, other) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceDeclaration(node->kind());
  }
};

// A `constraint` declaration, such as `constraint X { impl as Y; }`.
class ConstraintDeclaration : public ConstraintTypeDeclaration {
 public:
  using ImplementsCarbonValueNode = void;

  ConstraintDeclaration(Nonnull<Arena*> arena, SourceLocation source_loc,
                        DeclaredName name,
                        std::optional<Nonnull<TuplePattern*>> params,
                        std::vector<Nonnull<Declaration*>> members)
      : ConstraintTypeDeclaration(AstNodeKind::ConstraintDeclaration, arena,
                                  source_loc, std::move(name), params,
                                  std::move(members)) {}

  explicit ConstraintDeclaration(CloneContext& context,
                                 const ConstraintDeclaration& other)
      : ConstraintTypeDeclaration(context, other) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromConstraintDeclaration(node->kind());
  }
};

// An `extends` declaration in an interface.
class InterfaceExtendDeclaration : public Declaration {
 public:
  InterfaceExtendDeclaration(SourceLocation source_loc,
                             Nonnull<Expression*> base)
      : Declaration(AstNodeKind::InterfaceExtendDeclaration, source_loc),
        base_(base) {}

  explicit InterfaceExtendDeclaration(CloneContext& context,
                                      const InterfaceExtendDeclaration& other)
      : Declaration(context, other), base_(context.Clone(other.base_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceExtendDeclaration(node->kind());
  }

  auto base() const -> const Expression* { return base_; }
  auto base() -> Expression* { return base_; }

 private:
  Nonnull<Expression*> base_;
};

// A `require ... impls` declaration in an interface.
class InterfaceRequireDeclaration : public Declaration {
 public:
  InterfaceRequireDeclaration(SourceLocation source_loc,
                              Nonnull<Expression*> impl_type,
                              Nonnull<Expression*> constraint)
      : Declaration(AstNodeKind::InterfaceRequireDeclaration, source_loc),
        impl_type_(impl_type),
        constraint_(constraint) {}

  explicit InterfaceRequireDeclaration(CloneContext& context,
                                       const InterfaceRequireDeclaration& other)
      : Declaration(context, other),
        impl_type_(context.Clone(other.impl_type_)),
        constraint_(context.Clone(other.constraint_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceRequireDeclaration(node->kind());
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

  explicit AssociatedConstantDeclaration(
      CloneContext& context, const AssociatedConstantDeclaration& other)
      : Declaration(context, other), binding_(context.Clone(other.binding_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAssociatedConstantDeclaration(node->kind());
  }

  auto binding() const -> const GenericBinding& { return *binding_; }
  auto binding() -> GenericBinding& { return *binding_; }

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }

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
        deduced_parameters_(std::move(deduced_params)),
        impl_type_(impl_type),
        self_decl_(self_decl),
        interface_(interface),
        members_(std::move(members)) {}

  explicit ImplDeclaration(CloneContext& context, const ImplDeclaration& other);

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
  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }
  void set_impl_bindings(llvm::ArrayRef<Nonnull<const ImplBinding*>> imps) {
    impl_bindings_ = imps;
  }
  auto impl_bindings() const -> llvm::ArrayRef<Nonnull<const ImplBinding*>> {
    return impl_bindings_;
  }
  auto self() const -> Nonnull<const SelfDeclaration*> { return self_decl_; }
  auto self() -> Nonnull<SelfDeclaration*> { return self_decl_; }

  // Set the enclosing match_first declaration. Should only be called once,
  // during type-checking.
  void set_match_first(Nonnull<const MatchFirstDeclaration*> match_first) {
    match_first_ = match_first;
  }
  // Get the enclosing match_first declaration, if any exists.
  auto match_first() const
      -> std::optional<Nonnull<const MatchFirstDeclaration*>> {
    return match_first_;
  }

 private:
  ImplKind kind_;
  std::vector<Nonnull<GenericBinding*>> deduced_parameters_;
  Nonnull<Expression*> impl_type_;
  Nonnull<SelfDeclaration*> self_decl_;
  Nonnull<Expression*> interface_;
  std::optional<Nonnull<const ConstraintType*>> constraint_type_;
  std::vector<Nonnull<Declaration*>> members_;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings_;
  std::optional<Nonnull<const MatchFirstDeclaration*>> match_first_;
};

class MatchFirstDeclaration : public Declaration {
 public:
  MatchFirstDeclaration(
      SourceLocation source_loc,
      std::vector<Nonnull<ImplDeclaration*>> impl_declarations)
      : Declaration(AstNodeKind::MatchFirstDeclaration, source_loc),
        impl_declarations_(std::move(impl_declarations)) {}

  explicit MatchFirstDeclaration(CloneContext& context,
                                 const MatchFirstDeclaration& other)
      : Declaration(context, other),
        impl_declarations_(context.Clone(other.impl_declarations_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMatchFirstDeclaration(node->kind());
  }

  auto impl_declarations() const
      -> llvm::ArrayRef<Nonnull<const ImplDeclaration*>> {
    return impl_declarations_;
  }
  auto impl_declarations() -> llvm::ArrayRef<Nonnull<ImplDeclaration*>> {
    return impl_declarations_;
  }

 private:
  std::vector<Nonnull<ImplDeclaration*>> impl_declarations_;
};

class AliasDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  explicit AliasDeclaration(SourceLocation source_loc, DeclaredName name,
                            Nonnull<Expression*> target)
      : Declaration(AstNodeKind::AliasDeclaration, source_loc),
        name_(std::move(name)),
        target_(target) {}

  explicit AliasDeclaration(CloneContext& context,
                            const AliasDeclaration& other)
      : Declaration(context, other),
        name_(other.name_),
        target_(context.Clone(other.target_)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAliasDeclaration(node->kind());
  }

  auto name() const -> const DeclaredName& { return name_; }
  auto target() const -> const Expression& { return *target_; }
  auto target() -> Expression& { return *target_; }
  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Value;
  }
  // Sets the resolved declaration of alias target. Should only be called once,
  // during name resolution.
  void set_resolved_declaration(Nonnull<const Declaration*> decl) {
    CARBON_CHECK(!resolved_declaration_.has_value());
    resolved_declaration_ = decl;
  }
  // Get the resolved declaration of alias target, if any exists.
  auto resolved_declaration() const
      -> std::optional<Nonnull<const Declaration*>> {
    return resolved_declaration_;
  }

 private:
  DeclaredName name_;
  Nonnull<Expression*> target_;
  std::optional<Nonnull<const Declaration*>> resolved_declaration_;
};

// Return the unqualified name of a declaration, if it has one.
auto GetName(const Declaration&) -> std::optional<std::string_view>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_DECLARATION_H_
