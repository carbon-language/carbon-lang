// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <string>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/impl_binding.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/return_term.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/ast/static_scope.h"
#include "executable_semantics/ast/value_category.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/common/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

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
    CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

  // Sets the value returned by constant_value(). Can only be called once,
  // during typechecking.
  void set_constant_value(Nonnull<const Value*> value) {
    CHECK(!constant_value_.has_value());
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

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}

 private:
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const Value*>> constant_value_;
};

class FunctionDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  static auto Create(Nonnull<Arena*> arena, SourceLocation source_loc,
                     std::string name,
                     std::vector<Nonnull<AstNode*>> deduced_params,
                     std::optional<Nonnull<BindingPattern*>> me_pattern,
                     Nonnull<TuplePattern*> param_pattern,
                     ReturnTerm return_term,
                     std::optional<Nonnull<Block*>> body)
      -> ErrorOr<Nonnull<FunctionDeclaration*>>;

  // Use `Create()` instead. This is public only so Arena::New() can call it.
  FunctionDeclaration(SourceLocation source_loc, std::string name,
                      std::vector<Nonnull<GenericBinding*>> deduced_params,
                      std::optional<Nonnull<BindingPattern*>> me_pattern,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body)
      : Declaration(AstNodeKind::FunctionDeclaration, source_loc),
        name_(std::move(name)),
        deduced_parameters_(std::move(deduced_params)),
        me_pattern_(me_pattern),
        param_pattern_(param_pattern),
        return_term_(return_term),
        body_(body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFunctionDeclaration(node->kind());
  }

  void PrintDepth(int depth, llvm::raw_ostream& out) const;

  auto name() const -> const std::string& { return name_; }
  auto deduced_parameters() const
      -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_parameters_;
  }
  auto deduced_parameters() -> llvm::ArrayRef<Nonnull<GenericBinding*>> {
    return deduced_parameters_;
  }
  auto me_pattern() const -> const BindingPattern& { return **me_pattern_; }
  auto me_pattern() -> BindingPattern& { return **me_pattern_; }
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
  std::optional<Nonnull<BindingPattern*>> me_pattern_;
  Nonnull<TuplePattern*> param_pattern_;
  ReturnTerm return_term_;
  std::optional<Nonnull<Block*>> body_;
};

class SelfDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  explicit SelfDeclaration(SourceLocation source_loc)
      : Declaration(AstNodeKind::SelfDeclaration, source_loc) {}
  // FIXME: Call set_static_type() and set_constant_value() (from Declaration,
  // the parent class), possibly during typechecking.

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromSelfDeclaration(node->kind());
  }

  auto name() const -> const std::string { return "Self"; }
  auto value_category() const -> ValueCategory { return ValueCategory::Let; }
};

class ClassDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ClassDeclaration(SourceLocation source_loc, std::string name,
                   Nonnull<SelfDeclaration*> self_decl,
                   std::optional<Nonnull<TuplePattern*>> type_params,
                   std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::ClassDeclaration, source_loc),
        name_(std::move(name)),
        self_decl_(self_decl),
        type_params_(type_params),
        members_(std::move(members)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromClassDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
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

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  std::string name_;
  Nonnull<SelfDeclaration*> self_decl_;
  std::optional<Nonnull<TuplePattern*>> type_params_;
  std::vector<Nonnull<Declaration*>> members_;
};

class AlternativeSignature : public AstNode {
 public:
  AlternativeSignature(SourceLocation source_loc, std::string name,
                       Nonnull<Expression*> signature)
      : AstNode(AstNodeKind::AlternativeSignature, source_loc),
        name_(std::move(name)),
        signature_(signature) {}

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAlternativeSignature(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto signature() const -> const Expression& { return *signature_; }
  auto signature() -> Expression& { return *signature_; }

 private:
  std::string name_;
  Nonnull<Expression*> signature_;
};

class ChoiceDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  ChoiceDeclaration(SourceLocation source_loc, std::string name,
                    std::vector<Nonnull<AlternativeSignature*>> alternatives)
      : Declaration(AstNodeKind::ChoiceDeclaration, source_loc),
        name_(std::move(name)),
        alternatives_(std::move(alternatives)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromChoiceDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto alternatives() const
      -> llvm::ArrayRef<Nonnull<const AlternativeSignature*>> {
    return alternatives_;
  }
  auto alternatives() -> llvm::ArrayRef<Nonnull<AlternativeSignature*>> {
    return alternatives_;
  }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  std::string name_;
  std::vector<Nonnull<AlternativeSignature*>> alternatives_;
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

  InterfaceDeclaration(SourceLocation source_loc, std::string name,
                       Nonnull<GenericBinding*> self,
                       std::vector<Nonnull<Declaration*>> members)
      : Declaration(AstNodeKind::InterfaceDeclaration, source_loc),
        name_(std::move(name)),
        members_(std::move(members)),
        self_(self) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromInterfaceDeclaration(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Declaration*>> {
    return members_;
  }
  auto self() const -> Nonnull<const GenericBinding*> { return self_; }
  auto self() -> Nonnull<GenericBinding*> { return self_; }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  std::string name_;
  std::vector<Nonnull<Declaration*>> members_;
  Nonnull<GenericBinding*> self_;
};

enum class ImplKind { InternalImpl, ExternalImpl };

class ImplDeclaration : public Declaration {
 public:
  using ImplementsCarbonValueNode = void;

  static auto Create(Nonnull<Arena*> arena, SourceLocation source_loc,
                     ImplKind kind, Nonnull<Expression*> impl_type,
                     bool self_type_specified, Nonnull<Expression*> interface,
                     std::vector<Nonnull<AstNode*>> deduced_params,
                     std::vector<Nonnull<Declaration*>> members)
      -> ErrorOr<Nonnull<ImplDeclaration*>>;

  // Use `Create` instead.
  ImplDeclaration(SourceLocation source_loc, ImplKind kind,
                  Nonnull<Expression*> impl_type,
                  std::optional<Nonnull<SelfDeclaration*>> self_decl,
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
  void set_interface_type(Nonnull<const Value*> iface_type) {
    interface_type_ = iface_type;
  }
  auto interface_type() const -> Nonnull<const Value*> {
    return *interface_type_;
  }
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
  auto self() const -> std::optional<Nonnull<const SelfDeclaration*>> {
    return self_decl_;
  }
  auto self() -> std::optional<Nonnull<SelfDeclaration*>> { return self_decl_; }

 private:
  ImplKind kind_;
  Nonnull<Expression*> impl_type_;
  std::optional<Nonnull<SelfDeclaration*>> self_decl_;
  Nonnull<Expression*> interface_;
  std::optional<Nonnull<const Value*>> interface_type_;
  std::vector<Nonnull<GenericBinding*>> deduced_parameters_;
  std::vector<Nonnull<Declaration*>> members_;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings_;
};

// Return the name of a declaration, if it has one.
auto GetName(const Declaration&) -> std::optional<std::string>;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
