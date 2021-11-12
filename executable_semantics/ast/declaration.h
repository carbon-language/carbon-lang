// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <string>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/ast/static_scope.h"
#include "executable_semantics/common/nonnull.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class StaticScope;

// Abstract base class of all AST nodes representing patterns.
//
// Declaration and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Declaration must provide a `classof` operation, and
// every concrete derived class must have a corresponding enumerator
// in `Kind`; see https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html for
// details.
class Declaration : public NamedEntityInterface {
 public:
  enum class Kind {
    FunctionDeclaration,
    ClassDeclaration,
    ChoiceDeclaration,
    VariableDeclaration,
  };

  Declaration(const Member&) = delete;
  auto operator=(const Member&) -> Declaration& = delete;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto named_entity_kind() const -> NamedEntityKind override {
    return NamedEntityKind::Declaration;
  }

  auto source_loc() const -> SourceLocation override { return source_loc_; }

  // The static type of the declared entity. Cannot be called before
  // typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of the declared entity. Can only be called once,
  // during typechecking.
  void set_static_type(Nonnull<const Value*> type) { static_type_ = type; }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  const Kind kind_;
  SourceLocation source_loc_;
  std::optional<Nonnull<const Value*>> static_type_;
};

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
struct GenericBinding : public NamedEntityInterface {
 public:
  GenericBinding(SourceLocation source_loc, std::string name,
                 Nonnull<Expression*> type)
      : source_loc_(source_loc), name_(std::move(name)), type_(type) {}

  auto named_entity_kind() const -> NamedEntityKind override {
    return NamedEntityKind::GenericBinding;
  }

  auto source_loc() const -> SourceLocation override { return source_loc_; }
  auto name() const -> const std::string& { return name_; }
  auto type() const -> const Expression& { return *type_; }

 private:
  SourceLocation source_loc_;
  std::string name_;
  Nonnull<Expression*> type_;
};

// The syntactic representation of a function declaration's return type.
// This syntax can take one of three forms:
// - An _explicit_ term consists of `->` followed by a type expression.
// - An _auto_ term consists of `-> auto`.
// - An _omitted_ term consists of no tokens at all.
// Each of these forms has a corresponding factory function.
class ReturnTerm {
 public:
  ReturnTerm(const ReturnTerm&) = default;
  ReturnTerm& operator=(const ReturnTerm&) = default;

  // Represents an omitted return term at `source_loc`.
  static auto Omitted(SourceLocation source_loc) -> ReturnTerm {
    return ReturnTerm(ReturnKind::Omitted, source_loc);
  }

  // Represents an auto return term at `source_loc`.
  static auto Auto(SourceLocation source_loc) -> ReturnTerm {
    return ReturnTerm(ReturnKind::Auto, source_loc);
  }

  // Represents an explicit return term with the given type expression.
  static auto Explicit(Nonnull<Expression*> type_expression) -> ReturnTerm {
    return ReturnTerm(type_expression);
  }

  // Returns true if this represents an omitted return term.
  auto is_omitted() const -> bool { return kind_ == ReturnKind::Omitted; }

  // Returns true if this represents an auto return term.
  auto is_auto() const -> bool { return kind_ == ReturnKind::Auto; }

  // If this represents an explicit return term, returns the type expression.
  // Otherwise, returns nullopt.
  auto type_expression() const -> std::optional<Nonnull<const Expression*>> {
    return type_expression_;
  }
  auto type_expression() -> std::optional<Nonnull<Expression*>> {
    return type_expression_;
  }

  // The static return type this term resolves to. Cannot be called before
  // typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the value of static_type(). Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) { static_type_ = type; }

  // Returns whether static_type() has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

  auto source_loc() const -> SourceLocation { return source_loc_; }

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  enum class ReturnKind { Omitted, Auto, Expression };

  explicit ReturnTerm(ReturnKind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {
    CHECK(kind != ReturnKind::Expression);
  }

  explicit ReturnTerm(Nonnull<Expression*> type_expression)
      : kind_(ReturnKind::Expression),
        type_expression_(type_expression),
        source_loc_(type_expression->source_loc()) {}

  ReturnKind kind_;
  std::optional<Nonnull<Expression*>> type_expression_;
  std::optional<Nonnull<const Value*>> static_type_;

  SourceLocation source_loc_;
};

class FunctionDeclaration : public Declaration {
 public:
  FunctionDeclaration(SourceLocation source_loc, std::string name,
                      std::vector<Nonnull<GenericBinding*>> deduced_params,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body)
      : Declaration(Kind::FunctionDeclaration, source_loc),
        name_(std::move(name)),
        deduced_parameters_(std::move(deduced_params)),
        param_pattern_(param_pattern),
        return_term_(return_term),
        body_(body) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::FunctionDeclaration;
  }

  void PrintDepth(int depth, llvm::raw_ostream& out) const;

  auto name() const -> const std::string& { return name_; }
  auto deduced_parameters() const
      -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_parameters_;
  }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_term() const -> const ReturnTerm& { return return_term_; }
  auto return_term() -> ReturnTerm& { return return_term_; }
  auto body() const -> std::optional<Nonnull<const Block*>> { return body_; }
  auto body() -> std::optional<Nonnull<Block*>> { return body_; }

  // Only contains function parameters. Scoped variables are in the body.
  auto static_scope() const -> const StaticScope& { return static_scope_; }
  auto static_scope() -> StaticScope& { return static_scope_; }

 private:
  std::string name_;
  std::vector<Nonnull<GenericBinding*>> deduced_parameters_;
  Nonnull<TuplePattern*> param_pattern_;
  ReturnTerm return_term_;
  std::optional<Nonnull<Block*>> body_;
  StaticScope static_scope_;
};

class ClassDeclaration : public Declaration {
 public:
  ClassDeclaration(SourceLocation source_loc, std::string name,
                   std::vector<Nonnull<Member*>> members)
      : Declaration(Kind::ClassDeclaration, source_loc),
        name_(std::move(name)),
        members_(std::move(members)) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::ClassDeclaration;
  }

  auto name() const -> const std::string& { return name_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Member*>> { return members_; }

  // Contains class members. Scoped variables are in the body.
  auto static_scope() const -> const StaticScope& { return static_scope_; }
  auto static_scope() -> StaticScope& { return static_scope_; }

 private:
  std::string name_;
  std::vector<Nonnull<Member*>> members_;
  StaticScope static_scope_;
};

class ChoiceDeclaration : public Declaration {
 public:
  class Alternative : public NamedEntityInterface {
   public:
    Alternative(SourceLocation source_loc, std::string name,
                Nonnull<Expression*> signature)
        : source_loc_(source_loc),
          name_(std::move(name)),
          signature_(signature) {}

    auto named_entity_kind() const -> NamedEntityKind override {
      return NamedEntityKind::ChoiceDeclarationAlternative;
    }

    auto source_loc() const -> SourceLocation override { return source_loc_; }
    auto name() const -> const std::string& { return name_; }
    auto signature() const -> const Expression& { return *signature_; }

   private:
    SourceLocation source_loc_;
    std::string name_;
    Nonnull<Expression*> signature_;
  };

  ChoiceDeclaration(SourceLocation source_loc, std::string name,
                    std::vector<Nonnull<Alternative*>> alternatives)
      : Declaration(Kind::ChoiceDeclaration, source_loc),
        name_(std::move(name)),
        alternatives_(std::move(alternatives)) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::ChoiceDeclaration;
  }

  auto name() const -> const std::string& { return name_; }
  auto alternatives() const -> llvm::ArrayRef<Nonnull<const Alternative*>> {
    return alternatives_;
  }

  // Contains the alternatives.
  auto static_scope() const -> const StaticScope& { return static_scope_; }
  auto static_scope() -> StaticScope& { return static_scope_; }

 private:
  std::string name_;
  std::vector<Nonnull<Alternative*>> alternatives_;
  StaticScope static_scope_;
};

// Global variable definition implements the Declaration concept.
class VariableDeclaration : public Declaration {
 public:
  VariableDeclaration(SourceLocation source_loc,
                      Nonnull<BindingPattern*> binding,
                      Nonnull<Expression*> initializer)
      : Declaration(Kind::VariableDeclaration, source_loc),
        binding_(binding),
        initializer_(initializer) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::VariableDeclaration;
  }

  auto binding() const -> const BindingPattern& { return *binding_; }
  auto binding() -> BindingPattern& { return *binding_; }
  auto initializer() const -> const Expression& { return *initializer_; }
  auto initializer() -> Expression& { return *initializer_; }

 private:
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  Nonnull<BindingPattern*> binding_;
  Nonnull<Expression*> initializer_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
