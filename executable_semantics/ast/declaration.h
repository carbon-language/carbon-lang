// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <string>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/class_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/nonnull.h"
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
class Declaration {
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

  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  const Kind kind_;
  SourceLocation source_loc_;
};

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
struct GenericBinding {
  std::string name;
  Nonnull<const Expression*> type;
};

class FunctionDeclaration : public Declaration {
 public:
  FunctionDeclaration(SourceLocation source_loc, std::string name,
                      std::vector<GenericBinding> deduced_params,
                      Nonnull<TuplePattern*> param_pattern,
                      Nonnull<Pattern*> return_type,
                      bool is_omitted_return_type,
                      std::optional<Nonnull<Statement*>> body)
      : Declaration(Kind::FunctionDeclaration, source_loc),
        name_(std::move(name)),
        deduced_parameters_(std::move(deduced_params)),
        param_pattern_(param_pattern),
        return_type_(return_type),
        is_omitted_return_type_(is_omitted_return_type),
        body_(body) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::FunctionDeclaration;
  }

  void PrintDepth(int depth, llvm::raw_ostream& out) const;

  auto name() const -> const std::string& { return name_; }
  auto deduced_parameters() const -> llvm::ArrayRef<GenericBinding> {
    return deduced_parameters_;
  }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_type() const -> const Pattern& { return *return_type_; }
  auto return_type() -> Pattern& { return *return_type_; }
  auto is_omitted_return_type() const -> bool {
    return is_omitted_return_type_;
  }
  auto body() const -> std::optional<Nonnull<const Statement*>> {
    return body_;
  }
  auto body() -> std::optional<Nonnull<Statement*>> { return body_; }

  // The static type of this function. Cannot be called before typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of this expression. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) { static_type_ = type; }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

 private:
  std::string name_;
  std::vector<GenericBinding> deduced_parameters_;
  Nonnull<TuplePattern*> param_pattern_;
  Nonnull<Pattern*> return_type_;
  bool is_omitted_return_type_;
  std::optional<Nonnull<Statement*>> body_;

  std::optional<Nonnull<const Value*>> static_type_;
};

class ClassDeclaration : public Declaration {
 public:
  ClassDeclaration(SourceLocation source_loc, std::string name,
                   std::vector<Nonnull<Member*>> members)
      : Declaration(Kind::ClassDeclaration, source_loc),
        definition_(source_loc, std::move(name), std::move(members)) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::ClassDeclaration;
  }

  auto definition() const -> const ClassDefinition& { return definition_; }
  auto definition() -> ClassDefinition& { return definition_; }

 private:
  ClassDefinition definition_;
};

class ChoiceDeclaration : public Declaration {
 public:
  class Alternative {
   public:
    Alternative(std::string name, Nonnull<Expression*> signature)
        : name_(std::move(name)), signature_(signature) {}

    auto name() const -> const std::string& { return name_; }
    auto signature() const -> const Expression& { return *signature_; }

   private:
    std::string name_;
    Nonnull<Expression*> signature_;
  };

  ChoiceDeclaration(SourceLocation source_loc, std::string name,
                    std::vector<Alternative> alternatives)
      : Declaration(Kind::ChoiceDeclaration, source_loc),
        name_(std::move(name)),
        alternatives_(std::move(alternatives)) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::ChoiceDeclaration;
  }

  auto name() const -> const std::string& { return name_; }
  auto alternatives() const -> llvm::ArrayRef<Alternative> {
    return alternatives_;
  }

 private:
  std::string name_;
  std::vector<Alternative> alternatives_;
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
