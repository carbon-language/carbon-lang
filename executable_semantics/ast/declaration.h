// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <string>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/class_definition.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
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
  Declaration& operator=(const Member&) = delete;

  void Print(llvm::raw_ostream& out) const;

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  const Kind kind_;
  SourceLocation source_loc_;
};

class FunctionDeclaration : public Declaration {
 public:
  FunctionDeclaration(Nonnull<FunctionDefinition*> definition)
      : Declaration(Kind::FunctionDeclaration, definition->source_loc()),
        definition_(definition) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->kind() == Kind::FunctionDeclaration;
  }

  auto definition() const -> const FunctionDefinition& { return *definition_; }
  auto definition() -> FunctionDefinition& { return *definition_; }

 private:
  Nonnull<FunctionDefinition*> definition_;
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
        : name_(name), signature_(signature) {}

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
