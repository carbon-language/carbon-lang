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
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto tag() const -> Kind { return tag_; }

  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(Kind tag, SourceLocation source_loc)
      : tag_(tag), source_loc_(source_loc) {}

 private:
  const Kind tag_;
  SourceLocation source_loc_;
};

class FunctionDeclaration : public Declaration {
 public:
  FunctionDeclaration(Nonnull<FunctionDefinition*> definition)
      : Declaration(Kind::FunctionDeclaration, definition->source_loc()),
        definition(definition) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->tag() == Kind::FunctionDeclaration;
  }

  auto Definition() const -> const FunctionDefinition& { return *definition; }
  auto Definition() -> FunctionDefinition& { return *definition; }

 private:
  Nonnull<FunctionDefinition*> definition;
};

class ClassDeclaration : public Declaration {
 public:
  ClassDeclaration(SourceLocation source_loc, std::string name,
                   std::vector<Nonnull<Member*>> members)
      : Declaration(Kind::ClassDeclaration, source_loc),
        definition({.loc = source_loc,
                    .name = std::move(name),
                    .members = std::move(members)}) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->tag() == Kind::ClassDeclaration;
  }

  auto Definition() const -> const ClassDefinition& { return definition; }
  auto Definition() -> ClassDefinition& { return definition; }

 private:
  ClassDefinition definition;
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
        name(std::move(name)),
        alternatives(std::move(alternatives)) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->tag() == Kind::ChoiceDeclaration;
  }

  auto Name() const -> const std::string& { return name; }
  auto Alternatives() const -> const std::vector<Alternative>& {
    return alternatives;
  }

 private:
  std::string name;
  std::vector<Alternative> alternatives;
};

// Global variable definition implements the Declaration concept.
class VariableDeclaration : public Declaration {
 public:
  VariableDeclaration(SourceLocation source_loc,
                      Nonnull<BindingPattern*> binding,
                      Nonnull<Expression*> initializer)
      : Declaration(Kind::VariableDeclaration, source_loc),
        binding(binding),
        initializer(initializer) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->tag() == Kind::VariableDeclaration;
  }

  auto Binding() const -> Nonnull<const BindingPattern*> { return binding; }
  auto Binding() -> Nonnull<BindingPattern*> { return binding; }
  auto Initializer() const -> Nonnull<const Expression*> { return initializer; }
  auto Initializer() -> Nonnull<Expression*> { return initializer; }

 private:
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  Nonnull<BindingPattern*> binding;
  Nonnull<Expression*> initializer;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
