// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <list>
#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/class_definition.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
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

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return tag; }

  auto SourceLoc() const -> SourceLocation { return loc; }

  void Print(llvm::raw_ostream& out) const;

 protected:
  // Constructs a Declaration representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Declaration(Kind tag, SourceLocation loc) : tag(tag), loc(loc) {}

 private:
  const Kind tag;
  SourceLocation loc;
};

class FunctionDeclaration : public Declaration {
 public:
  FunctionDeclaration(const FunctionDefinition* definition)
      : Declaration(Kind::FunctionDeclaration, definition->loc),
        definition(definition) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->Tag() == Kind::FunctionDeclaration;
  }

  auto Definition() const -> const FunctionDefinition& { return *definition; }

 private:
  const FunctionDefinition* definition;
};

class ClassDeclaration : public Declaration {
 public:
  ClassDeclaration(SourceLocation loc, std::string name,
                   std::list<Member*> members)
      : Declaration(Kind::ClassDeclaration, loc),
        definition({.loc = loc,
                    .name = std::move(name),
                    .members = std::move(members)}) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->Tag() == Kind::ClassDeclaration;
  }

  auto Definition() const -> const ClassDefinition& { return definition; }

 private:
  ClassDefinition definition;
};

class ChoiceDeclaration : public Declaration {
 public:
  ChoiceDeclaration(
      SourceLocation loc, std::string name,
      std::list<std::pair<std::string, const Expression*>> alternatives)
      : Declaration(Kind::ChoiceDeclaration, loc),
        name(std::move(name)),
        alternatives(std::move(alternatives)) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->Tag() == Kind::ChoiceDeclaration;
  }

  auto Name() const -> const std::string& { return name; }
  auto Alternatives() const
      -> const std::list<std::pair<std::string, const Expression*>>& {
    return alternatives;
  }

 private:
  std::string name;
  std::list<std::pair<std::string, const Expression*>> alternatives;
};

// Global variable definition implements the Declaration concept.
class VariableDeclaration : public Declaration {
 public:
  VariableDeclaration(SourceLocation loc, const BindingPattern* binding,
                      const Expression* initializer)
      : Declaration(Kind::VariableDeclaration, loc),
        binding(binding),
        initializer(initializer) {}

  static auto classof(const Declaration* decl) -> bool {
    return decl->Tag() == Kind::VariableDeclaration;
  }

  auto Binding() const -> const BindingPattern* { return binding; }
  auto Initializer() const -> const Expression* { return initializer; }

 private:
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  const BindingPattern* binding;
  const Expression* initializer;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
