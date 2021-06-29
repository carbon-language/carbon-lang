// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <list>
#include <string>

#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/struct_definition.h"
#include "executable_semantics/interpreter/dictionary.h"

namespace Carbon {

struct Value;

using Address = unsigned int;
using TypeEnv = Dictionary<std::string, const Value*>;
using Env = Dictionary<std::string, Address>;

struct TypeCheckContext {
  // Symbol table mapping names of runtime entities to their type.
  TypeEnv types;
  // Symbol table mapping names of compile time entities to their value.
  Env values;
};

// An existential AST declaration satisfying the Declaration concept.
class Declaration {
 public:  // ValueSemantic concept API.
  Declaration(const Declaration& other) = default;
  Declaration& operator=(const Declaration& other) = default;

  // Constructs an instance equivalent to `d`, where `Model` satisfies the
  // Declaration concept.
  template <class Model>
  Declaration(Model d) : box(std::make_shared<Boxed<Model>>(d)) {}

 public:  // Declaration concept API, in addition to ValueSemantic.
  void Print() const { box->Print(); }
  auto Name() const -> std::string { return box->Name(); }

  // Signals a type error if the declaration is not well typed,
  // otherwise returns this declaration with annotated types.
  //
  // - Parameter env: types of run-time names.
  // - Paraemter ct_env: values of compile-time names.
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration {
    return box->TypeChecked(env, ct_env);
  }
  // Add an entry in the runtime global symbol table for this declaration.
  void InitGlobals(Env& globals) const { return box->InitGlobals(globals); }
  // Add an entry in the compile time global symbol tables for this declaration.
  auto TopLevel(TypeCheckContext& e) const -> void { return box->TopLevel(e); }

 private:  // types
  // A base class that erases the type of a `Boxed<Content>`, where `Content`
  // satisfies the Declaration concept.
  struct Box {
   protected:
    Box() {}

   public:
    Box(const Box& other) = delete;
    Box& operator=(const Box& other) = delete;

    virtual ~Box() {}
    virtual auto Print() const -> void = 0;
    virtual auto Name() const -> std::string = 0;
    virtual auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration = 0;
    virtual auto InitGlobals(Env& globals) const -> void = 0;
    virtual auto TopLevel(TypeCheckContext&) const -> void = 0;
  };

  // The derived class that holds an instance of `Content` satisfying the
  // Declaration concept.
  template <class Content>
  struct Boxed final : Box {
    const Content content;
    explicit Boxed(Content content) : Box(), content(content) {}

    auto Print() const -> void override { return content.Print(); }
    auto Name() const -> std::string override { return content.Name(); }
    auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration override {
      return content.TypeChecked(env, ct_env);
    }
    auto InitGlobals(Env& globals) const -> void override {
      content.InitGlobals(globals);
    }
    auto TopLevel(TypeCheckContext& e) const -> void override {
      content.TopLevel(e);
    }
  };

 private:  // data members
  // Note: the pointee is const as long as we have no mutating methods. When
  std::shared_ptr<const Box> box;
};

struct FunctionDeclaration {
  const FunctionDefinition* definition;
  explicit FunctionDeclaration(const FunctionDefinition* definition)
      : definition(definition) {}

  auto Print() const -> void;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  auto InitGlobals(Env& globals) const -> void;
  auto TopLevel(TypeCheckContext&) const -> void;
};

struct StructDeclaration {
  StructDefinition definition;
  StructDeclaration(int line_num, std::string name, std::list<Member*>* members)
      : definition{line_num, new std::string(name), members} {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  void InitGlobals(Env& globals) const;
  auto TopLevel(TypeCheckContext&) const -> void;
};

struct ChoiceDeclaration {
  int line_num;
  std::string name;
  std::list<std::pair<std::string, Expression>> alternatives;

  ChoiceDeclaration(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression>> alternatives)
      : line_num(line_num), name(name), alternatives(std::move(alternatives)) {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  void InitGlobals(Env& globals) const;
  auto TopLevel(TypeCheckContext&) const -> void;
};

// Global variable definition implements the Declaration concept.
class VariableDeclaration {
 public:
  VariableDeclaration(int source_location, std::string name,
                      const Expression* type, const Expression* initializer)
      : source_location(source_location),
        name(name),
        type(type),
        initializer(initializer) {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  void InitGlobals(Env& globals) const;
  auto TopLevel(TypeCheckContext&) const -> void;

 private:
  int source_location;
  std::string name;
  const Expression* type;
  const Expression* initializer;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
