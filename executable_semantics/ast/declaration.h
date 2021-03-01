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

namespace Carbon {

struct Value;

template <class K, class V>
class AssocList;

using Address = unsigned int;
using TypeEnv = AssocList<std::string, Value*>;
using Env = AssocList<std::string, Address>;

/// TODO:explain this. Also name it if necessary. Consult with jsiek.
using ExecutionEnvironment = std::pair<TypeEnv, Env>;

/// An existential AST declaration satisfying the Declaration concept.
class Declaration {
 public:  // ValueSemantic concept API.
  Declaration(const Declaration& other) = default;
  Declaration& operator=(const Declaration& other) = default;

  /// Constructs an instance equivalent to `d`, where `Model` satisfies the
  /// Declaration concept.
  template <class Model>
  Declaration(Model d) : box(std::make_shared<Boxed<Model>>(d)) {}

 public:  // Declaration concept API, in addition to ValueSemantic.
  void Print() const { box->Print(); }
  auto Name() const -> std::string { return box->Name(); }
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration {
    return box->TypeChecked(env, ct_env);
  }
  void InitGlobals(Env& globals) const { return box->InitGlobals(globals); }
  auto TopLevel(ExecutionEnvironment& e) const -> void {
    return box->TopLevel(e);
  }

 private:  // types
  /// A base class that erases the type of a `Boxed<Content>`, where `Content`
  /// satisfies the Declaration concept.
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
    virtual auto TopLevel(ExecutionEnvironment&) const -> void = 0;
  };

  /// The derived class that holds an instance of `Content` satisfying the
  /// Declaration concept.
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
    auto TopLevel(ExecutionEnvironment& e) const -> void override {
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
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

struct StructDeclaration {
  StructDefinition definition;
  StructDeclaration(int line_num, std::string name, std::list<Member*>* members)
      : definition{line_num, new std::string(name), members} {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  void InitGlobals(Env& globals) const;
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

struct ChoiceDeclaration {
  int line_num;
  std::string name;
  std::list<std::pair<std::string, Expression*>> alternatives;

  ChoiceDeclaration(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression*>> alternatives)
      : line_num(line_num), name(name), alternatives(alternatives) {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  void InitGlobals(Env& globals) const;
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
