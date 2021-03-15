// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <list>
#include <string>
#include <utility>

#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/struct_definition.h"
#include "executable_semantics/interpreter/dictionary.h"

namespace Carbon {

struct Value;

using Address = unsigned int;
using TypeEnv = Dictionary<std::string, Value*>;
using Env = Dictionary<std::string, Address>;

/// TODO:explain this. Also name it if necessary. Consult with jsiek.
using ExecutionEnvironment = std::pair<TypeEnv, Env>;

/// An existential AST declaration satisfying the Declaration concept.
class Declaration {
 public:  // ValueSemantic concept API.
  Declaration(const Declaration& other) = default;
  auto operator=(const Declaration& other) -> Declaration& = default;

  /// Constructs an instance equivalent to `d`, where `Model` satisfies the
  /// Declaration concept.
  template <class Model>
  // NOLINTNEXTLINE(google-explicit-constructor)
  Declaration(Model d) : box(std::make_shared<Boxed<Model>>(d)) {}

 public:  // Declaration concept API, in addition to ValueSemantic.
  void Print() const { box->Print(); }
  [[nodiscard]] auto Name() const -> std::string { return box->Name(); }
  [[nodiscard]] auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration {
    return box->TypeChecked(env, ct_env);
  }
  // TODO: This lint issue should probably be fixed by switching to a pointer.
  // NOLINTNEXTLINE(google-runtime-references)
  void InitGlobals(Env& globals) const { return box->InitGlobals(globals); }
  // TODO: This lint issue should probably be fixed by switching to a pointer.
  // NOLINTNEXTLINE(google-runtime-references)
  auto TopLevel(ExecutionEnvironment& e) const -> void {
    return box->TopLevel(e);
  }

 private:  // types
  /// A base class that erases the type of a `Boxed<Content>`, where `Content`
  /// satisfies the Declaration concept.
  struct Box {
   protected:
    Box() = default;

   public:
    Box(const Box& other) = delete;
    auto operator=(const Box& other) -> Box& = delete;

    virtual ~Box() = default;
    virtual auto Print() const -> void = 0;
    [[nodiscard]] virtual auto Name() const -> std::string = 0;
    [[nodiscard]] virtual auto TypeChecked(TypeEnv env, Env ct_env) const
        -> Declaration = 0;
    // NOLINTNEXTLINE(google-runtime-references)
    virtual auto InitGlobals(Env& globals) const -> void = 0;
    // NOLINTNEXTLINE(google-runtime-references)
    virtual auto TopLevel(ExecutionEnvironment&) const -> void = 0;
  };

  /// The derived class that holds an instance of `Content` satisfying the
  /// Declaration concept.
  template <class Content>
  struct Boxed final : Box {
    const Content content;
    explicit Boxed(Content content) : Box(), content(content) {}

    auto Print() const -> void override { return content.Print(); }
    [[nodiscard]] auto Name() const -> std::string override {
      return content.Name();
    }
    [[nodiscard]] auto TypeChecked(TypeEnv env, Env ct_env) const
        -> Declaration override {
      return content.TypeChecked(env, ct_env);
    }
    // NOLINTNEXTLINE(google-runtime-references)
    auto InitGlobals(Env& globals) const -> void override {
      content.InitGlobals(globals);
    }
    // NOLINTNEXTLINE(google-runtime-references)
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
  [[nodiscard]] auto Name() const -> std::string;
  [[nodiscard]] auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  // NOLINTNEXTLINE(google-runtime-references)
  auto InitGlobals(Env& globals) const -> void;
  // NOLINTNEXTLINE(google-runtime-references)
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

struct StructDeclaration {
  StructDefinition definition;
  StructDeclaration(int line_num, std::string name, std::list<Member*>* members)
      : definition{line_num, new std::string(std::move(name)), members} {}

  void Print() const;
  [[nodiscard]] auto Name() const -> std::string;
  [[nodiscard]] auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  // NOLINTNEXTLINE(google-runtime-references)
  void InitGlobals(Env& globals) const;
  // NOLINTNEXTLINE(google-runtime-references)
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

struct ChoiceDeclaration {
  int line_num;
  std::string name;
  std::list<std::pair<std::string, Expression*>> alternatives;

  ChoiceDeclaration(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression*>> alternatives)
      : line_num(line_num),
        name(std::move(std::move(name))),
        alternatives(std::move(std::move(alternatives))) {}

  void Print() const;
  [[nodiscard]] auto Name() const -> std::string;
  [[nodiscard]] auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
  // NOLINTNEXTLINE(google-runtime-references)
  void InitGlobals(Env& globals) const;
  // NOLINTNEXTLINE(google-runtime-references)
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
