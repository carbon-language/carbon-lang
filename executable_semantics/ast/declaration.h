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

enum class DeclarationKind {
  FunctionDeclaration,
  StructDeclaration,
  ChoiceDeclaration,
  VariableDeclaration,
};

struct FunctionDeclaration {
  static constexpr DeclarationKind Kind = DeclarationKind::FunctionDeclaration;
  FunctionDefinition definition;
};

struct StructDeclaration {
  static constexpr DeclarationKind Kind = DeclarationKind::StructDeclaration;
  StructDefinition definition;
};

struct ChoiceDeclaration {
  static constexpr DeclarationKind Kind = DeclarationKind::ChoiceDeclaration;
  int line_num;
  std::string name;
  std::list<std::pair<std::string, const Expression*>> alternatives;
};

// Global variable definition implements the Declaration concept.
struct VariableDeclaration {
  static constexpr DeclarationKind Kind = DeclarationKind::VariableDeclaration;
  int source_location;
  std::string name;
  const Expression* type;
  const Expression* initializer;
};

class Declaration {
 public:
  auto tag() const -> DeclarationKind;

  static auto MakeFunctionDeclaration() -> const Declaration*;
  static auto MakeStructDeclaration() -> const Declaration*;
  static auto MakeChoiceDeclaration() -> const Declaration*;
  static auto MakeVariableDeclaration() -> const Declaration*;

  auto GetFunctionDeclaration() const -> const FunctionDeclaration&;
  auto GetStructDeclaration() const -> const StructDeclaration&;
  auto GetChoiceDeclaration() const -> const ChoiceDeclaration&;
  auto GetVariableDeclaration() const -> const VariableDeclaration&;

  /*
   void Print() const;
   auto Name() const -> std::string;

   // Signals a type error if the declaration is not well typed,
   // otherwise returns this declaration with annotated types.
   //
   // - Parameter env: types of run-time names.
   // - Paraemter ct_env: values of compile-time names.
   auto TypeChecked(TypeEnv env, Env ct_env) const -> Declaration;
   // Add an entry in the runtime global symbol table for this declaration.
   void InitGlobals(Env& globals) const;
   // Add an entry in the compile time global symbol tables for this
   declaration. void TopLevel(TypeCheckContext& e) const;
   */

 private:
  std::variant<FunctionDeclaration, StructDeclaration, ChoiceDeclaration,
               VariableDeclaration>
      value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
