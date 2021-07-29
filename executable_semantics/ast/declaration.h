// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <list>
#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/struct_definition.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

struct Value;

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
  static auto MakeFunctionDeclaration(FunctionDefinition definition)
      -> const Declaration;
  static auto MakeStructDeclaration(int line_num, std::string name,
                                    std::list<Member*> members)
      -> const Declaration;
  static auto MakeChoiceDeclaration(
      int line_num, std::string name,
      std::list<std::pair<std::string, const Expression*>> alternatives)
      -> const Declaration;
  static auto MakeVariableDeclaration(int source_location, std::string name,
                                      const Expression* type,
                                      const Expression* initializer)
      -> const Declaration;

  auto GetFunctionDeclaration() const -> const FunctionDeclaration&;
  auto GetStructDeclaration() const -> const StructDeclaration&;
  auto GetChoiceDeclaration() const -> const ChoiceDeclaration&;
  auto GetVariableDeclaration() const -> const VariableDeclaration&;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  inline auto tag() const -> DeclarationKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

 private:
  std::variant<FunctionDeclaration, StructDeclaration, ChoiceDeclaration,
               VariableDeclaration>
      value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
