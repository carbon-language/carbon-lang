// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_

#include <set>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/ptr.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

using TypeEnv = Dictionary<std::string, const Value*>;

class TypeChecker {
 public:
  struct TypeCheckContext {
    // Symbol table mapping names of runtime entities to their type.
    TypeEnv types;
    // Symbol table mapping names of compile time entities to their value.
    Env values;
  };

  auto MakeTypeChecked(const Ptr<const Declaration> d, const TypeEnv& types,
                       const Env& values) -> Ptr<const Declaration>;

  auto TopLevel(const std::list<Ptr<const Declaration>>& fs)
      -> TypeCheckContext;

 private:
  struct TCExpression {
    TCExpression(Ptr<const Expression> e, const Value* t, TypeEnv types)
        : exp(e), type(t), types(types) {}

    Ptr<const Expression> exp;
    const Value* type;
    TypeEnv types;
  };

  struct TCPattern {
    Ptr<const Pattern> pattern;
    const Value* type;
    TypeEnv types;
  };

  struct TCStatement {
    TCStatement(const Statement* s, TypeEnv types) : stmt(s), types(types) {}

    const Statement* stmt;
    TypeEnv types;
  };

  auto TypeCheckExp(Ptr<const Expression> e, TypeEnv types, Env values)
      -> TCExpression;
  auto TypeCheckPattern(Ptr<const Pattern> p, TypeEnv types, Env values,
                        const Value* expected) -> TCPattern;

  auto TypeCheckStmt(const Statement* s, TypeEnv types, Env values,
                     const Value*& ret_type, bool is_omitted_ret_type)
      -> TCStatement;

  auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv types, Env values)
      -> Ptr<const FunctionDefinition>;

  auto TypeCheckCase(const Value* expected, Ptr<const Pattern> pat,
                     const Statement* body, TypeEnv types, Env values,
                     const Value*& ret_type, bool is_omitted_ret_type)
      -> std::pair<Ptr<const Pattern>, const Statement*>;

  auto TypeOfFunDef(TypeEnv types, Env values,
                    const FunctionDefinition* fun_def) -> const Value*;
  auto TypeOfClassDef(const ClassDefinition* sd, TypeEnv /*types*/, Env ct_top)
      -> const Value*;

  void TopLevel(const Declaration& d, TypeCheckContext* tops);

  Interpreter interpreter;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
