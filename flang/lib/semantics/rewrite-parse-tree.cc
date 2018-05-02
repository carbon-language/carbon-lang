// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>

namespace Fortran::semantics {

// Symbols collected during name resolution that are added to parse tree.
using symbolMap = std::map<const SourceName, Symbol *>;

/// Walk the parse tree and add symbols from the symbolMap in Name nodes.
/// Convert mis-identified statement functions to array element assignments.
class RewriteMutator {
public:
  RewriteMutator(const symbolMap &symbols) : symbols_{symbols} {}

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}

  // Fill in name.symbol if there is a corresponding symbol
  void Post(parser::Name &name) {
    const auto it = symbols_.find(name.source);
    if (it != symbols_.end()) {
      name.symbol = it->second;
    }
  }

  using stmtFuncType =
      parser::Statement<parser::Indirection<parser::StmtFunctionStmt>>;

  // Find mis-parsed statement functions and move to stmtFuncsToConvert list.
  void Post(parser::SpecificationPart &x) {
    auto &list = std::get<std::list<parser::DeclarationConstruct>>(x.t);
    for (auto it = list.begin(); it != list.end();) {
      if (auto stmt = std::get_if<stmtFuncType>(&it->u)) {
        Symbol *symbol{std::get<parser::Name>(stmt->statement->t).symbol};
        if (symbol && symbol->has<EntityDetails>()) {
          // not a stmt func: remove it here and add to ones to convert
          stmtFuncsToConvert.push_back(std::move(*stmt));
          it = list.erase(it);
          continue;
        }
      }
      ++it;
    }
  }

  // Insert converted assignments at start of ExecutionPart.
  bool Pre(parser::ExecutionPart &x) {
    auto origFirst = x.v.begin();  // insert each elem before origFirst
    for (stmtFuncType &sf : stmtFuncsToConvert) {
      auto &&stmt = sf.statement->ConvertToAssignment();
      stmt.source = sf.source;
      x.v.insert(origFirst,
          parser::ExecutionPartConstruct{parser::ExecutableConstruct{stmt}});
    }
    stmtFuncsToConvert.clear();
    return true;
  }

  void Post(parser::Variable &x) { ConvertFunctionRef(x); }

  void Post(parser::Expr &x) { ConvertFunctionRef(x); }

private:
  const symbolMap &symbols_;
  std::list<stmtFuncType> stmtFuncsToConvert;

  // For T = Variable or Expr, if x has a function reference that really
  // should be an array element reference (i.e. the name occurs in an
  // entity declaration, convert it.
  template<typename T> void ConvertFunctionRef(T &x) {
    auto *funcRef =
        std::get_if<parser::Indirection<parser::FunctionReference>>(&x.u);
    if (!funcRef) {
      return;
    }
    parser::Name *name = std::get_if<parser::Name>(
        &std::get<parser::ProcedureDesignator>((*funcRef)->v.t).u);
    if (!name || !name->symbol || !name->symbol->has<EntityDetails>()) {
      return;
    }
    x.u = parser::Indirection{(*funcRef)->ConvertToArrayElementRef()};
  }
};

static void CollectSymbols(Scope &scope, symbolMap &symbols) {
  for (auto &pair : scope) {
    Symbol &symbol{pair.second};
    for (const auto &name : symbol.occurrences()) {
      symbols.emplace(name, &symbol);
    }
  }
  for (auto &child : scope.children()) {
    CollectSymbols(child, symbols);
  }
}

void RewriteParseTree(parser::Program &program) {
  symbolMap symbols;
  CollectSymbols(Scope::globalScope, symbols);
  RewriteMutator mutator{symbols};
  parser::Walk(program, mutator);
}

}  // namespace Fortran::semantics
