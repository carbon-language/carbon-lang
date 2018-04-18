#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-mutator.h"
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
      x.v.insert(origFirst, ConvertToAssignment(sf));
    }
    stmtFuncsToConvert.clear();
    return true;
  }

private:
  const symbolMap &symbols_;
  std::list<stmtFuncType> stmtFuncsToConvert;

  // Convert a statement function statement to an ExecutionPartConstruct
  // containing an array element assignment statement.
  static parser::ExecutionPartConstruct ConvertToAssignment(stmtFuncType &x) {
    parser::StmtFunctionStmt &sf{*x.statement};
    auto &funcName = std::get<parser::Name>(sf.t);
    auto &funcArgs = std::get<std::list<parser::Name>>(sf.t);
    auto &funcExpr = std::get<parser::Scalar<parser::Expr>>(sf.t).thing;
    parser::ArrayElement arrayElement{
        funcName, std::list<parser::SectionSubscript>{}};
    for (parser::Name &arg : funcArgs) {
      arrayElement.subscripts.push_back(parser::SectionSubscript{
          parser::Scalar{parser::Integer{parser::Indirection{
              parser::Expr{parser::Indirection{parser::Designator{arg}}}}}}});
    }
    auto &&variable = parser::Variable{parser::Indirection{parser::Designator{
        parser::DataRef{parser::Indirection{std::move(arrayElement)}}}}};
    auto &&stmt = parser::Statement{std::nullopt,
        parser::ActionStmt{parser::Indirection{
            parser::AssignmentStmt{std::move(variable), std::move(funcExpr)}}}};
    stmt.source = x.source;
    return parser::ExecutionPartConstruct{parser::ExecutableConstruct{stmt}};
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
