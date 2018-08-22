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

#include "unparse-with-symbols.h"
#include "symbol.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include "../parser/unparse.h"
#include <map>
#include <ostream>
#include <set>

namespace Fortran::semantics {

// Walk the parse tree and collection information about which statements
// define and reference symbols. Then PrintSymbols outputs information
// by statement.
class SymbolDumpVisitor {
public:
  // Write out symbols defined or referenced at this statement.
  void PrintSymbols(const parser::CharBlock &stmt, std::ostream &, int) const;

  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}
  template<typename T> bool Pre(const parser::Statement<T> &stmt) {
    currStmt_ = &stmt.source;
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    currStmt_ = nullptr;
  }
  void Post(const parser::Name &);
  void Post(const parser::EndFunctionStmt &) { EndScope(); }
  void Post(const parser::EndModuleStmt &) { EndScope(); }
  void Post(const parser::EndMpSubprogramStmt &) { EndScope(); }
  void Post(const parser::EndProgramStmt &) { EndScope(); }
  void Post(const parser::EndSubmoduleStmt &) { EndScope(); }
  void Post(const parser::EndSubroutineStmt &) { EndScope(); }
  bool Pre(const parser::DataRef &) { return PreReference(); }
  void Post(const parser::DataRef &) { PostReference(); }
  bool Pre(const parser::Designator &) { return PreReference(); }
  void Post(const parser::Designator &) { PostReference(); }
  bool Pre(const parser::StructureComponent &) { return PreReference(); }
  void Post(const parser::StructureComponent &) { PostReference(); }
  bool Pre(const parser::ProcedureDesignator &) { return PreReference(); }
  void Post(const parser::ProcedureDesignator &) { PostReference(); }
  bool Pre(const parser::DerivedTypeSpec &) { return PreReference(); }
  void Post(const parser::DerivedTypeSpec &) { PostReference(); }
  bool Pre(const parser::UseStmt &) { return PreReference(); }
  void Post(const parser::UseStmt &) { PostReference(); }
  bool Pre(const parser::ImportStmt &) { return PreReference(); }
  void Post(const parser::ImportStmt &) { PostReference(); }

private:
  using symbolMap = std::multimap<const char *, const Symbol *>;

  const SourceName *currStmt_{nullptr};  // current statement we are processing
  int isRef_{0};  // > 0 means in the context of a reference
  symbolMap defs_;  // statement location to symbol defined there
  symbolMap refs_;  // statement location to symbol referenced there
  std::map<const Symbol *, const char *> symbolToStmt_;  // symbol to def

  void EndScope();
  bool PreReference();
  void PostReference();
  bool isRef() const { return isRef_ > 0; }
  void PrintSymbols(const parser::CharBlock &, std::ostream &, int,
      const symbolMap &, bool) const;
  void Indent(std::ostream &, int) const;
};

void SymbolDumpVisitor::PrintSymbols(
    const parser::CharBlock &location, std::ostream &out, int indent) const {
  PrintSymbols(location, out, indent, defs_, true);
  PrintSymbols(location, out, indent, refs_, false);
}
void SymbolDumpVisitor::PrintSymbols(const parser::CharBlock &location,
    std::ostream &out, int indent, const symbolMap &symbols, bool isDef) const {
  std::set<const Symbol *> done;  // used to prevent duplicates
  auto range{symbols.equal_range(location.begin())};
  for (auto it{range.first}; it != range.second; ++it) {
    auto *symbol{it->second};
    if (done.insert(symbol).second) {
      Indent(out, indent);
      out << '!' << (isDef ? "DEF"s : "REF"s) << ": ";
      DumpForUnparse(out, symbol->GetUltimate(), isDef);
      out << '\n';
    }
  }
}
void SymbolDumpVisitor::Indent(std::ostream &out, int indent) const {
  for (int i{0}; i < indent; ++i) {
    out << ' ';
  }
}

void SymbolDumpVisitor::Post(const parser::Name &name) {
  if (const auto *symbol{name.symbol}) {
    CHECK(currStmt_);
    // If this is the first reference to an implicitly defined symbol,
    // record it as a def.
    bool isImplicit{symbol->test(Symbol::Flag::Implicit) &&
        symbolToStmt_.find(symbol) == symbolToStmt_.end()};
    if (isRef() && !isImplicit) {
      refs_.emplace(currStmt_->begin(), symbol);
    } else {
      symbolToStmt_.emplace(symbol, currStmt_->begin());
    }
  }
}

// Defs are initially saved in symbolToStmt_ so that a symbol defined across
// multiple statements is associated with only one (the first). Now that we
// are at the end of a scope, move them into defs_.
void SymbolDumpVisitor::EndScope() {
  for (auto pair : symbolToStmt_) {
    defs_.emplace(pair.second, pair.first);
  }
  symbolToStmt_.clear();
}

// {Pre,Post}Reference() are called around constructs that contains symbols
// references. Sometimes those are nested (e.g. DataRef inside Designator)
// so we need to maintain a count to know when we are back out.
bool SymbolDumpVisitor::PreReference() {
  ++isRef_;
  return true;
}
void SymbolDumpVisitor::PostReference() {
  CHECK(isRef_ > 0);
  --isRef_;
}

void UnparseWithSymbols(std::ostream &out, const parser::Program &program,
    parser::Encoding encoding) {
  SymbolDumpVisitor visitor;
  parser::Walk(program, visitor);
  parser::preStatementType preStatement{
      [&](const parser::CharBlock &location, std::ostream &out, int indent) {
        visitor.PrintSymbols(location, out, indent);
      }};
  parser::Unparse(out, program, encoding, false, true, &preStatement);
}

}  // namespace Fortran::semantics
