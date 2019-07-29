// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
// reference symbols. Then PrintSymbols outputs information by statement.
// The first reference to a symbol is treated as its definition and more
// information is included.
class SymbolDumpVisitor {
public:
  // Write out symbols referenced at this statement.
  void PrintSymbols(const parser::CharBlock &, std::ostream &, int);

  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}
  template<typename T> bool Pre(const parser::Statement<T> &stmt) {
    currStmt_ = &stmt.source;
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    currStmt_ = nullptr;
  }
  void Post(const parser::Name &name);

private:
  const SourceName *currStmt_{nullptr};  // current statement we are processing
  std::multimap<const char *, const Symbol *> symbols_;  // location to symbol
  std::set<const Symbol *> symbolsDefined_;  // symbols that have been processed
  void Indent(std::ostream &, int) const;
};

void SymbolDumpVisitor::PrintSymbols(
    const parser::CharBlock &location, std::ostream &out, int indent) {
  std::set<const Symbol *> done;  // prevent duplicates on this line
  auto range{symbols_.equal_range(location.begin())};
  for (auto it{range.first}; it != range.second; ++it) {
    const auto *symbol{it->second};
    if (done.insert(symbol).second) {
      bool firstTime{symbolsDefined_.insert(symbol).second};
      Indent(out, indent);
      out << '!' << (firstTime ? "DEF"s : "REF"s) << ": ";
      DumpForUnparse(out, *symbol, firstTime);
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
    if (!symbol->has<MiscDetails>()) {
      symbols_.emplace(DEREF(currStmt_).begin(), symbol);
    }
  }
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
}
