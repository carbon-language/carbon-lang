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

#include "semantics.h"
#include "canonicalize-do.h"
#include "mod-file.h"
#include "resolve-labels.h"
#include "resolve-names.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"

namespace Fortran::semantics {

static void DoDumpSymbols(std::ostream &, const Scope &, int indent = 0);
static void PutIndent(std::ostream &, int indent);

Semantics &Semantics::set_searchDirectories(
    const std::vector<std::string> &directories) {
  for (auto directory : directories) {
    directories_.push_back(directory);
  }
  return *this;
}

Semantics &Semantics::set_moduleDirectory(const std::string &directory) {
  moduleDirectory_ = directory;
  directories_.insert(directories_.begin(), directory);
  return *this;
}

bool Semantics::Perform(parser::Program &program) {
  ValidateLabels(messages_, program);
  if (AnyFatalError()) {
    return false;
  }
  ResolveNames(messages_, globalScope_, program, directories_);
  if (AnyFatalError()) {
    return false;
  }
  RewriteParseTree(messages_, globalScope_, program);
  if (AnyFatalError()) {
    return false;
  }
  parser::CanonicalizeDo(program);
  ModFileWriter writer;
  writer.set_directory(moduleDirectory_);
  if (!writer.WriteAll(globalScope_)) {
    messages_.Annex(writer.errors());
    return false;
  }
  return true;
}

void Semantics::DumpSymbols(std::ostream &os) {
  DoDumpSymbols(os, globalScope_);
}

void DoDumpSymbols(std::ostream &os, const Scope &scope, int indent) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol{scope.symbol()}) {
    os << ' ' << symbol->name().ToString();
  }
  os << '\n';
  ++indent;
  for (const auto &pair : scope) {
    const auto &symbol{*pair.second};
    PutIndent(os, indent);
    os << symbol << '\n';
    if (const auto *details{symbol.detailsIf<GenericDetails>()}) {
      if (const auto &type{details->derivedType()}) {
        PutIndent(os, indent);
        os << *type << '\n';
      }
    }
  }
  for (const auto &child : scope.children()) {
    DoDumpSymbols(os, child, indent);
  }
  --indent;
}

static void PutIndent(std::ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}

}  // namespace Fortran::semantics
