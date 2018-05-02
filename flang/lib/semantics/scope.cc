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

#include "scope.h"
#include "symbol.h"
#include <memory>

namespace Fortran::semantics {

const Scope Scope::systemScope{
    Scope::systemScope, Scope::Kind::System, nullptr};
Scope Scope::globalScope{Scope::systemScope, Scope::Kind::Global, nullptr};

Scope &Scope::MakeScope(Kind kind, const Symbol *symbol) {
  children_.emplace_back(*this, kind, symbol);
  return children_.back();
}

std::ostream &operator<<(std::ostream &os, const Scope &scope) {
  os << Scope::EnumToString(scope.kind())
     << " scope: " << scope.children_.size() << " children\n";
  for (const auto &sym : scope.symbols_) {
    os << "  " << sym.second << "\n";
  }
  return os;
}

}  // namespace Fortran::semantics
