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

Scope Scope::systemScope{Scope::systemScope, Scope::Kind::System, nullptr};
Scope Scope::globalScope{Scope::systemScope, Scope::Kind::Global, nullptr};

Symbols<1024> Scope::allSymbols;

Scope &Scope::MakeScope(Kind kind, Symbol *symbol) {
  children_.emplace_back(*this, kind, symbol);
  return children_.back();
}

Scope::iterator Scope::find(const SourceName &name) {
  auto it{symbols_.find(name)};
  if (it != end()) {
    it->second->add_occurrence(name);
  }
  return it;
}
Scope::const_iterator Scope::find(const SourceName &name) const {
  return symbols_.find(name);
}
Scope::size_type Scope::erase(const SourceName &name) {
  auto it{symbols_.find(name)};
  if (it != end()) {
    it->second->remove_occurrence(name);
    symbols_.erase(it);
    return 1;
  } else {
    return 0;
  }
}
Scope *Scope::FindSubmodule(const SourceName &name) const {
  auto it{submodules_.find(name)};
  if (it == submodules_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}
bool Scope::AddSubmodule(const SourceName &name, Scope *submodule) {
  return submodules_.emplace(name, submodule).second;
}
DerivedTypeSpec &Scope::MakeDerivedTypeSpec(const SourceName &name) {
  derivedTypeSpecs_.emplace_back(name);
  return derivedTypeSpecs_.back();
}

std::ostream &operator<<(std::ostream &os, const Scope &scope) {
  os << Scope::EnumToString(scope.kind()) << " scope: ";
  if (auto *symbol{scope.symbol()}) {
    os << *symbol << ' ';
  }
  os << scope.children_.size() << " children\n";
  for (const auto &pair : scope.symbols_) {
    const auto *symbol{pair.second};
    os << "  " << *symbol << '\n';
  }
  return os;
}

}  // namespace Fortran::semantics
