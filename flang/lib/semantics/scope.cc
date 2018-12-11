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
#include <algorithm>
#include <memory>

namespace Fortran::semantics {

Symbols<1024> Scope::allSymbols;

bool Scope::IsModule() const {
  return kind_ == Kind::Module && !symbol_->get<ModuleDetails>().isSubmodule();
}

Scope &Scope::MakeScope(Kind kind, Symbol *symbol) {
  children_.emplace_back(*this, kind, symbol);
  return children_.back();
}

Scope::iterator Scope::find(const SourceName &name) {
  return symbols_.find(name);
}
Scope::const_iterator Scope::find(const SourceName &name) const {
  return symbols_.find(name);
}
Scope::size_type Scope::erase(const SourceName &name) {
  auto it{symbols_.find(name)};
  if (it != end()) {
    symbols_.erase(it);
    return 1;
  } else {
    return 0;
  }
}
Symbol *Scope::FindSymbol(const SourceName &name) const {
  if (kind() == Kind::DerivedType) {
    return parent_.FindSymbol(name);
  }
  const auto it{find(name)};
  if (it != end()) {
    return it->second;
  } else if (CanImport(name)) {
    return parent_.FindSymbol(name);
  } else {
    return nullptr;
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
bool Scope::AddSubmodule(const SourceName &name, Scope &submodule) {
  return submodules_.emplace(name, &submodule).second;
}
DeclTypeSpec &Scope::MakeDeclTypeSpec(TypeCategory category, int kind) {
  DeclTypeSpec type{IntrinsicTypeSpec{category, kind}};
  auto it{std::find(declTypeSpecs_.begin(), declTypeSpecs_.end(), type)};
  if (it != declTypeSpecs_.end()) {
    return *it;
  } else {
    declTypeSpecs_.push_back(type);
    return declTypeSpecs_.back();
  }
}
DeclTypeSpec &Scope::MakeDeclTypeSpec(
    DeclTypeSpec::Category category, const SourceName &name) {
  CHECK(category == DeclTypeSpec::TypeDerived ||
      category == DeclTypeSpec::ClassDerived);
  derivedTypeSpecs_.emplace_back(name);
  declTypeSpecs_.emplace_back(category, derivedTypeSpecs_.back());
  return declTypeSpecs_.back();
}
DeclTypeSpec &Scope::MakeDeclTypeSpec(DeclTypeSpec::Category category) {
  CHECK(category == DeclTypeSpec::TypeStar ||
      category == DeclTypeSpec::ClassStar);
  declTypeSpecs_.emplace_back(category);
  return declTypeSpecs_.back();
}
DerivedTypeSpec &Scope::MakeDerivedTypeSpec(const SourceName &name) {
  derivedTypeSpecs_.emplace_back(name);
  return derivedTypeSpecs_.back();
}

Scope::ImportKind Scope::GetImportKind() const {
  if (importKind_) {
    return *importKind_;
  }
  if (symbol_) {
    if (auto *details{symbol_->detailsIf<SubprogramDetails>()}) {
      if (details->isInterface()) {
        return ImportKind::None;  // default for interface body
      }
    }
  }
  return ImportKind::Default;
}

std::optional<parser::MessageFixedText> Scope::SetImportKind(ImportKind kind) {
  if (!importKind_.has_value()) {
    importKind_ = kind;
    return std::nullopt;
  }
  bool hasNone{kind == ImportKind::None || *importKind_ == ImportKind::None};
  bool hasAll{kind == ImportKind::All || *importKind_ == ImportKind::All};
  // Check C8100 and C898: constraints on multiple IMPORT statements
  if (hasNone || hasAll) {
    return hasNone
        ? "IMPORT,NONE must be the only IMPORT statement in a scope"_err_en_US
        : "IMPORT,ALL must be the only IMPORT statement in a scope"_err_en_US;
  } else if (kind != *importKind_ &&
      (kind != ImportKind::Only || kind != ImportKind::Only)) {
    return "Every IMPORT must have ONLY specifier if one of them does"_err_en_US;
  } else {
    return std::nullopt;
  }
}

void Scope::add_importName(const SourceName &name) {
  importNames_.insert(name);
}

// true if name can be imported or host-associated from parent scope.
bool Scope::CanImport(const SourceName &name) const {
  if (kind_ == Kind::Global) {
    return false;
  }
  switch (GetImportKind()) {
  case ImportKind::None: return false;
  case ImportKind::All:
  case ImportKind::Default: return true;
  case ImportKind::Only: return importNames_.count(name) > 0;
  default: CRASH_NO_CASE;
  }
}

const Scope *Scope::FindScope(const parser::CharBlock &source) const {
  if (!sourceRange_.Contains(source)) {
    return nullptr;
  }
  for (const auto &child : children_) {
    if (const auto *scope{child.FindScope(source)}) {
      return scope;
    }
  }
  return this;
}

void Scope::AddSourceRange(const parser::CharBlock &source) {
  sourceRange_.ExtendToCover(source);
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
}
