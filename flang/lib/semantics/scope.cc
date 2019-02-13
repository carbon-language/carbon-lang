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

#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "type.h"
#include "../evaluate/fold.h"
#include "../parser/characters.h"
#include <algorithm>
#include <memory>

namespace Fortran::semantics {

Symbols<1024> Scope::allSymbols;

bool Scope::IsModule() const {
  return kind_ == Kind::Module && !symbol_->get<ModuleDetails>().isSubmodule();
}

Scope &Scope::MakeScope(Kind kind, Symbol *symbol) {
  return children_.emplace_back(*this, kind, symbol);
}

Scope::iterator Scope::find(const SourceName &name) {
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
  auto it{find(name)};
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

const DeclTypeSpec &Scope::MakeNumericType(
    TypeCategory category, KindExpr &&kind) {
  return MakeLengthlessType(NumericTypeSpec{category, std::move(kind)});
}
const DeclTypeSpec &Scope::MakeLogicalType(KindExpr &&kind) {
  return MakeLengthlessType(LogicalTypeSpec{std::move(kind)});
}
const DeclTypeSpec &Scope::MakeTypeStarType() {
  return MakeLengthlessType(DeclTypeSpec{DeclTypeSpec::TypeStar});
}
const DeclTypeSpec &Scope::MakeClassStarType() {
  return MakeLengthlessType(DeclTypeSpec{DeclTypeSpec::ClassStar});
}
// Types that can't have length parameters can be reused without having to
// compare length expressions. They are stored in the global scope.
const DeclTypeSpec &Scope::MakeLengthlessType(DeclTypeSpec &&type) {
  auto it{std::find(declTypeSpecs_.begin(), declTypeSpecs_.end(), type)};
  if (it != declTypeSpecs_.end()) {
    return *it;
  } else {
    return declTypeSpecs_.emplace_back(std::move(type));
  }
}

const DeclTypeSpec &Scope::MakeCharacterType(
    ParamValue &&length, KindExpr &&kind) {
  return declTypeSpecs_.emplace_back(
      CharacterTypeSpec{std::move(length), std::move(kind)});
}

const DeclTypeSpec &Scope::MakeDerivedType(
    DeclTypeSpec::Category category, DerivedTypeSpec &&spec) {
  return MakeDerivedType(std::move(spec), category);
}

const DeclTypeSpec &Scope::MakeDerivedType(DeclTypeSpec::Category category,
    DerivedTypeSpec &&instance, SemanticsContext &semanticsContext) {
  DeclTypeSpec &type{declTypeSpecs_.emplace_back(
      category, DerivedTypeSpec{std::move(instance)})};
  type.derivedTypeSpec().Instantiate(*this, semanticsContext);
  return type;
}

DeclTypeSpec &Scope::MakeDerivedType(const Symbol &typeSymbol) {
  CHECK(typeSymbol.has<DerivedTypeDetails>());
  CHECK(typeSymbol.scope() != nullptr);
  return MakeDerivedType(
      DerivedTypeSpec{typeSymbol}, DeclTypeSpec::TypeDerived);
}

DeclTypeSpec &Scope::MakeDerivedType(
    DerivedTypeSpec &&spec, DeclTypeSpec::Category category) {
  return declTypeSpecs_.emplace_back(
      category, DerivedTypeSpec{std::move(spec)});
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

bool Scope::IsParameterizedDerivedType() const {
  if (kind_ != Kind::DerivedType) {
    return false;
  }
  if (const Scope * parent{GetDerivedTypeParent()}) {
    if (parent->IsParameterizedDerivedType()) {
      return true;
    }
  }
  for (const auto &pair : symbols_) {
    if (pair.second->has<TypeParamDetails>()) {
      return true;
    }
  }
  return false;
}

const DeclTypeSpec *Scope::FindInstantiatedDerivedType(
    const DerivedTypeSpec &spec, DeclTypeSpec::Category category) const {
  DeclTypeSpec type{category, spec};
  auto typeIter{std::find(declTypeSpecs_.begin(), declTypeSpecs_.end(), type)};
  if (typeIter != declTypeSpecs_.end()) {
    return &*typeIter;
  }
  return nullptr;
}

const DeclTypeSpec &Scope::FindOrInstantiateDerivedType(DerivedTypeSpec &&spec,
    DeclTypeSpec::Category category, SemanticsContext &semanticsContext) {
  spec.FoldParameterExpressions(semanticsContext.foldingContext());
  if (const DeclTypeSpec * type{FindInstantiatedDerivedType(spec, category)}) {
    return *type;
  }
  // Create a new instantiation of this parameterized derived type
  // for this particular distinct set of actual parameter values.
  DeclTypeSpec &type{MakeDerivedType(std::move(spec), category)};
  type.derivedTypeSpec().Instantiate(*this, semanticsContext);
  return type;
}

void Scope::InstantiateDerivedType(
    Scope &clone, SemanticsContext &semanticsContext) const {
  CHECK(kind_ == Kind::DerivedType);
  clone.sourceRange_ = sourceRange_;
  clone.chars_ = chars_;
  for (const auto &pair : symbols_) {
    pair.second->Instantiate(clone, semanticsContext);
  }
}

const DeclTypeSpec &Scope::InstantiateIntrinsicType(
    const DeclTypeSpec &spec, SemanticsContext &semanticsContext) {
  const IntrinsicTypeSpec *intrinsic{spec.AsIntrinsic()};
  CHECK(intrinsic != nullptr);
  if (evaluate::ToInt64(intrinsic->kind()).has_value()) {
    return spec;  // KIND is already a known constant
  }
  // The expression was not originally constant, but now it must be so
  // in the context of a parameterized derived type instantiation.
  KindExpr copy{intrinsic->kind()};
  evaluate::FoldingContext &foldingContext{semanticsContext.foldingContext()};
  copy = evaluate::Fold(foldingContext, std::move(copy));
  int kind{
      semanticsContext.defaultKinds().GetDefaultKind(intrinsic->category())};
  if (auto value{evaluate::ToInt64(copy)}) {
    if (evaluate::IsValidKindOfIntrinsicType(intrinsic->category(), *value)) {
      kind = *value;
    } else {
      foldingContext.messages().Say(
          "KIND parameter value (%jd) of intrinsic type %s "
          "did not resolve to a supported value"_err_en_US,
          static_cast<std::intmax_t>(*value),
          parser::ToUpperCaseLetters(
              common::EnumToString(intrinsic->category()))
              .data());
    }
  }
  switch (spec.category()) {
  case DeclTypeSpec::Numeric:
    return declTypeSpecs_.emplace_back(
        NumericTypeSpec{intrinsic->category(), KindExpr{kind}});
  case DeclTypeSpec::Logical:
    return declTypeSpecs_.emplace_back(LogicalTypeSpec{KindExpr{kind}});
  case DeclTypeSpec::Character:
    return declTypeSpecs_.emplace_back(CharacterTypeSpec{
        ParamValue{spec.characterTypeSpec().length()}, KindExpr{kind}});
  default: CRASH_NO_CASE;
  }
}

const Symbol *Scope::GetSymbol() const {
  if (symbol_ != nullptr) {
    return symbol_;
  }
  if (derivedTypeSpec_ != nullptr) {
    return &derivedTypeSpec_->typeSymbol();
  }
  return nullptr;
}

const Scope *Scope::GetDerivedTypeParent() const {
  if (const Symbol * symbol{GetSymbol()}) {
    if (const DerivedTypeSpec * parent{symbol->GetParentTypeSpec(this)}) {
      return parent->scope();
    }
  }
  return nullptr;
}
}
