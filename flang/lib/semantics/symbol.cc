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

#include "symbol.h"
#include "scope.h"
#include "../common/idioms.h"
#include "../evaluate/fold.h"
#include <ostream>
#include <string>

namespace Fortran::semantics {

std::ostream &operator<<(std::ostream &os, const parser::CharBlock &name) {
  return os << name.ToString();
}

const Scope *ModuleDetails::parent() const {
  return isSubmodule_ && scope_ ? &scope_->parent() : nullptr;
}
const Scope *ModuleDetails::ancestor() const {
  if (!isSubmodule_ || !scope_) {
    return nullptr;
  }
  for (auto *scope{scope_};;) {
    auto *parent{&scope->parent()};
    if (parent->kind() != Scope::Kind::Module) {
      return scope;
    }
    scope = parent;
  }
}
void ModuleDetails::set_scope(const Scope *scope) {
  CHECK(!scope_);
  bool scopeIsSubmodule{scope->parent().kind() == Scope::Kind::Module};
  CHECK(isSubmodule_ == scopeIsSubmodule);
  scope_ = scope;
}

void EntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = &type;
}

void ObjectEntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = &type;
}

void ObjectEntityDetails::ReplaceType(const DeclTypeSpec &type) {
  type_ = &type;
}

void ObjectEntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}

bool ObjectEntityDetails::IsDescriptor() const {
  if (type_ != nullptr) {
    if (const IntrinsicTypeSpec * typeSpec{type_->AsIntrinsic()}) {
      if (typeSpec->category() == TypeCategory::Character) {
        // TODO maybe character lengths won't be in descriptors
        return true;
      }
    } else if (const DerivedTypeSpec * typeSpec{type_->AsDerived()}) {
      if (isDummy()) {
        return true;
      }
      // Any length type parameter?
      if (const Scope * scope{typeSpec->scope()}) {
        if (const Symbol * symbol{scope->symbol()}) {
          if (const auto *details{symbol->detailsIf<DerivedTypeDetails>()}) {
            for (const Symbol *param : details->paramDecls()) {
              if (const auto *details{param->detailsIf<TypeParamDetails>()}) {
                if (details->attr() == common::TypeParamAttr::Len) {
                  return true;
                }
              }
            }
          }
        }
      }
    } else if (type_->category() == DeclTypeSpec::Category::TypeStar ||
        type_->category() == DeclTypeSpec::Category::ClassStar) {
      return true;
    }
  }
  if (IsAssumedShape() || IsDeferredShape() || IsAssumedRank()) {
    return true;
  }
  // TODO: Explicit shape component array dependent on length parameter
  // TODO: Automatic (adjustable) arrays
  return false;
}

ProcEntityDetails::ProcEntityDetails(const EntityDetails &d) {
  if (auto type{d.type()}) {
    interface_.set_type(*type);
  }
}

// A procedure pointer or dummy procedure must be a descriptor if
// and only if it requires a static link.
bool ProcEntityDetails::IsDescriptor() const { return HasExplicitInterface(); }

const Symbol &UseDetails::module() const {
  // owner is a module so it must have a symbol:
  return *symbol_->owner().symbol();
}

UseErrorDetails::UseErrorDetails(const UseDetails &useDetails) {
  add_occurrence(useDetails.location(), *useDetails.module().scope());
}
UseErrorDetails &UseErrorDetails::add_occurrence(
    const SourceName &location, const Scope &module) {
  occurrences_.push_back(std::make_pair(location, &module));
  return *this;
}

GenericDetails::GenericDetails(const SymbolList &specificProcs) {
  for (const auto *proc : specificProcs) {
    add_specificProc(*proc);
  }
}

void GenericDetails::set_specific(Symbol &specific) {
  CHECK(!specific_);
  specific_ = &specific;
}
void GenericDetails::set_derivedType(Symbol &derivedType) {
  CHECK(!derivedType_);
  derivedType_ = &derivedType;
}

const Symbol *GenericDetails::CheckSpecific() const {
  if (specific_) {
    for (const auto *proc : specificProcs_) {
      if (proc == specific_) {
        return nullptr;
      }
    }
    return specific_;
  } else {
    return nullptr;
  }
}

// The name of the kind of details for this symbol.
// This is primarily for debugging.
std::string DetailsToString(const Details &details) {
  return std::visit(
      common::visitors{
          [](const UnknownDetails &) { return "Unknown"; },
          [](const MainProgramDetails &) { return "MainProgram"; },
          [](const ModuleDetails &) { return "Module"; },
          [](const SubprogramDetails &) { return "Subprogram"; },
          [](const SubprogramNameDetails &) { return "SubprogramName"; },
          [](const EntityDetails &) { return "Entity"; },
          [](const ObjectEntityDetails &) { return "ObjectEntity"; },
          [](const ProcEntityDetails &) { return "ProcEntity"; },
          [](const DerivedTypeDetails &) { return "DerivedType"; },
          [](const UseDetails &) { return "Use"; },
          [](const UseErrorDetails &) { return "UseError"; },
          [](const HostAssocDetails &) { return "HostAssoc"; },
          [](const GenericDetails &) { return "Generic"; },
          [](const ProcBindingDetails &) { return "ProcBinding"; },
          [](const GenericBindingDetails &) { return "GenericBinding"; },
          [](const FinalProcDetails &) { return "FinalProc"; },
          [](const TypeParamDetails &) { return "TypeParam"; },
          [](const MiscDetails &) { return "Misc"; },
          [](const auto &) { return "unknown"; },
      },
      details);
}

const std::string Symbol::GetDetailsName() const {
  return DetailsToString(details_);
}

void Symbol::set_details(Details &&details) {
  CHECK(CanReplaceDetails(details));
  details_ = std::move(details);
}

bool Symbol::CanReplaceDetails(const Details &details) const {
  if (has<UnknownDetails>()) {
    return true;  // can always replace UnknownDetails
  } else {
    return std::visit(
        common::visitors{
            [](const UseErrorDetails &) { return true; },
            [=](const ObjectEntityDetails &) { return has<EntityDetails>(); },
            [=](const ProcEntityDetails &) { return has<EntityDetails>(); },
            [=](const SubprogramDetails &) {
              return has<SubprogramNameDetails>();
            },
            [](const auto &) { return false; },
        },
        details);
  }
}

Symbol &Symbol::GetUltimate() {
  return const_cast<Symbol &>(static_cast<const Symbol *>(this)->GetUltimate());
}
const Symbol &Symbol::GetUltimate() const {
  if (const auto *details{detailsIf<UseDetails>()}) {
    return details->symbol().GetUltimate();
  } else if (const auto *details{detailsIf<HostAssocDetails>()}) {
    return details->symbol().GetUltimate();
  } else {
    return *this;
  }
}

DeclTypeSpec *Symbol::GetType() {
  return const_cast<DeclTypeSpec *>(
      const_cast<const Symbol *>(this)->GetType());
}

const DeclTypeSpec *Symbol::GetType() const {
  return std::visit(
      common::visitors{
          [](const EntityDetails &x) { return x.type(); },
          [](const ObjectEntityDetails &x) { return x.type(); },
          [](const ProcEntityDetails &x) { return x.interface().type(); },
          [](const TypeParamDetails &x) { return x.type(); },
          [](const auto &) -> const DeclTypeSpec * { return nullptr; },
      },
      details_);
}

void Symbol::SetType(const DeclTypeSpec &type) {
  std::visit(
      common::visitors{
          [&](EntityDetails &x) { x.set_type(type); },
          [&](ObjectEntityDetails &x) { x.set_type(type); },
          [&](ProcEntityDetails &x) { x.interface().set_type(type); },
          [&](TypeParamDetails &x) { x.set_type(type); },
          [](auto &) {},
      },
      details_);
}

bool Symbol::IsSubprogram() const {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const UseDetails &x) { return x.symbol().IsSubprogram(); },
          [](const auto &) { return false; },
      },
      details_);
}

bool Symbol::HasExplicitInterface() const {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const ProcEntityDetails &x) { return x.HasExplicitInterface(); },
          [](const UseDetails &x) { return x.symbol().HasExplicitInterface(); },
          [](const auto &) { return false; },
      },
      details_);
}

bool Symbol::IsSeparateModuleProc() const {
  if (attrs().test(Attr::MODULE)) {
    if (auto *details{detailsIf<SubprogramDetails>()}) {
      return details->isInterface();
    }
  }
  return false;
}

bool Symbol::IsDescriptor() const {
  if (const auto *objectDetails{detailsIf<ObjectEntityDetails>()}) {
    return objectDetails->IsDescriptor();
  } else if (const auto *procDetails{detailsIf<ProcEntityDetails>()}) {
    if (attrs_.test(Attr::POINTER) || attrs_.test(Attr::EXTERNAL)) {
      return procDetails->IsDescriptor();
    }
  }
  return false;
}

int Symbol::Rank() const {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &sd) {
            if (sd.isFunction()) {
              return sd.result().Rank();
            } else {
              return 0;
            }
          },
          [](const GenericDetails &) {
            return 0; /*TODO*/
          },
          [](const UseDetails &x) { return x.symbol().Rank(); },
          [](const HostAssocDetails &x) { return x.symbol().Rank(); },
          [](const ObjectEntityDetails &oed) {
            return static_cast<int>(oed.shape().size());
          },
          [](const auto &) { return 0; },
      },
      details_);
}

ObjectEntityDetails::ObjectEntityDetails(const EntityDetails &d)
  : isDummy_{d.isDummy()}, type_{d.type()} {}

std::ostream &operator<<(std::ostream &os, const EntityDetails &x) {
  if (x.type()) {
    os << " type: " << *x.type();
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const ObjectEntityDetails &x) {
  if (x.type()) {
    os << " type: " << *x.type();
  }
  if (!x.shape().empty()) {
    os << " shape:";
    for (const auto &s : x.shape()) {
      os << ' ' << s;
    }
  }
  if (x.init_) {
    x.init_->AsFortran(os << " init:");
  }
  return os;
}

bool ProcEntityDetails::HasExplicitInterface() const {
  if (auto *symbol{interface_.symbol()}) {
    return symbol->HasExplicitInterface();
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, const ProcEntityDetails &x) {
  if (auto *symbol{x.interface_.symbol()}) {
    os << ' ' << symbol->name().ToString();
  } else if (auto *type{x.interface_.type()}) {
    os << ' ' << *type;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const DerivedTypeDetails &x) {
  if (!x.extends().empty()) {
    os << " extends:" << x.extends().ToString();
  }
  if (x.sequence()) {
    os << " sequence";
  }
  return os;
}

static std::ostream &DumpType(std::ostream &os, const Symbol &symbol) {
  if (const auto *type{symbol.GetType()}) {
    os << *type << ' ';
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Details &details) {
  os << DetailsToString(details);
  std::visit(
      common::visitors{
          [&](const UnknownDetails &x) {},
          [&](const MainProgramDetails &x) {},
          [&](const ModuleDetails &x) {
            if (x.isSubmodule()) {
              os << " (";
              if (x.ancestor()) {
                auto &ancestor{x.ancestor()->name()};
                os << ancestor;
                if (x.parent()) {
                  auto &parent{x.parent()->name()};
                  if (ancestor != parent) {
                    os << ':' << parent;
                  }
                }
              }
              os << ")";
            }
          },
          [&](const SubprogramDetails &x) {
            os << " (";
            int n = 0;
            for (const auto &dummy : x.dummyArgs()) {
              if (n++ > 0) os << ", ";
              DumpType(os, *dummy);
              os << dummy->name();
            }
            os << ')';
            if (x.isFunction()) {
              os << " result(";
              DumpType(os, x.result());
              os << x.result().name() << ')';
            }
            if (x.isInterface()) {
              os << " interface";
            }
          },
          [&](const SubprogramNameDetails &x) {
            os << ' ' << EnumToString(x.kind());
          },
          [&](const EntityDetails &x) { os << x; },
          [&](const ObjectEntityDetails &x) { os << x; },
          [&](const ProcEntityDetails &x) { os << x; },
          [&](const DerivedTypeDetails &x) { os << x; },
          [&](const UseDetails &x) {
            os << " from " << x.symbol().name() << " in " << x.module().name();
          },
          [&](const UseErrorDetails &x) {
            os << " uses:";
            for (const auto &[location, module] : x.occurrences()) {
              os << " from " << module->name() << " at " << location;
            }
          },
          [](const HostAssocDetails &) {},
          [&](const GenericDetails &x) {
            for (const auto *proc : x.specificProcs()) {
              os << ' ' << proc->name();
            }
          },
          [&](const ProcBindingDetails &x) {
            os << " => " << x.symbol().name();
          },
          [&](const GenericBindingDetails &x) {
            os << " =>";
            char sep{' '};
            for (const auto *proc : x.specificProcs()) {
              os << sep << proc->name().ToString();
              sep = ',';
            }
          },
          [&](const FinalProcDetails &) {},
          [&](const TypeParamDetails &x) {
            if (x.type()) {
              os << ' ' << *x.type();
            }
            os << ' ' << common::EnumToString(x.attr());
            if (x.init()) {
              x.init()->AsFortran(os << " init:");
            }
          },
          [&](const MiscDetails &x) {
            os << ' ' << MiscDetails::EnumToString(x.kind());
          },
      },
      details);
  return os;
}

std::ostream &operator<<(std::ostream &o, Symbol::Flag flag) {
  return o << Symbol::EnumToString(flag);
}

std::ostream &operator<<(std::ostream &o, const Symbol::Flags &flags) {
  std::size_t n{flags.count()};
  std::size_t seen{0};
  for (std::size_t j{0}; seen < n; ++j) {
    Symbol::Flag flag{static_cast<Symbol::Flag>(j)};
    if (flags.test(flag)) {
      if (seen++ > 0) {
        o << ", ";
      }
      o << flag;
    }
  }
  return o;
}

std::ostream &operator<<(std::ostream &os, const Symbol &symbol) {
  os << symbol.name();
  if (!symbol.attrs().empty()) {
    os << ", " << symbol.attrs();
  }
  if (!symbol.flags().empty()) {
    os << " (" << symbol.flags() << ')';
  }
  os << ": " << symbol.details_;
  return os;
}

// Output a unique name for a scope by qualifying it with the names of
// parent scopes. For scopes without corresponding symbols, use the kind
// with an index (e.g. Block1, Block2, etc.).
static void DumpUniqueName(std::ostream &os, const Scope &scope) {
  if (scope.kind() != Scope::Kind::Global) {
    DumpUniqueName(os, scope.parent());
    os << '/';
    if (auto *scopeSymbol{scope.symbol()}) {
      os << scopeSymbol->name().ToString();
    } else {
      int index{1};
      for (auto &child : scope.parent().children()) {
        if (child == scope) {
          break;
        }
        if (child.kind() == scope.kind()) {
          ++index;
        }
      }
      os << Scope::EnumToString(scope.kind()) << index;
    }
  }
}

// Dump a symbol for UnparseWithSymbols. This will be used for tests so the
// format should be reasonably stable.
std::ostream &DumpForUnparse(
    std::ostream &os, const Symbol &symbol, bool isDef) {
  DumpUniqueName(os, symbol.owner());
  os << '/' << symbol.name().ToString();
  if (isDef) {
    if (!symbol.attrs().empty()) {
      os << ' ' << symbol.attrs();
    }
    if (symbol.test(Symbol::Flag::Implicit)) {
      os << " (implicit)";
    }
    if (symbol.test(Symbol::Flag::LocalityLocal)) {
      os << " (local)";
    }
    if (symbol.test(Symbol::Flag::LocalityLocalInit)) {
      os << " (local_init)";
    }
    if (symbol.test(Symbol::Flag::LocalityShared)) {
      os << " (shared)";
    }
    os << ' ' << symbol.GetDetailsName();
    if (const auto *type{symbol.GetType()}) {
      os << ' ' << *type;
    }
  }
  return os;
}

Symbol &Symbol::Instantiate(Scope &scope, const DerivedTypeSpec &spec,
    evaluate::FoldingContext &foldingContext) const {
  auto pair{scope.try_emplace(name_, attrs_)};
  Symbol &symbol{*pair.first->second};
  if (!pair.second) {
    // Symbol was already present in the scope, which can only happen
    // in the case of type parameters that had actual values present in
    // the derived type spec.
    get<TypeParamDetails>();  // confirm or crash with message
    return symbol;
  }
  symbol.attrs_ = attrs_;
  symbol.flags_ = flags_;
  std::visit(
      common::visitors{
          [&](const ObjectEntityDetails &that) {
            symbol.details_ = that;
            ObjectEntityDetails &details{symbol.get<ObjectEntityDetails>()};
            details.set_init(
                evaluate::Fold(foldingContext, std::move(details.init())));
            for (ShapeSpec &dim : details.shape()) {
              if (dim.lbound().isExplicit()) {
                dim.lbound().SetExplicit(Fold(
                    foldingContext, std::move(dim.lbound().GetExplicit())));
              }
              if (dim.ubound().isExplicit()) {
                dim.ubound().SetExplicit(Fold(
                    foldingContext, std::move(dim.ubound().GetExplicit())));
              }
            }
            // TODO: fold cobounds too once we can represent them
          },
          [&](const ProcBindingDetails &that) {
            symbol.details_ = ProcBindingDetails{
                that.symbol().Instantiate(scope, spec, foldingContext)};
          },
          [&](const GenericBindingDetails &that) {
            symbol.details_ = GenericBindingDetails{};
            GenericBindingDetails &details{symbol.get<GenericBindingDetails>()};
            for (const Symbol *sym : that.specificProcs()) {
              details.add_specificProc(
                  sym->Instantiate(scope, spec, foldingContext));
            }
          },
          [&](const TypeParamDetails &that) {
            symbol.details_ = that;
            TypeParamDetails &details{symbol.get<TypeParamDetails>()};
            details.set_init(
                evaluate::Fold(foldingContext, std::move(details.init())));
          },
          [&](const FinalProcDetails &that) { symbol.details_ = that; },
          [&](const auto &) {
            get<ObjectEntityDetails>();  // crashes with actual details
          },
      },
      details_);
  return symbol;
}

const Symbol *Symbol::GetParent() const {
  const auto &details{get<DerivedTypeDetails>()};
  CHECK(scope_ != nullptr);
  if (!details.extends().empty()) {
    auto iter{scope_->find(details.extends())};
    CHECK(iter != scope_->end());
    const Symbol &parentComp{*iter->second};
    CHECK(parentComp.test(Symbol::Flag::ParentComp));
    const auto &object{parentComp.get<ObjectEntityDetails>()};
    const DerivedTypeSpec *derived{object.type()->AsDerived()};
    CHECK(derived != nullptr);
    return &derived->typeSymbol();
  }
  return nullptr;
}

std::list<SourceName> DerivedTypeDetails::OrderParameterNames(
    const Symbol &type) const {
  std::list<SourceName> result;
  if (const Symbol * parent{type.GetParent()}) {
    result = parent->get<DerivedTypeDetails>().OrderParameterNames(*parent);
  }
  for (const auto &name : paramNames_) {
    result.push_back(name);
  }
  return result;
}

std::list<Symbol *> DerivedTypeDetails::OrderParameterDeclarations(
    const Symbol &type) const {
  std::list<Symbol *> result;
  if (const Symbol * parent{type.GetParent()}) {
    result =
        parent->get<DerivedTypeDetails>().OrderParameterDeclarations(*parent);
  }
  for (Symbol *symbol : paramDecls_) {
    result.push_back(symbol);
  }
  return result;
}

void TypeParamDetails::set_type(const DeclTypeSpec &type) {
  CHECK(type_ == nullptr);
  type_ = &type;
}
}
