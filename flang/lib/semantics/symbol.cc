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

#include "symbol.h"
#include "scope.h"
#include "../common/idioms.h"
#include <memory>

namespace Fortran::semantics {

std::ostream &operator<<(std::ostream &os, const parser::CharBlock &name) {
  return os << name.ToString();
}

const Scope *ModuleDetails::parent() const {
  return isSubmodule_ ? &scope_->parent() : nullptr;
}
const Scope *ModuleDetails::ancestor() const {
  if (!isSubmodule_) {
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
  type_ = type;
}

void ObjectEntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = type;
}

void ObjectEntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}

ProcEntityDetails::ProcEntityDetails(const EntityDetails &d) {
  if (auto type{d.type()}) {
    interface_.set_type(*type);
  }
}

const Symbol &UseDetails::module() const {
  // owner is a module so it must have a symbol:
  return *symbol_->owner().symbol();
}

UseErrorDetails::UseErrorDetails(const UseDetails &useDetails) {
  add_occurrence(useDetails.location(), *useDetails.module().scope());
}
UseErrorDetails &UseErrorDetails::add_occurrence(
    const SourceName &location, const Scope &module) {
  occurrences_.push_back(std::make_pair(&location, &module));
  return *this;
}

GenericDetails::GenericDetails(const listType &specificProcs) {
  for (const auto *proc : specificProcs) {
    add_specificProc(proc);
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
          [](const GenericDetails &) { return "Generic"; },
          [](const ProcBindingDetails &) { return "ProcBinding"; },
          [](const GenericBindingDetails &) { return "GenericBinding"; },
          [](const FinalProcDetails &) { return "FinalProc"; },
          [](const auto &) { return "unknown"; },
      },
      details);
}

const std::string Symbol::GetDetailsName() const {
  return DetailsToString(details_);
}

void Symbol::set_details(const Details &details) {
  CHECK(CanReplaceDetails(details));
  details_ = details;
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

void Symbol::add_occurrence(const SourceName &name) {
  if (occurrences_.back().begin() != name.begin()) {
    occurrences_.push_back(name);
  }
}
void Symbol::remove_occurrence(const SourceName &name) {
  auto end{occurrences_.end()};
  for (auto it{occurrences_.begin()}; it != end; ++it) {
    if (it->begin() == name.begin()) {
      occurrences_.erase(it);
      return;
    }
  }
}
Symbol &Symbol::GetUltimate() {
  return const_cast<Symbol &>(static_cast<const Symbol *>(this)->GetUltimate());
}
const Symbol &Symbol::GetUltimate() const {
  if (const auto *details{detailsIf<UseDetails>()}) {
    return details->symbol().GetUltimate();
  } else {
    return *this;
  }
}

const DeclTypeSpec *Symbol::GetType() const {
  return std::visit(
      common::visitors{
          [](const EntityDetails &x) {
            return x.type().has_value() ? &x.type().value() : nullptr;
          },
          [](const ObjectEntityDetails &x) {
            return x.type().has_value() ? &x.type().value() : nullptr;
          },
          [](const ProcEntityDetails &x) { return x.interface().type(); },
          [](const auto &) {
            return static_cast<const DeclTypeSpec *>(nullptr);
          },
      },
      details_);
}

void Symbol::SetType(const DeclTypeSpec &type) {
  std::visit(
      common::visitors{
          [&](EntityDetails &x) { x.set_type(type); },
          [&](ObjectEntityDetails &x) { x.set_type(type); },
          [&](ProcEntityDetails &x) { x.interface().set_type(type); },
          [](auto &) {},
      },
      details_);
}

bool Symbol::isSubprogram() const {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const UseDetails &x) { return x.symbol().isSubprogram(); },
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
              auto &ancestor{x.ancestor()->name()};
              auto &parent{x.parent()->name()};
              os << " (" << ancestor.ToString();
              if (parent != ancestor) {
                os << ':' << parent.ToString();
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
            for (const auto &pair : x.occurrences()) {
              os << " from " << pair.second->name() << " at " << *pair.first;
            }
          },
          [&](const GenericDetails &x) {
            for (const auto *proc : x.specificProcs()) {
              os << ' ' << proc->name();
            }
          },
          [&](const ProcBindingDetails &x) {
            os << " => " << x.symbol().name();
          },
          [&](const GenericBindingDetails &) { /* TODO */ },
          [&](const FinalProcDetails &) {},
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
    os << ' ' << symbol.GetDetailsName();
    if (const auto *type{symbol.GetType()}) {
      os << ' ' << *type;
    }
  }
  return os;
}

}  // namespace Fortran::semantics
