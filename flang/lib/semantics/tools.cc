// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "tools.h"
#include "scope.h"
#include "../evaluate/variable.h"
#include <algorithm>
#include <set>
#include <variant>

namespace Fortran::semantics {

static const Symbol *FindCommonBlockInScope(
    const Scope &scope, const Symbol &object) {
  for (const auto &pair : scope.commonBlocks()) {
    const Symbol &block{*pair.second};
    if (IsCommonBlockContaining(block, object)) {
      return &block;
    }
  }
  return nullptr;
}

const Symbol *FindCommonBlockContaining(const Symbol &object) {
  for (const Scope *scope{&object.owner()};
       scope->kind() != Scope::Kind::Global; scope = &scope->parent()) {
    if (const Symbol * block{FindCommonBlockInScope(*scope, object)}) {
      return block;
    }
  }
  return nullptr;
}

const Scope *FindProgramUnitContaining(const Scope &start) {
  const Scope *scope{&start};
  while (scope != nullptr) {
    switch (scope->kind()) {
    case Scope::Kind::Module:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram: return scope;
    case Scope::Kind::Global:
    case Scope::Kind::System: return nullptr;
    case Scope::Kind::DerivedType:
    case Scope::Kind::Block:
    case Scope::Kind::Forall:
    case Scope::Kind::ImpliedDos: scope = &scope->parent();
    }
  }
  return nullptr;
}

const Scope *FindProgramUnitContaining(const Symbol &symbol) {
  return FindProgramUnitContaining(symbol.owner());
}

const Scope *FindPureFunctionContaining(const Scope *scope) {
  scope = FindProgramUnitContaining(*scope);
  while (scope != nullptr) {
    if (IsPureFunction(*scope)) {
      return scope;
    }
    scope = FindProgramUnitContaining(scope->parent());
  }
  return nullptr;
}

bool IsCommonBlockContaining(const Symbol &block, const Symbol &object) {
  const auto &objects{block.get<CommonBlockDetails>().objects()};
  auto found{std::find(objects.begin(), objects.end(), &object)};
  return found != objects.end();
}

bool IsUseAssociated(const Symbol &symbol, const Scope &scope) {
  const Scope *owner{FindProgramUnitContaining(symbol.GetUltimate().owner())};
  return owner != nullptr && owner->kind() == Scope::Kind::Module &&
      owner != FindProgramUnitContaining(scope);
}

bool IsAncestor(const Scope *maybeAncestor, const Scope &maybeDescendent) {
  if (maybeAncestor == nullptr) {
    return false;
  }
  const Scope *scope{&maybeDescendent};
  while (scope->kind() != Scope::Kind::Global) {
    scope = &scope->parent();
    if (scope == maybeAncestor) {
      return true;
    }
  }
  return false;
}

bool IsHostAssociated(const Symbol &symbol, const Scope &scope) {
  return IsAncestor(FindProgramUnitContaining(symbol), scope);
}

bool IsDummy(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    return details->isDummy();
  } else if (const auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    return details->isDummy();
  } else {
    return false;
  }
}

bool IsPointerDummy(const Symbol &symbol) {
  return symbol.attrs().test(Attr::POINTER) && IsDummy(symbol);
}

bool IsFunction(const Symbol &symbol) {
  if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    return procDetails->interface().type() != nullptr ||
        (procDetails->interface().symbol() != nullptr &&
            IsFunction(*procDetails->interface().symbol()));
  } else if (const auto *subprogram{symbol.detailsIf<SubprogramDetails>()}) {
    return subprogram->isFunction();
  } else {
    return false;
  }
}

bool IsPureFunction(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PURE) && IsFunction(symbol);
}

bool IsPureFunction(const Scope &scope) {
  if (const Symbol * symbol{scope.GetSymbol()}) {
    return IsPureFunction(*symbol);
  } else {
    return false;
  }
}

static bool HasPointerComponent(
    const Scope &scope, std::set<const Scope *> &visited) {
  if (scope.kind() != Scope::Kind::DerivedType) {
    return false;
  }
  if (!visited.insert(&scope).second) {
    return false;
  }
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (symbol.attrs().test(Attr::POINTER)) {
      return true;
    }
    if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          if (const Scope * nested{derived->scope()}) {
            if (HasPointerComponent(*nested, visited)) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

bool HasPointerComponent(const Scope &scope) {
  std::set<const Scope *> visited;
  return HasPointerComponent(scope, visited);
}

bool HasPointerComponent(const DerivedTypeSpec &derived) {
  if (const Scope * scope{derived.scope()}) {
    return HasPointerComponent(*scope);
  } else {
    return false;
  }
}

bool HasPointerComponent(const DeclTypeSpec &type) {
  if (const DerivedTypeSpec * derived{type.AsDerived()}) {
    return HasPointerComponent(*derived);
  } else {
    return false;
  }
}

bool HasPointerComponent(const DeclTypeSpec *type) {
  return type != nullptr && HasPointerComponent(*type);
}

bool IsOrHasPointerComponent(const Symbol &symbol) {
  return symbol.attrs().test(Attr::POINTER) ||
      HasPointerComponent(symbol.GetType());
}

// C1594 specifies several ways by which an object might be globally visible.
bool IsExternallyVisibleObject(const Symbol &object, const Scope &scope) {
  return IsUseAssociated(object, scope) || IsHostAssociated(object, scope) ||
      (IsPureFunction(scope) && IsPointerDummy(object)) ||
      (object.attrs().test(Attr::INTENT_IN) && IsDummy(object)) ||
      FindCommonBlockContaining(object) != nullptr;
  // TODO: Storage association with any object for which this predicate holds
}
}
