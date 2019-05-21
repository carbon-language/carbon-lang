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
#include "semantics.h"
#include "symbol.h"
#include "type.h"
#include "../common/Fortran.h"
#include "../common/indirection.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
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

bool DoesScopeContain(
    const Scope *maybeAncestor, const Scope &maybeDescendent) {
  if (maybeAncestor != nullptr) {
    const Scope *scope{&maybeDescendent};
    while (scope->kind() != Scope::Kind::Global) {
      scope = &scope->parent();
      if (scope == maybeAncestor) {
        return true;
      }
    }
  }
  return false;
}

bool DoesScopeContain(const Scope *maybeAncestor, const Symbol &symbol) {
  return DoesScopeContain(maybeAncestor, symbol.owner());
}

bool IsHostAssociated(const Symbol &symbol, const Scope &scope) {
  return DoesScopeContain(FindProgramUnitContaining(symbol), scope);
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
  return IsPointer(symbol) && IsDummy(symbol);
}

// variable-name
bool IsVariableName(const Symbol &symbol) {
  return symbol.has<ObjectEntityDetails>() && !IsParameter(symbol);
}

// proc-name
bool IsProcName(const Symbol &symbol) {
  return symbol.has<ProcEntityDetails>();
}

bool IsFunction(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &x) { return x.isFunction(); },
          [&](const SubprogramNameDetails &x) {
            return symbol.test(Symbol::Flag::Function);
          },
          [](const ProcEntityDetails &x) {
            const auto &ifc{x.interface()};
            return ifc.type() || (ifc.symbol() && IsFunction(*ifc.symbol()));
          },
          [](const UseDetails &x) { return IsFunction(x.symbol()); },
          [](const auto &) { return false; },
      },
      symbol.details());
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

bool IsProcedure(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const ProcEntityDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const ProcBindingDetails &) { return true; },
          [](const UseDetails &x) { return IsProcedure(x.symbol()); },
          [](const auto &) { return false; },
      },
      symbol.details());
}

bool IsProcedurePointer(const Symbol &symbol) {
  return symbol.has<ProcEntityDetails>() && IsPointer(symbol);
}

static const Symbol *FindPointerComponent(
    const Scope &scope, std::set<const Scope *> &visited) {
  if (scope.kind() != Scope::Kind::DerivedType) {
    return nullptr;
  }
  if (!visited.insert(&scope).second) {
    return nullptr;
  }
  // If there's a top-level pointer component, return it for clearer error
  // messaging.
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (IsPointer(symbol)) {
      return &symbol;
    }
  }
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          if (const Scope * nested{derived->scope()}) {
            if (const Symbol *
                pointer{FindPointerComponent(*nested, visited)}) {
              return pointer;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

const Symbol *FindPointerComponent(const Scope &scope) {
  std::set<const Scope *> visited;
  return FindPointerComponent(scope, visited);
}

const Symbol *FindPointerComponent(const DerivedTypeSpec &derived) {
  if (const Scope * scope{derived.scope()}) {
    return FindPointerComponent(*scope);
  } else {
    return nullptr;
  }
}

const Symbol *FindPointerComponent(const DeclTypeSpec &type) {
  if (const DerivedTypeSpec * derived{type.AsDerived()}) {
    return FindPointerComponent(*derived);
  } else {
    return nullptr;
  }
}

const Symbol *FindPointerComponent(const DeclTypeSpec *type) {
  return type ? FindPointerComponent(*type) : nullptr;
}

const Symbol *FindPointerComponent(const Symbol &symbol) {
  return IsPointer(symbol) ? &symbol : FindPointerComponent(symbol.GetType());
}

// C1594 specifies several ways by which an object might be globally visible.
const Symbol *FindExternallyVisibleObject(
    const Symbol &object, const Scope &scope) {
  // TODO: Storage association with any object for which this predicate holds,
  // once EQUIVALENCE is supported.
  if (IsUseAssociated(object, scope) || IsHostAssociated(object, scope) ||
      (IsPureFunction(scope) && IsPointerDummy(object)) ||
      (object.attrs().test(Attr::INTENT_IN) && IsDummy(object))) {
    return &object;
  } else if (const Symbol * block{FindCommonBlockContaining(object)}) {
    return block;
  } else {
    return nullptr;
  }
}

bool ExprHasTypeCategory(
    const SomeExpr &expr, const common::TypeCategory &type) {
  auto dynamicType{expr.GetType()};
  return dynamicType.has_value() && dynamicType->category() == type;
}

bool ExprTypeKindIsDefault(
    const SomeExpr &expr, const SemanticsContext &context) {
  auto dynamicType{expr.GetType()};
  return dynamicType.has_value() &&
      dynamicType->category() != common::TypeCategory::Derived &&
      dynamicType->kind() ==
      context.defaultKinds().GetDefaultKind(dynamicType->category());
}

const Symbol *FindFunctionResult(const Symbol &symbol) {
  if (const auto *procEntity{symbol.detailsIf<ProcEntityDetails>()}) {
    const ProcInterface &interface{procEntity->interface()};
    if (interface.symbol() != nullptr) {
      return FindFunctionResult(*interface.symbol());
    }
  } else if (const auto *subp{symbol.detailsIf<SubprogramDetails>()}) {
    if (subp->isFunction()) {
      return &subp->result();
    }
  }
  return nullptr;
}

bool IsDerivedTypeFromModule(
    const DerivedTypeSpec *derived, const char *module, const char *name) {
  if (!derived) {
    return false;
  } else {
    const auto &symbol{derived->typeSymbol()};
    return symbol.name() == name && symbol.owner().IsModule() &&
        symbol.owner().name() == module;
  }
}

bool IsTeamType(const DerivedTypeSpec *derived) {
  return IsDerivedTypeFromModule(derived, "iso_fortran_env", "team_type");
}

const Symbol *HasCoarrayUltimateComponent(
    const DerivedTypeSpec &derivedTypeSpec) {
  const Symbol &symbol{derivedTypeSpec.typeSymbol()};
  // TODO is it guaranteed that derived type symbol have a scope and is it the
  // right scope to look into?
  CHECK(symbol.scope());
  for (const Symbol *componentSymbol :
      symbol.get<DerivedTypeDetails>().OrderComponents(*symbol.scope())) {
    CHECK(componentSymbol);
    const ObjectEntityDetails &objectDetails{
        componentSymbol->get<ObjectEntityDetails>()};
    if (objectDetails.IsCoarray()) {
      // Coarrays are ultimate components because they must be allocatable
      // according to C746.
      return componentSymbol;
    }
    if (!IsAllocatableOrPointer(*componentSymbol)) {
      if (const DeclTypeSpec * declTypeSpec{objectDetails.type()}) {
        if (const DerivedTypeSpec *
            componentDerivedTypeSpec{declTypeSpec->AsDerived()}) {
          // Avoid infinite loop, though this should not happen due to C744
          CHECK(&symbol != &componentDerivedTypeSpec->typeSymbol());
          if (const Symbol *
              subcomponent{
                  HasCoarrayUltimateComponent(*componentDerivedTypeSpec)}) {
            return subcomponent;
          }
        }
      }
    }
  }
  return nullptr;
}

const bool IsEventTypeOrLockType(const DerivedTypeSpec *derivedTypeSpec) {
  return IsDerivedTypeFromModule(
             derivedTypeSpec, "iso_fortran_env", "event_type") ||
      IsDerivedTypeFromModule(derivedTypeSpec, "iso_fortran_env", "lock_type");
}

const Symbol *HasEventOrLockPotentialComponent(
    const DerivedTypeSpec &derivedTypeSpec) {

  const Symbol &symbol{derivedTypeSpec.typeSymbol()};
  // TODO is it guaranteed that derived type symbol have a scope and is it the
  // right scope to look into?
  CHECK(symbol.scope());
  for (const Symbol *componentSymbol :
      symbol.get<DerivedTypeDetails>().OrderComponents(*symbol.scope())) {
    CHECK(componentSymbol);
    if (!IsPointer(*componentSymbol)) {
      if (const DeclTypeSpec * declTypeSpec{componentSymbol->GetType()}) {
        if (const DerivedTypeSpec *
            componentDerivedTypeSpec{declTypeSpec->AsDerived()}) {
          // Avoid infinite loop, that may happen if the component
          // is an allocatable of the same type as the derived type.
          // TODO: Is it legal to have longer type loops: i.e type B has a
          // component of type A that has an allocatable component of type B?
          if (&symbol != &componentDerivedTypeSpec->typeSymbol()) {
            if (IsEventTypeOrLockType(componentDerivedTypeSpec)) {
              return componentSymbol;
            } else if (const Symbol *
                subcomponent{HasEventOrLockPotentialComponent(
                    *componentDerivedTypeSpec)}) {
              return subcomponent;
            }
          }
        }
      }
    }
  }
  return nullptr;
}
}
