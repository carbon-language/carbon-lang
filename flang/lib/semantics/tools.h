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

#ifndef FORTRAN_SEMANTICS_TOOLS_H_
#define FORTRAN_SEMANTICS_TOOLS_H_

// Simple predicates and look-up functions that are best defined
// canonically for use in semantic checking.

#include "attr.h"
#include "expression.h"
#include "semantics.h"
#include "../common/Fortran.h"
#include "../evaluate/expression.h"
#include "../evaluate/variable.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
#include <functional>

namespace Fortran::semantics {

class DeclTypeSpec;
class DerivedTypeSpec;
class Scope;
class Symbol;

const Symbol *FindCommonBlockContaining(const Symbol &object);
const Scope *FindProgramUnitContaining(const Scope &);
const Scope *FindProgramUnitContaining(const Symbol &);
const Scope *FindPureProcedureContaining(const Scope *);
const Symbol *FindPointerComponent(const Scope &);
const Symbol *FindPointerComponent(const DerivedTypeSpec &);
const Symbol *FindPointerComponent(const DeclTypeSpec &);
const Symbol *FindPointerComponent(const Symbol &);
const Symbol *FindInterface(const Symbol &);
const Symbol *FindSubprogram(const Symbol &);
const Symbol *FindFunctionResult(const Symbol &);
const Symbol *GetAssociationRoot(const Symbol &);

bool IsCommonBlockContaining(const Symbol &block, const Symbol &object);
bool DoesScopeContain(const Scope *maybeAncestor, const Scope &maybeDescendent);
bool DoesScopeContain(const Scope *, const Symbol &);
bool IsUseAssociated(const Symbol *, const Scope &);
bool IsHostAssociated(const Symbol &, const Scope &);
bool IsDummy(const Symbol &);
bool IsPointerDummy(const Symbol &);
bool IsFunction(const Symbol &);
bool IsPureProcedure(const Symbol &);
bool IsPureProcedure(const Scope &);
bool IsProcedure(const Symbol &);
bool IsProcName(const Symbol &symbol);  // proc-name
bool IsVariableName(const Symbol &symbol);  // variable-name
bool IsProcedurePointer(const Symbol &);
bool IsExtensibleType(const DerivedTypeSpec *);
// Is this a derived type from module with this name?
bool IsDerivedTypeFromModule(
    const DerivedTypeSpec *derived, const char *module, const char *name);
// Is this derived type TEAM_TYPE from module ISO_FORTRAN_ENV
bool IsTeamType(const DerivedTypeSpec *);
// Is this derived type either C_PTR or C_FUNPTR from module ISO_C_BINDING
bool IsIsoCType(const DerivedTypeSpec *);
const bool IsEventTypeOrLockType(const DerivedTypeSpec *);
// Returns an ultimate component symbol that is a
// coarray or nullptr if there are no such component.
// There is no guarantee regarding which ultimate coarray
// component is returned in case there are several because this
// does not really matter for the checks where it is needed.
const Symbol *HasCoarrayUltimateComponent(const DerivedTypeSpec &);
// Same logic as HasCoarrayUltimateComponent, but looking for
// potential component of EVENT_TYPE or LOCK_TYPE from
// ISO_FORTRAN_ENV module.
const Symbol *HasEventOrLockPotentialComponent(const DerivedTypeSpec &);
bool IsOrContainsEventOrLockComponent(const Symbol &);

// Return an ultimate component of type that matches predicate, or nullptr.
const Symbol *FindUltimateComponent(
    const DerivedTypeSpec &type, std::function<bool(const Symbol &)> predicate);

inline bool IsPointer(const Symbol &symbol) {
  return symbol.attrs().test(Attr::POINTER);
}
inline bool IsAllocatable(const Symbol &symbol) {
  return symbol.attrs().test(Attr::ALLOCATABLE);
}
inline bool IsAllocatableOrPointer(const Symbol &symbol) {
  return IsPointer(symbol) || IsAllocatable(symbol);
}
inline bool IsParameter(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PARAMETER);
}
inline bool IsOptional(const Symbol &symbol) {
  return symbol.attrs().test(Attr::OPTIONAL);
}
inline bool IsIntentIn(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_IN);
}
inline bool IsProtected(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PROTECTED);
}
bool IsFinalizable(const Symbol &symbol);
bool IsCoarray(const Symbol &symbol);
bool IsAssumedSizeArray(const Symbol &symbol);
std::optional<parser::MessageFixedText> WhyNotModifiable(
    const Symbol &symbol, const Scope &scope);
// Is the symbol modifiable in this scope
bool IsExternalInPureContext(const Symbol &symbol, const Scope &scope);

// Returns the complete list of derived type parameter symbols in
// the order in which their declarations appear in the derived type
// definitions (parents first).
SymbolVector OrderParameterDeclarations(const Symbol &);
// Returns the complete list of derived type parameter names in the
// order defined by 7.5.3.2.
std::list<SourceName> OrderParameterNames(const Symbol &);

// Create a new instantiation of this parameterized derived type
// for this particular distinct set of actual parameter values.
void InstantiateDerivedType(DerivedTypeSpec &, Scope &, SemanticsContext &);
// Return an existing or new derived type instance
const DeclTypeSpec &FindOrInstantiateDerivedType(Scope &, DerivedTypeSpec &&,
    SemanticsContext &, DeclTypeSpec::Category = DeclTypeSpec::TypeDerived);
void ProcessParameterExpressions(DerivedTypeSpec &, evaluate::FoldingContext &);

// Determines whether an object might be visible outside a
// PURE function (C1594); returns a non-null Symbol pointer for
// diagnostic purposes if so.
const Symbol *FindExternallyVisibleObject(const Symbol &, const Scope &);

template<typename A>
const Symbol *FindExternallyVisibleObject(const A &, const Scope &) {
  return nullptr;  // default base case
}

template<typename T>
const Symbol *FindExternallyVisibleObject(
    const evaluate::Designator<T> &designator, const Scope &scope) {
  if (const Symbol * symbol{designator.GetBaseObject().symbol()}) {
    return FindExternallyVisibleObject(*symbol, scope);
  } else if (std::holds_alternative<evaluate::CoarrayRef>(designator.u)) {
    // Coindexed values are visible even if their image-local objects are not.
    return designator.GetBaseObject().symbol();
  } else {
    return nullptr;
  }
}

template<typename T>
const Symbol *FindExternallyVisibleObject(
    const evaluate::Expr<T> &expr, const Scope &scope) {
  return std::visit(
      [&](const auto &x) { return FindExternallyVisibleObject(x, scope); },
      expr.u);
}

using SomeExpr = evaluate::Expr<evaluate::SomeType>;

bool ExprHasTypeCategory(
    const SomeExpr &expr, const common::TypeCategory &type);
bool ExprTypeKindIsDefault(
    const SomeExpr &expr, const SemanticsContext &context);

struct GetExprHelper {
  const SomeExpr *Get(const parser::Expr::TypedExpr &x) {
    CHECK(x);
    return x->v ? &*x->v : nullptr;
  }
  const SomeExpr *Get(const parser::Expr &x) { return Get(x.typedExpr); }
  const SomeExpr *Get(const parser::Variable &x) { return Get(x.typedExpr); }
  template<typename T> const SomeExpr *Get(const common::Indirection<T> &x) {
    return Get(x.value());
  }
  template<typename T> const SomeExpr *Get(const std::optional<T> &x) {
    return x.has_value() ? Get(x.value()) : nullptr;
  }
  template<typename T> const SomeExpr *Get(const T &x) {
    if constexpr (ConstraintTrait<T>) {
      return Get(x.thing);
    } else if constexpr (WrapperTrait<T>) {
      return Get(x.v);
    } else {
      return nullptr;
    }
  }
};

template<typename T> const SomeExpr *GetExpr(const T &x) {
  return GetExprHelper{}.Get(x);
}

template<typename T> std::optional<std::int64_t> GetIntValue(const T &x) {
  if (const auto *expr{GetExpr(x)}) {
    return evaluate::ToInt64(*expr);
  } else {
    return std::nullopt;
  }
}

}
#endif  // FORTRAN_SEMANTICS_TOOLS_H_
