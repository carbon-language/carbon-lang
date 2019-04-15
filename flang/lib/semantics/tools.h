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

#include "../common/Fortran.h"
#include "../evaluate/variable.h"

namespace Fortran::parser {
class Messages;
struct Expr;
struct Name;
struct Variable;
}

namespace Fortran::evaluate {
class GenericExprWrapper;
}

namespace Fortran::semantics {

class DeclTypeSpec;
class DerivedTypeSpec;
class Scope;
class Symbol;

const Symbol *FindCommonBlockContaining(const Symbol &object);
const Scope *FindProgramUnitContaining(const Scope &);
const Scope *FindProgramUnitContaining(const Symbol &);
const Scope *FindPureFunctionContaining(const Scope *);
const Symbol *FindPointerComponent(const Scope &);
const Symbol *FindPointerComponent(const DerivedTypeSpec &);
const Symbol *FindPointerComponent(const DeclTypeSpec &);
const Symbol *FindPointerComponent(const Symbol &);

bool IsCommonBlockContaining(const Symbol &block, const Symbol &object);
bool DoesScopeContain(const Scope *maybeAncestor, const Scope &maybeDescendent);
bool DoesScopeContain(const Scope *, const Symbol &);
bool IsUseAssociated(const Symbol *, const Scope &);
bool IsHostAssociated(const Symbol &, const Scope &);
bool IsDummy(const Symbol &);
bool IsPointer(const Symbol &);
bool IsPointerDummy(const Symbol &);
bool IsFunction(const Symbol &);
bool IsPureFunction(const Symbol &);
bool IsPureFunction(const Scope &);
bool IsProcName(const Symbol &symbol);  // proc-name
bool IsVariableName(const Symbol &symbol);  // variable-name
bool IsAllocatable(const Symbol &);
bool IsAllocatableOrPointer(const Symbol &);

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

bool ExprHasTypeCategory(
    const evaluate::GenericExprWrapper &expr, const common::TypeCategory &type);

// If this Expr or Variable represents a simple Name, return it.
parser::Name *GetSimpleName(parser::Expr &);
const parser::Name *GetSimpleName(const parser::Expr &);
parser::Name *GetSimpleName(parser::Variable &);
const parser::Name *GetSimpleName(const parser::Variable &);
}
#endif  // FORTRAN_SEMANTICS_TOOLS_H_
