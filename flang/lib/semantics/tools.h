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
bool IsFunctionResult(const Symbol &);
bool IsFunctionResultWithSameNameAsFunction(const Symbol &);
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
inline bool IsAssumedSizeArray(const Symbol &symbol) {
  const auto *details{symbol.detailsIf<ObjectEntityDetails>()};
  return details && details->IsAssumedSize();
}
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

// Derived type component iterator that provides a C++ LegacyForwadIterator
// iterator over the Ordered, Direct, Ultimate or Potential components of a
// DerivedTypeSpec. These iterators can be used with STL algorithms
// accepting LegacyForwadIterator.
// The kind of component is a template argument of the iterator factory
// ComponentIterator.
//
//
// - Ordered components are the components from the component order defined
// in 7.5.4.7, except that the parent components IS added between the parent
// component order and the components in order of declaration.
// This "deviation" is important for structure-constructor analysis.
// For this kind of iterator, the component tree is recursively visited in the
// following order:
//  - first, the Ordered components of the parent type (if relevant)
//  - then, the parent component (if relevant, different from 7.5.4.7!)
//  - then, the components in declaration order (without visiting subcomponents)
//
// - Ultimate, Direct and Potential components are as defined in 7.5.1.
// Parent and procedure components are considered against these definitions.
// For this kind of iterator, the component tree is recursively visited in the
// following order:
//  - the parent component first (if relevant)
//  - then, the components of the parent type (if relevant)
//      + visiting the component and then, if it is derived type data component,
//        visiting the subcomponents before visiting the next
//        component in declaration order.
//  - then, components in declaration order, similarly to components of parent
//    type.
//  Here, the parent component is visited first so that search for a component
//  verifying a property will never descend into a component that already
//  verifies the property (this helps giving clearer feedback).
//
// ComponentIterator::const_iterator remain valid during the whole lifetime of
// the DerivedTypeSpec passed by reference to the ComponentIterator factory.
// Their validity is independent of the ComponentIterator factory lifetime.
//
// For safety and simplicity, the iterators are read only and can only be
// incremented. This could be changed if desired.
//
// Note that iterators are made in such a way that one can easily test and build
// info message in the following way:
//    ComponentIterator<ComponentIterator> comp{derived}
//    if (auto it{std::find_if(comp.begin(), comp.end(), predicate)}) {
//       msg = it.BuildResultDesignatorName() + " verifies predicates";
//       const Symbol* component{*it};
//       ....
//    }

ENUM_CLASS(ComponentKind, Ordered, Direct, Ultimate, Potential)

template<ComponentKind componentKind> class ComponentIterator {
public:
  ComponentIterator(const DerivedTypeSpec &derived) : derived_{derived} {}
  class const_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Symbol *;
    using difference_type = void;
    using pointer = const value_type *;
    using reference = const value_type &;

    static const_iterator Create(const DerivedTypeSpec &);

    const_iterator &operator++() {
      Increment();
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp(*this);
      Increment();
      return tmp;
    }
    reference operator*() const {
      CHECK(!componentPath_.empty());
      return std::get<0>(componentPath_.back());
    }

    bool operator==(const const_iterator &other) const {
      return componentPath_ == other.componentPath_;
    }
    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    // bool() operator indicates if the iterator can be dereferenced without
    // having to check against an end() iterator.
    explicit operator bool() const {
      return !componentPath_.empty() &&
          GetComponentSymbol(componentPath_.back());
    }

    // Build a designator name of the referenced component for messages.
    // The designator helps when the component referred to by the iterator
    // may be "buried" into other components. This gives the full
    // path inside the iterated derived type: e.g "%a%b%c%ultimate"
    // when (*it)->names() only gives "ultimate". Parent component are
    // part of the path for clarity, even though they could be
    // skipped.
    std::string BuildResultDesignatorName() const;

  private:
    using name_iterator = typename std::list<SourceName>::const_iterator;
    using ComponentPathNode =
        std::tuple<const Symbol *, const DerivedTypeSpec *, name_iterator>;
    using ComponentPath = std::vector<ComponentPathNode>;

    static const Symbol *GetComponentSymbol(const ComponentPathNode &node) {
      return std::get<0>(node);
    }
    static void SetComponentSymbol(ComponentPathNode &node, const Symbol *sym) {
      std::get<0>(node) = sym;
    }
    static const Symbol &GetTypeSymbol(const ComponentPathNode &node) {
      return std::get<1>(node)->typeSymbol();
    }
    static const Scope *GetScope(const ComponentPathNode &node) {
      return std::get<1>(node)->scope();
    }
    static name_iterator &GetIterator(ComponentPathNode &node) {
      return std::get<2>(node);
    }
    bool PlanComponentTraversal(const Symbol &component);
    void Increment();
    ComponentPath componentPath_;
  };

  const_iterator begin() { return cbegin(); }
  const_iterator end() { return cend(); }
  const_iterator cbegin() { return const_iterator::Create(derived_); }
  const_iterator cend() { return const_iterator{}; }

private:
  const DerivedTypeSpec &derived_;
};

extern template class ComponentIterator<ComponentKind::Ordered>;
extern template class ComponentIterator<ComponentKind::Direct>;
extern template class ComponentIterator<ComponentKind::Ultimate>;
extern template class ComponentIterator<ComponentKind::Potential>;
using OrderedComponentIterator = ComponentIterator<ComponentKind::Ordered>;
using DirectComponentIterator = ComponentIterator<ComponentKind::Direct>;
using UltimateComponentIterator = ComponentIterator<ComponentKind::Ultimate>;
using PotentialComponentIterator = ComponentIterator<ComponentKind::Potential>;

// Common component searches, the iterator returned is referring to the first
// component, according to the order defined for the related ComponentIterator,
// that verifies the property from the name.
// If no components verifies the property, an end iterator (casting to false)
// is returned. Otherwise, the returned iterator cast to true and can be
// dereferenced.
PotentialComponentIterator::const_iterator FindEventOrLockPotentialComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindCoarrayUltimateComponent(
    const DerivedTypeSpec &);
}
#endif  // FORTRAN_SEMANTICS_TOOLS_H_
