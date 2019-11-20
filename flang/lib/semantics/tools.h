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
const Scope *FindPureProcedureContaining(const Scope &);
const Scope *FindPureProcedureContaining(const Symbol &);
const Symbol *FindPointerComponent(const Scope &);
const Symbol *FindPointerComponent(const DerivedTypeSpec &);
const Symbol *FindPointerComponent(const DeclTypeSpec &);
const Symbol *FindPointerComponent(const Symbol &);
const Symbol *FindInterface(const Symbol &);
const Symbol *FindSubprogram(const Symbol &);
const Symbol *FindFunctionResult(const Symbol &);
const Symbol *FindOverriddenBinding(const Symbol &);

const DeclTypeSpec *FindParentTypeSpec(const DerivedTypeSpec &);
const DeclTypeSpec *FindParentTypeSpec(const DeclTypeSpec &);
const DeclTypeSpec *FindParentTypeSpec(const Scope &);
const DeclTypeSpec *FindParentTypeSpec(const Symbol &);

// Return the Symbol of the variable of a construct association, if it exists
const Symbol *GetAssociationRoot(const Symbol &);

bool IsGenericDefinedOp(const Symbol &);
bool IsCommonBlockContaining(const Symbol &block, const Symbol &object);
bool DoesScopeContain(const Scope *maybeAncestor, const Scope &maybeDescendent);
bool DoesScopeContain(const Scope *, const Symbol &);
bool IsUseAssociated(const Symbol &, const Scope &);
bool IsHostAssociated(const Symbol &, const Scope &);
bool IsDummy(const Symbol &);
bool IsPointerDummy(const Symbol &);
bool IsFunction(const Symbol &);
bool IsPureProcedure(const Symbol &);
bool IsPureProcedure(const Scope &);
bool IsBindCProcedure(const Symbol &);
bool IsBindCProcedure(const Scope &);
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
bool IsEventTypeOrLockType(const DerivedTypeSpec *);
bool IsOrContainsEventOrLockComponent(const Symbol &);
// Has an explicit or implied SAVE attribute
bool IsSaved(const Symbol &);
bool CanBeTypeBoundProc(const Symbol *);

// Return an ultimate component of type that matches predicate, or nullptr.
const Symbol *FindUltimateComponent(const DerivedTypeSpec &type,
    const std::function<bool(const Symbol &)> &predicate);
const Symbol *FindUltimateComponent(
    const Symbol &symbol, const std::function<bool(const Symbol &)> &predicate);

// Returns an immediate component of type that matches predicate, or nullptr.
// An immediate component of a type is one declared for that type or is an
// immediate component of the type that it extends.
const Symbol *FindImmediateComponent(
    const DerivedTypeSpec &, const std::function<bool(const Symbol &)> &);

inline bool IsPointer(const Symbol &symbol) {
  return symbol.attrs().test(Attr::POINTER);
}
inline bool IsAllocatable(const Symbol &symbol) {
  return symbol.attrs().test(Attr::ALLOCATABLE);
}
inline bool IsAllocatableOrPointer(const Symbol &symbol) {
  return IsPointer(symbol) || IsAllocatable(symbol);
}
inline bool IsNamedConstant(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PARAMETER);
}
inline bool IsOptional(const Symbol &symbol) {
  return symbol.attrs().test(Attr::OPTIONAL);
}
inline bool IsIntentIn(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_IN);
}
inline bool IsIntentInOut(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_INOUT);
}
inline bool IsIntentOut(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_OUT);
}
inline bool IsProtected(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PROTECTED);
}
bool IsFinalizable(const Symbol &);
bool IsFinalizable(const DerivedTypeSpec &);
bool HasImpureFinal(const DerivedTypeSpec &);
bool IsCoarray(const Symbol &);
inline bool IsAssumedSizeArray(const Symbol &symbol) {
  const auto *details{symbol.detailsIf<ObjectEntityDetails>()};
  return details && details->IsAssumedSize();
}
bool IsAssumedLengthCharacter(const Symbol &);
bool IsAssumedLengthCharacterFunction(const Symbol &);
// Is the symbol modifiable in this scope
std::optional<parser::MessageFixedText> WhyNotModifiable(
    const Symbol &, const Scope &);
std::unique_ptr<parser::Message> WhyNotModifiable(SourceName, const SomeExpr &,
    const Scope &, bool vectorSubscriptIsOk = false);
const Symbol *IsExternalInPureContext(const Symbol &, const Scope &);
bool HasCoarray(const parser::Expr &);
bool IsPolymorphicAllocatable(const Symbol &);

// Analysis of image control statements
bool IsImageControlStmt(const parser::ExecutableConstruct &);
// Get the location of the image control statement in this ExecutableConstruct
parser::CharBlock GetImageControlStmtLocation(
    const parser::ExecutableConstruct &);
// Image control statements that reference coarrays need an extra message
// to clarify why they're image control statements.  This function returns
// std::nullopt for ExecutableConstructs that do not require an extra message.
std::optional<parser::MessageFixedText> GetImageControlStmtCoarrayMsg(
    const parser::ExecutableConstruct &);

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
    return x ? Get(*x) : nullptr;
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

// Derived type component iterator that provides a C++ LegacyForwardIterator
// iterator over the Ordered, Direct, Ultimate or Potential components of a
// DerivedTypeSpec. These iterators can be used with STL algorithms
// accepting LegacyForwardIterator.
// The kind of component is a template argument of the iterator factory
// ComponentIterator.
//
// - Ordered components are the components from the component order defined
// in 7.5.4.7, except that the parent component IS added between the parent
// component order and the components in order of declaration.
// This "deviation" is important for structure-constructor analysis.
// For this kind of iterator, the component tree is recursively visited in the
// following order:
//  - first, the Ordered components of the parent type (if relevant)
//  - then, the parent component (if relevant, different from 7.5.4.7!)
//  - then, the components in declaration order (without visiting subcomponents)
//
// - Ultimate, Direct and Potential components are as defined in 7.5.1.
//   - Ultimate components of a derived type are the closure of its components
//     of intrinsic type, its ALLOCATABLE or POINTER components, and the
//     ultimate components of its non-ALLOCATABLE non-POINTER derived type
//     components.  (No ultimate component has a derived type unless it is
//     ALLOCATABLE or POINTER.)
//   - Direct components of a derived type are all of its components, and all
//     of the direct components of its non-ALLOCATABLE non-POINTER derived type
//     components.  (Direct components are always present.)
//   - Potential subobject components of a derived type are the closure of
//     its non-POINTER components and the potential subobject components of
//     its non-POINTER derived type components.  (The lifetime of each
//     potential subobject component is that of the entire instance.)
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
//    ComponentIterator<ComponentKind::...> comp{derived}
//    if (auto it{std::find_if(comp.begin(), comp.end(), predicate)}) {
//       msg = it.BuildResultDesignatorName() + " verifies predicates";
//       const Symbol *component{*it};
//       ....
//    }

ENUM_CLASS(ComponentKind, Ordered, Direct, Ultimate, Potential, Scope)

template<ComponentKind componentKind> class ComponentIterator {
public:
  ComponentIterator(const DerivedTypeSpec &derived) : derived_{derived} {}
  class const_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = SymbolRef;
    using difference_type = void;
    using pointer = const Symbol *;
    using reference = const Symbol &;

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
      return DEREF(componentPath_.back().component());
    }
    pointer operator->() const { return &**this; }

    bool operator==(const const_iterator &other) const {
      return componentPath_ == other.componentPath_;
    }
    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    // bool() operator indicates if the iterator can be dereferenced without
    // having to check against an end() iterator.
    explicit operator bool() const { return !componentPath_.empty(); }

    // Builds a designator name of the referenced component for messages.
    // The designator helps when the component referred to by the iterator
    // may be "buried" into other components. This gives the full
    // path inside the iterated derived type: e.g "%a%b%c%ultimate"
    // when it->name() only gives "ultimate". Parent components are
    // part of the path for clarity, even though they could be
    // skipped.
    std::string BuildResultDesignatorName() const;

  private:
    using name_iterator =
        std::conditional_t<componentKind == ComponentKind::Scope,
            typename Scope::const_iterator,
            typename std::list<SourceName>::const_iterator>;

    class ComponentPathNode {
    public:
      explicit ComponentPathNode(const DerivedTypeSpec &derived)
        : derived_{derived} {
        if constexpr (componentKind == ComponentKind::Scope) {
          const Scope &scope{DEREF(derived.scope())};
          nameIterator_ = scope.cbegin();
          nameEnd_ = scope.cend();
        } else {
          const std::list<SourceName> &nameList{
              derived.typeSymbol().get<DerivedTypeDetails>().componentNames()};
          nameIterator_ = nameList.cbegin();
          nameEnd_ = nameList.cend();
        }
      }
      const Symbol *component() const { return component_; }
      void set_component(const Symbol &component) { component_ = &component; }
      bool visited() const { return visited_; }
      void set_visited(bool yes) { visited_ = yes; }
      bool descended() const { return descended_; }
      void set_descended(bool yes) { descended_ = yes; }
      name_iterator &nameIterator() { return nameIterator_; }
      name_iterator nameEnd() { return nameEnd_; }
      const Symbol &GetTypeSymbol() const { return derived_->typeSymbol(); }
      const Scope &GetScope() const { return DEREF(derived_->scope()); }
      bool operator==(const ComponentPathNode &that) const {
        return &*derived_ == &*that.derived_ &&
            nameIterator_ == that.nameIterator_ &&
            component_ == that.component_;
      }

    private:
      common::Reference<const DerivedTypeSpec> derived_;
      name_iterator nameEnd_;
      name_iterator nameIterator_;
      const Symbol *component_{nullptr};  // until Increment()
      bool visited_{false};
      bool descended_{false};
    };

    const DerivedTypeSpec *PlanComponentTraversal(
        const Symbol &component) const;
    // Advances to the next relevant symbol, if any.  Afterwards, the
    // iterator will either be at its end or contain no null component().
    void Increment();

    std::vector<ComponentPathNode> componentPath_;
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
extern template class ComponentIterator<ComponentKind::Scope>;
using OrderedComponentIterator = ComponentIterator<ComponentKind::Ordered>;
using DirectComponentIterator = ComponentIterator<ComponentKind::Direct>;
using UltimateComponentIterator = ComponentIterator<ComponentKind::Ultimate>;
using PotentialComponentIterator = ComponentIterator<ComponentKind::Potential>;
using ScopeComponentIterator = ComponentIterator<ComponentKind::Scope>;

// Common component searches, the iterator returned is referring to the first
// component, according to the order defined for the related ComponentIterator,
// that verifies the property from the name.
// If no component verifies the property, an end iterator (casting to false)
// is returned. Otherwise, the returned iterator casts to true and can be
// dereferenced.
PotentialComponentIterator::const_iterator FindEventOrLockPotentialComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindCoarrayUltimateComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindPointerUltimateComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindAllocatableUltimateComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator
FindPolymorphicAllocatableUltimateComponent(const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator
FindPolymorphicAllocatableNonCoarrayUltimateComponent(const DerivedTypeSpec &);

}
#endif  // FORTRAN_SEMANTICS_TOOLS_H_
