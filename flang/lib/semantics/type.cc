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

#include "type.h"
#include "expression.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "../common/restorer.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../evaluate/type.h"
#include "../parser/characters.h"
#include <algorithm>
#include <sstream>

namespace Fortran::semantics {

DerivedTypeSpec::DerivedTypeSpec(const DerivedTypeSpec &that)
  : typeSymbol_{that.typeSymbol_}, scope_{that.scope_}, parameters_{
                                                            that.parameters_} {}

DerivedTypeSpec::DerivedTypeSpec(DerivedTypeSpec &&that)
  : typeSymbol_{that.typeSymbol_}, scope_{that.scope_}, parameters_{std::move(
                                                            that.parameters_)} {
}

void DerivedTypeSpec::set_scope(const Scope &scope) {
  CHECK(!scope_);
  CHECK(scope.kind() == Scope::Kind::DerivedType);
  scope_ = &scope;
}

ParamValue &DerivedTypeSpec::AddParamValue(
    SourceName name, ParamValue &&value) {
  auto pair{parameters_.insert(std::make_pair(name, std::move(value)))};
  CHECK(pair.second);  // name was not already present
  return pair.first->second;
}

ParamValue *DerivedTypeSpec::FindParameter(SourceName target) {
  return const_cast<ParamValue *>(
      const_cast<const DerivedTypeSpec *>(this)->FindParameter(target));
}

void DerivedTypeSpec::FoldParameterExpressions(
    evaluate::FoldingContext &foldingContext) {
  for (auto &pair : parameters_) {
    if (MaybeIntExpr expr{pair.second.GetExplicit()}) {
      pair.second.SetExplicit(evaluate::Fold(foldingContext, std::move(*expr)));
    }
  }
}

void DerivedTypeSpec::Instantiate(
    Scope &containingScope, SemanticsContext &semanticsContext) {
  Scope &newScope{containingScope.MakeScope(Scope::Kind::DerivedType)};
  newScope.set_derivedTypeSpec(*this);
  scope_ = &newScope;
  const DerivedTypeDetails &typeDetails{typeSymbol_.get<DerivedTypeDetails>()};

  // Evaluate any necessary default initial value expressions for those
  // type parameters that lack explicit initialization.  These expressions
  // are evaluated in the scope of the derived type instance and follow the
  // order in which their declarations appeared so as to allow later
  // parameter values to depend on those of their predecessors.
  // The folded values of the expressions replace the init() expressions
  // of the parameters' symbols in the instantiation's scope.
  evaluate::FoldingContext &foldingContext{semanticsContext.foldingContext()};
  auto restorer{foldingContext.WithPDTInstance(*this)};

  for (const Symbol *symbol :
      typeDetails.OrderParameterDeclarations(typeSymbol_)) {
    const SourceName &name{symbol->name()};
    const TypeParamDetails &details{symbol->get<TypeParamDetails>()};
    MaybeIntExpr expr;
    ParamValue *paramValue{FindParameter(name)};
    if (paramValue != nullptr) {
      expr = paramValue->GetExplicit();
    } else {
      expr = details.init();
      expr = evaluate::Fold(foldingContext, std::move(expr));
    }
    // Ensure that any kind type parameters are constant by now.
    if (details.attr() == common::TypeParamAttr::Kind && expr.has_value()) {
      // Any errors in rank and type will have already elicited messages, so
      // don't complain further here.
      if (auto maybeDynamicType{expr->GetType()}) {
        if (expr->Rank() == 0 &&
            maybeDynamicType->category == TypeCategory::Integer &&
            !evaluate::ToInt64(expr).has_value()) {
          std::stringstream fortran;
          expr->AsFortran(fortran);
          if (auto *msg{foldingContext.messages().Say(
                  "Value of kind type parameter '%s' (%s) is not "
                  "scalar INTEGER constant"_err_en_US,
                  name.ToString().data(), fortran.str().data())}) {
            msg->Attach(name, "declared here"_en_US);
          }
        }
      }
    }
    if (expr.has_value()) {
      const Scope *typeScope{typeSymbol_.scope()};
      if (typeScope != nullptr &&
          typeScope->find(symbol->name()) != typeScope->end()) {
        // This type parameter belongs to the derived type itself, not
        // one of its parents.  Put the type parameter expression value
        // into the new scope as the initialization value for the parameter
        // so that type parameter inquiries can acquire it.
        TypeParamDetails instanceDetails{details.attr()};
        instanceDetails.set_init(std::move(*expr));
        Symbol *parameter{newScope.try_emplace(name, std::move(instanceDetails))
                              .first->second};
        CHECK(parameter != nullptr);
      } else if (paramValue != nullptr) {
        // Update the type parameter value in the spec for parent component
        // derived type instantiation later (in symbol.cc) and folding.
        paramValue->SetExplicit(std::move(*expr));
      } else {
        // Save the resolved value in the spec in case folding needs it.
        AddParamValue(symbol->name(), ParamValue{std::move(*expr)});
      }
    }
  }

  // Instantiate every non-parameter symbol from the original derived
  // type's scope into the new instance.
  const Scope *typeScope{typeSymbol_.scope()};
  CHECK(typeScope != nullptr);
  typeScope->InstantiateDerivedType(newScope, semanticsContext);
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  o << x.typeSymbol().name().ToString();
  if (!x.parameters_.empty()) {
    o << '(';
    bool first = true;
    for (const auto &[name, value] : x.parameters_) {
      if (first) {
        first = false;
      } else {
        o << ',';
      }
      o << name.ToString() << '=' << value;
    }
    o << ')';
  }
  return o;
}

Bound::Bound(int bound) : expr_{bound} {}

std::ostream &operator<<(std::ostream &o, const Bound &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else if (x.expr_) {
    x.expr_->AsFortran(o);
  } else {
    o << "<no-expr>";
  }
  return o;
}

std::ostream &operator<<(std::ostream &o, const ShapeSpec &x) {
  if (x.lb_.isAssumed()) {
    CHECK(x.ub_.isAssumed());
    o << "..";
  } else {
    if (!x.lb_.isDeferred()) {
      o << x.lb_;
    }
    o << ':';
    if (!x.ub_.isDeferred()) {
      o << x.ub_;
    }
  }
  return o;
}

std::ostream &operator<<(std::ostream &os, const ArraySpec &arraySpec) {
  char sep{'('};
  for (auto &shape : arraySpec) {
    os << sep << shape;
    sep = ',';
  }
  if (sep == ',') {
    os << ')';
  }
  return os;
}

bool IsExplicit(const ArraySpec &arraySpec) {
  for (const auto &shapeSpec : arraySpec) {
    if (!shapeSpec.isExplicit()) {
      return false;
    }
  }
  return true;
}

ParamValue::ParamValue(MaybeIntExpr &&expr) : expr_{std::move(expr)} {}
ParamValue::ParamValue(SomeIntExpr &&expr) : expr_{std::move(expr)} {}
ParamValue::ParamValue(std::int64_t value)
  : ParamValue(SomeIntExpr{evaluate::Expr<evaluate::SubscriptInteger>{value}}) {
}

void ParamValue::SetExplicit(SomeIntExpr &&x) {
  category_ = Category::Explicit;
  expr_ = std::move(x);
}

std::ostream &operator<<(std::ostream &o, const ParamValue &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else if (!x.GetExplicit()) {
    o << "<no-expr>";
  } else {
    x.GetExplicit()->AsFortran(o);
  }
  return o;
}

IntrinsicTypeSpec::IntrinsicTypeSpec(TypeCategory category, KindExpr &&kind)
  : category_{category}, kind_{std::move(kind)} {
  CHECK(category != TypeCategory::Derived);
}

std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x) {
  os << parser::ToUpperCaseLetters(common::EnumToString(x.category()));
  if (auto k{evaluate::ToInt64(x.kind())}) {
    return os << '(' << *k << ')';  // emit unsuffixed kind code
  } else {
    return x.kind().AsFortran(os << '(') << ')';
  }
}

std::ostream &operator<<(std::ostream &os, const CharacterTypeSpec &x) {
  os << "CHARACTER(" << x.length() << ',';
  if (auto k{evaluate::ToInt64(x.kind())}) {
    return os << *k << ')';  // emit unsuffixed kind code
  } else {
    return x.kind().AsFortran(os) << ')';
  }
}

DeclTypeSpec::DeclTypeSpec(NumericTypeSpec &&typeSpec)
  : category_{Numeric}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(LogicalTypeSpec &&typeSpec)
  : category_{Logical}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(const CharacterTypeSpec &typeSpec)
  : category_{Character}, typeSpec_{typeSpec} {}
DeclTypeSpec::DeclTypeSpec(CharacterTypeSpec &&typeSpec)
  : category_{Character}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(Category category, const DerivedTypeSpec &typeSpec)
  : category_{category}, typeSpec_{typeSpec} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category, DerivedTypeSpec &&typeSpec)
  : category_{category}, typeSpec_{std::move(typeSpec)} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category) : category_{category} {
  CHECK(category == TypeStar || category == ClassStar);
}
bool DeclTypeSpec::IsNumeric(TypeCategory tc) const {
  return category_ == Numeric && numericTypeSpec().category() == tc;
}
IntrinsicTypeSpec *DeclTypeSpec::AsIntrinsic() {
  return const_cast<IntrinsicTypeSpec *>(
      const_cast<const DeclTypeSpec *>(this)->AsIntrinsic());
}
const NumericTypeSpec &DeclTypeSpec::numericTypeSpec() const {
  CHECK(category_ == Numeric);
  return std::get<NumericTypeSpec>(typeSpec_);
}
const LogicalTypeSpec &DeclTypeSpec::logicalTypeSpec() const {
  CHECK(category_ == Logical);
  return std::get<LogicalTypeSpec>(typeSpec_);
}
const DerivedTypeSpec &DeclTypeSpec::derivedTypeSpec() const {
  CHECK(category_ == TypeDerived || category_ == ClassDerived);
  return std::get<DerivedTypeSpec>(typeSpec_);
}
DerivedTypeSpec &DeclTypeSpec::derivedTypeSpec() {
  CHECK(category_ == TypeDerived || category_ == ClassDerived);
  return std::get<DerivedTypeSpec>(typeSpec_);
}
bool DeclTypeSpec::operator==(const DeclTypeSpec &that) const {
  return category_ == that.category_ && typeSpec_ == that.typeSpec_;
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  switch (x.category()) {
  case DeclTypeSpec::Numeric: return o << x.numericTypeSpec();
  case DeclTypeSpec::Logical: return o << x.logicalTypeSpec();
  case DeclTypeSpec::Character: return o << x.characterTypeSpec();
  case DeclTypeSpec::TypeDerived:
    return o << "TYPE(" << x.derivedTypeSpec() << ')';
  case DeclTypeSpec::ClassDerived:
    return o << "CLASS(" << x.derivedTypeSpec() << ')';
  case DeclTypeSpec::TypeStar: return o << "TYPE(*)";
  case DeclTypeSpec::ClassStar: return o << "CLASS(*)";
  default: CRASH_NO_CASE; return o;
  }
}

void ProcInterface::set_symbol(const Symbol &symbol) {
  CHECK(!type_);
  symbol_ = &symbol;
}
void ProcInterface::set_type(const DeclTypeSpec &type) {
  CHECK(!symbol_);
  type_ = &type;
}
}
