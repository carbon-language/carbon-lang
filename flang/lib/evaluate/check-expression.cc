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

#include "check-expression.h"
#include "traverse.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"

using namespace std::literals::string_literals;

namespace Fortran::evaluate {

// Constant expression predicate IsConstantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
class IsConstantExprHelper : public AllTraverse<IsConstantExprHelper> {
public:
  using Base = AllTraverse<IsConstantExprHelper>;
  IsConstantExprHelper() : Base{*this} {}
  using Base::operator();

  template<int KIND> bool operator()(const TypeParamInquiry<KIND> &inq) const {
    return inq.parameter().template get<semantics::TypeParamDetails>().attr() ==
        common::TypeParamAttr::Kind;
  }
  bool operator()(const semantics::Symbol &symbol) const {
    return IsNamedConstant(symbol);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const semantics::ParamValue &param) const {
    return param.isExplicit() && (*this)(param.GetExplicit());
  }
  template<typename T> bool operator()(const FunctionRef<T> &call) const {
    if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
      return intrinsic->name == "kind";
      // TODO: other inquiry intrinsics
    } else {
      return false;
    }
  }

  // Forbid integer division by zero in constants.
  template<int KIND>
  bool operator()(
      const Divide<Type<TypeCategory::Integer, KIND>> &division) const {
    using T = Type<TypeCategory::Integer, KIND>;
    if (const auto divisor{GetScalarConstantValue<T>(division.right())}) {
      return !divisor->IsZero();
    } else {
      return false;
    }
  }
};

template<typename A> bool IsConstantExpr(const A &x) {
  return IsConstantExprHelper{}(x);
}
template bool IsConstantExpr(const Expr<SomeType> &);

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
// The caller is responsible for checking the base object symbol's
// characteristics (TARGET, SAVE, &c.) since this code can't use GetUltimate().
class IsInitialDataTargetHelper
  : public AllTraverse<IsInitialDataTargetHelper> {
public:
  using Base = AllTraverse<IsInitialDataTargetHelper>;
  using Base::operator();
  IsInitialDataTargetHelper() : Base{*this} {}

  bool operator()(const BOZLiteralConstant &) const { return false; }
  bool operator()(const NullPointer &) const { return true; }
  template<typename T> bool operator()(const Constant<T> &) const {
    return false;
  }
  bool operator()(const semantics::Symbol &) const { return true; }
  bool operator()(const StaticDataObject &) const { return false; }
  template<int KIND> bool operator()(const TypeParamInquiry<KIND> &) const {
    return false;
  }
  bool operator()(const Triplet &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        IsConstantExpr(x.stride());
  }
  bool operator()(const Subscript &x) const {
    return std::visit(
        common::visitors{
            [&](const Triplet &t) { return (*this)(t); },
            [&](const auto &y) {
              return y.value().Rank() == 0 && IsConstantExpr(y.value());
            },
        },
        x.u);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const Substring &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        (*this)(x.parent());
  }
  bool operator()(const DescriptorInquiry &) const { return false; }
  template<typename T> bool operator()(const ArrayConstructor<T> &) const {
    return false;
  }
  bool operator()(const StructureConstructor &) const { return false; }
  template<typename T> bool operator()(const FunctionRef<T> &) { return false; }
  template<typename D, typename R, typename... O>
  bool operator()(const Operation<D, R, O...> &) const {
    return false;
  }
  template<typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  bool operator()(const Relational<SomeType> &) const { return false; }

private:
  const semantics::Symbol &(*GetUltimate)(const semantics::Symbol &);
};

bool IsInitialDataTarget(const Expr<SomeType> &x) {
  return IsInitialDataTargetHelper{}(x);
}

// Specification expression validation (10.1.11(2), C1010)
struct CheckSpecificationExprHelper
  : public AnyTraverse<CheckSpecificationExprHelper,
        std::optional<std::string>> {
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<CheckSpecificationExprHelper, Result>;
  CheckSpecificationExprHelper() : Base{*this} {}
  using Base::operator();

  Result operator()(const ProcedureDesignator &) const {
    return "dummy procedure argument";
  }
  Result operator()(const CoarrayRef &) const { return "coindexed reference"; }

  Result operator()(const semantics::Symbol &symbol) const {
    if (semantics::IsNamedConstant(symbol)) {
      return std::nullopt;
    } else if (symbol.IsDummy()) {
      if (symbol.attrs().test(semantics::Attr::OPTIONAL)) {
        return "reference to OPTIONAL dummy argument '"s +
            symbol.name().ToString() + "'";
      } else if (symbol.attrs().test(semantics::Attr::INTENT_OUT)) {
        return "reference to INTENT(OUT) dummy argument '"s +
            symbol.name().ToString() + "'";
      } else if (symbol.has<semantics::ObjectEntityDetails>()) {
        return std::nullopt;
      } else {
        return "dummy procedure argument";
      }
    } else if (symbol.has<semantics::UseDetails>() ||
        symbol.has<semantics::HostAssocDetails>() ||
        symbol.owner().kind() == semantics::Scope::Kind::Module) {
      return std::nullopt;
    } else if (const auto *object{
                   symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      // TODO: what about EQUIVALENCE with data in COMMON?
      // TODO: does this work for blank COMMON?
      if (object->commonBlock() != nullptr) {
        return std::nullopt;
      }
    }
    return "reference to local entity '"s + symbol.name().ToString() + "'";
  }

  Result operator()(const Component &x) const {
    // Don't look at the component symbol.
    return (*this)(x.base());
  }

  template<typename T> Result operator()(const FunctionRef<T> &x) const {
    if (const auto *symbol{x.proc().GetSymbol()}) {
      if (!symbol->attrs().test(semantics::Attr::PURE)) {
        return "reference to impure function '"s + symbol->name().ToString() +
            "'";
      } else if (symbol->owner().kind() == semantics::Scope::Kind::Subprogram) {
        return "reference to internal function '"s + symbol->name().ToString() +
            "'";
      }
      // TODO: other checks for standard module procedures
    } else {
      const SpecificIntrinsic &intrin{DEREF(x.proc().GetSpecificIntrinsic())};
      if (intrin.name == "present") {
        return std::nullopt;  // no need to check argument(s)
      }
      if (IsConstantExpr(x)) {
        return std::nullopt;  // inquiry functions may not need to check
                              // argument(s)
      }
    }
    return (*this)(x.arguments());
  }
};

template<typename A>
void CheckSpecificationExpr(const A &x, parser::ContextualMessages &messages) {
  if (auto why{CheckSpecificationExprHelper{}(x)}) {
    std::stringstream ss;
    ss << x;
    messages.Say("The expression (%s) cannot be used as a "
                 "specification expression (%s)"_err_en_US,
        ss.str(), *why);
  }
}

template void CheckSpecificationExpr(
    const Expr<SomeType> &, parser::ContextualMessages &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SomeInteger>> &, parser::ContextualMessages &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &,
    parser::ContextualMessages &);
}
