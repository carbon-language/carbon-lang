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
#include "traversal.h"  // TODO pmk
#include "traverse.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"

namespace Fortran::evaluate {

// Constant expression predicate IsConstantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
class IsConstantExprVisitor : public virtual VisitorBase<bool> {
public:
  using Result = bool;
  explicit IsConstantExprVisitor(int) { result() = true; }

  template<int KIND> void Handle(const TypeParamInquiry<KIND> &inq) {
    Check(inq.parameter().template get<semantics::TypeParamDetails>().attr() ==
        common::TypeParamAttr::Kind);
  }
  void Handle(const semantics::Symbol &symbol) {
    Check(IsNamedConstant(symbol));
  }
  void Handle(const CoarrayRef &) { Return(false); }
  void Pre(const semantics::ParamValue &param) { Check(param.isExplicit()); }
  template<typename T> void Pre(const FunctionRef<T> &call) {
    if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
      Check(intrinsic->name == "kind");
      // TODO: Obviously many other intrinsics can be allowed
    } else {
      Return(false);
    }
  }

  // Forbid integer division by zero in constants.
  template<int KIND>
  void Handle(const Divide<Type<TypeCategory::Integer, KIND>> &division) {
    using T = Type<TypeCategory::Integer, KIND>;
    if (const auto divisor{GetScalarConstantValue<T>(division.right())}) {
      Check(!divisor->IsZero());
    }
  }

private:
  void Check(bool ok) {
    if (!ok) {
      Return(false);
    }
  }
};

template<typename A> bool IsConstantExpr(const A &x) {
  return Visitor<IsConstantExprVisitor>{0}.Traverse(x);
}

bool IsConstantExpr(const Expr<SomeType> &expr) {
  return Visitor<IsConstantExprVisitor>{0}.Traverse(expr);
}

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
// The caller is responsible for checking the base object symbol's
// characteristics (TARGET, SAVE, &c.) since this code can't use GetUltimate().
template<typename A> bool IsInitialDataTarget(const A &) { return false; }
template<typename... A> bool IsInitialDataTarget(const std::variant<A...> &);
bool IsInitialDataTarget(const DataRef &);
template<typename T> bool IsInitialDataTarget(const Expr<T> &);
bool IsInitialDataTarget(const semantics::Symbol &) { return true; }
bool IsInitialDataTarget(const semantics::Symbol *s) {
  return IsInitialDataTarget(*s);
}
bool IsInitialDataTarget(const Component &x) {
  return IsInitialDataTarget(x.base());
}
bool IsInitialDataTarget(const NamedEntity &x) {
  if (const Component * c{x.UnwrapComponent()}) {
    return IsInitialDataTarget(*c);
  } else {
    return IsInitialDataTarget(x.GetLastSymbol());
  }
}
bool IsInitialDataTarget(const Triplet &x) {
  if (auto lower{x.lower()}) {
    if (!IsConstantExpr(*lower)) {
      return false;
    }
  }
  if (auto upper{x.upper()}) {
    if (!IsConstantExpr(*upper)) {
      return false;
    }
  }
  return IsConstantExpr(x.stride());
}
bool IsInitialDataTarget(const Subscript &x) {
  return std::visit(
      common::visitors{
          [](const Triplet &t) { return IsInitialDataTarget(t); },
          [&](const auto &y) {
            return y.value().Rank() == 0 && IsConstantExpr(y.value());
          },
      },
      x.u);
}
bool IsInitialDataTarget(const ArrayRef &x) {
  for (const Subscript &ss : x.subscript()) {
    if (!IsInitialDataTarget(ss)) {
      return false;
    }
  }
  return IsInitialDataTarget(x.base());
}
bool IsInitialDataTarget(const DataRef &x) { return IsInitialDataTarget(x.u); }
bool IsInitialDataTarget(const Substring &x) {
  return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper());
}
bool IsInitialDataTarget(const ComplexPart &x) {
  return IsInitialDataTarget(x.complex());
}
template<typename T> bool IsInitialDataTarget(const Designator<T> &x) {
  return IsInitialDataTarget(x.u);
}
bool IsInitialDataTarget(const NullPointer &) { return true; }
template<typename T> bool IsInitialDataTarget(const Expr<T> &x) {
  return IsInitialDataTarget(x.u);
}
template<typename... A> bool IsInitialDataTarget(const std::variant<A...> &u) {
  return std::visit([](const auto &x) { return IsInitialDataTarget(x); }, u);
}

bool IsInitialDataTarget(const Expr<SomeType> &x) {
  return IsInitialDataTarget(x.u);
}

// Specification expression validation (10.1.11(2), C1010)
class CheckSpecificationExprHelper
  : public PredicateTraverse<CheckSpecificationExprHelper> {
public:
  using Base = PredicateTraverse<CheckSpecificationExprHelper>;
  using Base::operator();

  explicit CheckSpecificationExprHelper(std::string &why)
    : Base{*this}, why_{why} {
    why_.clear();
  }

  bool operator()(const ProcedureDesignator &) {
    return Say("dummy procedure argument");
  }
  bool operator()(const CoarrayRef &) { return Say("coindexed reference"); }

  bool operator()(const semantics::Symbol &symbol) {
    if (semantics::IsNamedConstant(symbol)) {
      return true;
    } else if (symbol.IsDummy()) {
      if (symbol.attrs().test(semantics::Attr::OPTIONAL)) {
        return Say("reference to OPTIONAL dummy argument '" +
            symbol.name().ToString() + "'");
      } else if (symbol.attrs().test(semantics::Attr::INTENT_OUT)) {
        return Say("reference to INTENT(OUT) dummy argument '" +
            symbol.name().ToString() + "'");
      } else if (symbol.has<semantics::ObjectEntityDetails>()) {
        return true;
      } else {
        return Say("dummy procedure argument");
      }
    } else if (symbol.has<semantics::UseDetails>() ||
        symbol.has<semantics::HostAssocDetails>() ||
        symbol.owner().kind() == semantics::Scope::Kind::Module) {
      return true;
    } else if (const auto *object{
                   symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      // TODO: what about EQUIVALENCE with data in COMMON?
      // TODO: does this work for blank COMMON?
      if (object->commonBlock() != nullptr) {
        return true;
      }
    }
    return Say("reference to local entity '" + symbol.name().ToString() + "'");
  }

  bool operator()(const Component &x) {
    // Don't look at the component symbol.
    return (*this)(x.base());
  }

  template<typename T> bool operator()(const FunctionRef<T> &x) {
    if (const auto *symbol{x.proc().GetSymbol()}) {
      if (!symbol->attrs().test(semantics::Attr::PURE)) {
        return Say(
            "reference to impure function '" + symbol->name().ToString() + "'");
      } else if (symbol->owner().kind() == semantics::Scope::Kind::Subprogram) {
        return Say("reference to internal function '" +
            symbol->name().ToString() + "'");
      }
      // TODO: other checks for standard module procedures
    } else {
      const SpecificIntrinsic &intrin{DEREF(x.proc().GetSpecificIntrinsic())};
      if (intrin.name == "present") {
        return true;  // no need to check argument(s)
      }
      if (IsConstantExpr(x)) {
        return true;  // inquiry functions may not need to check argument(s)
      }
    }
    return (*this)(x.arguments());
  }

private:
  bool Say(std::string &&s) {
    if (!why_.empty()) {
      why_ += "; ";
    }
    why_ += std::move(s);
    return false;
  }

  std::string &why_;
};

template<typename A>
void CheckSpecificationExpr(const A &x, parser::ContextualMessages &messages) {
  std::string why;
  if (!CheckSpecificationExprHelper{why}(x)) {
    std::stringstream ss;
    ss << x;
    if (!why.empty()) {
      why = " ("s + why + ')';
    }
    messages.Say(
        "The expression (%s) cannot be used as a "
        "specification expression%s"_err_en_US, ss.str(), why);
  }
}

template void CheckSpecificationExpr(
    const Expr<SomeType> &, parser::ContextualMessages &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SomeInteger>> &, parser::ContextualMessages &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &, parser::ContextualMessages &);
}
