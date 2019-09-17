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

#ifndef FORTRAN_EVALUATE_DESCENDER_H_
#define FORTRAN_EVALUATE_DESCENDER_H_

// Helper friend class templates for Visitor::Visit() and Rewriter::Traverse().

#include "expression.h"
#include "../semantics/type.h"
#include <algorithm>

namespace Fortran::evaluate {

template<typename VISITOR> class Descender {
public:
  explicit Descender(VISITOR &v) : visitor_{v} {}

  // Base cases
  void Descend(const semantics::Symbol &) {}
  void Descend(semantics::Symbol &) {}
  template<typename T> void Descend(const Constant<T> &) {}
  template<typename T> void Descend(Constant<T> &) {}
  void Descend(const std::shared_ptr<StaticDataObject> &) {}
  void Descend(std::shared_ptr<StaticDataObject> &) {}
  void Descend(const ImpliedDoIndex &) {}
  void Descend(ImpliedDoIndex &) {}
  void Descend(const BOZLiteralConstant &) {}
  void Descend(BOZLiteralConstant &) {}
  void Descend(const NullPointer &) {}
  void Descend(NullPointer &) {}

  template<typename X> void Descend(const X *p) {
    if (p != nullptr) {
      Visit(*p);
    }
  }
  template<typename X> void Descend(X *p) {
    if (p != nullptr) {
      Visit(*p);
    }
  }

  template<typename X> void Descend(const std::optional<X> &o) {
    if (o.has_value()) {
      Visit(*o);
    }
  }
  template<typename X> void Descend(std::optional<X> &o) {
    if (o.has_value()) {
      Visit(*o);
    }
  }

  template<typename X, bool COPY>
  void Descend(const common::Indirection<X, COPY> &p) {
    Visit(p.value());
  }
  template<typename X, bool COPY>
  void Descend(common::Indirection<X, COPY> &p) {
    Visit(p.value());
  }

  template<typename X, typename DELETER>
  void Descend(const std::unique_ptr<X, DELETER> &p) {
    if (p.get() != nullptr) {
      Visit(*p);
    }
  }
  template<typename X, typename DELETER>
  void Descend(std::unique_ptr<X, DELETER> &p) {
    if (p.get() != nullptr) {
      Visit(*p);
    }
  }

  template<typename... X> void Descend(const std::variant<X...> &u) {
    std::visit([&](const auto &x) { Visit(x); }, u);
  }
  template<typename... X> void Descend(std::variant<X...> &u) {
    std::visit([&](auto &x) { Visit(x); }, u);
  }

  template<typename X> void Descend(const std::vector<X> &xs) {
    for (const auto &x : xs) {
      Visit(x);
    }
  }
  template<typename X> void Descend(std::vector<X> &xs) {
    for (auto &x : xs) {
      Visit(x);
    }
  }

  void Descend(const GenericExprWrapper &w) { Visit(w.v); }
  void Descend(GenericExprWrapper &w) { Visit(w.v); }

  template<typename T> void Descend(const Expr<T> &expr) { Visit(expr.u); }
  template<typename T> void Descend(Expr<T> &expr) { Visit(expr.u); }

  template<typename D, typename R, typename X>
  void Descend(const Operation<D, R, X> &op) {
    Visit(op.left());
  }
  template<typename D, typename R, typename X>
  void Descend(Operation<D, R, X> &op) {
    Visit(op.left());
  }
  template<typename D, typename R, typename X, typename Y>
  void Descend(const Operation<D, R, X, Y> &op) {
    Visit(op.left());
    Visit(op.right());
  }
  template<typename D, typename R, typename X, typename Y>
  void Descend(Operation<D, R, X, Y> &op) {
    Visit(op.left());
    Visit(op.right());
  }

  void Descend(const Relational<SomeType> &r) { Visit(r.u); }
  void Descend(Relational<SomeType> &r) { Visit(r.u); }

  template<typename R> void Descend(const ImpliedDo<R> &ido) {
    Visit(ido.lower());
    Visit(ido.upper());
    Visit(ido.stride());
    Visit(ido.values());
  }
  template<typename R> void Descend(ImpliedDo<R> &ido) {
    Visit(ido.lower());
    Visit(ido.upper());
    Visit(ido.stride());
    Visit(ido.values());
  }

  template<typename R> void Descend(const ArrayConstructorValue<R> &av) {
    Visit(av.u);
  }
  template<typename R> void Descend(ArrayConstructorValue<R> &av) {
    Visit(av.u);
  }

  template<typename R> void Descend(const ArrayConstructorValues<R> &avs) {
    for (const auto &x : avs) {
      Visit(x);
    }
  }
  template<typename R> void Descend(ArrayConstructorValues<R> &avs) {
    for (auto &x : avs) {
      Visit(x);
    }
  }

  template<int KIND>
  void Descend(
      const ArrayConstructor<Type<TypeCategory::Character, KIND>> &ac) {
    const ArrayConstructorValues<Type<TypeCategory::Character, KIND>> &base{ac};
    Visit(base);
    Visit(ac.LEN());
  }
  template<int KIND>
  void Descend(ArrayConstructor<Type<TypeCategory::Character, KIND>> &ac) {
    ArrayConstructorValues<Type<TypeCategory::Character, KIND>> &base{ac};
    Visit(base);
    Visit(ac.LEN());
  }

  void Descend(const semantics::ParamValue &param) {
    Visit(param.GetExplicit());
  }
  void Descend(semantics::ParamValue &param) { Visit(param.GetExplicit()); }

  void Descend(const semantics::DerivedTypeSpec &derived) {
    for (const auto &pair : derived.parameters()) {
      Visit(pair.second);
    }
  }
  void Descend(semantics::DerivedTypeSpec &derived) {
    for (const auto &pair : derived.parameters()) {
      Visit(pair.second);
    }
  }

  void Descend(const StructureConstructor &sc) {
    Visit(sc.derivedTypeSpec());
    for (const auto &pair : sc) {
      Visit(pair.second);
    }
  }
  void Descend(StructureConstructor &sc) {
    Visit(sc.derivedTypeSpec());
    for (const auto &pair : sc) {
      Visit(pair.second);
    }
  }

  void Descend(const BaseObject &object) { Visit(object.u); }
  void Descend(BaseObject &object) { Visit(object.u); }

  void Descend(const Component &component) {
    Visit(component.base());
    Visit(component.GetLastSymbol());
  }
  void Descend(Component &component) {
    Visit(component.base());
    Visit(component.GetLastSymbol());
  }

  void Descend(const NamedEntity &x) {
    if (const Component * c{x.UnwrapComponent()}) {
      Visit(*c);
    } else {
      Visit(x.GetLastSymbol());
    }
  }
  void Descend(NamedEntity &x) {
    if (Component * c{x.UnwrapComponent()}) {
      Visit(*c);
    } else {
      Visit(x.GetLastSymbol());
    }
  }

  template<int KIND> void Descend(const TypeParamInquiry<KIND> &inq) {
    Visit(inq.base());
    Visit(inq.parameter());
  }
  template<int KIND> void Descend(TypeParamInquiry<KIND> &inq) {
    Visit(inq.base());
    Visit(inq.parameter());
  }

  void Descend(const DescriptorInquiry &inq) { Visit(inq.base()); }
  void Descend(DescriptorInquiry &inq) { Visit(inq.base()); }

  void Descend(const Triplet &triplet) {
    Visit(triplet.lower());
    Visit(triplet.upper());
    Visit(triplet.stride());
  }
  void Descend(Triplet &triplet) {
    if (auto x{triplet.lower()}) {
      Visit(*x);
      triplet.set_lower(std::move(*x));
    }
    if (auto x{triplet.upper()}) {
      Visit(*x);
      triplet.set_upper(std::move(*x));
    }
    triplet.set_stride(Visit(triplet.stride()));
  }

  void Descend(const Subscript &sscript) { Visit(sscript.u); }
  void Descend(Subscript &sscript) { Visit(sscript.u); }

  void Descend(const ArrayRef &aref) {
    Visit(aref.base());
    Visit(aref.subscript());
  }
  void Descend(ArrayRef &aref) {
    Visit(aref.base());
    Visit(aref.subscript());
  }

  void Descend(const CoarrayRef &caref) {
    Visit(caref.base());
    Visit(caref.subscript());
    Visit(caref.cosubscript());
    Visit(caref.stat());
    Visit(caref.team());
  }
  void Descend(CoarrayRef &caref) {
    Visit(caref.base());
    Visit(caref.subscript());
    Visit(caref.cosubscript());
    if (auto x{caref.stat()}) {
      Visit(*x);
      caref.set_stat(std::move(*x));
    }
    if (auto x{caref.team()}) {
      Visit(*x);
      caref.set_team(std::move(*x), caref.teamIsTeamNumber());
    }
  }

  void Descend(const DataRef &data) { Visit(data.u); }
  void Descend(DataRef &data) { Visit(data.u); }

  void Descend(const ComplexPart &z) { Visit(z.complex()); }
  void Descend(ComplexPart &z) { Visit(z.complex()); }

  void Descend(const Substring &ss) {
    Visit(ss.parent());
    Visit(ss.lower());
    Visit(ss.upper());
  }
  void Descend(Substring &ss) {
    Visit(ss.parent());
    auto lx{ss.lower()};
    Visit(lx);
    ss.set_lower(std::move(lx));
    if (auto ux{ss.upper()}) {
      Visit(ux);
      ss.set_upper(std::move(*ux));
    }
  }

  template<typename T> void Descend(const Designator<T> &designator) {
    Visit(designator.u);
  }
  template<typename T> void Descend(Designator<T> &designator) {
    Visit(designator.u);
  }

  template<typename T> void Descend(const Variable<T> &var) { Visit(var.u); }
  template<typename T> void Descend(Variable<T> &var) { Visit(var.u); }

  void Descend(const ActualArgument &arg) {
    if (const auto *expr{arg.UnwrapExpr()}) {
      Visit(*expr);
    } else {
      const semantics::Symbol *aType{arg.GetAssumedTypeDummy()};
      Visit(*aType);
    }
  }
  void Descend(ActualArgument &arg) {
    if (auto *expr{arg.UnwrapExpr()}) {
      Visit(*expr);
    } else {
      const semantics::Symbol *aType{arg.GetAssumedTypeDummy()};
      Visit(*aType);
    }
  }

  void Descend(const SpecificIntrinsic &) {}
  void Descend(SpecificIntrinsic &) {}

  void Descend(const ProcedureDesignator &p) { Visit(p.u); }
  void Descend(ProcedureDesignator &p) { Visit(p.u); }

  void Descend(const ProcedureRef &call) {
    Visit(call.proc());
    Visit(call.arguments());
  }
  void Descend(ProcedureRef &call) {
    Visit(call.proc());
    Visit(call.arguments());
  }

private:
  template<typename T> void Visit(const T &x) { return visitor_.Visit(x); }
  template<typename T> void Visit(T &x) { x = visitor_.Traverse(std::move(x)); }

  VISITOR &visitor_;
};

template<typename Derived> class ExpressionPredicateHelperBase {
private:
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

public:
  template<typename A, bool C>
  bool operator()(const common::Indirection<A, C> &x) {
    return derived()(x.value());
  }
  template<typename A> bool operator()(const A *x) {
    if (x != nullptr) {
      return derived()(*x);
    } else {
      return Derived::DefaultResult;
    }
  }
  template<typename A> bool operator()(const std::optional<A> &x) {
    if (x.has_value()) {
      return derived()(*x);
    } else {
      return Derived::DefaultResult;
    }
  }
  template<typename... A> bool operator()(const std::variant<A...> &u) {
    return std::visit(derived(), u);
  }
  template<typename A> bool operator()(const std::vector<A> &x) {
    if constexpr (Derived::IsConjunction) {
      return std::all_of(x.begin(), x.end(), derived());
    } else {
      return std::any_of(x.begin(), x.end(), derived());
    }
  }
};

template<typename Derived> struct ExpressionPredicateHelperSumTypeMixins {
private:
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

public:
  template<typename T> bool operator()(const ArrayConstructorValue<T> &x) {
    return derived()(x.u);
  }
  template<typename T> bool operator()(const ArrayConstructorValues<T> &x) {
    if constexpr (Derived::IsConjunction) {
      return std::all_of(x.begin(), x.end(), *this);
    } else {
      return std::any_of(x.begin(), x.end(), *this);
    }
  }
  template<typename T> bool operator()(const ImpliedDo<T> &x) {
    if constexpr (Derived::IsConjunction) {
      return derived()(x.lower()) && derived()(x.upper()) &&
          derived()(x.stride()) && derived()(x.values());
    } else {
      return derived()(x.lower()) || derived()(x.upper()) ||
          derived()(x.stride()) || derived()(x.values());
    }
  }
  bool operator()(const StructureConstructor &x) {
    if constexpr (Derived::IsConjunction) {
      return std::all_of(x.begin(), x.end(), *this);
    } else {
      return std::any_of(x.begin(), x.end(), *this);
    }
  }
  bool operator()(const StructureConstructorValues::value_type &x) {
    return derived()(x.second);
  }
  template<typename D, typename R, typename O>
  bool operator()(const Operation<D, R, O> &op) {
    return derived()(op.left());
  }
  template<typename D, typename R, typename LO, typename RO>
  bool operator()(const Operation<D, R, LO, RO> &op) {
    return derived()(op.left()) && derived()(op.right());
  }
  template<typename T> bool operator()(const Expr<T> &x) {
    return derived()(x.u);
  }
  bool operator()(const Relational<SomeType> &x) { return derived()(x.u); }
};

template<typename Derived> struct ExpressionPredicateHelperVariableMixins {
private:
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

public:
  bool operator()(const NamedEntity &x) {
    if (const Component * component{x.UnwrapComponent()}) {
      return derived()(*component);
    } else {
      return derived()(x.GetFirstSymbol());
    }
  }
  bool operator()(const Triplet &x) {
    if constexpr (Derived::IsConjunction) {
      return derived()(x.lower()) && derived()(x.upper()) &&
          derived()(x.stride());
    } else {
      return derived()(x.lower()) || derived()(x.upper()) ||
          derived()(x.stride());
    }
  }
  bool operator()(const Substring &x) {
    if constexpr (Derived::IsConjunction) {
      return derived()(x.parent()) && derived()(x.lower()) &&
          derived()(x.upper());
    } else {
      return derived()(x.parent()) || derived()(x.lower()) ||
          derived()(x.upper());
    }
  }
  bool operator()(const Subscript &x) { return derived()(x.u); }
  bool operator()(const DataRef &x) { return derived()(x.u); }
  bool operator()(const ComplexPart &x) { return derived()(x.complex()); }
  template<typename T> bool operator()(const Designator<T> &x) {
    return derived()(x.u);
  }
  template<typename T> bool operator()(const Variable<T> &x) {
    return derived()(x.u);
  }
  bool operator()(const DescriptorInquiry &x) { return derived()(x.base()); }
};
}
#endif  // FORTRAN_EVALUATE_DESCENDER_H_
