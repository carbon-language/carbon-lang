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

#ifndef FORTRAN_EVALUATE_TRAVERSAL_H_
#define FORTRAN_EVALUATE_TRAVERSAL_H_

#include "expression.h"

// Implements an expression traversal utility framework.
namespace Fortran::evaluate {

template<typename RESULT>
class TraversalBase {
public:
  using Result = RESULT;
  template<typename A> void Handle(const A &) { defaultHandle_ = true; }
  template<typename A> void Pre(const A &) {}
  template<typename A> void Post(const A &) {}
  template<typename... A> void Return(A &&...x) {
    result_.emplace(std::move(x)...);
  }
protected:
  std::optional<Result> result_;
  bool defaultHandle_{false};
};

// Descend() is a helper function template for Traversal::Visit().
// Do not use directly.
namespace descend {
template<typename VISITOR, typename EXPR> void Descend(VISITOR &, const EXPR &) {}
template<typename V, typename A> void Descend(V &visitor, const A *p) {
  if (p != nullptr) {
    visitor.Visit(*p);
  }
}
template<typename V, typename A> void Descend(V &visitor, const std::optional<A> *o) {
  if (o.has_value()) {
    visitor.Visit(*o);
  }
}
template<typename V, typename A> void Descend(V &visitor, const CopyableIndirection<A> &p) {
  visitor.Visit(p.value());
}
template<typename V, typename... A> void Descend(V &visitor, const std::variant<A...> &u) {
  std::visit([&](const auto &x){ visitor.Visit(x); }, u);
}
template<typename V, typename A> void Descend(V &visitor, const std::vector<A> &xs) {
  for (const auto &x : xs) {
    visitor.Visit(x);
  }
}
template<typename V, typename T> void Descend(V &visitor, const Expr<T> &expr) {
  visitor.Visit(expr.u);
}
template<typename V, typename D, typename R, typename... O>
void Descend(V &visitor, const Operation<D,R,O...> &op) {
  visitor.Visit(op.left());
  if constexpr (op.operands > 1) {
    visitor.Visit(op.right());
  }
}
template<typename V, typename R> void Descend(V &visitor, const ImpliedDo<R> &ido) {
  visitor.Visit(ido.lower());
  visitor.Visit(ido.upper());
  visitor.Visit(ido.stride());
  visitor.Visit(ido.values());
}
template<typename V, typename R> void Descend(V &visitor, const ArrayConstructorValue<R> &av) {
  visitor.Visit(av.u);
}
template<typename V, typename R> void Descend(V &visitor, const ArrayConstructorValues<R> &avs) {
  visitor.Visit(avs.values());
}
template<typename V, int KIND> void Descend(V &visitor, const ArrayConstructor<Type<TypeCategory::Character, KIND>> &ac) {
  visitor.Visit(static_cast<ArrayConstructorValues<Type<TypeCategory::Character, KIND>>>(ac));
  visitor.Visit(ac.LEN());
}
template<typename V> void Descend(V &visitor, const semantics::ParamValue &param) {
  visitor.Visit(param.GetExplicit());
}
template<typename V> void Descend(V &visitor, const semantics::DerivedTypeSpec &derived) {
  for (const auto &pair : derived.parameters()) {
    visitor.Visit(pair.second);
  }
}
template<typename V> void Descend(V &visitor, const StructureConstructor &sc) {
  visitor.Visit(sc.derivedTypeSpec());
  for (const auto &pair : sc.values()) {
    visitor.Visit(pair.second);
  }
}
template<typename V> void Descend(V &visitor, const BaseObject &object) {
  visitor.Visit(object.u);
}
template<typename V> void Descend(V &visitor, const Component &component) {
  visitor.Visit(component.base());
  visitor.Visit(component.GetLastSymbol());
}
template<typename V, int KIND> void Descend(V &visitor, const TypeParamInquiry<KIND> &inq) {
  visitor.Visit(inq.base());
  visitor.Visit(inq.parameter());
}
template<typename V> void Descend(V &visitor, const Triplet &triplet) {
  visitor.Visit(triplet.lower());
  visitor.Visit(triplet.upper());
  visitor.Visit(triplet.stride());
}
template<typename V> void Descend(V &visitor, const Subscript &sscript) {
  visitor.Visit(sscript.u);
}
template<typename V> void Descend(V &visitor, const ArrayRef &aref) {
  visitor.Visit(aref.base());
  visitor.Visit(aref.subscript());
}
template<typename V> void Descend(V &visitor, const CoarrayRef &caref) {
  visitor.Visit(caref.base());
  visitor.Visit(caref.subscript());
  visitor.Visit(caref.cosubscript());
  visitor.Visit(caref.stat());
  visitor.Visit(caref.team());
}
template<typename V> void Descend(V &visitor, const DataRef &data) {
  visitor.Visit(data.u);
}
template<typename V> void Descend(V &visitor, const ComplexPart &z) {
  visitor.Visit(z.complex());
}
template<typename V, typename T> void Descend(V &visitor, const Designator<T> &designator) {
  visitor.Visit(designator.u);
}
template<typename V, typename T> void Descend(V &visitor, const Variable<T> &var) {
  visitor.Visit(var.u);
}
template<typename V> void Descend(V &visitor, const ActualArgument &arg) {
  visitor.Visit(arg.value());
}
template<typename V> void Descend(V &visitor, const ProcedureDesignator &p) {
  visitor.Visit(p.u);
}
template<typename V> void Descend(V &visitor, const ProcedureRef &call) {
  visitor.Visit(call.proc());
  visitor.Visit(call.arguments());
}
}

template<typename RESULT, typename... A>
class Traversal : public virtual TraversalBase<RESULT>, public virtual A... {
public:
  using Result = RESULT;
  using A::Handle..., A::Pre..., A::Post...;
private:
  using TraversalBase<Result>::result_, TraversalBase<Result>::defaultHandle_;
public:
  template<typename... B> Traversal(B... x) : A{x}... {}
  template<typename B> std::optional<Result> Traverse(const B &x) {
    Visit(x);
    return std::move(result_);
  }

  // TODO: make private, make Descend instances friends
  template<typename B> void Visit(const B &x) {
    if (!result_.has_value()) {
      defaultHandle_ = false;
      Handle(x);
      if (defaultHandle_) {
        Pre(x);
        if (!result_.has_value()) {
          descend::Descend(*this, x);
          if (!result_.has_value()) {
            Post(x);
          }
        }
      }
    }
  }
};
}
#endif  // FORTRAN_EVALUATE_TRAVERSAL_H_
