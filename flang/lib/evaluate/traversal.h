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
#include <type_traits>

// Implements an expression traversal utility framework.
// See fold.cc to see how this framework is used to implement detection
// of constant expressions.
//
// To use, define one or more client visitation classes of the form:
//   class MyVisitor : public virtual TraversalBase<RESULT> {
//     explicit MyVisitor(ARGTYPE);  // single-argument constructor
//     void Handle(const T1 &);  // callback for type T1 objects
//     void Pre(const T2 &);  // callback before visiting T2
//     void Post(const T2 &);  // callback after visiting T2
//     ...
//   };
// RESULT should have some default-constructible type.
// Then instantiate and construct a Traversal and its embedded MyVisitor via:
//   Traversal<RESULT, MyVisitor, ...> t{value};  // value is ARGTYPE &&
// and call:
//   RESULT result{t.Traverse(topLevelExpr)};
// Within the callback routines (Handle, Pre, Post), one may call
//   void Return(RESULT &&);  // to define the result and end traversal
//   void Return();  // to end traversal with current result
//   RESULT &result();  // to reference the result to define or update it
// For any given expression object type T for which a callback is defined
// in any visitor class, the callback must be distinct from all others.
// Further, if there is a Handle(const T &) callback, there cannot be a
// Pre() or a Post().

namespace Fortran::evaluate {

template<typename RESULT> class TraversalBase {
public:
  using Result = RESULT;

  Result &result() { return result_; }

  // Note the odd return type; it distinguishes these default callbacks
  // from any void-valued client callback.
  template<typename A> std::nullptr_t Handle(const A &) { return nullptr; }
  template<typename A> std::nullptr_t Pre(const A &) { return nullptr; }
  template<typename A> std::nullptr_t Post(const A &) { return nullptr; }

  void Return() { done_ = true; }
  void Return(RESULT &&x) {
    result_ = std::move(x);
    done_ = true;
  }

protected:
  bool done_{false};
  Result result_;
};

template<typename RESULT, typename... A>
class Traversal : public virtual TraversalBase<RESULT>, public A... {
public:
  using Result = RESULT;
  using Base = TraversalBase<Result>;
  using Base::Handle, Base::Pre, Base::Post;
  using A::Handle..., A::Pre..., A::Post...;

private:
  using TraversalBase<Result>::done_, TraversalBase<Result>::result_;

public:
  template<typename... B> Traversal(B... x) : A{x}... {}
  template<typename B> Result Traverse(const B &x) {
    Visit(x);
    return std::move(result_);
  }

private:
  template<typename B> void Visit(const B &x) {
    if (!done_) {
      if constexpr (std::is_same_v<std::decay_t<decltype(Handle(x))>,
                        std::nullptr_t>) {
        // No visitation class defines Handle(B), so try Pre()/Post().
        Pre(x);
        if (!done_) {
          Descend(x);
          if (!done_) {
            Post(x);
          }
        }
      } else {
        static_assert(
            std::is_same_v<std::decay_t<decltype(Pre(x))>, std::nullptr_t>);
        static_assert(
            std::is_same_v<std::decay_t<decltype(Post(x))>, std::nullptr_t>);
        Handle(x);
      }
    }
  }

  template<typename X> void Descend(const X &) {}  // default case

  template<typename X> void Descend(const X *p) {
    if (p != nullptr) {
      Visit(*p);
    }
  }
  template<typename X> void Descend(const std::optional<X> &o) {
    if (o.has_value()) {
      Visit(*o);
    }
  }
  template<typename X> void Descend(const CopyableIndirection<X> &p) {
    Visit(p.value());
  }
  template<typename... X> void Descend(const std::variant<X...> &u) {
    std::visit([&](const auto &x) { Visit(x); }, u);
  }
  template<typename X> void Descend(const std::vector<X> &xs) {
    for (const auto &x : xs) {
      Visit(x);
    }
  }
  template<typename T> void Descend(const Expr<T> &expr) { Visit(expr.u); }
  template<typename D, typename R, typename... O>
  void Descend(const Operation<D, R, O...> &op) {
    Visit(op.left());
    if constexpr (op.operands > 1) {
      Visit(op.right());
    }
  }
  template<typename R> void Descend(const ImpliedDo<R> &ido) {
    Visit(ido.lower());
    Visit(ido.upper());
    Visit(ido.stride());
    Visit(ido.values());
  }
  template<typename R> void Descend(const ArrayConstructorValue<R> &av) {
    Visit(av.u);
  }
  template<typename R> void Descend(const ArrayConstructorValues<R> &avs) {
    Visit(avs.values());
  }
  template<int KIND>
  void Descend(
      const ArrayConstructor<Type<TypeCategory::Character, KIND>> &ac) {
    Visit(static_cast<
        ArrayConstructorValues<Type<TypeCategory::Character, KIND>>>(ac));
    Visit(ac.LEN());
  }
  void Descend(const semantics::ParamValue &param) {
    Visit(param.GetExplicit());
  }
  void Descend(const semantics::DerivedTypeSpec &derived) {
    for (const auto &pair : derived.parameters()) {
      Visit(pair.second);
    }
  }
  void Descend(const StructureConstructor &sc) {
    Visit(sc.derivedTypeSpec());
    for (const auto &pair : sc.values()) {
      Visit(pair.second);
    }
  }
  void Descend(const BaseObject &object) { Visit(object.u); }
  void Descend(const Component &component) {
    Visit(component.base());
    Visit(component.GetLastSymbol());
  }
  template<int KIND> void Descend(const TypeParamInquiry<KIND> &inq) {
    Visit(inq.base());
    Visit(inq.parameter());
  }
  void Descend(const Triplet &triplet) {
    Visit(triplet.lower());
    Visit(triplet.upper());
    Visit(triplet.stride());
  }
  void Descend(const Subscript &sscript) { Visit(sscript.u); }
  void Descend(const ArrayRef &aref) {
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
  void Descend(const DataRef &data) { Visit(data.u); }
  void Descend(const ComplexPart &z) { Visit(z.complex()); }
  template<typename T> void Descend(const Designator<T> &designator) {
    Visit(designator.u);
  }
  template<typename T> void Descend(const Variable<T> &var) { Visit(var.u); }
  void Descend(const ActualArgument &arg) { Visit(arg.value()); }
  void Descend(const ProcedureDesignator &p) { Visit(p.u); }
  void Descend(const ProcedureRef &call) {
    Visit(call.proc());
    Visit(call.arguments());
  }
};
}
#endif  // FORTRAN_EVALUATE_TRAVERSAL_H_
