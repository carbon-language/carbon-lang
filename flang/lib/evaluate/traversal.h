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

namespace descend {
template<typename VISITOR, typename EXPR>
void Descend(VISITOR &, const EXPR &) {}
}

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
          descend::Descend(*this, x);
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

  template<typename B> friend void descend::Descend(Traversal &, const B &);
};
}

// Helper friend function template definitions
#include "traversal-descend.h"
#endif  // FORTRAN_EVALUATE_TRAVERSAL_H_
