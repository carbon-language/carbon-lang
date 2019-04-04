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

#include "descender.h"
#include <type_traits>

// Implements an expression traversal utility framework.
// See fold.cc to see how this framework is used to implement detection
// of constant expressions.
//
// To use for non-mutating visitation, define one or more client visitation
// classes of the form:
//   class MyVisitor : public virtual VisitorBase<RESULT> {
//     using Result = RESULT;
//     explicit MyVisitor(ARGTYPE);  // single-argument constructor
//     void Handle(const T1 &);  // callback for type T1 objects
//     void Pre(const T2 &);  // callback before visiting T2
//     void Post(const T2 &);  // callback after visiting T2
//     ...
//   };
// RESULT should have some default-constructible type, and it must be
// the same type in all of the visitors that you combine in the next step.
//
// Then instantiate and construct a Visitor and its embedded visitors via:
//   Visitor<MyVisitor, ...> v{value...};  // value is/are ARGTYPE &&
// and call:
//   RESULT result{v.Traverse(topLevelExpr)};
// Within the callback routines (Handle, Pre, Post), one may call
//   void Return(RESULT &&);  // to define the result and end traversal
//   void Return();  // to end traversal with current result
//   RESULT &result();  // to reference the result to define or update it
// For any given expression object type T for which a callback is defined
// in any visitor class, the callback must be distinct from all others.
// Further, if there is a Handle(const T &) callback, there cannot be a
// Pre(const T &) or a Post(const T &).
//
// For rewriting traversals, the paradigm is similar; however, the
// argument types are rvalues and the non-void result types match
// the arguments:
//   class MyRewriter : public virtual RewriterBase<RESULT> {
//     using Result = RESULT;
//     explicit MyRewriter(ARGTYPE);  // single-argument constructor
//     T1 Handle(T1 &&);  // rewriting callback for type T1 objects
//     void Pre(T2 &);  // in-place mutating callback before visiting T2
//     T2 Post(T2 &&);  // rewriting callback after visiting T2
//     ...
//   };
//   Rewriter<MyRewriter, ...> rw{value};
//   topLevelExpr = rw.Traverse(std::move(topLevelExpr));

namespace Fortran::evaluate {

template<typename RESULT> class VisitorBase {
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

template<typename A, typename... B> struct VisitorResultTypeHelper {
  using type = typename A::Result;
  static_assert(common::AreSameType<type, typename B::Result...>);
};
template<typename... A>
using VisitorResultType = typename VisitorResultTypeHelper<A...>::type;

template<typename... A>
class Visitor : public virtual VisitorBase<VisitorResultType<A...>>,
                public A... {
public:
  using Result = VisitorResultType<A...>;
  using Base = VisitorBase<Result>;
  using Base::Handle, Base::Pre, Base::Post;
  using A::Handle..., A::Pre..., A::Post...;

private:
  using VisitorBase<Result>::done_, VisitorBase<Result>::result_;

public:
  template<typename... B> Visitor(B... x) : A{x}... {}
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
          descender_.Descend(x);
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

  friend class Descender<Visitor>;
  Descender<Visitor> descender_{*this};
};

class RewriterBase {
public:
  template<typename A> A Handle(A &&x) {
    defaultHandleCalled_ = true;
    return std::move(x);
  }
  template<typename A> void Pre(const A &) {}
  template<typename A> A Post(A &&x) { return std::move(x); }

  void Return() { done_ = true; }

protected:
  bool done_{false};
  bool defaultHandleCalled_{false};
};

template<typename... A>
class Rewriter : public virtual RewriterBase, public A... {
public:
  using RewriterBase::Handle, RewriterBase::Pre, RewriterBase::Post;
  using A::Handle..., A::Pre..., A::Post...;

  template<typename... B> Rewriter(B... x) : A{x}... {}

private:
  using RewriterBase::done_, RewriterBase::defaultHandleCalled_;

public:
  template<typename B> B Traverse(B &&x) {
    if (!done_) {
      defaultHandleCalled_ = false;
      x = Handle(std::move(x));
      if (defaultHandleCalled_) {
        Pre(x);
        if (!done_) {
          descender_.Descend(x);
          if (!done_) {
            x = Post(std::move(x));
          }
        }
      }
    }
    return x;
  }

  Descender<Rewriter> descender_{*this};
};
}
#endif  // FORTRAN_EVALUATE_TRAVERSAL_H_
