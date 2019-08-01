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
// See fold.cc to see an example of how this framework was used to
// implement then detection of constant expressions.
//
// The bases of references (component, array, coarray, substring, &
// procedures) are visited before any subscript, cosubscript, or actual
// arguments.  Visitors may rely on this ordering of descent.
//
// To use for non-mutating visitation, define one or more client visitation
// classes of the form:
//   class MyVisitor : public virtual VisitorBase<RESULT> {
//   public:
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
// If Handle() and Pre()/Post() are defined for the same type,
// Handle() has precedence.  This can arise when member function
// templates are used as catch-alls.
//
// Then instantiate and construct a Visitor and its embedded visitors via:
//   Visitor<MyVisitor, ...> v{value...};  // value is/are ARGTYPE &&
// and call:
//   RESULT result{v.Traverse(topLevelExpr)};
// Within the callback routines (Handle, Pre, Post), one may call
//   void Return(A &&);  // to assign to the result and end traversal
//   void Return();  // to end traversal with current result
//   RESULT &result();  // to reference the result to define or update it
// For any given expression object type T for which a callback is defined
// in any visitor class, the callback must be distinct from all others.
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

  // These dummies prevent the "using A::Handle..., "
  // statements in Visitor (below) from failing, while
  // their odd result and argument types prevent them
  // from clashing with actual member function callbacks
  // and member function template callbacks in visitor
  // instances.
  std::nullptr_t Handle(std::nullptr_t);
  std::nullptr_t Pre(std::nullptr_t);
  std::nullptr_t Post(std::nullptr_t);

  void Return() { done_ = true; }

  template<typename A> common::IfNoLvalue<void, A> Return(A &&x) {
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

// Some SFINAE-fu to enable detection of Handle(), Pre() and Post()
// callbacks in "if constexpr ()" predicates that guard calls to them below.
// These have to be declared outside Visitor because they rely on
// specialization.
template<typename A, typename B, typename = void>
struct HasVisitorHandle : std::false_type {};
template<typename A, typename B>
struct HasVisitorHandle<A, B,
    decltype(static_cast<A *>(nullptr)->Handle(
        *static_cast<const B *>(nullptr)))> : std::true_type {};

template<typename A, typename B, typename = void>
struct HasVisitorPre : std::false_type {};
template<typename A, typename B>
struct HasVisitorPre<A, B,
    decltype(static_cast<A *>(nullptr)->Pre(*static_cast<const B *>(nullptr)))>
  : std::true_type {};

template<typename A, typename B, typename = void>
struct HasVisitorPost : std::false_type {};
template<typename A, typename B>
struct HasVisitorPost<A, B,
    decltype(static_cast<A *>(nullptr)->Post(*static_cast<const B *>(nullptr)))>
  : std::true_type {};

template<typename... A>
class Visitor : public virtual VisitorBase<VisitorResultType<A...>>,
                public A... {
public:
  using Result = VisitorResultType<A...>;
  using Base = VisitorBase<Result>;
  using A::Handle..., A::Pre..., A::Post...;

private:
  using VisitorBase<Result>::done_, VisitorBase<Result>::result_;

public:
  template<typename... B> Visitor(B... x) : A{x}... {}
  template<typename B> Result Traverse(const B &x) {
    Visit(x);
    return std::move(result_);
  }

  template<typename B> void Visit(const B &x) {
    if (!done_) {
      if constexpr ((... || HasVisitorHandle<A, B, void>::value)) {
        // At least one visitor declares a member function
        // or member function template Handle() for B.  This call
        // will fail if more than one visitor has done so.
        Handle(x);
      } else {
        if constexpr ((... || HasVisitorPre<A, B, void>::value)) {
          Pre(x);
        }
        if (!done_) {
          descender_.Descend(x);
          if (!done_) {
            if constexpr ((... || HasVisitorPost<A, B, void>::value)) {
              Post(x);
            }
          }
        }
      }
    }
  }

private:
  friend class Descender<Visitor>;
  Descender<Visitor> descender_{*this};
};

class RewriterBase {
public:
  void Return() { done_ = true; }

  // Dummy declarations to ensure that "using A::Handle..." &c.
  // do not fail in Rewriter below.
  std::nullptr_t Handle(std::nullptr_t);
  std::nullptr_t Pre(std::nullptr_t);
  std::nullptr_t Post(std::nullptr_t);

protected:
  bool done_{false};
};

template<typename A, typename B, typename = B>
struct HasMutatorHandle : std::false_type {};
template<typename A, typename B>
struct HasMutatorHandle<A, B,
    decltype(static_cast<A *>(nullptr)->Handle(
        static_cast<B &&>(*static_cast<B *>(nullptr))))> : std::true_type {};

template<typename A, typename B, typename = void>
struct HasMutatorPre : std::false_type {};
template<typename A, typename B>
struct HasMutatorPre<A, B,
    decltype(static_cast<A *>(nullptr)->Pre(*static_cast<const B *>(nullptr)))>
  : std::true_type {};

template<typename A, typename B, typename = B>
struct HasMutatorPost : std::false_type {};
template<typename A, typename B>
struct HasMutatorPost<A, B,
    decltype(static_cast<A *>(nullptr)->Post(
        static_cast<B &&>(*static_cast<B *>(nullptr))))> : std::true_type {};

template<typename... A>
class Rewriter : public virtual RewriterBase, public A... {
public:
  using A::Handle..., A::Pre..., A::Post...;

  template<typename... B> Rewriter(B... x) : A{x}... {}

private:
  using RewriterBase::done_;

public:
  template<typename B> common::IfNoLvalue<B, B> Traverse(B &&x) {
    if (!done_) {
      if constexpr ((... || HasMutatorHandle<A, B, B>::value)) {
        x = Handle(std::move(x));
      } else {
        if constexpr ((... || HasMutatorPre<A, B, B>::value)) {
          Pre(x);
        }
        if (!done_) {
          descender_.Descend(x);
          if (!done_) {
            if constexpr ((... || HasMutatorPost<A, B, B>::value)) {
              x = Post(std::move(x));
            }
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
