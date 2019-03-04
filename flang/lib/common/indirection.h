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

#ifndef FORTRAN_COMMON_INDIRECTION_H_
#define FORTRAN_COMMON_INDIRECTION_H_

// Defines several smart pointer class templates that are rather like
// std::unique_ptr<>.
// - Indirection<> is, like a C++ reference type, restricted to be non-null
//   when constructed or assigned.
// - OwningPointer<> is like a std::unique_ptr<> with an out-of-line destructor.
//   This makes it suitable for use with forward-declared content types
//   in a way that bare C pointers allow but std::unique_ptr<> cannot.
// - ForwardReference<> is a kind of Indirection<> that, like OwningPointer<>,
//   accommodates the use of forward declarations.
// Users of Indirection<> and ForwardReference<> need to check whether their
// pointers are null.  Like a C++ reference, they are meant to be as invisible
// as possible.
// All of these can optionally support copy construction
// and copy assignment.

#include "../common/idioms.h"
#include <memory>
#include <type_traits>
#include <utility>

namespace Fortran::common {

// The default case does not support (deep) copy construction and assignment.
template<typename A, bool COPY = false> class Indirection {
public:
  using element_type = A;
  Indirection() = delete;
  Indirection(A *&&p) : p_{p} {
    CHECK(p_ && "assigning null pointer to Indirection");
    p = nullptr;
  }
  Indirection(A &&x) : p_{new A(std::move(x))} {}
  Indirection(Indirection &&that) : p_{that.p_} {
    CHECK(p_ && "move construction of Indirection from null Indirection");
    that.p_ = nullptr;
  }
  ~Indirection() {
    delete p_;
    p_ = nullptr;
  }
  Indirection &operator=(Indirection &&that) {
    CHECK(that.p_ && "move assignment of null Indirection to Indirection");
    auto tmp{p_};
    p_ = that.p_;
    that.p_ = tmp;
    return *this;
  }

  A &value() { return *p_; }
  const A &value() const { return *p_; }
  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

  bool operator==(const A &x) const { return *p_ == x; }
  bool operator==(const Indirection &that) const { return *p_ == *that.p_; }

  template<typename... ARGS> static Indirection Make(ARGS &&... args) {
    return {new A(std::forward<ARGS>(args)...)};
  }

private:
  A *p_{nullptr};
};

// Variant with copy construction and assignment
template<typename A> class Indirection<A, true> {
public:
  using element_type = A;

  Indirection() = delete;
  Indirection(A *&&p) : p_{p} {
    CHECK(p_ && "assigning null pointer to Indirection");
    p = nullptr;
  }
  Indirection(const A &x) : p_{new A(x)} {}
  Indirection(A &&x) : p_{new A(std::move(x))} {}
  Indirection(const Indirection &that) {
    CHECK(that.p_ && "copy construction of Indirection from null Indirection");
    p_ = new A(*that.p_);
  }
  Indirection(Indirection &&that) : p_{that.p_} {
    CHECK(p_ && "move construction of Indirection from null Indirection");
    that.p_ = nullptr;
  }
  ~Indirection() {
    delete p_;
    p_ = nullptr;
  }
  Indirection &operator=(const Indirection &that) {
    CHECK(that.p_ && "copy assignment of Indirection from null Indirection");
    *p_ = *that.p_;
    return *this;
  }
  Indirection &operator=(Indirection &&that) {
    CHECK(that.p_ && "move assignment of null Indirection to Indirection");
    auto tmp{p_};
    p_ = that.p_;
    that.p_ = tmp;
    return *this;
  }

  A &value() { return *p_; }
  const A &value() const { return *p_; }
  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

  bool operator==(const A &x) const { return *p_ == x; }
  bool operator==(const Indirection &that) const { return *p_ == *that.p_; }

  template<typename... ARGS> static Indirection Make(ARGS &&... args) {
    return {new A(std::forward<ARGS>(args)...)};
  }

private:
  A *p_{nullptr};
};

// A variant of Indirection suitable for use with forward-referenced types.
// These are nullable pointers, not references.  Allocation is not available,
// and a single externalized destructor must be defined.  Copyable if an
// external copy constructor and operator= are implemented.
template<typename A> class OwningPointer {
public:
  using element_type = A;

  OwningPointer() {}
  OwningPointer(OwningPointer &&that) : p_{that.p_} { that.p_ = nullptr; }
  explicit OwningPointer(std::unique_ptr<A> &&that) : p_{that.release()} {}
  explicit OwningPointer(A *&&p) : p_{p} { p = nullptr; }

  // Must be externally defined; see DEFINE_OWNING_DESTRUCTOR below
  ~OwningPointer();

  // Must be externally defined if copying is needed.
  OwningPointer(const A &);
  OwningPointer(const OwningPointer &);
  OwningPointer &operator=(const A &);
  OwningPointer &operator=(const OwningPointer &);

  OwningPointer &operator=(OwningPointer &&that) {
    auto tmp{p_};
    p_ = that.p_;
    that.p_ = tmp;
    return *this;
  }
  OwningPointer &operator=(A *&&p) {
    return *this = OwningPointer(std::move(p));
  }

  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

  A *get() const { return p_; }

  bool operator==(const A &x) const { return p_ != nullptr && *p_ == x; }
  bool operator==(const OwningPointer &that) const {
    return (p_ == nullptr && that.p_ == nullptr) ||
        (that.p_ != nullptr && *this == *that.p_);
  }

private:
  A *p_{nullptr};
};

// ForwardReference can be viewed as either a non-nullable variant of
// OwningPointer or as a variant of Indirection that accommodates use with
// a forward-declared content type.
template<typename A> class ForwardReference {
public:
  using element_type = A;

  explicit ForwardReference(std::unique_ptr<A> &&that) : p_{that.release()} {}
  explicit ForwardReference(A *&&p) : p_{p} {
    CHECK(p_ && "assigning null pointer to ForwardReference");
    p = nullptr;
  }
  ForwardReference(ForwardReference<A> &&that) : p_{that.p_} {
    CHECK(p_ &&
        "move construction of ForwardReference from null ForwardReference");
    that.p_ = nullptr;
  }

  // Must be externally defined; see DEFINE_OWNING_DESTRUCTOR below
  ~ForwardReference();

  // Must be externally defined if copying is needed.
  ForwardReference(const A &);
  ForwardReference(const ForwardReference &);
  ForwardReference &operator=(const A &);
  ForwardReference &operator=(const ForwardReference &);

  ForwardReference &operator=(ForwardReference &&that) {
    CHECK(that.p_ &&
        "move assignment of null ForwardReference to ForwardReference");
    auto tmp{p_};
    p_ = that.p_;
    that.p_ = tmp;
    return *this;
  }
  ForwardReference &operator=(A *&&p) {
    return *this = ForwardReference(std::move(p));
  }

  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

  A &value() { return *p_; }
  const A &value() const { return *p_; }

  bool operator==(const A &x) const { return *p_ == x; }
  bool operator==(const ForwardReference &that) const {
    return *p_ == *that.p_;
  }

private:
  A *p_{nullptr};
};
}

// Mandatory instantiation and definition -- put somewhere, not in a namespace
// CLASS here is OwningPointer or ForwardReference.
#define DEFINE_OWNING_DESTRUCTOR(CLASS, A) \
  namespace Fortran::common { \
  template class CLASS<A>; \
  template<> CLASS<A>::~CLASS() { \
    delete p_; \
    p_ = nullptr; \
  } \
  }

// Optional definitions for OwningPointer and ForwardReference
#define DEFINE_OWNING_COPY_CONSTRUCTORS(CLASS, A) \
  namespace Fortran::common { \
  template<> CLASS<A>::CLASS(const A &that) : p_{new A(that)} {} \
  template<> \
  CLASS<A>::CLASS(const CLASS<A> &that) \
    : p_{that.p_ ? new A(*that.p_) : nullptr} {} \
  }
#define DEFINE_OWNING_COPY_ASSIGNMENTS(CLASS, A) \
  namespace Fortran::common { \
  template<> CLASS<A> &CLASS<A>::operator=(const A &that) { \
    delete p_; \
    p_ = new A(that); \
    return *this; \
  } \
  template<> CLASS<A> &CLASS<A>::operator=(const CLASS<A> &that) { \
    delete p_; \
    p_ = that.p_ ? new A(*that.p_) : nullptr; \
    return *this; \
  } \
  }

#define DEFINE_OWNING_COPY_FUNCTIONS(CLASS, A) \
  DEFINE_OWNING_COPY_CONSTRUCTORS(CLASS, A) \
  DEFINE_OWNING_COPY_ASSIGNMENTS(CLASS, A)
#define DEFINE_OWNING_SPECIAL_FUNCTIONS(CLASS, A) \
  DEFINE_OWNING_DESTRUCTOR(CLASS, A) \
  DEFINE_OWNING_COPY_FUNCTIONS(CLASS, A)

#endif  // FORTRAN_COMMON_INDIRECTION_H_
