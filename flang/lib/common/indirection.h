// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

// Defines a smart pointer class template that's rather like std::unique_ptr<>
// but further restricted, like a C++ reference, to be non-null when constructed
// or assigned.  Users need not check whether these pointers are null.
// Supports copy construction, too.
// Intended to be as invisible as a reference, wherever possible.

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
  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

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
  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

  bool operator==(const Indirection &that) const { return *p_ == *that.p_; }

  template<typename... ARGS> static Indirection Make(ARGS &&... args) {
    return {new A(std::forward<ARGS>(args)...)};
  }

private:
  A *p_{nullptr};
};

// A variant of Indirection suitable for use with forward-referenced types.
// These are nullable pointers, not references.  Allocation is not available,
// and a single externalized destructor must be defined.
template<typename A> class OwningPointer {
public:
  using element_type = A;

  OwningPointer() {}
  OwningPointer(OwningPointer &&that) : p_{that.release()} {}
  explicit OwningPointer(std::unique_ptr<A> &&that) : p_{that.release()} {}
  explicit OwningPointer(A *&&p) : p_{p} { p = nullptr; }
  ~OwningPointer();
  OwningPointer &operator=(OwningPointer &&that) {
    reset(that.release());
    return *this;
  }

  A &operator*() { return *p_; }
  const A &operator*() const { return *p_; }
  A *operator->() { return p_; }
  const A *operator->() const { return p_; }

  A *get() const { return p_; }

  A *release() {
    A *result{p_};
    p_ = nullptr;
    return result;
  }

  void reset(A *p) {
    this->~OwningPointer();
    p_ = p;
  }

  bool operator==(const A &x) const {
    return p_ != nullptr && (p_ == &x || *p_ == x);
  }
  bool operator==(const OwningPointer &that) const {
    return (p_ == nullptr && that.p_ == nullptr) ||
        (that.p_ != nullptr && *this == *that.p_);
  }

private:
  A *p_{nullptr};
};
}
#endif  // FORTRAN_COMMON_INDIRECTION_H_
