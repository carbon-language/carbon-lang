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

// Define a smart pointer class template that is rather like
// non-nullable std::unique_ptr<>.  Indirection<> is, like a C++ reference
// type, restricted to be non-null when constructed or assigned.
// Indirection<> optionally supports copy construction and copy assignment.
//
// To use Indirection<> with forward-referenced types, add
//    extern template class Fortran::common::Indirection<FORWARD_TYPE>;
// outside any namespace in a header before use, and
//    template class Fortran::common::Indirection<FORWARD_TYPE>;
// in one C++ source file later where a definition of the type is visible.

#include "idioms.h"
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

  bool operator==(const A &that) const { return *p_ == that; }
  bool operator==(const Indirection &that) const { return *p_ == *that.p_; }
  bool operator!=(const A &that) const { return *p_ != that; }
  bool operator!=(const Indirection &that) const { return *p_ != *that.p_; }

  template<typename... ARGS>
  static common::IfNoLvalue<Indirection, ARGS...> Make(ARGS &&... args) {
    return {new A(std::move(args)...)};
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

  bool operator==(const A &that) const { return *p_ == that; }
  bool operator==(const Indirection &that) const { return *p_ == *that.p_; }
  bool operator!=(const A &that) const { return *p_ != that; }
  bool operator!=(const Indirection &that) const { return *p_ != *that.p_; }

  template<typename... ARGS>
  static common::IfNoLvalue<Indirection, ARGS...> Make(ARGS &&... args) {
    return {new A(std::move(args)...)};
  }

private:
  A *p_{nullptr};
};

template<typename A> using CopyableIndirection = Indirection<A, true>;

// For use with std::unique_ptr<> when declaring owning pointers to
// forward-referenced types, here's a minimal custom deleter that avoids
// some of the drama with std::default_delete<>.  Invoke DEFINE_DELETER()
// later in exactly one C++ source file where a complete definition of the
// type is visible.  Be advised, std::unique_ptr<> does not have copy
// semantics; if you need ownership, copy semantics, and nullability,
// std::optional<CopyableIndirection<>> works.
template<typename A> class Deleter {
public:
  void operator()(A *) const;
};
}
#define DEFINE_DELETER(A) \
  template<> void Fortran::common::Deleter<A>::operator()(A *p) const { \
    delete p; \
  }
#endif  // FORTRAN_COMMON_INDIRECTION_H_
