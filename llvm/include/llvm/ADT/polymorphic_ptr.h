//===- llvm/ADT/polymorphic_ptr.h - Smart copyable owned ptr ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides a polymorphic_ptr class template. See the class comments
/// for details about this API, its intended use cases, etc.
///
/// The primary motivation here is to work around the necessity of copy
/// semantics in C++98. This is typically used where any actual copies are
/// incidental or unnecessary. As a consequence, it is expected to cease to be
/// useful and be removed when we can directly rely on move-only types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POLYMORPHIC_PTR_H
#define LLVM_ADT_POLYMORPHIC_PTR_H

#include "llvm/Support/Compiler.h"

namespace llvm {

/// \brief An owning, copyable polymorphic smart pointer.
///
/// This pointer exists to provide copyable owned smart pointer. Rather than
/// shared ownership semantics, it has unique ownership semantics and deep copy
/// semantics. It is copyable by requiring that the underlying type exposes
/// a method which can produce a (heap allocated) clone.
///
/// Note that in almost all scenarios use of this could be avoided if we could
/// build move-only containers of a std::unique_ptr, but until then this
/// provides an effective way to place polymorphic objects in a container.
template <typename T> class polymorphic_ptr {
  T *ptr;

public:
  polymorphic_ptr(T *ptr = 0) : ptr(ptr) {}
  polymorphic_ptr(const polymorphic_ptr &arg) : ptr(arg ? arg->clone() : 0) {}
#if LLVM_HAS_RVALUE_REFERENCES
  polymorphic_ptr(polymorphic_ptr &&arg) : ptr(arg.take()) {}
#endif
  ~polymorphic_ptr() { delete ptr; }

  polymorphic_ptr &operator=(polymorphic_ptr arg) {
    swap(arg);
    return *this;
  }
  polymorphic_ptr &operator=(T *arg) {
    if (arg != ptr) {
      delete ptr;
      ptr = arg;
    }
    return *this;
  }

  T &operator*() const { return *ptr; }
  T *operator->() const { return ptr; }
  LLVM_EXPLICIT operator bool() const { return ptr != 0; }
  bool operator!() const { return ptr == 0; }

  T *get() const { return ptr; }

  T *take() {
    T *tmp = ptr;
    ptr = 0;
    return tmp;
  }

  void swap(polymorphic_ptr &arg) {
    T *tmp = ptr;
    ptr = arg.ptr;
    arg.ptr = tmp;
  }
};

template <typename T>
void swap(polymorphic_ptr<T> &lhs, polymorphic_ptr<T> &rhs) {
  lhs.swap(rhs);
}

template <typename T, typename U>
bool operator==(const polymorphic_ptr<T> &lhs, const polymorphic_ptr<U> &rhs) {
  return lhs.get() == rhs.get();
}

template <typename T, typename U>
bool operator!=(const polymorphic_ptr<T> &lhs, const polymorphic_ptr<U> &rhs) {
  return lhs.get() != rhs.get();
}

template <typename T, typename U>
bool operator==(const polymorphic_ptr<T> &lhs, U *rhs) {
  return lhs.get() == rhs;
}

template <typename T, typename U>
bool operator!=(const polymorphic_ptr<T> &lhs, U *rhs) {
  return lhs.get() != rhs;
}

template <typename T, typename U>
bool operator==(T *lhs, const polymorphic_ptr<U> &rhs) {
  return lhs == rhs.get();
}

template <typename T, typename U>
bool operator!=(T *lhs, const polymorphic_ptr<U> &rhs) {
  return lhs != rhs.get();
}

}

#endif
