//===-- runtime/memory.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Thin wrapper around malloc()/free() to isolate the dependency,
// ease porting, and provide an owning pointer.

#ifndef FORTRAN_RUNTIME_MEMORY_H_
#define FORTRAN_RUNTIME_MEMORY_H_

#include <memory>

namespace Fortran::runtime {

class Terminator;

void *AllocateMemoryOrCrash(Terminator &, std::size_t bytes);
template<typename A> A &AllocateOrCrash(Terminator &t) {
  return *reinterpret_cast<A *>(AllocateMemoryOrCrash(t, sizeof(A)));
}
void FreeMemory(void *);
void FreeMemoryAndNullify(void *&);

template<typename A> struct New {
  template<typename... X> A &operator()(Terminator &terminator, X &&... x) {
    return *new (AllocateMemoryOrCrash(terminator, sizeof(A)))
        A{std::forward<X>(x)...};
  }
};

template<typename A> struct OwningPtrDeleter {
  void operator()(A *p) { FreeMemory(p); }
};

template<typename A> using OwningPtr = std::unique_ptr<A, OwningPtrDeleter<A>>;
}

#endif  // FORTRAN_RUNTIME_MEMORY_H_
