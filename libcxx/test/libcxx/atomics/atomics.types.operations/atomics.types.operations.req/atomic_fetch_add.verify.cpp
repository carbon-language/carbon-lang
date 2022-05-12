//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <class T>
//     T* atomic_fetch_add(volatile atomic<T*>* obj, ptrdiff_t op)
// template <class T>
//     T* atomic_fetch_add(atomic<T*>* obj, ptrdiff_t op);

#include <atomic>

void void_pointer() {
  {
    volatile std::atomic<void*> obj;
    // expected-error@atomic:* {{incomplete type 'void' where a complete type is required}}
    std::atomic_fetch_add(&obj, 0);
  }
  {
    std::atomic<void*> obj;
    // expected-error@atomic:* {{incomplete type 'void' where a complete type is required}}
    std::atomic_fetch_add(&obj, 0);
  }
}

struct Incomplete;

void pointer_to_incomplete_type() {
  {
    volatile std::atomic<Incomplete*> obj;
    // expected-error@atomic:* {{incomplete type 'Incomplete' where a complete type is required}}
    std::atomic_fetch_add(&obj, 0);
  }
  {
    std::atomic<Incomplete*> obj;
    // expected-error@atomic:* {{incomplete type 'Incomplete' where a complete type is required}}
    std::atomic_fetch_add(&obj, 0);
  }
}

void function_pointer() {
  {
    volatile std::atomic<void (*)(int)> fun;
    // expected-error@atomic:* {{static_assert failed due to requirement '!is_function<void (int)>::value' "Pointer to function isn't allowed"}}
    std::atomic_fetch_add(&fun, 0);
  }
  {
    std::atomic<void (*)(int)> fun;
    // expected-error@atomic:* {{static_assert failed due to requirement '!is_function<void (int)>::value' "Pointer to function isn't allowed"}}
    std::atomic_fetch_add(&fun, 0);
  }
}

struct S {
  void fun(int);
};

void member_function_pointer() {
  {
    volatile std::atomic<void (S::*)(int)> fun;
    // expected-error@atomic:* {{no member named 'fetch_add' in}}
    std::atomic_fetch_add(&fun, 0);
  }
  {
    std::atomic<void (S::*)(int)> fun;
    // expected-error@atomic:* {{no member named 'fetch_add' in}}
    std::atomic_fetch_add(&fun, 0);
  }
}
