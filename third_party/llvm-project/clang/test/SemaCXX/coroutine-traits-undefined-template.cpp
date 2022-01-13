// This file contains references to sections of the Coroutines TS, which can be
// found at http://wg21.link/coroutines.

// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -verify %s -fcxx-exceptions -fexceptions -Wunused-result

namespace std {
namespace experimental {

template<typename ...T>
struct coroutine_traits {
  struct promise_type {};
};

template<> struct coroutine_traits<void>; // expected-note {{forward declaration of 'std::experimental::coroutine_traits<void>'}}
}} // namespace std::experimental

void uses_forward_declaration() {
  co_return; // expected-error {{this function cannot be a coroutine: missing definition of specialization 'coroutine_traits<void>'}}
}
