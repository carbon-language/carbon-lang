// RUN: %clang_cc1 -std=c++11 -fcoroutines %s -verify

template<typename T, typename U>
U f(T t) {
  co_await t;
  co_yield t;

  1 + co_await t;
  1 + co_yield t; // expected-error {{expected expression}}

  auto x = co_await t;
  auto y = co_yield t;

  for co_await (int x : t) {}
  for co_await (int x = 0; x != 10; ++x) {} // expected-error {{'co_await' modifier can only be applied to range-based for loop}}

  if (t)
    co_return t;
  else
    co_return {t};
}
