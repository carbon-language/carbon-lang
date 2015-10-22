// RUN: %clang_cc1 -std=c++14 -fcoroutines -verify %s

void mixed_yield() {
  // FIXME: diagnose
  co_yield 0;
  return;
}

void mixed_await() {
  // FIXME: diagnose
  co_await 0;
  return;
}

void only_coreturn() {
  // FIXME: diagnose
  co_return;
}

void mixed_coreturn(bool b) {
  // FIXME: diagnose
  if (b)
    co_return;
  else
    return;
}

struct CtorDtor {
  CtorDtor() {
    co_yield 0; // expected-error {{'co_yield' cannot be used in a constructor}}
  }
  CtorDtor(int n) {
    // The spec doesn't say this is ill-formed, but it must be.
    co_await n; // expected-error {{'co_await' cannot be used in a constructor}}
  }
  ~CtorDtor() {
    co_return 0; // expected-error {{'co_return' cannot be used in a destructor}}
  }
  // FIXME: The spec says this is ill-formed.
  void operator=(CtorDtor&) {
    co_yield 0;
  }
};

constexpr void constexpr_coroutine() {
  co_yield 0; // expected-error {{'co_yield' cannot be used in a constexpr function}}
}

void varargs_coroutine(const char *, ...) {
  co_await 0; // expected-error {{'co_await' cannot be used in a varargs function}}
}
