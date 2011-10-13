// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void fn() = default; // expected-error {{only special member}}
struct foo {
  void fn() = default; // expected-error {{only special member}}

  foo() = default;
  foo(const foo&) = default;
  foo(foo&) = default;
  foo& operator = (const foo&) = default;
  foo& operator = (foo&) = default;
  ~foo() = default;
};

struct bar {
  bar();
  bar(const bar&);
  bar(bar&);
  bar& operator = (const bar&);
  bar& operator = (bar&);
  ~bar();
};

bar::bar() = default;
bar::bar(const bar&) = default;
bar::bar(bar&) = default;
bar& bar::operator = (const bar&) = default;
bar& bar::operator = (bar&) = default;
bar::~bar() = default;

// FIXME: static_assert(__is_trivial(foo), "foo should be trivial");

static_assert(!__has_trivial_destructor(bar), "bar's destructor isn't trivial");
static_assert(!__has_trivial_constructor(bar),
              "bar's default constructor isn't trivial");
static_assert(!__has_trivial_copy(bar), "bar has no trivial copy");
static_assert(!__has_trivial_assign(bar), "bar has no trivial assign");

void tester() {
  foo f, g(f);
  bar b, c(b);
  f = g;
  b = c;
}

