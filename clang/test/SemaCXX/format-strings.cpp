// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral -pedantic %s

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
}

void f(char **sp, float *fp) {
  // TODO: Warn that the 'a' length modifier is an extension.
  scanf("%as", sp);

  // TODO: Warn that the 'a' conversion specifier is a C++11 feature.
  printf("%a", 1.0);
  scanf("%afoobar", fp);
}

void g() {
  printf("%ls", "foo"); // expected-warning{{format specifies type 'wchar_t *' but the argument has type 'const char *'}}
}

// Test that we properly handle format_idx on C++ members.
class Foo {
public:
  const char *gettext(const char *fmt) __attribute__((format_arg(2)));

  int scanf(const char *restrict, ...) __attribute__((format(scanf, 2, 3)));
  int printf(const char *restrict, ...) __attribute__((format(printf, 2, 3)));

  static const char *gettext_static(const char *fmt) __attribute__((format_arg(1)));
  static int printf_static(const char *restrict, ...) __attribute__((format(printf, 1, 2)));
};

void h(int *i) {
  Foo foo;
  foo.scanf("%d"); // expected-warning{{more '%' conversions than data arguments}}
  foo.printf("%d", i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}
  Foo::printf_static("%d", i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}

  printf(foo.gettext("%d"), i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}
  printf(Foo::gettext_static("%d"), i); // expected-warning{{format specifies type 'int' but the argument has type 'int *'}}
}
