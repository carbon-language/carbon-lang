// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

namespace test1 {
  static int abc = 42; // expected-warning {{variable 'abc' is not needed and will not be emitted}}
  template <typename T>
  int foo(void) {
    return abc;
  }
}

namespace test2 {
  struct bah {
  };
  namespace {
    struct foo : bah {
      static char bar;
      virtual void zed();
    };
    void foo::zed() {
      bar++;
    }
    char foo::bar=0;
  }
  bah *getfoo() {
    return new foo();
  }
}
