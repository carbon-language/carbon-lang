// RUN: %clang_cc1 -std=c++1z -verify %s

namespace pr32864_0 {
  struct transfer_t {
    void *fctx;
  };
  template <typename Ctx> class record {
    void run() {
      transfer_t t;
      Ctx from{t.fctx};
    }
  };
}

namespace pr33140_0a {
  struct S {
    constexpr S(const int &a = 0) {}
  };
  void foo(void) { S s[2] = {}; }
}

namespace pr33140_0b {
  bool bar(float const &f = 0);
  bool foo() { return bar() && bar(); }
}

namespace pr33140_2 {
  // FIXME: The declaration of 'b' below should lifetime-extend two int
  // temporaries.
  struct A { int &&r = 0; }; // expected-note 2{{initializing field 'r' with default member initializer}}
  struct B { A x, y; };
  B b = {}; // expected-warning 2{{not supported}}
}

namespace pr33140_3 {
  typedef struct Y { unsigned int c; } Y_t;
  struct X {
    Y_t a;
  };
  struct X foo[2] = {[0 ... 1] = {.a = (Y_t){.c = 0}}}; // expected-warning {{C99 extension}}
}

namespace pr33140_6 {
  struct Y { unsigned int c; };
  struct X { struct Y *p; };
  int f() {
    // FIXME: This causes clang to crash.
    //return (struct X[2]){ [0 ... 1] = { .p = &(struct Y&)(struct Y&&)(struct Y){0} } }[0].p->c;
    return 0;
  }
}

namespace pr33140_10 {
  int a(const int &n = 0);
  bool b() { return a() == a(); }
}
