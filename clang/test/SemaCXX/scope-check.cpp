// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s -Wno-unreachable-code

namespace test0 {
  struct D { ~D(); };

  int f(bool b) {
    if (b) {
      D d;
      goto end;
    }

  end:
    return 1;
  }
}

namespace test1 {
  struct C { C(); };

  int f(bool b) {
    if (b)
      goto foo; // expected-error {{illegal goto into protected scope}}
    C c; // expected-note {{jump bypasses variable initialization}}
  foo:
    return 1;
  }
}

namespace test2 {
  struct C { C(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };

    C c;
    goto *ip;
  lbl1:
    return 0;
  lbl2:
    return 1;
  }
}

namespace test3 {
  struct C { C(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };

    goto *ip;
  lbl1: {
    C c;
    return 0;
  }
  lbl2:
    return 1;
  }
}

namespace test4 {
  struct C { C(); };
  struct D { ~D(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };

    C c0;

    goto *ip; // expected-warning {{indirect goto might cross protected scopes}}
    C c1; // expected-note {{jump bypasses variable initialization}}
  lbl1: // expected-note {{possible target of indirect goto}}
    return 0;
  lbl2:
    return 1;
  }
}

namespace test5 {
  struct C { C(); };
  struct D { ~D(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };
    C c0;

    goto *ip;
  lbl1: // expected-note {{possible target of indirect goto}}
    return 0;
  lbl2:
    if (ip[1]) {
      D d; // expected-note {{jump exits scope of variable with non-trivial destructor}}
      ip += 2;
      goto *ip; // expected-warning {{indirect goto might cross protected scopes}}
    }
    return 1;
  }
}

namespace test6 {
  struct C { C(); };

  unsigned f(unsigned s0, unsigned s1, void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2, &&lbl3, &&lbl4 };
    C c0;

    goto *ip;
  lbl1:
    s0++;
    goto *++ip;
  lbl2:
    s0 -= s1;
    goto *++ip;
  lbl3: {
    unsigned tmp = s0;
    s0 = s1;
    s1 = tmp;
    goto *++ip;
  }
  lbl4:
    return s0;
  }
}

// C++0x says it's okay to skip non-trivial initializers on static
// locals, and we implement that in '03 as well.
namespace test7 {
  struct C { C(); };

  void test() {
    goto foo;
    static C c;
  foo:
    return;
  }
}
