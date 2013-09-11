// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-gcc-compat

namespace N {
  void c1(int *a) {}
}

class C {
  static void c2(int *a) {}  // expected-note {{implicitly declared private here}} expected-note {{implicitly declared private here}}
};

void t1() {
  int v1 __attribute__((cleanup(N::c1)));
  int v2 __attribute__((cleanup(N::c2)));  // expected-error {{no member named 'c2' in namespace 'N'}}
  int v3 __attribute__((cleanup(C::c2)));  // expected-error {{'c2' is a private member of 'C'}}
}

class D : public C {
  void t2() {
    int v1 __attribute__((cleanup(c2)));  // expected-error {{'c2' is a private member of 'C'}}
  }
};
