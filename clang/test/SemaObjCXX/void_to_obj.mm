// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/6463729>
@class XX;

void func() {
  XX *obj;
  void *vv;

  obj = vv; // expected-error{{assigning to 'XX *' from incompatible type 'void *'}}
}
