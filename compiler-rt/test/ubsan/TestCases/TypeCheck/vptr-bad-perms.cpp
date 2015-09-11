// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Tests that we consider vtable pointers in writable memory to be invalid.

// REQUIRES: vptr-validation

#include <string.h>

struct A {
  virtual void f();
};

void A::f() {}

struct B {
  virtual void f();
};

void B::f() {}

int main() {
  // Create a fake vtable for A in writable memory and copy A's vtable into it.
  void *fake_vtable[3];
  A a;
  void ***vtp = (void ***)&a;
  memcpy(fake_vtable, *vtp - 2, sizeof(void *) * 3);
  *vtp = fake_vtable + 2;

  // A's vtable is invalid because it lives in writable memory.
  // CHECK: invalid vptr
  reinterpret_cast<B*>(&a)->f();
}
