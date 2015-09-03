// Defines diamond multiple inheritance structure
//   A
//  / \
// B   C
//  \ /
//   Derived

// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>

class A {
public:
  int x;
  int *y_ptr;
  int *z_ptr;
  int *w_ptr;
  A() { x = 5; }
  void set_ptrs(int *y_ptr, int *z_ptr, int *w_ptr) {
    this->y_ptr = y_ptr;
    this->z_ptr = z_ptr;
    this->w_ptr = w_ptr;
  }
  virtual ~A() {
    assert(__msan_test_shadow(&this->x, sizeof(this->x) == -1));
    // bad access subclass member
    assert(__msan_test_shadow(this->y_ptr, sizeof(*this->y_ptr)) != -1);
    assert(__msan_test_shadow(this->z_ptr, sizeof(*this->z_ptr)) != -1);
    assert(__msan_test_shadow(this->w_ptr, sizeof(*this->w_ptr)) != -1);
  }
};

struct B : virtual public A {
public:
  int y;
  B() { y = 10; }
  virtual ~B() {
    assert(__msan_test_shadow(&this->x, sizeof(this->x)) == -1);
    assert(__msan_test_shadow(&this->y, sizeof(this->y)) == -1);
    assert(__msan_test_shadow(this->y_ptr, sizeof(*this->y_ptr)) == -1);

    // memory in subclasses is poisoned
    assert(__msan_test_shadow(this->z_ptr, sizeof(*this->z_ptr)) != -1);
    assert(__msan_test_shadow(this->w_ptr, sizeof(*this->w_ptr)) != -1);
  }
};

struct C : virtual public A {
public:
  int z;
  C() { z = 15; }
  virtual ~C() {
    assert(__msan_test_shadow(&this->x, sizeof(this->x)) == -1);
    assert(__msan_test_shadow(&this->z, sizeof(this->z)) == -1);
    assert(__msan_test_shadow(this->y_ptr, sizeof(*this->y_ptr)) == -1);
    assert(__msan_test_shadow(this->z_ptr, sizeof(*this->z_ptr) == -1));

    // memory in subclasses is poisoned
    assert(__msan_test_shadow(this->w_ptr, sizeof(*this->w_ptr)) != -1);
  }
};

class Derived : public B, public C {
public:
  int w;
  Derived() { w = 10; }
  ~Derived() {
    assert(__msan_test_shadow(&this->x, sizeof(this->x)) == -1);
    assert(__msan_test_shadow(&this->y, sizeof(this->y)) == -1);
    assert(__msan_test_shadow(&this->w, sizeof(this->w)) == -1);
  }
};

int main() {
  Derived *d = new Derived();
  d->set_ptrs(&d->y, &d->z, &d->w);

  // Order of destruction: Derived, C, B, A
  d->~Derived();
  // Verify that local pointer is unpoisoned, and that the object's
  // members are.
  assert(__msan_test_shadow(&d, sizeof(d)) == -1);
  assert(__msan_test_shadow(&d->x, sizeof(d->x)) != -1);
  assert(__msan_test_shadow(&d->y, sizeof(d->y)) != -1);
  assert(__msan_test_shadow(&d->z, sizeof(d->z)) != -1);
  assert(__msan_test_shadow(&d->w, sizeof(d->w)) != -1);
  assert(__msan_test_shadow(&d->y_ptr, sizeof(d->y_ptr)) != -1);
  assert(__msan_test_shadow(&d->z_ptr, sizeof(d->z_ptr)) != -1);
  assert(__msan_test_shadow(&d->w_ptr, sizeof(d->w_ptr)) != -1);
  return 0;
}
