#ifndef UTILS_H
#define UTILS_H

inline void break_optimization(void *arg) {
  __asm__ __volatile__("" : : "r" (arg) : "memory");
}

// Tests will instantiate this class to pad out bit sets to test out the various
// ways we can represent the bit set (32-bit inline, 64-bit inline, memory).
// This class has 37 virtual member functions, which forces us to use a
// pointer-aligned bitset.
template <typename T, unsigned I>
class Deriver : T {
  virtual void f() {}
  virtual void g() {}
  virtual void f1() {}
  virtual void f2() {}
  virtual void f3() {}
  virtual void f4() {}
  virtual void f5() {}
  virtual void f6() {}
  virtual void f7() {}
  virtual void f8() {}
  virtual void f9() {}
  virtual void f10() {}
  virtual void f11() {}
  virtual void f12() {}
  virtual void f13() {}
  virtual void f14() {}
  virtual void f15() {}
  virtual void f16() {}
  virtual void f17() {}
  virtual void f18() {}
  virtual void f19() {}
  virtual void f20() {}
  virtual void f21() {}
  virtual void f22() {}
  virtual void f23() {}
  virtual void f24() {}
  virtual void f25() {}
  virtual void f26() {}
  virtual void f27() {}
  virtual void f28() {}
  virtual void f29() {}
  virtual void f30() {}
  virtual void f31() {}
  virtual void f32() {}
  virtual void f33() {}
  virtual void f34() {}
  virtual void f35() {}
};

#endif
