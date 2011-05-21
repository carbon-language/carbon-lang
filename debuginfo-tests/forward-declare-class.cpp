// RUN: %clangxx -O0 -g %s -c -o %t.o
// RUN: %test_debuginfo %s %t.o
// Radar 9168773

// DEBUGGER: ptype A
// CHECK: type = class A {
// CHECK-NEXT: public:
// CHECK-NEXT: int MyData;
// CHECK-NEXT: }
class A;
class B {
public:
  void foo(const A *p);
};

B iEntry;

class A {
public:
  int MyData;
};

A irp;

