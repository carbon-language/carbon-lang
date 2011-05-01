// RUN: %clang_cc1 -emit-llvm -fexceptions -fcxx-exceptions -std=c++0x -o - %s | FileCheck %s

struct non_trivial {
  non_trivial();
  ~non_trivial();
};
non_trivial::non_trivial() {}
non_trivial::~non_trivial() {}

// We use a virtual base to ensure that the constructor
// delegation optimization (complete->base) can't be
// performed.
struct delegator {
  non_trivial n; 
  delegator();
  delegator(int);
  delegator(char);
  delegator(bool);
};

delegator::delegator() {
  throw 0;
}

// CHECK: define void @_ZN9delegatorC1Ei
// CHECK: call void @_ZN9delegatorC1Ev
// CHECK-NOT: lpad
// CHECK: ret
// CHECK-NOT: lpad
// CHECK: define void @_ZN9delegatorC2Ei
// CHECK: call void @_ZN9delegatorC2Ev
// CHECK-NOT: lpad
// CHECK: ret
// CHECK-NOT: lpad
delegator::delegator(int)
  : delegator()
{}

delegator::delegator(bool)
{}

// CHECK: define void @_ZN9delegatorC2Ec
// CHECK: call void @_ZN9delegatorC2Eb
// CHECK: call void @__cxa_throw
delegator::delegator(char)
  : delegator(true) {
  throw 0;
}
