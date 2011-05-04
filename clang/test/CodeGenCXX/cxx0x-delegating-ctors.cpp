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


delegator::delegator(bool)
{}

// CHECK: define void @_ZN9delegatorC1Ec
// CHECK: void @_ZN9delegatorC1Eb
// CHECK: void @__cxa_throw
// CHECK: void @_ZSt9terminatev
// CHECK: void @_ZN9delegatorD1Ev
// CHECK: define void @_ZN9delegatorC2Ec
// CHECK: void @_ZN9delegatorC2Eb
// CHECK: void @__cxa_throw
// CHECK: void @_ZSt9terminatev
// CHECK: void @_ZN9delegatorD2Ev
delegator::delegator(char)
  : delegator(true) {
  throw 0;
}

// CHECK: define void @_ZN9delegatorC1Ei
// CHECK: void @_ZN9delegatorC1Ev
// CHECK-NOT: void @_ZSt9terminatev
// CHECK: ret
// CHECK-NOT: void @_ZSt9terminatev
// CHECK: define void @_ZN9delegatorC2Ei
// CHECK: void @_ZN9delegatorC2Ev
// CHECK-NOT: void @_ZSt9terminatev
// CHECK: ret
// CHECK-NOT: void @_ZSt9terminatev
delegator::delegator(int)
  : delegator()
{}
