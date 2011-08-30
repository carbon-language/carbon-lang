// RUN: %clang_cc1 -emit-llvm -o - -std=c++0x %s | FileCheck %s

// construct

struct E {
  E();
  E(E&&);
};

struct F {
  F();
  F(F&&);
};

struct G {
  E e;
};

struct H : G {
  F l;
  E m;
  F ar[2];
};

void f() {
  H s;
  // CHECK: call void @_ZN1HC1EOS_
  H t(static_cast<H&&>(s));
}


// assign

struct A {
  A &operator =(A&&);
};

struct B {
  B &operator =(B&&);
};

struct C {
  A a;
};

struct D : C {
  A a;
  B b;
  A ar[2];
};

void g() {
  D d;
  // CHECK: call {{.*}} @_ZN1DaSEOS_
  d = D();
}


// move assignment ops

// CHECK: define linkonce_odr {{.*}} @_ZN1DaSEOS_
// CHECK: call {{.*}} @_ZN1CaSEOS_
// CHECK: call {{.*}} @_ZN1AaSEOS_
// CHECK: call {{.*}} @_ZN1BaSEOS_
// array loop
// CHECK: br i1
// CHECK: call {{.*}} @_ZN1AaSEOS_

// CHECK: define linkonce_odr {{.*}} @_ZN1CaSEOS_
// CHECK: call {{.*}} @_ZN1AaSEOS_


// move ctors

// CHECK: define linkonce_odr void @_ZN1HC2EOS_
// CHECK: call void @_ZN1GC2EOS_
// CHECK: call void @_ZN1FC1EOS_
// CHECK: call void @_ZN1EC1EOS_
// array loop
// CHECK: br i1
// CHECK: call void @_ZN1FC1EOS_

// CHECK: define linkonce_odr void @_ZN1GC2EOS_
// CHECK: call void @_ZN1EC1EOS_
