// RUN: %clang_cc1 -emit-llvm -o - -std=c++0x %s | FileCheck -check-prefix=CHECK-ASSIGN %s
// RUN: %clang_cc1 -emit-llvm -o - -std=c++0x %s | FileCheck -check-prefix=CHECK-CTOR %s

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

// PR10822
struct I {
  unsigned var[1];
};

void h() {
  I i;
  i = I();
}

// move assignment ops

// CHECK-ASSIGN: define linkonce_odr {{.*}} @_ZN1DaSEOS_
// CHECK-ASSIGN: call {{.*}} @_ZN1CaSEOS_
// CHECK-ASSIGN: call {{.*}} @_ZN1AaSEOS_
// CHECK-ASSIGN: call {{.*}} @_ZN1BaSEOS_
// array loop
// CHECK-ASSIGN: br i1
// CHECK-ASSIGN: call {{.*}} @_ZN1AaSEOS_

// CHECK-ASSIGN: define linkonce_odr {{.*}} @_ZN1IaSEOS_
// call void @llvm.memcpy.

// CHECK-ASSIGN: define linkonce_odr {{.*}} @_ZN1CaSEOS_
// CHECK-ASSIGN: call {{.*}} @_ZN1AaSEOS_

// move ctors

// CHECK-CTOR: define linkonce_odr void @_ZN1HC2EOS_
// CHECK-CTOR: call void @_ZN1GC2EOS_
// CHECK-CTOR: call void @_ZN1FC1EOS_
// CHECK-CTOR: call void @_ZN1EC1EOS_
// array loop
// CHECK-CTOR: br i1
// CHECK-CTOR: call void @_ZN1FC1EOS_

// CHECK-CTOR: define linkonce_odr void @_ZN1GC2EOS_
// CHECK-CTOR: call void @_ZN1EC1EOS_
