// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKX86 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=arm-linux-gnueabihf | FileCheck --check-prefix=CHECKARM %s

template <class T> struct A { A(); };
struct B { A<int> x; };
void a() {   
  B b;
}

// CHECKX86: call {{.*}} @_ZN1BC1Ev
// CHECKX86: define linkonce_odr {{.*}} @_ZN1BC1Ev(%struct.B* %this) unnamed_addr
// CHECKX86: call {{.*}} @_ZN1AIiEC1Ev

// CHECKARM: call {{.*}} @_ZN1BC1Ev
// CHECKARM: define linkonce_odr {{.*}} @_ZN1BC1Ev(%struct.B* returned %this) unnamed_addr
// CHECKARM: call {{.*}} @_ZN1AIiEC1Ev
