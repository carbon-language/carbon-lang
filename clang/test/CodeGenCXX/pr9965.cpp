// RUN: %clang_cc1 -std=c++0x -emit-llvm -o - %s | FileCheck %s
template<typename T>
struct X
{
    X() = default;
};

X<int> x;
// CHECK: define internal void @__cxx_global_var_init()
// CHECK: call {{.*}} @_ZN1XIiEC1Ev
// CHECK: define linkonce_odr {{.*}} @_ZN1XIiEC1Ev
// CHECK: define linkonce_odr {{.*}} @_ZN1XIiEC2Ev
