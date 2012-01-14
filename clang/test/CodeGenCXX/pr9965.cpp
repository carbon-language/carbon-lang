// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s | FileCheck %s
template<typename T>
struct X
{
    X() = default;
    ~X() {} // not a literal type
};

X<int> x;
// CHECK: define internal void @__cxx_global_var_init()
// CHECK: call {{.*}} @_ZN1XIiEC1Ev
// CHECK: define linkonce_odr {{.*}} @_ZN1XIiEC1Ev
// CHECK: define linkonce_odr {{.*}} @_ZN1XIiEC2Ev
