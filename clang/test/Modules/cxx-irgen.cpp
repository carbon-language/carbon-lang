// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c++ -std=c++11 -fmodules-cache-path=%t -I %S/Inputs -triple %itanium_abi_triple -disable-llvm-optzns -emit-llvm -o - %s | FileCheck %s
// FIXME: When we have a syntax for modules in C++, use that.

@import cxx_irgen_top;

// CHECK-DAG: call {{[a-z]*[ ]?i32}} @_ZN8CtorInitIiE1fEv(
CtorInit<int> x;

@import cxx_irgen_left;
@import cxx_irgen_right;

// CHECK-DAG: define available_externally hidden {{signext i32|i32}} @_ZN1SIiE1gEv({{.*}} #[[ALWAYS_INLINE:.*]] align
int a = S<int>::g();

// CHECK-DAG: define available_externally {{signext i32|i32}} @_ZN1SIiE1fEv({{.*}} #[[ALWAYS_INLINE]] align
int b = h();

// CHECK-DAG: define linkonce_odr {{signext i32|i32}} @_Z3minIiET_S0_S0_(i32
int c = min(1, 2);

// CHECK: attributes #[[ALWAYS_INLINE]] = {{.*}} alwaysinline
