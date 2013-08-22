// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK: ; ModuleID

template<typename> struct A { static int a; };

// CHECK-NOT: @_ZN1AIcE1aE
template<> int A<char>::a;

// CHECK: @_ZN1AIbE1aE = global i32 10
template<> int A<bool>::a = 10;

// CHECK: @llvm.global_ctors = appending global [2 x { i32, void ()* }]
// CHECK: [{ i32, void ()* } { i32 65535, void ()* @__cxx_global_var_init },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

extern "C" int foo();
template<> int A<short>::a = foo();  // Separate global_ctor entry
int b = foo();  // Goes in _GLOBAL__I_a
int c = foo();  // Goes in _GLOBAL__I_a

// An explicit specialization is ordered, and goes in __GLOBAL_I_a.
template<> struct A<int> { static int a; };
int A<int>::a = foo();

// CHECK: define internal void @__cxx_global_var_init()
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_ZN1AIsE1aE
// CHECK: ret

// CHECK: define internal void @_GLOBAL__I_a()
//   We call unique stubs for every ordered dynamic initializer in the TU.
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK-NOT: call
// CHECK: ret
