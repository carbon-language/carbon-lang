// RUN: %clang_cc1 %s -std=c++1y -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK: ; ModuleID

extern "C" int foo();

template<typename T> struct A { static int a; };
template<typename T> int A<T>::a = foo();

// CHECK-NOT: @_ZN1AIcE1aE
template<> int A<char>::a;

// CHECK: @_ZN1AIbE1aE = global i32 10
template<> int A<bool>::a = 10;

// CHECK: @llvm.global_ctors = appending global [7 x { i32, void ()* }]
// CHECK: [{ i32, void ()* } { i32 65535, void ()* @[[unordered1:[^ ]*]] },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @[[unordered2:[^ ]*]] },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @[[unordered3:[^ ]*]] },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @[[unordered4:[^ ]*]] },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @[[unordered5:[^ ]*]] },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @[[unordered6:[^ ]*]] },
// CHECK:  { i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

template int A<short>::a;  // Unordered
int b = foo();
int c = foo();
int d = A<void>::a; // Unordered

// An explicit specialization is ordered, and goes in __GLOBAL_I_a.
template<> struct A<int> { static int a; };
int A<int>::a = foo();

template<typename T> struct S { static T x; static T y; };
template<> int S<int>::x = foo();
template<> int S<int>::y = S<int>::x;

template<typename T> T x = foo();
template short x<short>;  // Unordered
template<> int x<int> = foo();
int e = x<char>; // Unordered

namespace ns {
template <typename T> struct a {
  static int i;
};
template<typename T> int a<T>::i = foo();
template struct a<int>;

struct b {
  template <typename T> static T i;
};
template<typename T> T b::i = foo();
template int b::i<int>;
}
// CHECK: define internal void @[[unordered1]]
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_ZN1AIsE1aE
// CHECK: ret

// CHECK: define internal void @[[unordered2]]
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_Z1xIsE
// CHECK: ret

// CHECK: define internal void @[[unordered3]]
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_ZN2ns1aIiE1iE
// CHECK: ret

// CHECK: define internal void @[[unordered4]]
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_ZN2ns1b1iIiEE
// CHECK: ret

// CHECK: define internal void @[[unordered5]]
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_ZN1AIvE1aE
// CHECK: ret

// CHECK: define internal void @[[unordered6]]
// CHECK: call i32 @foo()
// CHECK: store {{.*}} @_Z1xIcE
// CHECK: ret

// CHECK: define internal void @_GLOBAL__I_a()
//   We call unique stubs for every ordered dynamic initializer in the TU.
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK-NOT: call
// CHECK: ret
