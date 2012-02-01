// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

#pragma GCC visibility push(hidden)
struct x {
  static int y;
};
#pragma GCC visibility pop
int x::y = 10;
// CHECK: @_ZN1x1yE = hidden global

#pragma GCC visibility push(hidden)
struct __attribute((visibility("default"))) x2 {
  static int y;
};
int x2::y = 10;
// CHECK: @_ZN2x21yE = global
#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
struct x3 {
  static int y;
} __attribute((visibility("default")));
int x3::y = 10;
// CHECK: @_ZN2x31yE = global
#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
template<class T> struct x4 {
  static int y;
};
#pragma GCC visibility pop
template<> int x4<int>::y = 10;
// CHECK: @_ZN2x4IiE1yE = hidden global i32

#pragma GCC visibility push(hidden)
template<int x> int f() { return x; }
extern "C" int g() { return f<3>(); }
#pragma GCC visibility pop
// CHECK: define hidden i32 @g()
// CHECK: define linkonce_odr hidden i32 @_Z1fILi3EEiv()

#pragma GCC visibility push(hidden)
template<class T> struct x5 {
  void y();
};
#pragma GCC visibility pop
template<> void x5<int>::y() {}
// CHECK: define hidden void @_ZN2x5IiE1yEv

#pragma GCC visibility push(hidden)
namespace n __attribute((visibility("default"))) {
  void f() {}
  // CHECK: define void @_ZN1n1fEv
}
#pragma GCC visibility pop

namespace n __attribute((visibility("default")))  {
#pragma GCC visibility push(hidden)
  void g() {}
  // CHECK: define hidden void @_ZN1n1gEv
#pragma GCC visibility pop
}
