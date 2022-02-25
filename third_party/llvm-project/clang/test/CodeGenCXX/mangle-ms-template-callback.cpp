// RUN: %clang_cc1 -fblocks -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

template<typename Signature>
class C;

template<typename Ret>
class C<Ret(void)> {};
typedef C<void(void)> C0;

template<typename Ret, typename Arg1>
class C<Ret(Arg1)> {};

template<typename Ret, typename Arg1, typename Arg2>
class C<Ret(Arg1, Arg2)> {};

C0 callback_void;
// CHECK: "?callback_void@@3V?$C@$$A6AXXZ@@A"

volatile C0 callback_void_volatile;
// CHECK: "?callback_void_volatile@@3V?$C@$$A6AXXZ@@C"

class Type {};

C<int(void)> callback_int;
// CHECK: "?callback_int@@3V?$C@$$A6AHXZ@@A"
C<Type(void)> callback_Type;
// CHECK: "?callback_Type@@3V?$C@$$A6A?AVType@@XZ@@A"

C<void(int)> callback_void_int;
// CHECK: "?callback_void_int@@3V?$C@$$A6AXH@Z@@A"
C<int(int)> callback_int_int;
// CHECK: "?callback_int_int@@3V?$C@$$A6AHH@Z@@A"
C<void(Type)> callback_void_Type;
// CHECK: "?callback_void_Type@@3V?$C@$$A6AXVType@@@Z@@A"

void foo(C0 c) {}
// CHECK: "?foo@@YAXV?$C@$$A6AXXZ@@@Z"

// Here be dragons!
// Let's face the magic of template partial specialization...

void function(C<void(void)>) {}
// CHECK: "?function@@YAXV?$C@$$A6AXXZ@@@Z"

template<typename Ret> class C<Ret(*)(void)> {};
void function_pointer(C<void(*)(void)>) {}
// CHECK: "?function_pointer@@YAXV?$C@P6AXXZ@@@Z"

// Block equivalent to the previous definitions.
template<typename Ret> class C<Ret(^)(void)> {};
void block(C<void(^)(void)>) {}
// CHECK: "?block@@YAXV?$C@P_EAXXZ@@@Z"
// FYI blocks are not present in MSVS, so we're free to choose the spec.

template<typename T> class C<void (T::*)(void)> {};
class Z {
 public:
  void method() {}
};
void member_pointer(C<void (Z::*)(void)>) {}
// CHECK: "?member_pointer@@YAXV?$C@P8Z@@AEXXZ@@@Z"

template<typename T> void bar(T) {}

void call_bar() {
  bar<int (*)(int)>(0);
// CHECK: "??$bar@P6AHH@Z@@YAXP6AHH@Z@Z"

  bar<int (^)(int)>(0);
// CHECK: "??$bar@P_EAHH@Z@@YAXP_EAHH@Z@Z"
// FYI blocks are not present in MSVS, so we're free to choose the spec.
}

template <void (*Fn)()> void WrapFnPtr() { Fn(); }
template <void (&Fn)()> void WrapFnRef() { Fn(); }
struct Thing {
  static void VoidStaticMethod();
};
void VoidFn();
void CallWrapper() {
  WrapFnPtr<VoidFn>();
  WrapFnRef<VoidFn>();
  WrapFnPtr<Thing::VoidStaticMethod>();
  WrapFnRef<Thing::VoidStaticMethod>();
}
// CHECK: call {{.*}} @"??$WrapFnPtr@$1?VoidFn@@YAXXZ@@YAXXZ"
// CHECK: call {{.*}} @"??$WrapFnRef@$1?VoidFn@@YAXXZ@@YAXXZ"
// CHECK: call {{.*}} @"??$WrapFnPtr@$1?VoidStaticMethod@Thing@@SAXXZ@@YAXXZ"
// CHECK: call {{.*}} @"??$WrapFnRef@$1?VoidStaticMethod@Thing@@SAXXZ@@YAXXZ"
