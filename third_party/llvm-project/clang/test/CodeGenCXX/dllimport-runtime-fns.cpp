// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -fms-compatibility-version=19.20 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -fms-compatibility-version=19.20 -triple aarch64-windows-msvc -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-windows-itanium -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -no-opaque-pointers -triple aarch64-windows-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=GNU

void foo1() { throw 1; }
// _CxxThrowException should not be marked dllimport.
// MSVC-LABEL: define dso_local void @"?foo1@@YAXXZ"
// MSVC: call void @_CxxThrowException
// MSVC: declare dso_local void @_CxxThrowException(i8*, %eh.ThrowInfo*)

// __cxa_throw should be marked dllimport for *-windows-itanium.
// ITANIUM-LABEL: define dso_local void @_Z4foo1v()
// ITANIUM: call void @__cxa_throw({{.*}})
// ITANIUM: declare dllimport void @__cxa_throw({{.*}})

// ... but not for *-windows-gnu.
// GNU-LABEL: define dso_local void @_Z4foo1v()
// GNU: call void @__cxa_throw({{.*}})
// GNU: declare dso_local void @__cxa_throw({{.*}})


void bar();
void foo2() noexcept(true) { bar(); }
// __std_terminate should not be marked dllimport.
// MSVC-LABEL: define dso_local void @"?foo2@@YAXXZ"
// MSVC: call void @__std_terminate()
// MSVC: declare dso_local void @__std_terminate()

// _ZSt9terminatev and __cxa_begin_catch should be marked dllimport.
// ITANIUM-LABEL: define linkonce_odr hidden void @__clang_call_terminate(i8* %0)
// ITANIUM: call i8* @__cxa_begin_catch({{.*}})
// ITANIUM: call void @_ZSt9terminatev()
// ITANIUM: declare dllimport i8* @__cxa_begin_catch(i8*)
// ITANIUM: declare dllimport void @_ZSt9terminatev()

// .. not for mingw.
// GNU-LABEL: define linkonce_odr hidden void @__clang_call_terminate(i8* %0)
// GNU: call i8* @__cxa_begin_catch({{.*}})
// GNU: call void @_ZSt9terminatev()
// GNU: declare dso_local i8* @__cxa_begin_catch(i8*)
// GNU: declare dso_local void @_ZSt9terminatev()


struct A {};
struct B { virtual void f(); };
struct C : A, virtual B {};
struct T {};
T *foo3() { return dynamic_cast<T *>((C *)0); }
// __RTDynamicCast should not be marked dllimport.
// MSVC-LABEL: define dso_local noundef %struct.T* @"?foo3@@YAPEAUT@@XZ"
// MSVC: call i8* @__RTDynamicCast({{.*}})
// MSVC: declare dso_local i8* @__RTDynamicCast(i8*, i32, i8*, i8*, i32)

// Again, imported
// ITANIUM-LABEL: define dso_local noundef %struct.T* @_Z4foo3v()
// ITANIUM: call i8* @__dynamic_cast({{.*}})
// ITANIUM: declare dllimport i8* @__dynamic_cast({{.*}})

// Not imported
// GNU-LABEL: define dso_local noundef %struct.T* @_Z4foo3v()
// GNU: call i8* @__dynamic_cast({{.*}})
// GNU: declare dso_local i8* @__dynamic_cast({{.*}})
