// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++11 %s 2>&1 | FileCheck %s -check-prefix=CHECKDEF -check-prefix=CHECK11
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++11 -fvisibility hidden %s 2>&1 | FileCheck %s -check-prefix=CHECKHID -check-prefix=CHECK11
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++14 -fno-sized-deallocation %s 2>&1 | FileCheck %s -check-prefix=CHECKDEF -check-prefix=CHECK11
// RU N: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++14 %s 2>&1 | FileCheck %s -check-prefix=CHECKDEF -check-prefix=CHECK14 -check-prefix=CHECK14UND
// RU N: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++14 -fvisibility hidden %s 2>&1 | FileCheck %s -check-prefix=CHECKHID -check-prefix=CHECK14 -check-prefix=CHECK14UND
// RU N: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++14 -fdefine-sized-deallocation %s 2>&1 | FileCheck %s -check-prefix=CHECKDEF -check-prefix=CHECK14 -check-prefix=CHECK14DEFCOMDAT
// RU N: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown -o - -std=c++14 -fdefine-sized-deallocation -fvisibility hidden %s 2>&1 | FileCheck %s -check-prefix=CHECKHID -check-prefix=CHECK14 -check-prefix=CHECK14DEFCOMDAT
// RU N: %clang_cc1 -emit-llvm -triple x86_64-apple-macosx -o - -std=c++14 -fdefine-sized-deallocation %s | FileCheck %s -check-prefix=CHECKDEF -check-prefix=CHECK14 -check-prefix=CHECK14DEFNOCOMDAT

// PR22419: Implicit sized deallocation functions always have default visibility.
//   Generalized to all implicit allocation functions.

// CHECK14-DAG: %struct.A = type { i8 }
struct A { };

// CHECKDEF-DAG: define void @_Z3fooP1A(%struct.A* %is)
// CHECKHID-DAG: define hidden void @_Z3fooP1A(%struct.A* %is)
void foo(A* is) {

  // CHECK11-DAG: call noalias i8* @_Znwm(i64 1)
  // CHECK14-DAG: call noalias i8* @_Znwm(i64 1)
  is = new A();

  // CHECK11-DAG: call void @_ZdlPv(i8* %{{.+}})
  // CHECK14UND-DAG: br i1 icmp ne (void (i8*, i64)* @_ZdlPvm, void (i8*, i64)* null),
  // CHECK14-DAG: call void @_ZdlPvm(i8* %{{.+}}, i64 1)
  // CHECK14UND-DAG: call void @_ZdlPv(i8* %{{.+}})
  delete is;
}

// CHECK11-DAG: declare noalias i8* @_Znwm(i64)
// CHECK11-DAG: declare void @_ZdlPv(i8*)

// CHECK14-DAG: declare noalias i8* @_Znwm(i64)
// CHECK14UND-DAG: declare extern_weak void @_ZdlPvm(i8*, i64)
// CHECK14DEFCOMDAT-DAG: define linkonce void @_ZdlPvm(i8*, i64) #{{[0-9]+}} comdat {
// CHECK14DEFCOMDAT-DAG: declare void @_ZdlPv(i8*)
// CHECK14DEFNOCOMDAT-DAG: define linkonce void @_ZdlPvm(i8*, i64) #{{[0-9]+}} {
// CHECK14DEFNOCOMDAT-DAG: declare void @_ZdlPv(i8*)

// CHECK14-DAG: %struct.B = type { i8 }
struct B { ~B() { }};

// CHECKDEF-DAG: define void @_Z1fP1B(%struct.B* %p)
// CHECKHID-DAG: define hidden void @_Z1fP1B(%struct.B* %p)
void f(B *p) {

  // CHECK11-DAG: call noalias i8* @_Znam(i64 13)
  // CHECK14-DAG: call noalias i8* @_Znam(i64 13)
  p = new B[5];

  // CHECK11-DAG: call void @_ZdaPv(i8* %{{.+}})
  // CHECK14UND-DAG: br i1 icmp ne (void (i8*, i64)* @_ZdaPvm, void (i8*, i64)* null),
  // CHECK14-DAG: call void @_ZdaPvm(i8* %{{.+}}, i64 %{{.+}})
  // CHECK14UND-DAG: call void @_ZdaPv(i8* %{{.+}})
  delete[] p;
}

// CHECK11-DAG: declare noalias i8* @_Znam(i64)
// CHECK11-DAG: declare void @_ZdaPv(i8*)

// CHECK14-DAG: declare noalias i8* @_Znam(i64)
// CHECK14UND-DAG: declare extern_weak void @_ZdaPvm(i8*, i64)
// CHECK14DEF-DAG: define linkonce void @_ZdaPvm(i8*, i64) #{{[0-9]+}} comdat {
// CHECK14DEF-DAG: declare void @_ZdaPv(i8*)
// CHECK14DEFNOCOMDAT-DAG: define linkonce void @_ZdaPvm(i8*, i64) #{{[0-9]+}} {
// CHECK14DEFNOCOMDAT-DAG: declare void @_ZdaPv(i8*)
