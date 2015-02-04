// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - -std=c++14 %s 2>&1 | FileCheck %s -check-prefix=CHECKDEF -check-prefix=CHECK
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - -std=c++14 -fvisibility hidden %s 2>&1 | FileCheck %s -check-prefix=CHECKHID -check-prefix=CHECK

// PR22419: Implicit sized deallocation functions always have default visibility.

// CHECKDEF-DAG: define void @_Z3fooPi(i32* %is)
// CHECKHID-DAG: define hidden void @_Z3fooPi(i32* %is)
void foo(int* is) {
  
  // CHECK-DAG: call void @_ZdlPvm(i8* %{{.+}}, i64 4)
  delete is;
}

// CHECK-DAG: define linkonce void @_ZdlPvm(i8*, i64)

// CHECK-DAG: %struct.A = type { i8 }
struct A { ~A() { }};

// CHECKDEF-DAG: define void @_Z1fP1A(%struct.A* %p)
// CHECKHID-DAG: define hidden void @_Z1fP1A(%struct.A* %p)
void f(A *p) {
  
  // CHECK-DAG: call void @_ZdaPvm(i8* %{{.+}}, i64 %{{.+}})
  delete[] p;
}

// CHECK-DAG: define linkonce void @_ZdaPvm(i8*, i64)
