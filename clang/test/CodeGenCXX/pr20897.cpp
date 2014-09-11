// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s
struct Base {};

// __declspec(dllexport) causes us to export the implicit constructor.
struct __declspec(dllexport) Derived : virtual Base {
// CHECK-LABEL: define weak_odr dllexport x86_thiscallcc %struct.Derived* @"\01??0Derived@@QAE@ABU0@@Z"
// CHECK:      %[[this:.*]] = load %struct.Derived** {{.*}}
// CHECK-NEXT: store %struct.Derived* %[[this]], %struct.Derived** %[[retval:.*]]
// CHECK:      %[[dest_a_gep:.*]] = getelementptr inbounds %struct.Derived* %[[this]], i32 0, i32 1
// CHECK-NEXT: %[[src_load:.*]]   = load %struct.Derived** {{.*}}
// CHECK-NEXT: %[[src_a_gep:.*]]  = getelementptr inbounds %struct.Derived* %[[src_load:.*]], i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %[[dest_a_gep]], i8* %[[src_a_gep]], i64 1, i32 4, i1 false)
// CHECK-NEXT: %[[dest_this:.*]] = load %struct.Derived** %[[retval]]
// CHECK-NEXT: ret %struct.Derived* %[[dest_this]]
  bool a : 1;
  bool b : 1;
};
