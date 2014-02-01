// RUN: %clang_cc1 -mconstructor-aliases -std=c++11 -fexceptions -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

struct A {
  A(int a);
  ~A();
  int a;
};

void foo(A a, A b, A c) {
}

// Order of destruction should be left to right.
//
// CHECK-LABEL: define void @"\01?foo@@YAXUA@@00@Z"
// CHECK:          ([[argmem_ty:<{ %struct.A, %struct.A, %struct.A }>]]* inalloca)
// CHECK: %[[a:[^ ]*]] = getelementptr inbounds [[argmem_ty]]* %0, i32 0, i32 0
// CHECK: %[[b:[^ ]*]] = getelementptr inbounds [[argmem_ty]]* %0, i32 0, i32 1
// CHECK: %[[c:[^ ]*]] = getelementptr inbounds [[argmem_ty]]* %0, i32 0, i32 2
// CHECK: call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[a]])
// CHECK: call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[b]])
// CHECK: call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[c]])
// CHECK: ret void


void call_foo() {
  foo(A(1), A(2), A(3));
}

// Order of evaluation should be right to left, and we should clean up the right
// things as we unwind.
//
// CHECK-LABEL: define void @"\01?call_foo@@YAXXZ"()
// CHECK: call i8* @llvm.stacksave()
// CHECK: %[[argmem:[^ ]*]] = alloca [[argmem_ty]], inalloca
// CHECK: %[[arg3:[^ ]*]] = getelementptr inbounds [[argmem_ty]]* %[[argmem]], i32 0, i32 2
// CHECK: invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@H@Z"(%struct.A* %[[arg3]], i32 3)
// CHECK: %[[arg2:[^ ]*]] = getelementptr inbounds [[argmem_ty]]* %[[argmem]], i32 0, i32 1
// CHECK: invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@H@Z"(%struct.A* %[[arg2]], i32 2)
// CHECK: %[[arg1:[^ ]*]] = getelementptr inbounds [[argmem_ty]]* %[[argmem]], i32 0, i32 0
// CHECK: invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@H@Z"(%struct.A* %[[arg1]], i32 1)
// CHECK: invoke void @"\01?foo@@YAXUA@@00@Z"([[argmem_ty]]* inalloca %[[argmem]])
// CHECK: call void @llvm.stackrestore
// CHECK: ret void
//
//   lpad2:
// CHECK: call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[arg2]])
// CHECK: br label
//
//   ehcleanup:
// CHECK: call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[arg3]])
