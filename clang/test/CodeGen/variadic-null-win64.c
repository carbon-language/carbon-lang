// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s --check-prefix=WINDOWS
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s --check-prefix=LINUX

// Make it possible to pass NULL through variadic functions on platforms where
// NULL has an integer type that is more narrow than a pointer. On such
// platforms we widen null pointer constants to a pointer-sized integer.

#define NULL 0

void v(const char *f, ...);
void f(const char *f) {
  v(f, 1, 2, 3, NULL);
}
// WINDOWS: define void @f(i8* %f)
// WINDOWS: call void (i8*, ...) @v(i8* {{.*}}, i32 1, i32 2, i32 3, i64 0)
// LINUX: define void @f(i8* %f)
// LINUX: call void (i8*, ...) @v(i8* {{.*}}, i32 1, i32 2, i32 3, i32 0)
