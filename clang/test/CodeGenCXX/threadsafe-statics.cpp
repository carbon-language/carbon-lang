// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck -check-prefix=WITH-TSS %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fno-threadsafe-statics | FileCheck -check-prefix=NO-TSS %s

int f();

// WITH-TSS: define void @_Z1gv() nounwind
// WITH-TSS: call i32 @__cxa_guard_acquire
// WITH-TSS: call void @__cxa_guard_release
// WITH-TSS: ret void
void g() { 
  static int a = f();
}

// NO-TSS: define void @_Z1gv() nounwind
// NO-TSS-NOT: call i32 @__cxa_guard_acquire
// NO-TSS-NOT: call void @__cxa_guard_release
// NO-TSS: ret void
