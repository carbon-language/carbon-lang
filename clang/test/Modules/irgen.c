// RUN: %clang_cc1 -emit-module -triple x86_64-apple-darwin10 -o %t/module.pcm -DBUILD_MODULE %s
// RUN: %clang_cc1 -fmodule-cache-path %t -triple x86_64-apple-darwin10 -fdisable-module-hash -emit-llvm -o - %s | FileCheck %s

#ifdef BUILD_MODULE
static inline int triple(int x) { return x * 3; }
#else
__import_module__ module;

// CHECK: define void @triple_value
void triple_value(int *px) {
  *px = triple(*px);
}

// CHECK: define internal i32 @triple(i32
#endif
