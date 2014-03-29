// RUN: %clang_cc1 -triple arm64-apple-ios -O3 -emit-llvm -o - %s | FileCheck %s

void f0(void *a, void *b) {
	__clear_cache(a,b);
// CHECK: call {{.*}} @__clear_cache
}
