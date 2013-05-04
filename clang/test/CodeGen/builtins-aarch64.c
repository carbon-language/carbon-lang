// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -O3 -emit-llvm -o - %s | FileCheck %s

void f0(char *a, char *b) {
	__clear_cache(a,b);
// CHECK: call {{.*}} @__clear_cache
}
