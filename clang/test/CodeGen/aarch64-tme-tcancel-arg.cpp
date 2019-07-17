// RUN: %clang_cc1 -triple aarch64-eabi -target-feature +tme -S -emit-llvm %s -o - | FileCheck %s

#define A -1
constexpr int f() { return 65536; }

void t_cancel() {
	__builtin_arm_tcancel(f() + A);
}

// CHECK: call void @llvm.aarch64.tcancel(i64 65535)
