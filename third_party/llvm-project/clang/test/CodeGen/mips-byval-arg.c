// RUN: %clang_cc1 -triple mipsel-unknown-linux -O3 -S -o - -emit-llvm %s | FileCheck %s -check-prefix=O32
// RUN: %clang_cc1 -triple mips64el-unknown-linux -O3 -S -target-abi n64 -o - -emit-llvm %s | FileCheck %s -check-prefix=N64

typedef struct {
  float f[3];
} S0;

extern void foo2(S0);

// O32-LABEL: define{{.*}} void @foo1(i32 inreg %a0.coerce0, i32 inreg %a0.coerce1, i32 inreg %a0.coerce2)
// N64-LABEL: define{{.*}} void @foo1(i64 inreg %a0.coerce0, i32 inreg %a0.coerce1)

void foo1(S0 a0) {
  foo2(a0);
}
