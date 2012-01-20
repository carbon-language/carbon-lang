// RUN: %clang -target mips64el-unknown-linux -ccc-clang-archs mips64el -O3 -S -mabi=n64 -o - -emit-llvm %s | FileCheck %s

typedef long double LD;

// CHECK: ret fp128

LD foo0() {
  return 2.625L;
}
