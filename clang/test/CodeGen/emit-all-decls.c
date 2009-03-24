// RUN: clang-cc -emit-llvm -o %t %s &&
// RUN: not grep "@foo" %t &&
// RUN: clang-cc -femit-all-decls -emit-llvm -o %t %s &&
// RUN: grep "@foo" %t

static void foo() {
  
}
