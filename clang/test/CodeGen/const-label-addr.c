// RUN: %clang_cc1 %s -emit-llvm -o %t
int a() {
A:;static void* a = &&A;
}
