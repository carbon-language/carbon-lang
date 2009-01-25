// RUN: clang %s -emit-llvm -o %t
int a() {
A:;static void* a = &&A;
}
