// RUN: clang-cc %s -emit-llvm -o %t
int a() {
A:;static void* a = &&A;
}
