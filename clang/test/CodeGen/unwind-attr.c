// RUN: clang-cc -fexceptions -emit-llvm -o - %s | grep "@foo() {" | count 1 &&
// RUN: clang-cc -emit-llvm -o - %s | grep "@foo() nounwind {" | count 1

int foo(void) {
}
