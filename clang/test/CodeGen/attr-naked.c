// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -o - | FileCheck %s

void t1() __attribute__((naked));

// Basic functionality check
// (Note that naked needs to imply noinline to work properly.)
// CHECK: define void @t1() naked noinline nounwind {
void t1()
{
}

// Make sure this doesn't explode in the verifier.
// (It doesn't really make sense, but it isn't invalid.)
// CHECK: define void @t2() naked noinline nounwind {
__attribute((naked, always_inline)) void t2()  {
}
