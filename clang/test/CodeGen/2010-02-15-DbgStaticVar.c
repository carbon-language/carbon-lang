// RUN: %clang_cc1 -g -emit-llvm %s -o - | FileCheck %s
// Test to check intentionally empty linkage name for a static variable.
// Radar 7651244.
static int foo(int a)
{
	static int b = 1;
	return b+a;
}

int main() {
	int j = foo(1);
	return 0;
}
// CHECK: !DIGlobalVariable(name: "b",
// CHECK-NOT:               linkageName:
// CHECK-SAME:              ){{$}}
