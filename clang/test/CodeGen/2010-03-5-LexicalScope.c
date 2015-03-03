// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// CHECK: !MDLexicalBlock(
// CHECK: !MDLexicalBlock(
int foo(int i) {
	if (i) {
		int j = 2;
	}
	else {
		int j = 3;
	}
	return i;
}
