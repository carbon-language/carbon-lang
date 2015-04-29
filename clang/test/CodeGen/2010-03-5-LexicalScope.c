// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// CHECK: !DILexicalBlock(
// CHECK: !DILexicalBlock(
int foo(int i) {
	if (i) {
		int j = 2;
	}
	else {
		int j = 3;
	}
	return i;
}
