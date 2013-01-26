// RUN: %clang_cc1 -emit-llvm -O0 -g %s -o - | FileCheck %s
// CHECK: DW_TAG_lexical_block
// CHECK: DW_TAG_lexical_block
int foo(int i) {
	if (i) {
		int j = 2;
	}
	else {
		int j = 3;
	}
	return i;
}
