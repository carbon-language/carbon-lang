// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -emit-llvm -o - %s | FileCheck %s
// pr8707

// CHECK: @__block_global_0.test = internal global i32
int (^block)(void) = ^ {
	static int test=0;
	return test;
};
// CHECK: @__block_global_1.test = internal global i32
void (^block1)(void) = ^ {
	static int test = 2;
	return;
};
// CHECK: @__block_global_2.test = internal global i32
int (^block2)(void) = ^ {
	static int test = 5;
	return test;
};

