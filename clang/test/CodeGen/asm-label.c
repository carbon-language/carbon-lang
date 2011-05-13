// RUN: %clang_cc1 -triple=i686-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm %s -o - | FileCheck %s --check-prefix=DARWIN

char *strerror(int) asm("alias");

void test(void)
{
	strerror(-1);
}

// LINUX: declare i8* @alias(i32)
// DARWIN: declare i8* @"\01alias"(i32)
