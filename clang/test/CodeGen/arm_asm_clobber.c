// RUN: clang-cc -triple armv6-unknown-unknown -emit-llvm -o %t %s

void test0(void) {
	asm volatile("mov r0, r0" :: );
}
void test1(void) {
	asm volatile("mov r0, r0" :::
				 "cc", "memory" );
}
void test2(void) {
	asm volatile("mov r0, r0" :::
				 "r0", "r1", "r2", "r3");
	asm volatile("mov r0, r0" :::
				 "r4", "r5", "r6", "r8");
}
void test3(void) {
	asm volatile("mov r0, r0" :::
				 "a1", "a2", "a3", "a4");
	asm volatile("mov r0, r0" :::
				 "v1", "v2", "v3", "v5");
}
