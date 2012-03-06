// Make sure -fno-inline is behaving correctly.
// rdar://10972766

// RUN: %clang_cc1 -O3 -fno-inline -emit-llvm %s -o - | FileCheck -check-prefix=NOINLINE %s

int dont_inline_me(int a, int b) { return(a+b); }

volatile int *pa = (int*) 0x1000;
void foo() {
// NOINLINE: @foo
// NOINLINE: dont_inline_me
    pa[0] = dont_inline_me(pa[1],pa[2]);	
}

