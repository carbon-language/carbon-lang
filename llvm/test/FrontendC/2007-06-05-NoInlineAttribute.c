// RUN: %llvmgcc -c -emit-llvm %s -o - | llvm-dis | grep llvm.noinline 

static int bar(int x, int y) __attribute__((noinline));

static int bar(int x, int y)  
{
 return x + y;
}

int foo(int a, int b) {
 return  bar(b, a);
}

