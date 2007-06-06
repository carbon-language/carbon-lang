// RUN: %llvmgxx -c -emit-llvm %s -o - | llvm-dis | grep llvm.noinline 

int bar(int x, int y); __attribute__((noinline))

int bar(int x, int y)  
{
 return x + y;
}

int foo(int a, int b) {
 return  bar(b, a);
}

