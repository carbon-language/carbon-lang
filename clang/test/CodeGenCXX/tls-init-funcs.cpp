// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -std=c++11 -S -emit-llvm %s -o - | FileCheck -check-prefix=BITCODE %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -std=c++11 -S %s -o - | FileCheck -check-prefix=ASM %s

// BITCODE: @llvm.tls_init_funcs = appending global [1 x void ()*] [void ()* @__tls_init]

struct A {
  A();
};

struct B {
  int i;
  B(int i);
};

thread_local int i = 37;
thread_local A a;
thread_local B b(927);

// ASM: .section __DATA,__thread_init,thread_local_init_function_pointers
// ASM: .align 3
// ASM: .quad ___tls_init
