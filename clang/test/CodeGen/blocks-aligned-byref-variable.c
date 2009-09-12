// RUN: clang-cc -emit-llvm -o - 
typedef int __attribute__((aligned(32)))  ai;

void f() {
  __block ai a = 10;

  ^{
    a = 20;
  }();
}

