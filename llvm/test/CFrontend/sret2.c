// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep sret | count 2

struct abc {
 int a;
 int b;
 int c;
};
 
struct abc foo2(){}
