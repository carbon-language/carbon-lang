// RUN: clang %s -emit-llvm -o - | grep "i32 @a(i32)"

int a();
int a(x) short x; {return x;}

