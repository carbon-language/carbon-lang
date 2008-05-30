// RUN: clang -emit-llvm < %s -triple=i686-apple-darwin9 | grep "global \[1 x i32\]"

int r[];
int (*a)[] = &r;
