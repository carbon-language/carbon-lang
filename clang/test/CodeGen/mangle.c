// RUN: clang -arch i386 -emit-llvm -o %t %s &&
// RUN: grep '@_Z2f0i' %t &&
// RUN: grep '@_Z2f0l' %t

// Make sure we mangle overloadable, even in C system headers.

# 1 "somesystemheader.h" 1 3 4
void __attribute__((__overloadable__)) f0(int a) {}
void __attribute__((__overloadable__)) f0(long b) {}
