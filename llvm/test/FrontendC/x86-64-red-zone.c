// RUN: $llvmgcc -m64 -fomit-frame-pointer -O2 %s -S -o - > %t
// RUN: not grep subq %t
// RUN: not grep addq %t
// RUN: grep {\\-4(%%rsp)} %t | count 2
// RUN: $llvmgcc -m64 -fomit-frame-pointer -O2 %s -S -o - -mno-red-zone > %t
// RUN: grep subq %t | count 1
// RUN: grep addq %t | count 1
// This is a test for x86-64, add your target below if it FAILs.
// XFAIL: alpha|ia64|arm|powerpc|sparc|x86

long double f0(float f) { return f; }
